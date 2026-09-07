"""
compression.py
=================================================

Compress the shifting errors (the lower-m-bit integer codes) to cut their
storage overhead. Reproduces the four techniques compared in the paper's
Table 1: Huffman, Deflate, LZMA, ZSTD.

Interface (unified):
    comp = get_compressor("huffman")
    payload = comp.encode(codes)          # codes: 1-D uint8 numpy array (0..2^m-1)
    codes2  = comp.decode(payload)        # -> 1-D uint8 numpy array (same length)
    ratio   = comp.ratio(codes, payload)  # original_bytes / compressed_bytes

Notes:
  * Huffman is implemented from scratch (heapq + real bit-packing) so the package
    is dependency-light and the compression ratio reflects genuine bit savings.
  * Deflate -> zlib, LZMA -> lzma (both stdlib).
  * The paper runs compression/decompression on CPU while inference runs on GPU,
    so latency should be timed on CPU (see evaluate.py).
"""

from __future__ import annotations

import heapq
import lzma
import pickle
import zlib
from abc import ABC, abstractmethod

import numpy as np


class Compressor(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, codes: np.ndarray) -> bytes: ...

    @abstractmethod
    def decode(self, payload: bytes) -> np.ndarray: ...

    @staticmethod
    def _to_bytes(codes: np.ndarray) -> bytes:
        # shifting-error codes fit in a byte (2^m <= 32 for N<=5)
        return np.asarray(codes, dtype=np.uint8).tobytes()

    @staticmethod
    def _from_bytes(raw: bytes) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def ratio(self, codes: np.ndarray, payload: bytes) -> float:
        original = np.asarray(codes, dtype=np.uint8).nbytes
        return original / max(1, len(payload))


# --------------------------------------------------------------------------- #
# Huffman (self-contained)
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("freq", "sym", "left", "right")

    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq, self.sym, self.left, self.right = freq, sym, left, right

    def __lt__(self, other):
        return self.freq < other.freq


def _build_codebook(freqs: dict[int, int]) -> dict[int, str]:
    if len(freqs) == 1:                       # single-symbol edge case -> 1 bit
        (sym,) = freqs
        return {sym: "0"}
    heap = [_Node(f, s) for s, f in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, _Node(a.freq + b.freq, None, a, b))
    root = heap[0]
    codes: dict[int, str] = {}

    def walk(node, prefix):
        if node.sym is not None:
            codes[node.sym] = prefix or "0"
            return
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")

    walk(root, "")
    return codes


class HuffmanCompressor(Compressor):
    name = "huffman"

    def encode(self, codes: np.ndarray) -> bytes:
        arr = np.asarray(codes, dtype=np.uint8)
        n = int(arr.size)
        syms, counts = np.unique(arr, return_counts=True)
        freqs = {int(s): int(c) for s, c in zip(syms, counts)}
        codebook = _build_codebook(freqs)
        bitstring = "".join(codebook[int(v)] for v in arr)
        pad = (-len(bitstring)) % 8
        bitstring += "0" * pad
        packed = int(bitstring, 2).to_bytes(len(bitstring) // 8, "big") if bitstring else b""
        payload = {"n": n, "cb": codebook, "pad": pad, "bits": packed}
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def decode(self, payload: bytes) -> np.ndarray:
        obj = pickle.loads(payload)
        n, codebook, pad, packed = obj["n"], obj["cb"], obj["pad"], obj["bits"]
        if n == 0:
            return np.zeros(0, dtype=np.uint8)
        total_bits = len(packed) * 8
        bitstring = bin(int.from_bytes(packed, "big"))[2:].zfill(total_bits) if packed else ""
        if pad:
            bitstring = bitstring[: len(bitstring) - pad]
        inv = {code: sym for sym, code in codebook.items()}
        out = np.empty(n, dtype=np.uint8)
        i, cur, idx = 0, "", 0
        for ch in bitstring:
            cur += ch
            if cur in inv:
                out[idx] = inv[cur]
                idx += 1
                cur = ""
            if idx == n:
                break
        return out


# --------------------------------------------------------------------------- #
# Deflate / LZMA (stdlib) and ZSTD (optional)
# --------------------------------------------------------------------------- #
class DeflateCompressor(Compressor):
    name = "deflate"

    def encode(self, codes: np.ndarray) -> bytes:
        return len(codes).to_bytes(8, "big") + zlib.compress(self._to_bytes(codes), level=9)

    def decode(self, payload: bytes) -> np.ndarray:
        n = int.from_bytes(payload[:8], "big")
        return self._from_bytes(zlib.decompress(payload[8:]))[:n]


class LZMACompressor(Compressor):
    name = "lzma"

    def encode(self, codes: np.ndarray) -> bytes:
        return len(codes).to_bytes(8, "big") + lzma.compress(self._to_bytes(codes))

    def decode(self, payload: bytes) -> np.ndarray:
        n = int.from_bytes(payload[:8], "big")
        return self._from_bytes(lzma.decompress(payload[8:]))[:n]


class ZstdCompressor(Compressor):
    name = "zstd"

    def __init__(self, level: int = 19):
        try:
            import zstandard  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ZSTD requires the `zstandard` package: pip install zstandard"
            ) from e
        import zstandard
        self._c = zstandard.ZstdCompressor(level=level)
        self._d = zstandard.ZstdDecompressor()

    def encode(self, codes: np.ndarray) -> bytes:
        return len(codes).to_bytes(8, "big") + self._c.compress(self._to_bytes(codes))

    def decode(self, payload: bytes) -> np.ndarray:
        n = int.from_bytes(payload[:8], "big")
        return self._from_bytes(self._d.decompress(payload[8:]))[:n]


_REGISTRY = {
    "huffman": HuffmanCompressor,
    "deflate": DeflateCompressor,
    "lzma": LZMACompressor,
    "zstd": ZstdCompressor,
}


def get_compressor(name: str, **kwargs) -> Compressor:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown compressor '{name}', choose from {list(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


class IdentityCompressor(Compressor):
    """No-op passthrough — useful for debugging the eDQA path without compression."""

    name = "identity"

    def encode(self, codes: np.ndarray) -> bytes:
        return len(codes).to_bytes(8, "big") + self._to_bytes(codes)

    def decode(self, payload: bytes) -> np.ndarray:
        n = int.from_bytes(payload[:8], "big")
        return self._from_bytes(payload[8:])[:n]


_REGISTRY["identity"] = IdentityCompressor
