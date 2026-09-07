"""
Reproduction skeleton of the eDQA method. Module map:

  quantizer.py      symmetric quant, right-shift, shifting-error table
  edqa_layer.py     eDQA layer quantize / de-quantize
  ranking.py        greedy channel-importance ranking (IQ search)
  compression.py    Huffman / Deflate / LZMA / ZSTD
  hooks.py          forward-hook activation-quantization manager
  data.py           CIFAR-10 / TinyImageNet + calibration subset
  baselines.py      Direct / PoT / NoisyQuant
  evaluate.py       Table 2, Figure 3/4, Table 1
"""

from .baselines import fake_quantize_noisyquant, fake_quantize_pot
from .compression import get_compressor
from .edqa_layer import edqa_dequantize_layer, edqa_fake_quantize, edqa_quantize_layer
from .hooks import QuantManager, default_target_layers
from .quantizer import (
    compute_scale,
    dequantize_direct,
    fake_quantize_direct,
    quantize_direct,
    right_shift_with_error,
    shifting_error_table,
)
from .ranking import evaluate_accuracy, load_ranks, rank_channels, save_ranks

__all__ = [
    "compute_scale",
    "quantize_direct",
    "dequantize_direct",
    "right_shift_with_error",
    "shifting_error_table",
    "fake_quantize_direct",
    "edqa_quantize_layer",
    "edqa_dequantize_layer",
    "edqa_fake_quantize",
    "rank_channels",
    "evaluate_accuracy",
    "save_ranks",
    "load_ranks",
    "get_compressor",
    "QuantManager",
    "default_target_layers",
    "fake_quantize_pot",
    "fake_quantize_noisyquant",
]

__version__ = "0.1.0"
