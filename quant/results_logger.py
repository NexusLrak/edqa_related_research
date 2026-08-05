"""
results_logger.py
=================================================

Structured result logging for run_experiments.py / run_experiments_tuned.py --
separate from the free-text console logs in quant/experiment_logs/*.log (those
are still useful as a narrative record, but aren't good for citing exact
numbers or feeding a plot).

Three outputs, all under quant/experiment_logs/:
  * results.csv  -- one row per accuracy measurement, TIDY/long format, append-
                    only across every run ever made. Load with pandas for
                    dissertation figures: df[df.section=="figure3"].plot(...).
  * timing.csv   -- one row per timed stage (ranking, table2 eval, etc), same
                    append-only long format. Previously this only existed as
                    scattered, inconsistently-formatted print() statements in
                    quant/experiment_logs/*.log -- not queryable. Added
                    2026-07-18; earlier runs' timings are NOT backfilled here,
                    only what's in the .log files' free text.
  * {run_id}.json -- one full structured snapshot per run (every number, plus
                    the config that produced it), for precise citation.

Schema (results.csv):
  timestamp, run_id, experiment, pipeline, variant, ranking, seed, eval_batches,
  section, method, bits, r, m, accuracy, std

  * pipeline   : "paper" (run_experiments.py) | "tuned" (run_experiments_tuned.py)
  * variant    : clip_percentile variant name, "n/a" for the paper pipeline
  * ranking    : "greedy" | "surrogate"
  * eval_batches: how many test batches the accuracy was measured over -- matters
                  a lot (see the 2026-07 finding: 8-batch subset gave 69.34%,
                  full 79-batch set gave 63.20% for the same config). ALWAYS
                  check this column before comparing numbers across runs.
  * section    : "table2" | "figure3" | "figure4"
  * method     : "direct" | "pot" | "noisyquant" | "edqa" (figure3/4 are eDQA-only)
  * bits, r, m : whichever of these varies is the x-axis for that section;
                 the other two are held fixed at that row's value.

Schema (timing.csv):
  timestamp, run_id, experiment, pipeline, variant, ranking, seed, stage, seconds

  * stage      : e.g. "ranking", "table2", "figure3", "figure4" -- whatever
                 label the caller passes to log_timing() / the `timed()`
                 context manager.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import time
import uuid
from dataclasses import dataclass, field

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_logs")
CSV_PATH = os.path.join(LOG_DIR, "results.csv")
CSV_FIELDS = [
    "timestamp", "run_id", "experiment", "pipeline", "variant", "ranking", "seed",
    "eval_batches", "section", "method", "bits", "r", "m", "accuracy", "std",
]
TIMING_CSV_PATH = os.path.join(LOG_DIR, "timing.csv")
TIMING_CSV_FIELDS = [
    "timestamp", "run_id", "experiment", "pipeline", "variant", "ranking", "seed", "stage", "seconds",
]


@dataclass
class RunContext:
    """Carries the config shared by every row logged in one run() call."""
    experiment: str
    pipeline: str          # "paper" | "tuned"
    variant: str = "n/a"
    ranking: str = "greedy"
    seed: int = 0
    eval_batches: "int | None" = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    _rows: list = field(default_factory=list)
    _timing_rows: list = field(default_factory=list)

    def log_timing(self, stage: str, seconds: float):
        """Record how long one stage (e.g. "ranking", "table2") took."""
        self._timing_rows.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "experiment": self.experiment,
            "pipeline": self.pipeline,
            "variant": self.variant,
            "ranking": self.ranking,
            "seed": self.seed,
            "stage": stage,
            "seconds": round(seconds, 2),
        })
        print(f"  [timing] {stage}: {seconds:.1f}s")

    @contextlib.contextmanager
    def timed(self, stage: str):
        """Usage: with ctx.timed('ranking'): ranks = rank_channels(...)"""
        t0 = time.time()
        yield
        self.log_timing(stage, time.time() - t0)

    def _base_row(self, section: str) -> dict:
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "experiment": self.experiment,
            "pipeline": self.pipeline,
            "variant": self.variant,
            "ranking": self.ranking,
            "seed": self.seed,
            "eval_batches": self.eval_batches if self.eval_batches is not None else "full",
            "section": section,
        }

    def log_table2(self, table2: dict):
        """table2: {method: {bits: (mean_acc, std_acc)}} -- compare_methods' return value."""
        for method, by_bits in table2.items():
            for bits, (acc, std) in by_bits.items():
                row = self._base_row("table2")
                row.update(method=method, bits=bits, r="", m="", accuracy=acc, std=std)
                self._rows.append(row)

    def log_figure3(self, fig3: dict, m: int = 3):
        """fig3: {r: accuracy} -- sweep_ratio's return value."""
        for r, acc in fig3.items():
            row = self._base_row("figure3")
            row.update(method="edqa", bits=3, r=round(r, 2), m=m, accuracy=acc, std="")
            self._rows.append(row)

    def log_figure4(self, fig4: dict, r: float):
        """fig4: {m: accuracy} -- sweep_extra_bits' return value."""
        for m, acc in fig4.items():
            row = self._base_row("figure4")
            row.update(method="edqa", bits=3, r=round(r, 2), m=m, accuracy=acc, std="")
            self._rows.append(row)

    def flush(self):
        """Append all rows collected so far to the shared CSVs, and write this
        run's full JSON snapshot. Call once at the end of run()."""
        os.makedirs(LOG_DIR, exist_ok=True)

        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                w.writeheader()
            for row in self._rows:
                w.writerow(row)

        if self._timing_rows:
            write_timing_header = not os.path.exists(TIMING_CSV_PATH)
            with open(TIMING_CSV_PATH, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TIMING_CSV_FIELDS)
                if write_timing_header:
                    w.writeheader()
                for row in self._timing_rows:
                    w.writerow(row)

        json_path = os.path.join(LOG_DIR, f"{self.run_id}_{self.experiment}_{self.pipeline}.json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "experiment": self.experiment,
                    "pipeline": self.pipeline,
                    "variant": self.variant,
                    "ranking": self.ranking,
                    "seed": self.seed,
                    "eval_batches": self.eval_batches if self.eval_batches is not None else "full",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "rows": self._rows,
                    "timing": self._timing_rows,
                },
                f,
                indent=2,
            )
        print(f"\n[results_logger] appended {len(self._rows)} rows to {CSV_PATH}")
        print(f"[results_logger] full snapshot: {json_path}")
