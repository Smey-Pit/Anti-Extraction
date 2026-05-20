"""
vlm_suppress/watermark/steplog.py

Minimal JSONL logger for the targeted PGD loop.

One JSON record per line, flushed immediately so partial runs are readable.
The first record (step=0) is a manifest carrying run-level metadata.

Usage
-----
    from vlm_suppress.watermark.steplog import StepLogger

    log = StepLogger(
        path        = Path("runs/targeted_pgd/banking_0000.jsonl"),
        image_id    = "banking_0000",
        source_word = "Thompson",
        target_word = "Henderson",
        surrogate   = "qwen2_5vl",
        n_steps     = 200,
        epsilon     = 0.03,
        alpha_pgd   = 0.003,
    )

    for step in range(n_steps):
        loss = ...   # scalar float or torch.Tensor
        # optional eval every N steps
        outcome    = "clean_read"   # or None if not evaluated this step
        transcript = "..."          # or None
        log.write(step=step, loss=loss, outcome=outcome, transcript=transcript)

    log.close()

Record schema
-------------
Manifest (step == 0):
    {
        "step": 0,
        "type": "manifest",
        "image_id": str,
        "source_word": str,
        "target_word": str,
        "surrogate": str,
        "n_steps": int,
        "epsilon": float,
        "alpha_pgd": float,
        "ghost_alpha": float | null,
        "timestamp": str,          # ISO-8601 UTC
        **extra_kwargs              # any additional metadata passed to __init__
    }

Step records:
    {
        "step": int,
        "type": "step",
        "loss": float,
        "outcome": str | null,     # null when no eval this step
        "transcript": str | null   # null when no eval this step
    }
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StepLogger:
    """Write one JSONL record per PGD step, plus a manifest at step 0."""

    def __init__(
        self,
        path: Path | str,
        image_id: str,
        source_word: str,
        target_word: str,
        surrogate: str,
        n_steps: int,
        epsilon: float,
        alpha_pgd: float,
        ghost_alpha: float | None = None,
        **extra_kwargs: Any,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", buffering=1)   # line-buffered

        manifest = {
            "step":        0,
            "type":        "manifest",
            "image_id":    image_id,
            "source_word": source_word,
            "target_word": target_word,
            "surrogate":   surrogate,
            "n_steps":     n_steps,
            "epsilon":     epsilon,
            "alpha_pgd":   alpha_pgd,
            "ghost_alpha": ghost_alpha,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            **extra_kwargs,
        }
        self._write_record(manifest)

    def write(
        self,
        step: int,
        loss: float | Any,
        outcome: str | None = None,
        transcript: str | None = None,
    ) -> None:
        """Append one step record. Accepts torch.Tensor for loss (calls .item())."""
        try:
            loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
        except Exception:
            loss_val = float("nan")

        record = {
            "step":       step,
            "type":       "step",
            "loss":       loss_val,
            "outcome":    outcome,
            "transcript": transcript,
        }
        self._write_record(record)

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "StepLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _write_record(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def path(self) -> Path:
        return self._path
