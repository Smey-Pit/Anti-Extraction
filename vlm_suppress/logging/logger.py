"""
Structured JSON run logger.

Every run produces:
  outputs/{run_id}/
    config.json         — full serialised ExperimentConfig
    results.jsonl       — one JSON object per (image, model, epsilon, kappa) tuple
    trajectories/       — one JSON per (image, epsilon, kappa) with full step log
    images/             — diff heatmaps, trajectory plots, comparison strips

Design principle: log everything from day one.
If you need a field later, it should already be in the log.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


class RunLogger:
    def __init__(self, cfg) -> None:
        """
        cfg: ExperimentConfig
        Creates the output directory and writes config.json immediately.
        """
        self.run_id = cfg.run_id
        self.run_dir = Path(cfg.log.output_dir) / cfg.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.img_dir   = self.run_dir / "images"
        self.traj_dir  = self.run_dir / "trajectories"
        self.img_dir.mkdir(exist_ok=True)
        self.traj_dir.mkdir(exist_ok=True)

        self._results_path = self.run_dir / "results.jsonl"
        self._results_fh   = self._results_path.open("a")

        # Serialise config immediately — before any results
        cfg_dict = _serialise(asdict(cfg))
        (self.run_dir / "config.json").write_text(
            json.dumps(cfg_dict, indent=2)
        )

        self._start_time = time.time()
        self.log_event("run_started", {"run_id": self.run_id})

    def log_result(self, record: dict[str, Any]) -> None:
        """
        Append one result record to results.jsonl.
        record should contain at minimum:
          image_id, domain, lexical_difficulty, model_name, epsilon, kappa,
          objective_config, proxy_stage, cer_clean, cer_adv, wer_clean, wer_adv,
          cer_delta, constraint_satisfied, fm_ens_final, lh_final
        """
        record["_timestamp"] = time.time()
        self._results_fh.write(json.dumps(_serialise(record)) + "\n")
        self._results_fh.flush()

    def log_trajectory(
        self,
        image_id: str,
        epsilon: float,
        kappa: float,
        trajectory: list,  # list[StepLog]
    ) -> None:
        """Saves the full PGD step-log for one attack run."""
        fname = f"{image_id}_eps{epsilon:.5f}_kappa{kappa:.5f}.json"
        traj_data = {
            "image_id": image_id,
            "epsilon": epsilon,
            "kappa": kappa,
            "steps": [dataclasses.asdict(s) for s in trajectory],
        }
        (self.traj_dir / fname).write_text(json.dumps(traj_data, indent=2))

    def log_event(self, event: str, data: dict = {}) -> None:
        """Log a free-form event (run_started, phase_complete, error, etc.)."""
        record = {"_event": event, "_timestamp": time.time(), **data}
        self._results_fh.write(json.dumps(record) + "\n")
        self._results_fh.flush()

    def image_path(self, name: str) -> Path:
        return self.img_dir / name

    def trajectory_path(self, name: str) -> Path:
        return self.traj_dir / name

    def close(self) -> None:
        elapsed = time.time() - self._start_time
        self.log_event("run_finished", {"elapsed_seconds": elapsed})
        self._results_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _serialise(obj: Any) -> Any:
    """Recursively make a dict/list JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj