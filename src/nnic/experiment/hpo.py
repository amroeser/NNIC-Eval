from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import itertools
import json

from nnic.experiment.runner import run_experiment_from_config


@dataclass
class HPOResult:
    config: Dict[str, Any]
    summary: Dict[str, Any]


def _iter_grid(search_space: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(search_space.keys())
    values = [list(v) for v in search_space.values()]
    for combo in itertools.product(*values):
        yield {k: v for k, v in zip(keys, combo)}


def grid_search(
    base_config: Dict[str, Any],
    search_space: Dict[str, Iterable[Any]],
    metric_path: Tuple[str, ...] = ("metrics_val", "accuracy"),
    save_path: str | None = None,
) -> List[HPOResult]:
    """Einfache Grid-Search über einen diskreten Suchraum.

    Für jede Kombination im `search_space` wird eine Kopie der `base_config` erzeugt,
    mit den entsprechenden Hyperparametern überschrieben und anschließend ein
    Experiment ausgeführt. Das Ergebnis wird gemeinsam mit der verwendeten Config
    zurückgegeben, sortiert nach dem Zielmetrikwert (absteigend).
    """
    results: List[HPOResult] = []
    for cfg_patch in _iter_grid(search_space):
        cfg = dict(base_config)
        # flache Überschreibung: keys wie "training.learning_rate" erlauben
        for key, value in cfg_patch.items():
            if "." in key:
                top, sub = key.split(".", 1)
                sub_cfg = dict(cfg.get(top, {}))
                sub_cfg[sub] = value
                cfg[top] = sub_cfg
            else:
                cfg[key] = value

        summary = run_experiment_from_config(cfg)
        results.append(HPOResult(config=cfg, summary=summary))

    def _score(res: HPOResult) -> float:
        d: Any = res.summary
        for part in metric_path:
            d = d.get(part, None) if isinstance(d, dict) else None
            if d is None:
                return float("-inf")
        try:
            return float(d)
        except Exception:
            return float("-inf")

    results.sort(key=_score, reverse=True)

    if save_path is not None:
        try:
            path = Path(save_path)
            payload = [{"config": r.config, "summary": r.summary} for r in results]
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    return results
