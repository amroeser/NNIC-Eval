import random
from pathlib import Path

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dirs(base_dir: str, experiment_name: str) -> dict:
    base = Path(base_dir)
    results_dir = base / "results" / experiment_name
    plots_dir = base / "plots" / experiment_name
    ic_rules_dir = base / "ic_rules" / experiment_name
    logs_dir = base / "logs" / experiment_name
    for directory in (results_dir, plots_dir, ic_rules_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "results": results_dir,
        "plots": plots_dir,
        "ic_rules": ic_rules_dir,
        "logs": logs_dir,
    }
