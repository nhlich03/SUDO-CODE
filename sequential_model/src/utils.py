import os
import json
import random
from typing import Any, Dict

import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file into dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def select_device(name: str = "auto") -> torch.device:
    """Select computation device."""
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto mode
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict, path: str) -> None:
    """Save dictionary to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
