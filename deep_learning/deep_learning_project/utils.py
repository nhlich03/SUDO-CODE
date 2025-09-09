import os
import random
import logging
import time
from typing import Tuple

import numpy as np
import torch

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove existing handlers (for re-entry in notebooks)
    logger.handlers = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}-{ts}.log"))
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import torch
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None) -> dict:
    import torch
    return torch.load(path, map_location=map_location)
