from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Iterable, List

def load_texts(path: str | Path, text_column: str = "text") -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]