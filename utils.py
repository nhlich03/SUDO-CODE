import json
import kagglehub
import pandas as pd
from pathlib import Path
from typing import Set
from config import DATASET_ID, DATASET_FILE, STOPWORDS_TXT

def download_dataset_kaggle() -> Path:
    path = Path(kagglehub.dataset_download(DATASET_ID))
    return path

def load_json_as_df(dataset_dir: Path, json_filename: str) -> pd.DataFrame:
    json_path = dataset_dir / json_filename
    with open(json_path, "r", encoding="utf-8") as f:
        _ = json.load(f) 
    df = pd.read_json(json_path)
    return df

def load_stopwords_txt(path: str = STOPWORDS_TXT) -> Set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    words: Set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return words

def load_data_from_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["content"])
    return df

def save_dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def output_results(df: pd.DataFrame, title: str, relpath: str, save_flag: bool, print_flag: bool):
    """
    Save results to CSV and/or print to console.
    """
    if save_flag:
        out_path = Path("results") / relpath
        save_dataframe_to_csv(df, out_path)
    if print_flag:
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))