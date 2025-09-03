
import os, json, yaml
import joblib

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def save_model(model, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
