
import argparse, json
from .utils import load_yaml
from .train import run_training

def build_parser():
    p = argparse.ArgumentParser(description="Train SVM or Naive Bayes for sentiment classification")
    p.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")

    # Optional overrides
    p.add_argument("--train_path", type=str)
    p.add_argument("--test_path", type=str)
    p.add_argument("--text_col", type=str)
    p.add_argument("--label_col", type=str)
    p.add_argument("--model", type=str, choices=["svm", "nb"])
    p.add_argument("--ngram_min", type=int)
    p.add_argument("--ngram_max", type=int)
    p.add_argument("--min_df", type=int)
    p.add_argument("--max_df", type=float)
    p.add_argument("--artifacts_dir", type=str)
    p.add_argument("--outputs_dir", type=str)
    p.add_argument("--model_name", type=str)
    p.add_argument("--replace_list_path", type=str)
    return p

def apply_overrides(cfg: dict, args):
    # Top-level overrides
    for key in ["train_path", "test_path", "text_col", "label_col", "model",
                "artifacts_dir", "outputs_dir", "model_name", "replace_list_path"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    # tfidf overrides
    tfidf = cfg.get("tfidf", {}) or {}
    for key in ["ngram_min", "ngram_max", "min_df", "max_df"]:
        val = getattr(args, key)
        if val is not None:
            tfidf[key] = val
    cfg["tfidf"] = tfidf
    return cfg

def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    cfg = apply_overrides(cfg, args)

    res = run_training(cfg)
    print(json.dumps({
        "accuracy": res["accuracy"],
        "model_path": res["model_path"],
        "predictions_csv": res["predictions_csv"],
        "metrics_csv": res.get("metrics_csv"),
        "metrics_json": res.get("metrics_json"),
        "report": res.get("report"), 
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
