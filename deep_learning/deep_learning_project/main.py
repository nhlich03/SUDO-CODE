import argparse
import os
import yaml
import torch
import torch.nn as nn

from utils import seed_all, get_logger
from data import build_loaders_and_assets
from model import TextClassifier
from train import Trainer
from eval import evaluate as eval_fn

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cmd_train(cfg):
    seed_all(cfg["seed"])
    logger = get_logger(os.path.join("outputs", "logs"), name="train")

    train_loader, val_loader, test_loader, assets = build_loaders_and_assets(cfg, logger)
    vocab = assets["vocab"]
    num_classes = assets["num_classes"]

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = TextClassifier(
        vocab_size=len(vocab.itos),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=num_classes,
        pad_idx=cfg["data"]["PAD"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    trainer = Trainer(model, criterion, optimizer, device=device, logger=logger)
    trainer.fit(train_loader, val_loader, epochs=cfg["train"]["epochs"],
                save_path=os.path.join("outputs", "checkpoints", "best.pth"),
                patience=cfg["train"]["patience"])
    logger.info(f"Training done. Best val_acc={trainer.best_val_acc:.4f}")

def cmd_eval(cfg):
    seed_all(cfg["seed"])
    logger = get_logger(os.path.join("outputs", "logs"), name="eval")
    acc = eval_fn(cfg, logger)
    logger.info(f"Eval done. Test acc={acc:.2f}")

def cmd_infer(cfg, input_csv: str = None):
    # Minimal CSV inference: expects a CSV with the same text_col as in config.
    import pandas as pd
    seed_all(cfg["seed"])
    logger = get_logger(os.path.join("outputs", "logs"), name="infer")

    if input_csv is None:
        input_csv = cfg.get("infer", {}).get("input_csv", None)
    assert input_csv is not None, "Please provide --input or set infer.input_csv in config."

    # Rebuild assets (vocab) and model
    # We don't need labels here, so we fake test split by reading only CSV and not using labels.
    # For simplicity, reuse build_loaders_and_assets and ignore returned loaders.
    train_loader, val_loader, test_loader, assets = build_loaders_and_assets(cfg, logger)
    vocab = assets["vocab"]
    num_classes = assets["num_classes"]
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = TextClassifier(
        vocab_size=len(vocab.itos),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=num_classes,
        pad_idx=cfg["data"]["PAD"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    ckpt = os.path.join("outputs", "checkpoints", "best.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint {ckpt}")
    model.eval()

    # Read input CSV and preprocess same way
    from data import Preprocessing
    pre = Preprocessing(cfg["data"]["stopwords_path"])
    raw_col = cfg["data"]["raw_text_col"]
    text_col = cfg["data"]["text_col"]
    df = pd.read_csv(input_csv)
    if text_col not in df.columns and raw_col in df.columns:
        df[text_col] = df[raw_col].astype(str).apply(pre.preprocessing_text)
    elif text_col not in df.columns:
        raise ValueError(f"Input CSV must contain either '{text_col}' or '{raw_col}'.")

    PAD = cfg["data"]["PAD"]
    from torch.nn.utils.rnn import pad_sequence
    import torch

    results = []
    for idx, row in df.iterrows():
        ids = torch.tensor(vocab.numericalize(str(row[text_col])), dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([ids.size(1)], dtype=torch.long)
        ids, lengths = ids.to(device), lengths.to(device)
        with torch.no_grad():
            logits = model(ids, lengths)
            pred = logits.argmax(1).item()
        results.append(pred)
    df["pred_label_idx"] = results

    out_csv = cfg["infer"]["output_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved inference results to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Text Classification Runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Train the model")
    sub.add_parser("eval", help="Evaluate the model")
    p_infer = sub.add_parser("infer", help="Run CSV inference")
    p_infer.add_argument("--input", type=str, default=None, help="CSV to infer (must contain text col)")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "train":
        cmd_train(cfg)
    elif args.cmd == "eval":
        cmd_eval(cfg)
    elif args.cmd == "infer":
        cmd_infer(cfg, input_csv=args.input)

if __name__ == "__main__":
    main()
