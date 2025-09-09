import os
import json
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from data import build_loaders_and_assets
from model import TextClassifier

@torch.no_grad()
def evaluate(cfg, logger):
    # Build loaders & assets (will re-preprocess and rebuild vocab to match training split)
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

    # Load best
    ckpt_path = os.path.join("outputs", "checkpoints", "best.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.info("No checkpoint found; evaluating random-initialized model.")

    model.eval()

    # Evaluate
    def _run(loader):
        total_loss, correct, total = 0.0, 0, 0
        ce = nn.CrossEntropyLoss()
        all_preds, all_true = [], []
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = ce(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().tolist())
            all_true.extend(y.cpu().tolist())
        avg_loss = total_loss / total
        acc = correct / total * 100.0
        return avg_loss, acc, all_true, all_preds

    test_loss, test_acc, y_true, y_pred = _run(test_loader)
    logger.info(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}")

    reports_dir = cfg["eval"]["save_reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    # Load label names
    label_path = "outputs/label_classes.json"
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            classes = json.load(f)["classes"]
    else:
        classes = None

    # Save reports
    report_txt = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(os.path.join(reports_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    import numpy as np
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(reports_dir, "confusion_matrix.csv"), index=False)

    return test_acc
