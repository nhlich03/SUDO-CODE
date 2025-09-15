import argparse
import os

import kagglehub
import torch

from utils import ensure_dir, load_config, select_device, set_seed, save_json
from data import (
    BOS,
    EOS,
    PAD,
    PARA,
    DataSplitter,
    TextPreprocessor,
    Tokenizer,
    Vocab,
    load_books,
    make_loaders,
)
from model import LSTMLM
from train import LMTrainer


def maybe_download(cfg):
    """Download dataset from KaggleHub if enabled."""
    if not cfg["data"].get("use_kagglehub", True):
        return cfg["data"]["data_dir"]

    path = kagglehub.dataset_download(cfg["data"]["kaggle_dataset"])
    print("Kaggle dataset path:", path)

    # dataset has /output folder containing .txt files
    default_dir = os.path.join(path, "output")
    return default_dir


def main(config_path: str):
    # -----------------
    # Load config
    # -----------------
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])

    out_dir = cfg["project"]["out_dir"]
    ckpt_dir = cfg["project"]["ckpt_dir"]
    vocab_path = cfg["project"]["vocab_path"]

    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)

    device = select_device(cfg["project"].get("device", "auto"))
    print("Device:", device)

    # -----------------
    # Load data
    # -----------------
    data_dir = maybe_download(cfg)
    print("Data dir:", data_dir)

    dataset = load_books(data_dir)

    splitter = DataSplitter(
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        seed=cfg["project"]["seed"],
        shuffle=cfg["split"].get("shuffle", True),
    )
    train_df, val_df, test_df = splitter.split(dataset)

    # -----------------
    # Preprocess
    # -----------------
    pre = TextPreprocessor(
        para_token=cfg["preprocess"]["para_token"],
        bos_token=cfg["preprocess"]["bos_token"],
        eos_token=cfg["preprocess"]["eos_token"],
        min_words=cfg["preprocess"]["min_words"],
        drop_patterns=cfg["preprocess"].get("drop_patterns", []),
    )
    train_records = pre.process_series(train_df["Texts"])
    val_records = pre.process_series(val_df["Texts"])
    test_records = pre.process_series(test_df["Texts"])

    # -----------------
    # Tokenizer & Vocab
    # -----------------
    tokenizer = Tokenizer(
        specials=[
            cfg["preprocess"]["bos_token"],
            cfg["preprocess"]["eos_token"],
            cfg["preprocess"]["para_token"],
        ]
    )
    vocab = Vocab(
        min_freq=cfg["vocab"]["min_freq"],
        max_size=cfg["vocab"]["max_size"],
    )
    vocab.build(train_records, tokenizer)
    vocab.save(vocab_path)

    pad_id = vocab.stoi[PAD]

    # -----------------
    # DataLoaders
    # -----------------
    train_loader, val_loader, test_loader = make_loaders(
        (train_records, val_records, test_records),
        tokenizer,
        vocab,
        max_len=cfg["windowing"]["max_len"],
        stride=cfg["windowing"].get("stride", cfg["windowing"]["max_len"]),
        batch_size=cfg["windowing"]["batch_size"],
        pad_id=pad_id,
    )

    # -----------------
    # Model
    # -----------------
    model = LSTMLM(
        vocab_size=len(vocab),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        pad_id=pad_id,
        dropout=cfg["model"]["dropout"],
    )

    # -----------------
    # Trainer
    # -----------------
    trainer = LMTrainer(
        model,
        pad_id=pad_id,
        device=device,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        use_scheduler=cfg["train"].get("use_scheduler", True),
    )

    ckpt_path = os.path.join(ckpt_dir, "best.pt")

    trainer.fit(
        train_loader,
        val_loader,
        epochs=cfg["train"]["epochs"],
        ckpt_path=ckpt_path,
        grad_clip=cfg["train"]["grad_clip"],
        patience=cfg["train"]["patience"],
    )

    # save training history
    history_path = os.path.join(out_dir, "history.json")
    save_json(trainer.history, history_path)

    # -----------------
    # Test
    # -----------------
    trainer.test(test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
