import argparse
import os
from typing import List

import torch

from utils import load_config
from data import Tokenizer, Vocab, PAD
from model import LSTMLM


def sample_next(
    logits_1d: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> int:
    """
    Temperature + top-k + top-p (nucleus) sampling from 1D logits.
    """
    logits = logits_1d.clone() / max(temperature, 1e-8)

    # Top-k
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        cutoff = values[-1]
        logits[logits < cutoff] = -float("inf")

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # mask tokens beyond nucleus
        mask = cum_probs > top_p
        if mask.any():
            # keep at least the largest logit
            mask[0] = False
        sorted_logits[mask] = -float("inf")
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id


def generate_text(
    cfg_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab & tokenizer
    vocab = Vocab.load(cfg["project"]["vocab_path"])
    tokenizer = Tokenizer(
        specials=[
            cfg["preprocess"]["bos_token"],
            cfg["preprocess"]["eos_token"],
            cfg["preprocess"]["para_token"],
        ]
    )

    # Build model & load checkpoint
    model = LSTMLM(
        vocab_size=len(vocab),
        emb_dim=cfg["model"]["emb_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        pad_id=vocab.stoi[PAD],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    ckpt_path = os.path.join(cfg["project"]["ckpt_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])  # type: ignore
    model.eval()

    # Tokenize prompt → ids
    ids = vocab.encode(tokenizer.tokenize(prompt))
    if not ids:
        raise ValueError("Prompt is empty after tokenization.")
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    generated: List[int] = []
    h = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, h = model(x, h)     # [1, T, V]
            next_logits = logits[0, -1] # [V]
        next_id = sample_next(
            next_logits.detach().cpu(),
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        generated.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)

    tokens = vocab.decode(generated)
    return " ".join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--prompt", type=str, default="<BOS> ")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    text = generate_text(
        cfg_path=args.config,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)
