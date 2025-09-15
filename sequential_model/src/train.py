import math
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn


class LMTrainer:
    def __init__(
        self,
        model: nn.Module,
        pad_id: int,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        use_scheduler: bool = True,
    ):
        self.model = model.to(device)
        self.pad_id = pad_id
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=1
            )
            if use_scheduler
            else None
        )

        # training logs
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_ppl": [],
            "val_ppl": [],
        }

    def _step(self, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        """One forward pass + compute loss (no optimizer step here)."""
        logits, _ = self.model(xb)  # [B, T, V]
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
        )
        return loss

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float]:
        """Evaluate average loss & perplexity over a dataloader."""
        self.model.eval()
        total_loss, total_tok = 0.0, 0
        for xb, yb, mask in loader:
            xb, yb, mask = xb.to(self.device), yb.to(self.device), mask.to(self.device)
            loss = self._step(xb, yb)
            ntok = mask.sum().item()
            total_loss += loss.item() * ntok
            total_tok += ntok
        avg_loss = total_loss / max(total_tok, 1)
        ppl = math.exp(avg_loss)
        return avg_loss, ppl

    def train_one_epoch(self, loader, grad_clip: float = 1.0) -> Tuple[float, float]:
        """Train for a single epoch and return avg loss & ppl."""
        self.model.train()
        total_loss, total_tok = 0.0, 0
        for xb, yb, mask in loader:
            xb, yb, mask = xb.to(self.device), yb.to(self.device), mask.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self._step(xb, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            ntok = mask.sum().item()
            total_loss += loss.item() * ntok
            total_tok += ntok

        avg_loss = total_loss / max(total_tok, 1)
        ppl = math.exp(avg_loss)
        return avg_loss, ppl

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 6,
        ckpt_path: str = "output/checkpoints/best.pt",
        grad_clip: float = 1.0,
        patience: int = 2,
    ) -> None:
        """Train with early stopping by validation perplexity."""
        best_val = float("inf")
        bad_epochs = 0

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        for ep in range(1, epochs + 1):
            tr_loss, tr_ppl = self.train_one_epoch(train_loader, grad_clip=grad_clip)
            va_loss, va_ppl = self.evaluate(val_loader)

            # log
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_ppl"].append(tr_ppl)
            self.history["val_ppl"].append(va_ppl)

            print(f"Epoch {ep:02d} | Train PPL: {tr_ppl:.2f} | Val PPL: {va_ppl:.2f}")

            if self.scheduler:
                self.scheduler.step(va_loss)

            # early stopping by val perplexity
            if va_ppl < best_val:
                best_val = va_ppl
                torch.save({"model": self.model.state_dict()}, ckpt_path)
                bad_epochs = 0
                print("  -> Saved best model")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(
                        f"Early stopping at epoch {ep} (no improvement for {patience} epochs)."
                    )
                    break

    def test(self, test_loader, ckpt_path: str = "output/checkpoints/best.pt") -> Tuple[float, float]:
        """Load best checkpoint and evaluate on test set."""
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])  # type: ignore
        test_loss, test_ppl = self.evaluate(test_loader)
        print(f"Test PPL: {test_ppl:.2f}")
        return test_loss, test_ppl
