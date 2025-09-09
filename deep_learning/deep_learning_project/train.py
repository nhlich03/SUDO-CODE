import os
import torch
import torch.nn as nn

from utils import save_checkpoint

class Trainer:
    def __init__(self, model, criterion, optimizer, device="cpu", logger=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger

        # save results
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []
        self.best_val_acc = 0.0

    def run_epoch(self, loader, train=True):
        if train: self.model.train()
        else:     self.model.eval()

        total_loss, correct, total = 0.0, 0, 0

        for x, lengths, y in loader:
            x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(train):
                logits = self.model(x, lengths)
                loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=10, save_path="outputs/checkpoints/best.pth", patience=3):
        patience_counter = 0
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(1, epochs+1):
            tr_loss, tr_acc = self.run_epoch(train_loader, train=True)
            va_loss, va_acc = self.run_epoch(val_loader, train=False)

            self.train_losses.append(tr_loss); self.train_accs.append(tr_acc)
            self.val_losses.append(va_loss);   self.val_accs.append(va_acc)

            msg = f"Epoch {epoch:02d} | Train Loss={tr_loss:.4f}, Acc={tr_acc:.4f} | Val Loss={va_loss:.4f}, Acc={va_acc:.4f}"
            print(msg)
            if self.logger: self.logger.info(msg)

            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0 
                if self.logger: self.logger.info(f"Saved best checkpoint to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stop_msg = f"Early stopping at epoch {epoch} (no improvement in {patience} epochs)."
                    print(stop_msg)
                    if self.logger: self.logger.info(stop_msg)
                    break
