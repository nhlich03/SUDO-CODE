import torch
import torch.nn as nn


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=256, hidden_dim=384, num_layers=2, pad_id=0, dropout=0.3):
        """
        LSTM-based Language Model.
        Args:
            vocab_size (int): size of vocabulary
            emb_dim (int): embedding dimension
            hidden_dim (int): hidden dimension of LSTM
            num_layers (int): number of LSTM layers
            pad_id (int): padding token index
            dropout (float): dropout probability
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        """
        Forward pass.
        Args:
            x (Tensor): input token ids [B, T]
            h (tuple): optional hidden state
        Returns:
            logits (Tensor): unnormalized scores [B, T, V]
            h: hidden state
        """
        emb = self.embed(x)              # [B, T, emb_dim]
        out, h = self.lstm(emb, h)       # [B, T, hidden_dim]
        out = self.drop(out)
        logits = self.fc(out)            # [B, T, vocab_size]
        return logits, h