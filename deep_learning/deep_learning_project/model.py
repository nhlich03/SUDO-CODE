import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes, pad_idx=0, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # x: (B, L)
        emb = self.emb(x)  # (B, L, E)
        mask = (x != 0).unsqueeze(-1).float()   # assumes PAD=0
        summed = (emb * mask).sum(dim=1)        # (B, E)
        lengths = lengths.unsqueeze(1).clamp(min=1)
        avg = summed / lengths                  # (B, E)
        h = self.dropout(self.relu(self.fc1(avg)))
        out = self.fc2(h)                       # (B, C)
        return out
