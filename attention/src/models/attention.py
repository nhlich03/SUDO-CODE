import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size * 2, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, hidden):
        hidden = hidden.permute(1, 0, 2)
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(hidden)))
        attn_weights = torch.softmax(score, dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context, attn_weights
