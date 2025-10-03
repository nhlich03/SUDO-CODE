import torch
import torch.nn as nn
from .attention import BahdanauAttention

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(embed_size + hidden_size*2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size*3, vocab_size)
        self.attention = BahdanauAttention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        emb = self.embedding(x)
        context, _ = self.attention(encoder_outputs, hidden)
        context = context.unsqueeze(1)
        rnn_input = torch.cat((emb, context), dim=-1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = torch.cat((output, context), dim=-1)
        output = self.fc(output.squeeze(1))
        return output, hidden
