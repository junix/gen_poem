import torch
import torch.nn as nn
from conf import device


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedd = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedd, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, dtype=torch.float32, device=device)
