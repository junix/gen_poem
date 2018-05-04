import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import device


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, in_tensor, hidden):
        embed = self.embedding(in_tensor).view(1, 1, -1)
        output, hidden = self.gru(embed, hidden)
        output = self.linear(output.view(1, -1))
        output = self.softmax(F.relu(output))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
