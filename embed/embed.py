import torch
import torch.nn as nn
import torch.nn.functional as F


class Embed(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(context_size * embed_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)
