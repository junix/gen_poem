import re

import torch
import torch.nn as nn
import torch.optim as optim

from conf import device, dataset_path, HIDDEN_SIZE, change_to_device
from dataset import char2index
from .embed import Embed

CONTEXT_SIZE = 3


def gen_dataset():
    with open(dataset_path, 'r') as f:
        for line in f:
            for seg in re.split('[，。]', line):
                for index in range(len(seg) - CONTEXT_SIZE):
                    yield seg[index:index + CONTEXT_SIZE], seg[index + CONTEXT_SIZE]


def train(model, dataset):
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    loss_func = nn.NLLLoss()
    count = 0
    for epoch in range(300):
        for context, target in dataset:
            context = torch.tensor([char2index[w] for w in context], dtype=torch.long, device=device)
            target = torch.tensor([char2index[target]], dtype=torch.long, device=device)
            model.zero_grad()
            log_probs = model(context)
            loss = loss_func(log_probs, target)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 50000 == 0:
                torch.save(model.embedding.state_dict(), 'embedding_state.pt')
                print(count)


def train_and_dump():
    model = Embed(vocab_size=len(char2index), context_size=CONTEXT_SIZE, embed_dim=HIDDEN_SIZE)
    change_to_device(model)
    dataset = list(gen_dataset())
    train(model, dataset)
