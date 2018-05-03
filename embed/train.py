import re

import torch
import torch.nn as nn
import torch.optim as optim

from conf import device, dataset_path
from dataset import char2index
from .embed import Embed

CONTEXT_SIZE = 2


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
        for ctxs, target in dataset:
            ctxs = torch.tensor([char2index[w] for w in ctxs], dtype=torch.long, device=device)
            target = torch.tensor([char2index[target]], dtype=torch.long, device=device)
            model.zero_grad()
            log_probs = model(ctxs)
            loss = loss_func(log_probs, target)
            loss.backward()
            optimizer.step()
            count += 1
            if count % 50000 == 0:
                torch.save(model, 'model.pt')
                print(count)


def train_and_dump():
    model = Embed(vocab_size=len(char2index), context_size=CONTEXT_SIZE, embed_dim=200)
    if device.type == 'cpu':
        model.cpu()
    else:
        model.cuda()
    dataset = list(gen_dataset())
    train(model, dataset)
