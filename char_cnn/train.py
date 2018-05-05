import re
import torch
import torch.optim as optim
import torch.nn as nn

from conf import dataset_path, change_to_device
from dataset import indexes_from_sentence, vocab_size

from .model import Model

_model_dump = 'model.pt'


def train(model, input_tensor, optimizer, criterion):
    optimizer.zero_grad()
    input_len = input_tensor.size(0)
    hidden = model.init_hidden()
    loss = 0
    for i in range(input_len - 1):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, input_tensor[i + 1].view(1))
    loss.backward()
    optimizer.step()


def train_iter(model):
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    criterion = nn.NLLLoss()
    count = 1
    dataset = list(gen_dataset())
    for epoch in range(400):
        for X in dataset:
            train(model, X, optimizer, criterion)
            count += 1
            if count % 40000 == 0:
                torch.save(model, _model_dump)
                print(count)


def gen_dataset():
    with open(dataset_path, 'r') as f:
        for line in f:
            for seg in re.split('[，。]', line):
                seg = seg.strip()
                if not seg:
                    continue
                yield indexes_from_sentence(seg)


def train_and_dump(load_old=False):
    if load_old:
        model = torch.load(_model_dump)
    else:
        model = Model(vocab_size=vocab_size, hidden_size=512)
    change_to_device(model)
    train_iter(model)
