import re
import random
import types
import torch
import torch.optim as optim
import torch.nn as nn

from conf import dataset_path, change_to_device
from dataset import indexes_from_sentence, vocab_size, EOS

from utils import change_lr
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
        topv, topi = output.topk(1)
        if topi.item() == EOS:
            break
    loss.backward()
    optimizer.step()
    return loss.item()


def train_iter(model):
    lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer.change_lr = types.MethodType(change_lr, optimizer)
    criterion = nn.NLLLoss()
    count = 1
    dataset = list(gen_dataset())
    loss = 0
    for epoch in range(400):
        random.shuffle(dataset)
        print('begin epoch=>', epoch)
        for X in dataset:
            loss += train(model, X, optimizer, criterion)
            count += 1
            if count % 20000 == 0:
                torch.save(model, _model_dump)
            if count % 100000 == 0:
                lr = lr * 0.95
                if lr < 0.0001:
                    lr = 0.0001
                optimizer.change_lr(lr)
            if count % 2000 == 0:
                print(count, ' => ', loss)
                loss = 0


def gen_dataset_sentence():
    with open(dataset_path, 'r') as f:
        for line in f:
            for seg in re.split('[，。]', line):
                seg = seg.strip()
                if not seg:
                    continue
                yield seg


def gen_dataset():
    for seg in gen_dataset_sentence():
        yield indexes_from_sentence(seg)


def train_and_dump(load_old=False):
    if load_old:
        model = torch.load(_model_dump)
    else:
        model = Model(vocab_size=vocab_size, hidden_size=512)
    change_to_device(model)
    model.train()
    train_iter(model)
