import random
import re
import types

import torch
import torch.nn as nn
import torch.optim as optim

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
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    loss.backward()
    optimizer.step()
    return loss.item()


def make_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    return optim.SGD(model.parameters(), lr=lr)


def train_iter(model, optimizer):
    lr = 0.001
    optimizer = make_optimizer(model, optimizer, lr=lr)
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
            if count % 80000 == 0:
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


def train_and_dump(load_old=False, optimizer='sgd'):
    global _model_dump
    _model_dump = optimizer + '.' + _model_dump
    if load_old:
        model = torch.load(_model_dump)
    else:
        model = Model(vocab_size=vocab_size, hidden_size=1024)
    change_to_device(model)
    model.train()
    train_iter(model, optimizer)
