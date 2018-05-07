import types

import torch
import torch.nn as nn
import torch.optim as optim

from conf import change_to_device, device
from dataset import vocab_size
from utils import change_lr
from .lang import gen_dataset
from .model import Net

_model_dump = 'model.pt'


def train(model, input_tensor, target_tensor, optimizer, criterion):
    target_tensor = target_tensor.unsqueeze_(-1).to(device)
    optimizer.zero_grad()
    input_len = input_tensor.size(0)
    hidden = model.init_hidden()
    loss = 0
    for i in range(input_len):
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    loss.backward()
    optimizer.step()
    return loss.item() / input_len


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
    loss = 0
    for input_tensor, target_tensor in gen_dataset():
        loss += train(model, input_tensor, target_tensor, optimizer, criterion)
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


def train_and_dump(load_old=False, optimizer='sgd'):
    global _model_dump
    _model_dump = optimizer + '.' + _model_dump
    if load_old:
        model = torch.load(_model_dump)
    else:
        model = Net(input_size=vocab_size, hidden_size=1024, output_size=vocab_size)
    change_to_device(model)
    model.train()
    train_iter(model, optimizer)
