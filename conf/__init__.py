import os
import torch
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = os.path.dirname(__file__) + '/../data/dataset.txt'

teacher_forcing_ratio = 0.5


def change_to_device(model):
    if device.type == 'cpu':
        model.cpu()
    else:
        model.cuda()
