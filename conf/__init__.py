import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = os.path.dirname(__file__) + '/../data/dataset.txt'
