import torch
from conf import device


def _read_vocab():
    chr2idx = {'<SOS>': 0, '<EOS>': 1}
    with open('data/dataset.txt', 'r') as f:
        for ch in set(f.read()):
            chr2idx[ch] = len(chr2idx)
        idx2chr = dict([(v, k) for k, v in chr2idx.items()])
        return chr2idx, idx2chr


char2index, index2char = _read_vocab()


def indexes_from_sentence(sentence):
    indexes = [char2index[c] for c in sentence]
    return torch.tensor(indexes, dtype=torch.long, device=device)
