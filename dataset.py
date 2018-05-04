import torch
from conf import device, dataset_path

SOS = 0
EOS = 1


def _read_vocab(file):
    chr2idx = {'<SOS>': SOS, '<EOS>': EOS}
    with open(file, 'r') as f:
        for ch in set(f.read()):
            chr2idx[ch] = len(chr2idx)
        idx2chr = dict([(v, k) for k, v in chr2idx.items()])
        return chr2idx, idx2chr


char2index, index2char = _read_vocab(dataset_path)

vocab_size = len(char2index)


def indexes_from_sentence(sentence, append_eos=True):
    indexes = [char2index[c] for c in sentence]
    if append_eos:
        indexes.append(EOS)
    return torch.tensor(indexes, dtype=torch.long, device=device)
