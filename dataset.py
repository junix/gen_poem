import torch
from conf import device, dataset_path


def _read_vocab(file):
    chr2idx = {}
    with open(file, 'r') as f:
        cs = list(set(f.read()))
        cs.append('<EOS>')
        for ch in cs:
            chr2idx[ch] = len(chr2idx)
        idx2chr = dict([(v, k) for k, v in chr2idx.items()])
        return chr2idx, idx2chr


char2index, index2char = _read_vocab(dataset_path)

vocab_size = len(char2index)
EOS = vocab_size - 1


def indexes_from_sentence(sentence, append_eos=True):
    indexes = [char2index[c] for c in sentence]
    if append_eos:
        indexes.append(EOS)
    return torch.tensor(indexes, dtype=torch.long, device=device)


def encoder_sentence(sentence):
    vec = torch.zeros(vocab_size, dtype=torch.float)
    for c in sentence:
        vec[char2index[c]] = 1.0
    return vec
