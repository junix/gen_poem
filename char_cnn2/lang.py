import re
import random
import torch
from dataset import vocab_size, char2index, EOS
from conf import dataset_path


# One-hot matrix of first to last letters (not including EOS) for input
def input_tensor(line):
    tensor = torch.zeros(len(line), 1, vocab_size)
    for index, ch in enumerate(line):
        tensor[index][0][char2index[ch]] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def target_tensor(line):
    letter_indexes = [char2index[line[i]] for i in range(1, len(line))]
    letter_indexes.append(EOS)
    return torch.tensor(letter_indexes, dtype=torch.long)


def gen_dataset_sentence():
    with open(dataset_path, 'r') as f:
        for line in f:
            for seg in re.split('[，。]', line):
                seg = seg.strip()
                if not seg:
                    continue
                yield seg


def gen_dataset():
    xs = list(gen_dataset_sentence())
    while True:
        line = random.choice(xs)
        yield input_tensor(line), target_tensor(line)
