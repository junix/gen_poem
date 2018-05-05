import random

import torch

from char_cnn.train import gen_dataset_sentence
from conf import change_to_device, device
from dataset import indexes_from_sentence, EOS, index2char


def load_model():
    model = torch.load('model.pt', map_location=lambda storage, loc: storage)
    change_to_device(model)
    return model


def load_predict():
    model = load_model()
    model.eval()

    def predict(sentence):
        in_tensor = indexes_from_sentence(sentence, append_eos=False)
        hidden = model.init_hidden()

        out_words = []

        with torch.no_grad():
            output = None
            for i in range(in_tensor.size(0)):
                output, hidden = model(in_tensor[i], hidden)
            while True:
                topv, topi = output.topk(1)
                if topi.item() == EOS:
                    break
                else:
                    out_words.append(topi.item())
                    input = torch.tensor([[topi.item()]], dtype=torch.long, device=device)
                    output, hidden = model(input, hidden)
        return ''.join([index2char[i] for i in out_words])

    return predict


def test():
    pred = load_predict()
    xs = list(gen_dataset_sentence())
    random.shuffle(xs)

    for seg in xs[:1000]:
        prefix = seg[:-2]
        suffix = pred(prefix)
        if suffix:
            print(seg, '=>', prefix + suffix)
