import random

import torch

from char_cnn.train import gen_dataset_sentence
from conf import change_to_device
from dataset import EOS, index2char
from .lang import input_tensor, int_input_tensor


def load_model():
    model = torch.load('sgd.model.pt', map_location=lambda storage, loc: storage)
    change_to_device(model)
    return model


def load_predict():
    model = load_model()
    model.eval()

    def predict(sentence):
        if not sentence:
            return ''
        in_tensor = input_tensor(sentence[0])
        hidden = model.init_hidden()

        out_words = []
        with torch.no_grad():
            for i in range(len(sentence)):
                output, hidden = model(in_tensor, hidden)
                topv, topi = output.topk(1)
                cid = topi.item()
                if cid == EOS:
                    break
                out_words.append(cid)
                in_tensor = int_input_tensor([cid])
            return sentence[0] + ''.join([index2char[i] for i in out_words])

    return predict


def test():
    pred = load_predict()
    xs = list(gen_dataset_sentence())
    random.shuffle(xs)

    count = 1
    for seg in xs[:1000]:
        gseg = pred(seg)
        if len(gseg) > 1:
            print(seg, '=>', gseg)
            count += 1
            if count >= 40:
                return
