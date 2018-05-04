import torch

from conf import device, change_to_device
from dataset import indexes_from_sentence


def load_models():
    encoder = torch.load('encoder.pt', map_location=lambda storage, loc: storage)
    decoder = torch.load('decoder.pt', map_location=lambda storage, loc: storage)
    change_to_device(encoder)
    change_to_device(decoder)
    return encoder, decoder


def load_predict():
    encoder, decoder = load_models()

    def predict(sentence):
        in_tensor = indexes_from_sentence(sentence)
        in_tensor_len = in_tensor.size(0)
        in_hidden = encoder.init_hidden()

