import torch

from conf import device, change_to_device
from dataset import indexes_from_sentence, SOS, EOS, index2char


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

        for i in range(in_tensor_len):
            output, in_hidden = encoder(in_tensor[i], in_hidden)

        decoder_input = torch.tensor([[SOS]], device=device)
        decoder_hidden = in_hidden
        decoder_words = []

        for i in range(20):
            out, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = out.topk(1)
            index = topi.item()
            if index == EOS:
                break
            else:
                decoder_words.append(index2char[index])
            decoder_input = topi.squeeze().detach()
        return ''.join(decoder_words)

    return predict
