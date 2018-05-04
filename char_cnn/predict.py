import torch
from conf import change_to_device
from dataset import indexes_from_sentence, EOS, index2char


def load_model():
    model = torch.load('model.pt', map_location=lambda storage, loc: storage)
    change_to_device(model)
    return model


def load_predict():
    model = load_model()

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
                out_words.append(topi)
                if topi == EOS:
                    break
                else:
                    input = output.squeeze.detach()
                    output, hidden = model(input, hidden)
        return ''.join([index2char(i) for i in out_words])

    return predict
