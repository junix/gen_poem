import re
import random
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import SOS, EOS, indexes_from_sentence, vocab_size
from conf import device, teacher_forcing_ratio, dataset_path, change_to_device, HIDDEN_SIZE

from encoder import Encoder
from decoder import Decoder


def train(encoder, decoder,
          input_tensor, output_tensor,
          encoder_optim, decoder_optim,
          criterion):
    encoder_hidden = encoder.init_hidden()
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    input_len = input_tensor.size(0)
    target_len = output_tensor.size(0)

    for i in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)

    decoder_input = torch.tensor([SOS], dtype=torch.long, device=device)
    # decoder_hidden = decoder.init_hidden()
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    loss = 0
    if use_teacher_forcing:
        for i in range(target_len):
            decoder_output, decoder_hidden = decoder(output_tensor[i], decoder_hidden)
            loss += criterion(decoder_output, output_tensor[i].view(-1))
    else:
        for i in range(target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, output_tensor[i].view(-1))
            if decoder_input.item() == EOS:
                break
    loss.backward()

    encoder_optim.step()
    decoder_optim.step()


def gen_dataset():
    pattern = re.compile('([^， ]+)，([^，。 ]+)。')
    with open(dataset_path, 'r') as f:
        for line in f:
            matcher = pattern.search(line)
            if matcher:
                input, target = matcher.group(1), matcher.group(2)
                yield indexes_from_sentence(input), indexes_from_sentence(target)


def train_and_dump():
    encoder = Encoder(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE)
    decoder = Decoder(output_size=vocab_size, hidden_size=HIDDEN_SIZE)
    embed = nn.Embedding(vocab_size, HIDDEN_SIZE)
    embed.load_state_dict(torch.load('embedding_state.pt', map_location=lambda storage, loc: storage))
    encoder.embedding = embed
    decoder.embedding = embed
    ignore_ids = set([id(p) for p in embed.parameters()])
    change_to_device(encoder)
    change_to_device(decoder)
    dataset = list(gen_dataset())
    enc_params = [p for p in encoder.parameters() if id(p) not in ignore_ids]
    encoder_optim = optim.SGD([{'params': enc_params}], lr=0.0001)
    dec_params = [p for p in decoder.parameters() if id(p) not in ignore_ids]
    decoder_optim = optim.SGD([{'params': dec_params}], lr=0.0001)
    criterion = nn.NLLLoss()
    count = 1
    for epoch in range(300):
        dataset0 = [random.choice(dataset) for _ in range(len(dataset))]
        for input, target in dataset0:
            count += 1
            train(encoder, decoder,
                  input, target,
                  encoder_optim, decoder_optim, criterion)
            if count % 40000 == 0:
                torch.save(encoder, "encoder.pt")
                torch.save(decoder, "decoder.pt")
                print(count)
