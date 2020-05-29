from io import open
import torch
import torch.nn as nn
from torch import optim

import string
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, self.initHidden(input.size(0)))
        return hidden

    def initHidden(self, batchsize):
        return (torch.zeros(1, batchsize, self.hidden_size, device=device), 
                torch.zeros(1, batchsize, self.hidden_size, device=device))

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output).permute(0, 2, 1)
        return output, hidden

#class AttnDecoderRNN(nn.Module):
#    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#        super(AttnDecoderRNN, self).__init__()
#        self.hidden_size = hidden_size
#        self.output_size = output_size
#        self.dropout_p = dropout_p
#        self.max_length = max_length
#
#        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#        self.dropout = nn.Dropout(self.dropout_p)
#        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
#        self.out = nn.Linear(self.hidden_size, self.output_size)
#
#    def forward(self, input, hidden, encoder_outputs):
#        embedded = self.embedding(input).view(1, 1, -1)
#        embedded = self.dropout(embedded)
#
#        attn_weights = F.softmax(
#            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                 encoder_outputs.unsqueeze(0))
#
#        output = torch.cat((embedded[0], attn_applied[0]), 1)
#        output = self.attn_combine(output).unsqueeze(0)
#
#        output = F.relu(output)
#        output, hidden = self.lstm(output, hidden)
#
#        output = F.log_softmax(self.out(output[0]), dim=1)
#        return output, hidden, attn_weights
#
#    def initHidden(self):
#        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class SpellChecker(object):
    def __init__(self, encoder, decoder, i2c, c2i):
        self.encoder = encoder
        self.decoder = decoder
        self.i2c = i2c
        self.c2i = c2i
        
    def check(self, line, max_sub_len=2):
        max_target_length = len(line) + max_sub_len
        decoder_input = torch.LongTensor([[EOS_TOKEN]])
        encode = encode_lines([line])
        output = []
        with torch.no_grad():
            decoder_hidden = self.encoder(encode)
            for di in range(max_target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_input = decoder_output.argmax(dim=1)

                if decoder_input.item() == c2i[EOS_TOKEN]:
                    break
                else:
                    output.append(decoder_input.item())
        return "".join([ i2c[i] for i in output ])

    def evaluate(self):
        return


SOS_TOKEN = 1
EOS_TOKEN = 0
BATCH = 8
chars = string.ascii_lowercase + "ıüşğçö "

c2i = {c :i for i,c in enumerate(chars)}
c2i[EOS_TOKEN] = len(c2i) + 1
c2i[SOS_TOKEN] = len(c2i) + 1
i2c = list(chars) 

def encode_lines(lines, prepad=True): 
    encoded_lines = [] 
    max_length = max(len(l) for l in lines) 
    for l in lines:
        if prepad:
            encoded_lines.append([c2i[SOS_TOKEN]]*(max_length - len(l)) + [c2i[c] for c in l] + [c2i[EOS_TOKEN]]) 
        else:
            encoded_lines.append([c2i[c] for c in l] + [c2i[EOS_TOKEN]] * (max_length - len(l)))
    return torch.LongTensor(encoded_lines)     


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_batched_pairs(fileName):
    
    lines = open(fileName).read().strip().split('\n') 
    pairs = [[s for s in l.split('|')] for l in lines]
    for pairs in batch(pairs, n=BATCH):
        in_lines, output_lines = zip(*pairs)
        yield encode_lines(in_lines), encode_lines(output_lines)
    
    return

def get_sets():
    training_pairs = list(get_batched_pairs('train.txt'))
    validation_pairs = list(get_batched_pairs('val.txt'))
    return training_pairs, validation_pairs




def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion,teacher_forcing_ratio = 0.5, update=True ):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(1)

    loss = 0
    decoder_hidden = encoder(input_tensor)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        decoder_output, decoder_hidden = decoder(target_tensor, decoder_hidden)
        loss = criterion(decoder_output, target_tensor)
    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_input = torch.LongTensor([[EOS_TOKEN] for i in range(target_tensor.size(0)) ])
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.argmax(dim=1)

            loss += criterion(decoder_output, target_tensor[:, di].unsqueeze(-1))
            if decoder_input.sum() == EOS_TOKEN:
                break
    if update:
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / target_length

    
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder,training_pairs, validation_pairs, sc, epochs=10, dev_line="ben geldim", print_every=100, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    report_val = len(training_pairs)
    training_pairs = [training_pairs for i in range(epochs)]
    for l in training_pairs: random.shuffle(l)
    training_pairs = sum(training_pairs, [])
    criterion = nn.CrossEntropyLoss()
    n_iters = len(training_pairs)
    bestloss = 1e10
    for iter, training_pair in enumerate(training_pairs, start=1):
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=(1 - iter/len(training_pairs) ))
        if iter % report_val == 0:
            encoder.eval()
            decoder.eval()
            val_loss = 0
            for _iter, validation_pair in enumerate(validation_pairs, start = 1):
                input_tensor = validation_pair[0]
                target_tensor = validation_pair[1]
                val_loss += train(input_tensor, target_tensor, encoder, 
                                 decoder, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio=-1, criterion, update=False)
            val_loss /= len(validation_pairs)
            if val_loss < bestloss:
                bestloss = val_loss
            else:
                print('bigger than best loss')
                return
            
            encoder.train()
            decoder.train()
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print(sc.check(dev_line))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


hidden_size = 32
encoder1 = EncoderRNN(len(c2i) + 1, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, len(c2i) + 1).to(device)
sc = SpellChecker(encoder1, decoder1, i2c, c2i)
training_pairs, validation_pairs, = get_sets()
trainIters(encoder1, decoder1, training_pairs, validation_pairs ,sc, print_every=100)

# TODOs
# - Split data into train/dev/test
# - Calculate dev loss without calling loss.backward to know when to stop training
# - Write function to decode step by step to test the model

