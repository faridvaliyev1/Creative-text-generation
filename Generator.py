from __future__ import print_function
from math import ceil
import numpy as np
import sys

import torch
import torch.optim as optim
import torch.nn as nn

import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.init as init

import helpers

class Generator(nn.Module):

  def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False):
    super(Generator, self).__init__()
    self.hidden_dim = hidden_dim # количество элементов на скрытом слое
    self.embedding_dim = embedding_dim # размер слоя embedding
    self.max_seq_len = max_seq_len # длина генерируемых примеров
    self.vocab_size = vocab_size # размер словаря использующегося при генерации
    self.gpu = gpu # использование cuda (True/False)
    self.lstm_num_layers = 1 # количество LSTM слоев

    self.embeddings = nn.Embedding(vocab_size, embedding_dim) # объявление слоя embeddings
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.lstm_num_layers) # объявление LSTM слоев
    #self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers=self.lstm_num_layers, batch_first=True, dropout=drop_prob)
    self.lstm2out = nn.Linear(hidden_dim, vocab_size) # объявление выходного слоя

  def init_hidden(self, batch_size=1):
    # инициализация состояния LSTM слоев
    h = autograd.Variable(torch.zeros(self.lstm_num_layers, batch_size, self.hidden_dim)) 
    c = autograd.Variable(torch.zeros(self.lstm_num_layers, batch_size, self.hidden_dim))

    if self.gpu:
        return h.cuda(), c.cuda()
    else:
        return h, c

  def forward(self, inp, hidden, c):
    """
    Embeds input and applies LSTM one token at a time (seq_len = 1)
    """
    # input dim                                             # batch_size
    emb = self.embeddings(inp)                              # batch_size x embedding_dim
    emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
    out, (hidden, c) = self.lstm(emb, (hidden, c))                    # 1 x batch_size x hidden_dim (out)
    out = self.lstm2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
    out = F.log_softmax(out, dim=1)
    return out, hidden, c

  def sample(self, num_samples, start_letter=0, degree=1):
    """
    Samples the network and returns num_samples samples of length max_seq_len.

    Outputs: samples, hidden
        - samples: num_samples x max_seq_length (a sampled sequence in each row)
    """

    samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

    h, c = self.init_hidden(num_samples)
    inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))
    if self.gpu:
        samples = samples.cuda()
        inp = inp.cuda()

    for i in range(self.max_seq_len):
        out, h, c = self.forward(inp, h, c)               # out: num_samples x vocab_size
        out = torch.exp(out)**degree
        out = torch.multinomial(out, 1)  # num_samples x 1 (sampling from each row)
        samples[:, i] = out.view(-1).data

        inp = out.view(-1)

    return samples

  def batchNLLLoss(self, inp, target):
    """
    Returns the NLL Loss for predicting target sequence.

    Inputs: inp, target
        - inp: batch_size x seq_len
        - target: batch_size x seq_len

        inp should be target with <s> (start letter) prepended
    """

    loss_fn = nn.NLLLoss()
    batch_size, seq_len = inp.size()
    inp = inp.permute(1, 0)           # seq_len x batch_size
    target = target.permute(1, 0)     # seq_len x batch_size
    h, c = self.init_hidden(batch_size)

    loss = 0
    for i in range(seq_len):
        out, h, c = self.forward(inp[i], h, c)
        loss += loss_fn(out, target[i])

    return loss     # per batch

  def batchPGLoss(self, inp, target, reward):
    """
    Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
    Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

    Inputs: inp, target
        - inp: batch_size x seq_len
        - target: batch_size x seq_len
        - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                  sentence)

        inp should be target with <s> (start letter) prepended
    """

    batch_size, seq_len = inp.size()
    inp = inp.permute(1, 0)          # seq_len x batch_size
    target = target.permute(1, 0)    # seq_len x batch_size
    h, c = self.init_hidden(batch_size)

    loss = 0
    for i in range(seq_len):
        out, h, c = self.forward(inp[i], h, c)
        # TODO: should h be detached from graph (.detach())?
        for j in range(batch_size):
            loss += -out[j][target.data[i][j]]*reward[j]#     # log(P(y_t|Y_1:Y_{t-1})) * Q

    return loss/batch_size