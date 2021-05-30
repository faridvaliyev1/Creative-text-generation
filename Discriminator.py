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
class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim # размер скрытого слоя
        self.embedding_dim = embedding_dim # ширина слоя embedding
        self.max_seq_len = max_seq_len # длина входного примера
        self.gpu = gpu # использование cuda (True/False)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # объявление слоя Embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, bidirectional=True, dropout=dropout) # объявление LSTM слоев

        self.lstm2hidden = nn.Linear(2*3*hidden_dim, hidden_dim) # объявление скрытого слоя
        # self.lstm2hidden = nn.Linear(max_seq_len*hidden_dim*2, hidden_dim) # объявление скрытого слоя

        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1) # объявление выходного слоя

    def init_hidden(self, batch_size):
        # инициализация LSTM слоев
        h = autograd.Variable(torch.zeros(2*3*1, batch_size, self.hidden_dim))
        c = autograd.Variable(torch.zeros(2*3*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def forward(self, input, hidden, c):
        # input dim                                                # batch_size x seq_len
        # print(input.shape)
        emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
        # print(emb.shape)
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        # print(emb.shape)
        out_lstm, (hidden, c) = self.lstm(emb, (hidden, c))                          # 4 x batch_size x hidden_dim

        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        # hidden = out_lstm
        # print(hidden.shape, hidden.view(-1, self.max_seq_len*self.hidden_dim).shape)
        # print(hidden.permute(1, 0, 2).contiguous().shape, hidden.permute(1, 0, 2).contiguous().view(-1, self.max_seq_len*self.hidden_dim*2).shape)

        out = self.lstm2hidden(hidden.view(-1, 6*self.hidden_dim))  # batch_size x 4*hidden_dim
        # out = self.lstm2hidden(hidden.view(-1, self.max_seq_len*self.hidden_dim*2))  # batch_size x 4*hidden_dim
        
        out = torch.relu(out)
        # out = out * torch.sigmoid(0.1 * out) # функция активации Swish: x * sigmoid(b*x)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h, c = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h, c)
        return out.view(-1)

    def batchBCELoss(self, inp, target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h, c = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h, c)
        return loss_fn(out, target)