# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from minibert import BERT
from utils import gumbel_softmax


class Generator(nn.Module):
    """Generator model"""

    def __init__(self, bert: BERT, GAN_param, Pretrain_param):
        super(Generator, self).__init__()
        self.seq_len = GAN_param["train_len"]
        self.hz = GAN_param["hidden_size"]
        self.n_chars = Pretrain_param["vocab"]
        self.bz = GAN_param['batch_size']
        self.gen_num = GAN_param['gen_num']
        self.hidden = Pretrain_param["hidden"]
        self.emb = bert
        self.trans = nn.Linear(self.hidden, self.hz)

        self.lstm = nn.LSTM(self.hz, self.hz, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(self.hz * 2, self.n_chars)
        self.highway = nn.Linear(self.hz * 2, self.hz * 2)
        self.dropout = nn.Dropout(p=0.3)

        self.init_params()

    def forward(self, seq):
        """
        Forward pass of the generator
        Args:
            seq: (batch_size, seq_len), sequence of tokens
        """
        seq = seq.long().cuda()
        emb = self.trans(self.emb(seq))
        h0, c0 = self.init_hidden()
        output, _ = self.lstm(emb, (h0, c0))

        highway = self.highway(output)
        output = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * output
        output = self.dropout(output)

        pred = gumbel_softmax(self.lin(output.contiguous().view(-1, self.hz * 2)), 0.5)
        return pred

    def step(self, x, h, c):
        """
        Single step generation
        Args:
            x: (batch_size, 1), single token
            h: (1, batch_size, hidden_dim), lstm hidden state
            c: (1, batch_size, hidden_dim), lstm cell state
        """
        emb = self.trans(self.emb(x))
        output, (h, c) = self.lstm(emb, (h, c))
        pred = gumbel_softmax(self.lin(output.contiguous().view(-1, self.hz * 2)), 0.5)
        return pred, h, c

    def init_hidden(self):
        """Initialize hidden states for LSTM"""
        h = Variable(torch.zeros((2, self.bz, self.hz))).cuda()
        c = Variable(torch.zeros((2, self.bz, self.hz))).cuda()
        return h, c

    def init_params(self):
        """Initialize model parameters"""
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def gen_sample(self, gen_seq_len, x=None):
        """Generate a sample sequence"""
        flag = x is None
        if flag:
            x = Variable(torch.randint(low=2, high=22, size=(self.bz, 1))).long().cuda()

        h, c = self.init_hidden()
        samples = []

        if flag:
            for _ in range(gen_seq_len):
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(given_len, dim=1)
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = output.multinomial(1)
            for _ in range(given_len, gen_seq_len):
                samples.append(x)
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)

        output = torch.cat(samples, dim=1)
        return output

    def gen_seqs(self, start_end: tuple = (5, 10)):
        """Generate multiple sequences"""
        start_, end_ = start_end
        x = Variable(torch.randint(low=2, high=22, size=(self.gen_num, 1, 1))).long().cuda()
        h, c = self.init_hidden_gen()
        samples = []

        for seq in x:
            seqs = []
            nn = np.random.randint(start_, end_ + 1)
            for i in range(nn):
                if i == 0:
                    seqs.append(seq)
                else:
                    output, h, c = self.step(seq, h, c)
                    seq = output.multinomial(1)
                    seqs.append(seq)
            samples.append(seqs)
        return samples

    def init_hidden_gen(self):
        """Initialize hidden states for generation"""
        h = Variable(torch.zeros((2, 1, self.hz))).cuda()
        c = Variable(torch.zeros((2, 1, self.hz))).cuda()
        return h, c


class ResBlock(nn.Module):
    """Residual block for the discriminator"""

    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
        )

    def forward(self, input):
        return input + (0.3 * self.res_block(input))


class Discriminator(nn.Module):
    """Discriminator model"""

    def __init__(self, GAN_param, Pretrain_param):
        super(Discriminator, self).__init__()
        self.seq_len = GAN_param['train_len']
        self.seq_len_gs = GAN_param['gen_len']
        self.n_chars = Pretrain_param["vocab"]
        self.hz = GAN_param["hidden_size"]
        self.dropout_p = GAN_param["dropout"]
        self.res_layer = GAN_param["res_layers"]
        self.bz = GAN_param['batch_size']
        self.latent_dim = GAN_param["latent_dim"]
        self.hidden = Pretrain_param["hidden"]

        kernel = [1, 3, 5, 7]
        padding = [0, 1, 2, 3]
        self.trans = nn.Linear(self.hidden, self.hz)
        self.lin_gs = nn.Linear(self.hz * self.seq_len_gs * 4, 1)
        self.conv1d = nn.ModuleList(
            [nn.Conv1d(self.hz, self.latent_dim, kernel_size=k, padding=p) for (k, p) in zip(kernel, padding)])
        self.highway = nn.Linear(self.hz, self.hz)
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.lin = nn.Linear(self.seq_len, 1)
        self.lin2 = nn.Linear(self.hz, 1)
        self.block = nn.Sequential(*[ResBlock(self.hz) for _ in range(self.res_layer)])
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hz)
        self.emb = nn.Linear(self.n_chars, self.hz)

        self.init_parameters()

    def forward(self, seq):
        """
        Forward pass of the discriminator
        Args:
            seq: (batch_size, seq_len, vocab_size)
        """
        emb = self.emb(seq)
        convs = [F.relu(conv(emb.transpose(-1, -2))) for conv in self.conv1d]
        pred = torch.cat(convs, 1)
        pred = self.block(pred)
        pred = self.LayerNorm(pred.transpose(-1, -2))
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        pred = self.lin2(pred)
        validity = nn.Sigmoid()(self.lin(self.dropout(pred.reshape(self.bz, -1))))
        return validity

    def init_parameters(self):
        """Initialize model parameters"""
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


# Create necessary directories
def create_directories():
    directories = ['models', 'data', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
