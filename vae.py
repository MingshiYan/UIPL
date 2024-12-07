#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : vae.py
# @Author: yanms
# @Date  : 2024/10/30 16:01
# @Desc  :
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()

        # encoder
        self.input_2hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_2mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_2sigma = nn.Linear(hidden_dim, z_dim)

        # decoder
        self.z_2hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_2output = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()

    def encode(self, x):
        h = self.relu(self.input_2hidden(x))
        mu, logvar = self.hidden_2mu(h), self.hidden_2sigma(h)
        return mu, logvar

    def decode(self, x):
        h = self.relu(self.z_2hidden(x))
        h = self.hidden_2output(h)
        return h

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(logvar)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        return recon_x, mu, logvar


# kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
# recon_loss = lambda recon_x, x: F.binary_cross_entropy(recon_x, x, size_average=False)
