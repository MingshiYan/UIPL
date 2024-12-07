#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author:
# @Date  : 2023/9/23 11:26
# @Desc  :
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = 0
        for embedding in embeddings:
            tmp = torch.norm(embedding, p=self.norm)
            tmp = tmp / embedding.shape[0]
            emb_loss += tmp
        return emb_loss


class ContrastiveLoss(nn.Module):
    '''
    infoNCELoss
    '''
    def __init__(self, size, device='cpu', temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(size, size, dtype=bool).to(device)).float())

    def forward(self, p_emb, n_emb):

        b_size = p_emb.shape[1]
        random_numbers = random.sample(range(1, b_size), 1)
        p_1 = p_emb[:, 0]
        p_2 = p_emb[:, random_numbers[0]]

        p1_emb = F.normalize(p_1, dim=-1)
        p2_emb = F.normalize(p_2, dim=-1)
        n_emb = F.normalize(n_emb, dim=-1)
        n_emb = n_emb.view(-1, n_emb.shape[-1])

        z_i = torch.sum(p1_emb * p2_emb, dim=-1)
        z_j = p1_emb.matmul(n_emb.T)
        z_i = torch.exp(z_i / self.temperature)
        z_j = torch.sum(torch.exp(z_j / self.temperature), dim=-1)
        ssl_loss = torch.log(z_i / (z_i + z_j))
        ssl_loss = -torch.mean(ssl_loss)
        return ssl_loss


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()


    def forward(self, mu, logvar):
        loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        loss = torch.sum(loss, dim=-1)
        loss = loss.mean()

        return loss