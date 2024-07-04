#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2023/9/23 16:16
# @Desc  :
import os.path

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, autograd

from data_set import DataSet
from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN


class UIPL(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(UIPL, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.reg_weight = args.reg_weight
        self.lamb = args.lamb
        self.beta = args.beta
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.global_Graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.all_inter_matrix)
        self.behavior_Graphs = nn.ModuleList({
            LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[index]) for index, _ in enumerate(self.behaviors)
        })

        self.M = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh()
        )

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.cross_loss = nn.BCELoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data[1:])
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def forward(self, batch_data):

        batch_data = batch_data.long()
        user_ids = batch_data[:, 0]
        inv_p_item_ids = batch_data[:, 1]
        inv_n_item_ids = batch_data[:, 2:-2]
        rec_item_ids = batch_data[:, -2:]

        ini_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.global_Graph(ini_embeddings)
        user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
        user_embs = [user_embedding]
        for graph in self.behavior_Graphs:
            behavior_embeddings = graph(ini_embeddings)
            b_user_embedding, b_item_embedding = torch.split(behavior_embeddings, [self.n_users + 1, self.n_items + 1])
            user_embs.append(b_user_embedding)

        user_embs = torch.stack(user_embs)
        invariant_user_emb = self.M(user_embs)

        inv_p_item_emb = item_embedding[inv_p_item_ids]
        inv_n_item_emb = item_embedding[inv_n_item_ids]
        p_scores, n_scores = [], []
        for i in range(invariant_user_emb.shape[0]):
            tmp_user_emb = invariant_user_emb[i][user_ids]
            p_score = torch.sum(tmp_user_emb * inv_p_item_emb, dim=-1)
            p_scores.append(p_score)
            n_score = torch.sum(tmp_user_emb.unsqueeze(1) * inv_n_item_emb, dim=-1)
            n_score = n_score.reshape(-1)
            n_scores.append(n_score)

        p_scores = torch.cat(p_scores)
        n_scores = torch.cat(n_scores)
        scores = torch.cat([p_scores, n_scores])
        y = torch.cat([torch.ones(p_scores.shape[0]), torch.zeros(n_scores.shape[0])]).to(self.device)
        log_loss = self.cross_loss(torch.sigmoid(scores), y)

        # BPRLoss for recommendation
        tar_user_emb = b_user_embedding[user_ids]
        inv_user_emb = invariant_user_emb[-1][user_ids]
        var_user_emb = tar_user_emb - inv_user_emb

        invariant_user_emb = torch.mean(invariant_user_emb, dim=0)
        inv_user_emb = invariant_user_emb[user_ids]

        item_emb = item_embedding[rec_item_ids]
        inv_scores = torch.sum(inv_user_emb.unsqueeze(1) * item_emb, dim=-1)
        tar_scores = torch.sum(var_user_emb.unsqueeze(1) * item_emb, dim=-1)
        scores = self.beta * inv_scores + (1 - self.beta) * tar_scores
        p_scores, n_scores = torch.chunk(scores, 2, dim=-1)
        bpr_loss = self.bpr_loss(p_scores, n_scores)

        reg_loss = self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        loss = self.lamb * log_loss + (1 - self.lamb) * bpr_loss + reg_loss

        return loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            ini_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.global_Graph(ini_embeddings)
            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            user_embs = [user_embedding]
            for graph in self.behavior_Graphs:
                behavior_embeddings = graph(ini_embeddings)
                b_user_embedding, b_item_embedding = torch.split(behavior_embeddings,
                                                                 [self.n_users + 1, self.n_items + 1])
                user_embs.append(b_user_embedding)

            user_embs = torch.stack(user_embs)
            invariant_user_emb = self.M(user_embs)
            tar_inv_enb = invariant_user_emb[-1]
            invariant_user_emb = torch.mean(invariant_user_emb, dim=0)

            b_user_embedding = b_user_embedding - tar_inv_enb

            self.storage_user_embeddings = torch.stack([self.beta * invariant_user_emb, (1 - self.beta) * b_user_embedding]).transpose(0, 1)
            self.storage_item_embeddings = item_embedding.transpose(0, 1).unsqueeze(0)

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings)
        scores = torch.sum(scores, dim=1)

        return scores

