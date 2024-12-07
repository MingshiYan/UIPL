#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2023/9/23 16:16
# @Desc  :
import os.path

import random
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from data_set import DataSet
from utils import BPRLoss, EmbLoss, ContrastiveLoss, KLLoss
from lightGCN import LightGCN
from vae import VAE


class UIPL(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(UIPL, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.reg_weight = args.reg_weight
        self.kl_reg = args.kl_reg
        self.ort_reg = args.ort_reg
        self.log_reg = args.log_reg
        self.nce_reg = args.nce_reg
        self.bpr_reg = args.bpr_reg
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.behaviors = args.behaviors
        degrees = dataset.item_behaviour_degree
        degrees = degrees / (degrees.sum(1, keepdim=True) + 1e-6)
        self.behavior_weight = degrees.T.unsqueeze(-1).to(self.device)
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.env_Graphs = nn.ModuleList([
            LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, inter) for inter in
            dataset.inter_matrix
        ])

        self.M = VAE(self.embedding_size, self.embedding_size // 2, self.embedding_size // 4)

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.cross_loss = nn.BCELoss()
        self.kl_loss = KLLoss()
        self.ort_loss = nn.MSELoss()
        self.nce_loss = ContrastiveLoss(size=len(self.behaviors) + 1, device=self.device,
                                        temperature=args.nce_temperature)

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
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def forward(self, batch_data):

        batch_data = batch_data.long()

        ini_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        pt_embedding = self.env_Graphs[-1](ini_embeddings)
        user_embs, item_emb = [], []
        for idx, graph in enumerate(self.env_Graphs):
            behavior_embeddings = graph(pt_embedding)
            b_user_embedding, b_item_embedding = torch.split(behavior_embeddings, [self.n_users + 1, self.n_items + 1])
            user_embs.append(b_user_embedding)
            if idx < len(self.behaviors):
                item_emb.append(b_item_embedding)

        agg_item_embedding = torch.stack(item_emb)
        agg_item_embedding = torch.sum(agg_item_embedding * self.behavior_weight, dim=0)

        user_embs = torch.stack(user_embs)
        user_embs = user_embs.permute(1, 0, 2)
        train_user_embs = user_embs[batch_data[:, 0, 0]]
        invariant_user_emb, mu, logvar = self.M(train_user_embs)
        kl_loss = self.kl_loss(mu, logvar)

        variant_user_emb = train_user_embs - invariant_user_emb

        # contrastive loss
        nce_loss = self.nce_loss(invariant_user_emb, variant_user_emb)

        sim = torch.sum(invariant_user_emb * variant_user_emb, dim=-1).view(-1)
        target = torch.zeros_like(sim)
        ort_loss = self.ort_loss(sim, target)

        # invariant_user_emb = invariant_user_emb.permute(1, 0, 2)
        # variant_user_emb = variant_user_emb.permute(1, 0, 2)

        inv_p_item_emb = agg_item_embedding[batch_data[:, 0, 1]]
        inv_n_item_emb = agg_item_embedding[batch_data[:, 0, 2:]]
        cross_p_scores = torch.sum(invariant_user_emb * inv_p_item_emb.unsqueeze(1), dim=-1)
        cross_n_scores = torch.sum(invariant_user_emb.unsqueeze(2) * inv_n_item_emb.unsqueeze(1), dim=-1)
        cross_p_scores = cross_p_scores.view(-1)
        cross_n_scores = cross_n_scores.view(-1)
        scores = torch.cat([cross_p_scores, cross_n_scores])
        y = torch.cat([torch.ones(cross_p_scores.shape[0]), torch.zeros(cross_n_scores.shape[0])]).to(self.device)
        log_loss = self.cross_loss(torch.sigmoid(scores), y)

        # BPRLoss for recommendation
        inv_user_emb = torch.mean(invariant_user_emb, dim=1)
        var_user_emb = variant_user_emb[:, 0]

        item_emb = agg_item_embedding[batch_data[:, -1, 1:3]]
        inv_scores = torch.sum(inv_user_emb.unsqueeze(1) * item_emb, dim=-1)
        tar_scores = torch.sum(var_user_emb.unsqueeze(1) * item_emb, dim=-1)
        scores = inv_scores + tar_scores
        p_scores, n_scores = torch.chunk(scores, 2, dim=-1)
        bpr_loss = self.bpr_loss(p_scores, n_scores)

        reg_loss = self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        loss = self.kl_reg * kl_loss + self.ort_reg * ort_loss + self.log_reg * log_loss + self.nce_reg * nce_loss + self.bpr_reg * bpr_loss + reg_loss

        return loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            ini_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            pt_embedding = self.env_Graphs[-1](ini_embeddings)
            user_embs, item_emb = [], []
            for idx, graph in enumerate(self.env_Graphs):
                behavior_embeddings = graph(pt_embedding)
                b_user_embedding, b_item_embedding = torch.split(behavior_embeddings,
                                                                 [self.n_users + 1, self.n_items + 1])
                user_embs.append(b_user_embedding)
                if idx < len(self.behaviors):
                    item_emb.append(b_item_embedding)

            item_embedding = torch.stack(item_emb)
            item_embedding = torch.sum(item_embedding * self.behavior_weight, dim=0)

            user_embs = torch.stack(user_embs)
            invariant_user_emb, _, _ = self.M(user_embs)
            b_user_embedding = user_embs[0] - invariant_user_emb[0]
            invariant_user_emb = torch.mean(invariant_user_emb, dim=0)

            self.storage_user_embeddings = torch.stack(
                [invariant_user_emb, b_user_embedding]).transpose(0, 1)
            self.storage_item_embeddings = item_embedding.transpose(0, 1).unsqueeze(0)

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings)
        scores = torch.sum(scores, dim=1)

        return scores

