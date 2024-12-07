#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author:
# @Date  : 2021/11/1 11:38
# @Desc  :
import argparse
import os
import random
import json
import torch
import scipy.sparse as sp

from collections import defaultdict
from itertools import combinations

from torch.utils.data import Dataset, DataLoader
import numpy as np

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class TestDate(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)

# class TrainDate(Dataset):
#     def __init__(self, user_count, item_count, neg_count, behavior_dict=None, behaviors=None):
#         self.user_count = user_count
#         self.item_count = item_count
#         self.behavior_dict = behavior_dict
#         self.behaviors = behaviors
#         self.neg_count = neg_count
#         self.total_items = np.arange(1, item_count + 1)
#
#     def __getitem__(self, idx):
#
#         items = self.behavior_dict['all'].get(str(idx + 1), None)
#         if items is None:
#             signal = [0] * (self.neg_count + 2)
#         else:
#             pos = random.sample(items, 1)[0]
#             signal = [idx + 1, pos]
#             for i in range(self.neg_count):
#                 neg = random.randint(1, self.item_count)
#                 while np.isin(neg, items):
#                     neg = random.randint(1, self.item_count)
#                 signal.append(neg)
#
#         target_items = self.behavior_dict[self.behaviors[-1]].get(str(idx + 1), None)
#         if target_items is None:
#             signal = [0] * (self.neg_count + 4)
#         else:
#             pos = random.sample(target_items, 1)[0]
#             neg = random.randint(1, self.item_count)
#             while np.isin(neg, items):
#                 neg = random.randint(1, self.item_count)
#             signal.append(pos)
#             signal.append(neg)
#         return np.array(signal)
#
#     def __len__(self):
#         return self.user_count


# class TrainDate(Dataset):
#     def __init__(self, user_count, item_count, neg_count, behavior_dict=None, behaviors=None):
#         self.user_count = user_count
#         self.item_count = item_count
#         self.behavior_dict = behavior_dict
#         self.behaviors = behaviors
#         self.neg_count = neg_count
#         self.total_items = np.arange(1, item_count + 1)
#
#     def __getitem__(self, idx):
#
#         def __samples(item_list):
#             if item_list is None:
#                 single = [0, 0, 0]
#             else:
#                 pos = random.sample(item_list, 1)[0]
#                 single = [idx + 1, pos]
#                 neg = random.randint(1, self.item_count)
#                 while np.isin(neg, item_list):
#                     neg = random.randint(1, self.item_count)
#                 single.append(neg)
#             return single
#
#         result = []
#         items = self.behavior_dict['all'].get(str(idx + 1), None)
#         result.append(__samples(items))
#
#         for behavior in self.behaviors:
#             target_items = self.behavior_dict[behavior].get(str(idx + 1), None)
#             result.append(__samples(target_items))
#
#         return np.array(result)
#
#     def __len__(self):
#         return self.user_count

# class TrainDate(Dataset):
#     def __init__(self, user_count, item_count, neg_count, behavior_dict=None, behaviors=None):
#         self.user_count = user_count
#         self.item_count = item_count
#         self.behavior_dict = behavior_dict
#         self.behaviors = behaviors
#         self.neg_count = neg_count
#         self.total_items = np.arange(1, item_count + 1)
#
#     def __getitem__(self, idx):
#
#
#         items = self.behavior_dict['all'].get(str(idx + 1), None)
#         if items is None:
#             single = [0] * (self.neg_count + 2)
#         else:
#             pos = random.sample(items, 1)[0]
#             single = [idx + 1, pos]
#
#             for i in range(self.neg_count):
#                 neg = random.randint(1, self.item_count)
#                 while np.isin(neg, items):
#                     neg = random.randint(1, self.item_count)
#                 single.append(neg)
#         result = [single]
#
#         place_holder = [0] * (self.neg_count - 1)
#         for behavior in self.behaviors:
#             items = self.behavior_dict[behavior].get(str(idx + 1), None)
#             if items is None:
#                 # To facilitate training, keep user ids consistent across all behaviors
#                 single = [idx + 1, 0, 0]
#                 single.extend(place_holder)
#             else:
#                 single = [idx + 1]
#                 pos = random.sample(items, 1)[0]
#                 neg = random.randint(1, self.item_count)
#                 while np.isin(neg, items):
#                     neg = random.randint(1, self.item_count)
#                 single.append(pos)
#                 single.append(neg)
#                 single.extend(place_holder)
#             result.append(single)
#
#         return np.array(result)
#
#     def __len__(self):
#         return self.user_count


class TrainDate(Dataset):
    def __init__(self, user_count, item_count, neg_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors
        self.neg_count = neg_count
        self.total_items = np.arange(1, item_count + 1)

    def __getitem__(self, idx):

        all_items = self.behavior_dict['all'].get(str(idx + 1), None)

        tar_items = self.behavior_dict[self.behaviors[-1]].get(str(idx + 1), None)
        if all_items is None:
            single = [0] * (self.neg_count + 2)
        else:
            pos = random.sample(all_items, 1)[0]
            single = [idx + 1, pos]
            for i in range(self.neg_count):
                neg = random.randint(1, self.item_count)
                while np.isin(neg, all_items):
                    neg = random.randint(1, self.item_count)
                single.append(neg)
        result = [single]

        place_holder = [0] * (self.neg_count - 1)
        if tar_items is None:
            # To facilitate training, keep user ids consistent across all behaviors
            single = [idx + 1, 0, 0]
            single.extend(place_holder)
        else:
            single = [idx + 1]
            pos = random.sample(tar_items, 1)[0]
            neg = random.randint(1, self.item_count)
            while np.isin(neg, all_items):
                neg = random.randint(1, self.item_count)
            single.append(pos)
            single.append(neg)
            single.extend(place_holder)
        result.append(single)
        return np.array(result)

    def __len__(self):
        return self.user_count


class DataSet(object):

    def __init__(self, args):

        self.behaviors = args.behaviors
        self.path = args.data_path
        self.loss_type = args.loss_type
        self.neg_count = args.neg_count

        self.__get_count()
        # self.__get_pos_sampling()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_mask_dict()
        self.__get_sparse_interact_dict()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items()])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_pos_sampling(self):
        with open(os.path.join(self.path, 'pos_sampling.txt'), encoding='utf-8') as f:
            data = f.readlines()
            arr = []
            for line in data:
                line = line.strip('\n').strip().split()
                arr.append([int(x) for x in line])
            self.pos_sampling = arr

    def __get_behavior_items(self):
        """
        load the list of items corresponding to the user under each behavior
        :return:
        """
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                self.train_behavior_dict[behavior] = b_dict

        with open(os.path.join(self.path, 'all_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.train_behavior_dict['all'] = b_dict

    # def __get_all_item_users(self):
    #     with open(os.path.join(self.path, 'all.txt'), encoding='utf-8') as f:

    def __get_test_dict(self):
        """
        load the list of items that the user has interacted with in the test set
        :return:
        """
        with open(os.path.join(self.path, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.test_interacts = b_dict

    def __get_validation_dict(self):
        """
        load the list of items that the user has interacted with in the validation set
        :return:
        """
        with open(os.path.join(self.path, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            self.validation_interacts = b_dict

    def __get_mask_dict(self):

        valid_mask = defaultdict(list)
        for key, value in self.train_behavior_dict[self.behaviors[-1]].items():
            valid_mask[key].extend(value)
        for key, value in self.test_interacts.items():
            valid_mask[key].extend(value)
        self.valid_mask = dict(valid_mask)

        test_mask = defaultdict(list)
        for key, value in self.train_behavior_dict[self.behaviors[-1]].items():
            test_mask[key].extend(value)
        for key, value in self.validation_interacts.items():
            test_mask[key].extend(value)
        self.test_mask = dict(test_mask)

    def __get_sparse_interact_dict(self):

        self.item_behaviour_degree = []
        self.inter_matrix = []

        # index_list = list(range(len(self.behaviors) - 1))
        index_list = list(range(len(self.behaviors)))[::-1]

        inter_matrix_list = []
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
                data = f.readlines()
                row = []
                col = []
                for line in data:
                    line = line.strip('\n').strip().split()
                    row.append(int(line[0]))
                    col.append(int(line[1]))
                inter_matrix_list.append([row, col])

        new_set = [list(subset) for i in range(1, len(index_list) + 1) for subset in combinations(index_list, i)]
        # new_set = [[x] for x in index_list]
        # new_set.append(index_list)

        # for row, col in inter_matrix_list:
        #     values = np.ones_like(row)
        #     inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
        #     self.inter_matrix.append(inter_matrix)

        # row, col = inter_matrix_list[-1]
        # values = np.ones_like(row)
        # inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
        # self.inter_matrix.append(inter_matrix)

        # for sub_set in new_set:
        #     tmp_idx = [inter_matrix_list[-1]]
        #     for i in sub_set:
        #         tmp_idx.append(inter_matrix_list[i])
        #     idxs = np.concatenate(tmp_idx, axis=1)
        #     tmp_data = np.ones_like(idxs[0])
        #     tmp_matrix = sp.coo_matrix((tmp_data, idxs), [self.user_count + 1, self.item_count + 1])
        #     self.inter_matrix.append(tmp_matrix)

        for idx, sub_set in enumerate(new_set):
            tmp_idx = []
            for i in sub_set:
                tmp_idx.append(inter_matrix_list[i])
            idxs = np.concatenate(tmp_idx, axis=1)
            tmp_data = np.ones_like(idxs[0])
            tmp_matrix = sp.coo_matrix((tmp_data, idxs), [self.user_count + 1, self.item_count + 1])
            if idx < len(index_list):
                item_degree = tmp_matrix.sum(axis=0)
                item_degree = torch.from_numpy(item_degree).squeeze()
                self.item_behaviour_degree.append(item_degree)
            self.inter_matrix.append(tmp_matrix)

        self.item_behaviour_degree = torch.stack(self.item_behaviour_degree, dim=0).T


    # def __get_sparse_interact_dict(self):
    #     """
    #     load graphs
    #
    #     :return:
    #     """
    #     # self.edge_index = {}
    #     self.item_behaviour_degree = []
    #     # self.user_behaviour_degree = []
    #     # self.all_item_user = {}
    #     # self.behavior_item_user = {}
    #     self.inter_matrix = []
    #     # self.user_item_inter_set = []
    #     all_row = []
    #     all_col = []
    #     for behavior in self.behaviors:
    #         # tmp_dict = {}
    #         with open(os.path.join(self.path, behavior + '.txt'), encoding='utf-8') as f:
    #             data = f.readlines()
    #             row = []
    #             col = []
    #             for line in data:
    #                 line = line.strip('\n').strip().split()
    #                 row.append(int(line[0]))
    #                 col.append(int(line[1]))
    #
    #                 # if line[1] in self.all_item_user:
    #                 #     self.all_item_user[line[1]].append(int(line[0]))
    #                 # else:
    #                 #     self.all_item_user[line[1]] = [int(line[0])]
    #
    #                 # if line[1] in tmp_dict:
    #                 #     tmp_dict[line[1]].append(int(line[0]))
    #                 # else:
    #                 #     tmp_dict[line[1]] = [int(line[0])]
    #             # self.behavior_item_user[behavior] = tmp_dict
    #             # indices = np.vstack((row, col))
    #             # indices = torch.LongTensor(indices)
    #
    #             values = torch.ones(len(row), dtype=torch.float32)
    #             inter_matrix = sp.coo_matrix((values, (row, col)), [self.user_count + 1, self.item_count + 1])
    #             # user_item_set = [list(row.nonzero()[1]) for row in inter_matrix.tocsr()]
    #             self.inter_matrix.append(inter_matrix)
    #             # self.user_item_inter_set.append(user_item_set)
    #             # user_degree = inter_matrix.sum(axis=1)
    #             item_degree = inter_matrix.sum(axis=0)
    #             # user_degree = torch.from_numpy(user_degree).squeeze()
    #             item_degree = torch.from_numpy(item_degree).squeeze()
    #             self.item_behaviour_degree.append(item_degree)
    #             # self.user_behaviour_degree.append(user_degree)
    #
    #             # user_item_set = [set(row.nonzero()[1]) for row in user_item_inter_matrix]
    #             # item_user_set = [set(user_inter[:, j].nonzero()[0]) for j in range(user_inter.shape[1])]
    #
    #             # user_inter = torch.sparse.FloatTensor(indices, values, [self.user_count + 1, self.item_count + 1]).to_dense()
    #             # self.item_behaviour_degree.append(user_inter.sum(dim=0))
    #             # self.user_behaviour_degree.append(user_inter.sum(dim=1))
    #             # col = [x + self.user_count + 1 for x in col]
    #             # row, col = [row, col], [col, row]
    #             # row = torch.LongTensor(row).view(-1)
    #             all_row.extend(row)
    #             # col = torch.LongTensor(col).view(-1)
    #             all_col.extend(col)
    #             # edge_index = torch.stack([row, col])
    #             # self.edge_index[behavior] = edge_index
    #     # self.all_item_user = {key: list(set(value)) for key, value in self.all_item_user.items()}
    #     self.item_behaviour_degree = torch.stack(self.item_behaviour_degree, dim=0).T
    #     # self.user_behaviour_degree = torch.stack(self.user_behaviour_degree, dim=0).T
    #     # all_row = torch.cat(all_row, dim=-1)
    #     # all_col = torch.cat(all_col, dim=-1)
    #     # all_row = all_row.tolist()
    #     # all_col = all_col.tolist()
    #     # self.all_edge_index = list(set(zip(all_row, all_col)))
    #     all_edge_index = list(set(zip(all_row, all_col)))
    #     all_row = [sub[0] for sub in all_edge_index]
    #     all_col = [sub[1] for sub in all_edge_index]
    #     values = torch.ones(len(all_row), dtype=torch.float32)
    #     self.all_inter_matrix = sp.coo_matrix((values, (all_row, all_col)), [self.user_count + 1, self.item_count + 1])
    #
    #     # self.all_edge_index = torch.LongTensor(self.all_edge_index).T



    def train_dataset(self):
        return TrainDate(self.user_count, self.item_count, self.neg_count, self.train_behavior_dict, self.behaviors)


    def validate_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.validation_interacts.keys()))

    def test_dataset(self):
        return TestDate(self.user_count, self.item_count, samples=list(self.test_interacts.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)
    parser.add_argument('--behaviors', type=list, default=['cart', 'click', 'collect', 'buy'], help='')
    parser.add_argument('--data_path', type=str, default='../data/Tmall', help='')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--neg_count', type=int, default=4)
    args = parser.parse_args()
    dataset = DataSet(args)
    loader = DataLoader(dataset=dataset.train_dataset(), batch_size=5, shuffle=True)
    for index, item in enumerate(loader):
        print(index, '-----', item)
