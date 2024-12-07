#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author:
# @Date  : 2023/9/23 15:25
# @Desc  :
import argparse
import ast
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from data_set import DataSet

from trainer import Trainer

seed = 2021
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # True can improve train speed
    torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='')
    parser.add_argument('--beta', type=float, default=0.5, help='')
    parser.add_argument('--kl_reg', type=float, default=1.0, help='')
    parser.add_argument('--ort_reg', type=float, default=1.0, help='')
    parser.add_argument('--log_reg', type=float, default=1.0, help='')
    parser.add_argument('--nce_reg', type=float, default=1.0, help='')
    parser.add_argument('--bpr_reg', type=float, default=1.0, help='')
    parser.add_argument('--layers', type=int, default=2)

    parser.add_argument('--data_name', type=str, default='yelp', help='')
    parser.add_argument('--behaviors', type=str, help='')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--neg_count', type=int, default=10)
    parser.add_argument('--nce_temperature', type=float, default=0.5)

    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--gpu_no', type=int, default=1, help='')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=200, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='', help='')
    parser.add_argument('--log_name', type=str, default='tmall', help='')
    parser.add_argument('--model_name', type=str, default='model_nce_all_pt_item_linear', help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')

    args = parser.parse_args()
    if args.data_name == 'tmall':
        args.data_path = '../data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'yelp':
        args.data_path = '../data/Yelp'
        args.behaviors = ['tip', 'neutral', 'neg', 'pos']
    elif args.data_name == 'ml':
        args.data_path = '../data/ML10M'
        args.behaviors = ['neutral', 'neg', 'pos']
    elif args.data_name == 'taobao':
        args.data_path = '../data/taobao'
        args.behaviors = ['view', 'cart', 'buy']
    elif args.data_name == 'tmall_cold':
        args.data_path = '../data/Tmall_cold_1000'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'yelp_cold':
        args.data_path = '../data/Yelp_cold_1000'
        args.behaviors = ['tip', 'neutral', 'neg', 'pos']
    elif args.data_name == 'ml_cold':
        args.data_path = '../data/ML10M_cold_1000'
        args.behaviors = ['neutral', 'neg', 'pos']
    elif args.data_name == 'taobao_cold':
        args.data_path = '../data/taobao_cold_1000'
        args.behaviors = ['view', 'cart', 'buy']
    else:
        raise Exception('data_name cannot be None')

    # args.behaviors = ast.literal_eval(args.behaviors)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    # args.train_writer = SummaryWriter('./log/train/' + logfile)
    # args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.log_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # model = UIPL(args, dataset).to(args.device)
    modules = __import__(args.model_name)
    model = getattr(modules, 'UIPL')(args, dataset).to(args.device)

    trainer = Trainer(model, dataset, args)

    logger.info(args.__str__())
    logger.info(model)

    trainer.train_model()
    # trainer.evaluate(0, 5, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, dataset.valid_mask)
    logger.info('train end total cost time: {}'.format(time.time() - start))



