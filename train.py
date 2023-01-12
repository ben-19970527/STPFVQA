#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:35:32 2019

@author: xuguangying
"""
import datetime
import os
import random
import warnings
from argparse import ArgumentParser

# from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from scipy import stats
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from net import MMT

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings("ignore")


class dataset(Dataset):
    """UCF101 Landmarks dataset."""

    def __init__(self, name, root_dir, list_score_dis):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.scale = 82.80008516
        self.max_len = 125
        self.feat_dim = 4096
        self.list_score_dis = list_score_dis
        self.ref_features = np.zeros((len(name), self.max_len, self.feat_dim))
        self.dis_features = np.zeros((len(name), self.max_len, self.feat_dim))
        self.mask = np.zeros((len(name), self.max_len))
        self.length = np.zeros((len(name), 1))
        self.mos = np.zeros((len(name), 1))
        for i, disname in enumerate(name):
            # print(disname)
            refname = str.split(disname, "_")[0] + "_" + str.split(disname, "_")[1] + "_ref.yuv"
            disfeatures = np.load(root_dir + '/' + disname + '_resnet-50_res5c.npy')
            # print(disfeatures.shape)
            reffeatures = np.load(root_dir + '/' + refname + '_resnet-50_res5c.npy')
            # print(reffeatures.shape)
            self.length[i] = disfeatures.shape[0]
            self.dis_features[i, :disfeatures.shape[0], :] = disfeatures
            self.ref_features[i, :reffeatures.shape[0], :] = reffeatures
            self.mask[i, :reffeatures.shape[0]] = reffeatures.shape[0]
            self.mos[i] = self.list_score_dis[i]  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    # get (16,240,320,3)
    def __getitem__(self, idx):
        sample = self.ref_features[idx], self.dis_features[idx], self.mask[idx], self.length[idx], self.label[idx]
        return sample


class dataset_live(Dataset):
    """UCF101 Landmarks dataset."""

    def __init__(self, name, root_dir):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.scale = 81.1601
        self.max_len = 500
        self.feat_dim = 4096
        self.ref_features = np.zeros((len(name), self.max_len, self.feat_dim))
        self.dis_features = np.zeros((len(name), self.max_len, self.feat_dim))
        self.mask = np.zeros((len(name), self.max_len))
        self.length = np.zeros((len(name), 1))
        self.mos = np.zeros((len(name), 1))
        for i, disname in enumerate(name):
            refname = str.split(disname, "_")[0][0:2] + '1_' + str.split(disname, "_")[1]
            disname = str.split(disname, ".")[0]
            refname = str.split(refname, ".")[0]
            disfeatures = np.load(root_dir + '/' + disname + '_resnet-50_res5c.npy')
            # print(disfeatures.shape)
            reffeatures = np.load(root_dir + '/' + refname + '_resnet-50_res5c.npy')
            # print(reffeatures.shape)
            self.length[i] = disfeatures.shape[0]
            self.dis_features[i, :disfeatures.shape[0], :] = disfeatures
            self.ref_features[i, :reffeatures.shape[0], :] = reffeatures

            self.mask[i, :reffeatures.shape[0]] = reffeatures.shape[0]
            self.mos[i] = np.load(root_dir + '/' + disname + '_score.npy')  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    # get (16,240,320,3)
    def __getitem__(self, idx):
        sample = self.ref_features[idx], self.dis_features[idx], self.mask[idx], self.length[idx], self.label[idx]
        return sample


if __name__ == "__main__":
    print("在数据集上跑")
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.00003,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='csiq_live', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='csiq_live', type=str,
                        help='model name (default: VSFA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    args.decay_interval = int(args.epochs / 10)
    args.decay_ratio = 0.8

    # torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'live':
        features_dir = 'CNN_features_live4/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'

    import pandas as pd

    for kk in range(10):
        args.exp_id = kk
        print('EXP ID: {}'.format(args.exp_id))
        print(args.database)
        print(args.model)

        device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
        scores = pd.read_table("csiq_DMOS.txt", names=['a'])
        scores = list(scores['a'])
        video_names = pd.read_table("csiq_name.txt", names=['a'])
        video_names = list(video_names['a'])
        list_name_ref = []
        list_score_ref = []
        list_name_dis = []
        list_score_dis = []
        for i in range(len(scores)):
            name, disext = os.path.splitext(video_names[i])
            video_names[i] = video_names[i].rstrip()
            if name.endswith('00'):
                list_name_ref.append(video_names[i])
                list_score_ref.append(scores[i])
            else:
                list_name_dis.append(video_names[i])
                list_score_dis.append(scores[i])
        list_zong = []
        for i in range(len(list_score_dis)):
            list_ = []
            list_.append(list_name_dis[i])
            list_.append(list_score_dis[i])
            list_zong.append(list_)

        random.shuffle(list_zong)
        # print(list)
        list_name_dis = []
        list_score_dis = []

        for i in range(len(list_zong)):
            list_name_dis.append(list_zong[i][0])
            list_score_dis.append(list_zong[i][1])

        root_list1 = r'/remote-home/cs_cs_shy/lc/ssim_resnet_lstm_vqa/CNN_features_csiq2'
        traindata = dataset(list_name_dis, root_list1, list_score_dis)
        index = []
        ref = ['pa1_25fps.yuv', 'rb1_25fps.yuv',
               'rh1_25fps.yuv', 'tr1_25fps.yuv', 'st1_25fps.yuv',
               'sf1_25fps.yuv', 'bs1_25fps.yuv', 'sh1_50fps.yuv', 'mc1_50fps.yuv', 'pr1_50fps.yuv'
               ]
        dis_name = []
        for i in range(len(ref)):
            for j in range(2, 17):
                dis_name.append(ref[i].replace("1", str(j)))
        # print(dis_name)
        for i in range(150):
            index.append(i)
        random.shuffle(dis_name)
        # print(len(dis_name))
        root_list = r'/remote-home/cs_cs_shy/lc/ssim_resnet_lstm_vqa/CNN_features_live3'

        testdata = dataset_live(dis_name, root_list)
        traindataloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testdataloader = DataLoader(testdata, batch_size=args.batch_size, shuffle=True, num_workers=2)
        device_ids = range(0, torch.cuda.device_count())
        base_config = edict(
            hidden_size=1536,
            vocab_size=None,  # get from word2idx
            video_feature_size=4096,
            max_position_embeddings=1202,  # get from max_seq_len
            type_vocab_size=2,
            layer_norm_eps=1e-12,  # bert layernorm
            hidden_dropout_prob=0.1,  # applies everywhere except attention
            num_hidden_layers=2,  # number of transformer layers
            attention_probs_dropout_prob=0.3,  # applies only to self attention
            intermediate_size=1536,  # after each self attention
            num_attention_heads=12,
            memory_dropout_prob=0.1
        )
        print("video_feature_size:", 4096, "num_attention_heads:", 12, "num_hidden_layers=2")
        model = torch.nn.DataParallel(MMT(base_config).to(device))
        total = sum(p.numel() for p in model.parameters())
        print("Total params: %.2fM" % (total / 1e6))
        print("==============")
        if not os.path.exists('models'):
            os.makedirs('models')
        trained_model_file = 'models/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)
        if not os.path.exists('results'):
            os.makedirs('results')
        save_result_file = 'results/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)
        if not args.disable_visualization:  # Tensorboard Visualization
            writer = SummaryWriter(log_dir='{}/XP{}-{}-{}-{}-{}-{}-{}'
                                   .format(args.log_dir, args.exp_id, args.database, args.model,
                                           args.lr, args.batch_size, args.epochs,
                                           datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

        criterion = nn.L1Loss()  # L1 loss
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)

        best_val_criterion = -1  # SROCC min
        scale = 82.80008516

        for epoch in range(args.epochs):
            # for epoch in tqdm(range(1, 1 + args.epochs), unit='epoch', initial=1, total=1 + args.epochs):
            model.train()
            L = 0
            for i, (ref, dis, mask, length, mos) in enumerate(traindataloader):
                label = mos.to(device).float()
                ref = ref.to(device).float()
                dis = dis.to(device).float()
                optimizer.zero_grad()
                # print("sss22222222", ref.shape, dis.shape)  # torch.Size([16, 500, 4096]) torch.Size([16, 500, 4096])
                # print(ref.shape, dis.shape)
                # length = length.to(device).float()
                # print("kkkkkkkkkkkk", length.shape)
                outputs = model(ref, dis, mask, length.float())
                # outputs = model(ref, dis, mask, length.float())
                quality_loss = criterion(outputs, label)
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                loss = quality_loss + 0.00001 * regularization_loss
                loss.backward()
                optimizer.step()
                L = L + loss.item()
            train_loss = L / (i + 1)
            # print(epoch, train_loss)

            model.eval()

            if args.test_ratio > 0 and not args.notest_during_training:
                # y_pred = np.zeros(len(testdata), dtype=np.float64)
                # y_test = np.zeros(len(testdata), dtype=np.float64)
                y_pred = np.zeros((int(len(testdata) / args.batch_size), args.batch_size))
                y_test = np.zeros((int(len(testdata) / args.batch_size), args.batch_size))
                L = 0
                with torch.no_grad():
                    for i, (ref, dis, mask, length, mos) in enumerate(testdataloader):
                        # print(mos.shape, mos)
                        y_test[i] = scale * mos.squeeze().cpu().numpy()  #
                        label = mos.to(device).float()
                        ref = ref.to(device).float()
                        dis = dis.to(device).float()
                        # length = length.to(device).float()
                        # print("lllllll", length.shape)
                        outputs = model(ref, dis, mask, length.float())
                        # outputs = torch.mean(outputs, 0, keepdim=True)
                        y_pred[i] = scale * outputs.squeeze().cpu().numpy()
                        loss = criterion(outputs, label)
                        L = L + loss.item()
                test_loss = L / (i + 1)
                y_pred = y_pred.flatten()
                y_test = y_test.flatten()
                PLCC = stats.pearsonr(y_pred, y_test)[0]
                SROCC = stats.spearmanr(y_pred, y_test)[0]
                RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
                KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

            writer.add_scalar("loss/test", test_loss, epoch)  #
            writer.add_scalar("loss/train", train_loss, epoch)  #
            writer.add_scalar("SROCC/test", SROCC, epoch)  #
            writer.add_scalar("KROCC/test", KROCC, epoch)  #
            writer.add_scalar("PLCC/test", PLCC, epoch)  #
            writer.add_scalar("RMSE/test", RMSE, epoch)  #
            # Update the model with the best val_SROCC
            if SROCC > best_val_criterion:
                print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(args.exp_id, epoch))
                print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
                if args.test_ratio > 0 and not args.notest_during_training:
                    np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE))
                trained_model_file = 'models/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)
                torch.save(model.state_dict(), trained_model_file)
                best_val_criterion = SROCC  # update best val SROCC
