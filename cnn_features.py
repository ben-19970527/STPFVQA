
import os

import skimage.io as io
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os
import numpy as np
import random
from argparse import ArgumentParser
import math
'''使用resnet50提取原始视频和失真视频特征'''


class datasetjian(Dataset):
    """UCF101 Landmarks dataset."""

    def __init__(self, name, root_dir, transform=None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            root_dir (string): Directory with all the video frames.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # print("aaaaaaaaaaaaaaaaaaa")
        self.name = name
        self.root_dir = root_dir
        self.transform = transform
        self.root_list = root_dir
        self.temp = 0
        self.length = 125

    def __len__(self):
        return len(self.name)

    # get (16,240,320,3)
    def __getitem__(self, idx):

        disname = self.name[idx]
        # scoredir = disname + "_score.npy"
        # scoredir = os.path.join(self.root_list, scoredir)
        # print(scoredir)
        # score = np.load(scoredir) / 81.1601
        # print(score)
        dis_video = self.get_single_video_x(disname)

        # dis_video = dis_video.transpose((0, 3, 1, 2))
        # ref_video = ref_video.transpose((0, 3, 1, 2))
        # sample = {'ref_video': ref_video, 'dis_video': dis_video, 'video_label': score}

        # if self.transform:
        #     sample = self.transform(sample)
        sample = {'name': disname,
                  'video': dis_video,
                  }
        return sample

    def get_single_video_x(self, disname):
        # print(disname) pa2_25fps.yuv
        disname, disext = os.path.splitext(disname)
        dis_dir = os.path.join(disname + '/')
        # dis_dir = os.path.join(disname)
        # print(dis_dir)
        dis_pic_path = os.path.join(self.root_list, dis_dir)
        dis_pic_names = os.listdir(dis_pic_path)
        num = len(dis_pic_names)
        # print("hhhhhhhhhhhhhhhhhhhhhhh",dis_pic_path, dis_pic_names, num)
        count = int(num / self.length)
        image_id = 1
        video_x = np.zeros((self.length, 3, 480, 832))
        # video_x = np.zeros((self.length, 3, 480, 640))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for i in range(self.length):
            if num != 250 or num != 500:
                if i % 2 == 0:
                    count = int(num / 125)
                else:
                    count = math.ceil(num / 125)
            s = image_id
            image_name = str(s) + '.jpg'
            image_path = os.path.join(dis_pic_path, image_name)
            tmp_image = io.imread(image_path)
            tmp_image = transform(tmp_image)
            video_x[i, :, :, :] = tmp_image
            image_id += count
            if image_id > num:
                image_id = num
        video_x = torch.from_numpy(np.asarray(video_x))
        return video_x


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

        self.model_layer1 = self.features[0:-2]
        self.model_layer2 = self.features[-2:-1]
        self.model_layer3 = self.features[-1]

    def forward(self, x):
        # features@: 7->res5c
        y1 = self.model_layer1(x)  # torch.Size([2, 64, 120, 120])
        y2 = self.model_layer2(y1)  # torch.Size([2, 128, 60, 60])
        y3 = self.model_layer3(y2)
        # features_mean1 = nn.functional.adaptive_avg_pool2d(y1, 1)
        # features_std1 = global_std_pool2d(y1)
        # features_mean2 = nn.functional.adaptive_avg_pool2d(y2, 1)
        # features_std2 = global_std_pool2d(y2)
        features_mean3 = nn.functional.adaptive_avg_pool2d(y3, 1)
        features_std3 = global_std_pool2d(y3)
        return features_mean3, features_std3


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def global_std_pool3d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], x.size()[2], -1, 1),
                     dim=3, keepdim=True)


def get_features(video_data, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].float().to(device)
            features_mean, features_std = extractor(batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:frame_end].float().to(device)
        features_mean, features_std = extractor(last_batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        output = torch.cat((output1, output2), 1).squeeze()

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='csiq', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'live':
        videos_dir = r'/remote-home/cs_cs_shy/lc/ssim_resnet_lstm_vqa/CNN_features_live1'  # videos dir
        features_dir = 'CNN_features_live4/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'ivc':
        videos_dir = r'/remote-home/cs_cs_shy/lc/ssim_resnet_lstm_vqa/CNN_features_ivc1'
        features_dir = 'CNN_features_ivc2/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'csiq':
        videos_dir = '/remote-home/cs_cs_shy/lc/ssim_resnet_lstm_vqa/CNN_features_csiq1'
        features_dir = 'CNN_features_csiq2/'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda:0" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    import pandas as pd

    '''live数据集'''
    scores = pd.read_table("csiq_DMOS.txt", names=['a'])
    scores = list(scores['a'])
    video_names = pd.read_table("csiq_name.txt", names=['a'])
    video_names = list(video_names['a'])
    ref = ['BasketballDrive_832x480_ref.yuv', 'BQMall_832x480_ref.yuv', 'BQTerrace_832x480_ref.yuv',
           'Cactus_832x480_ref.yuv', 'Carving_832x480_ref.yuv',
           'Chipmunks_832x480_ref.yuv', 'Flowervase_832x480_ref.yuv', 'Keiba_832x480_ref.yuv', 'Kimono_832x480_ref.yuv',
           'ParkScene_832x480_ref.yuv', 'PartyScene_832x480_ref.yuv', 'Timelapse_832x480_ref.yuv']
    '''ivc数据集'''
    # scores = pd.read_table("ivc_score.txt", names=['a'])
    # scores = list(scores['a'])
    # video_names = pd.read_table("ivc_name.txt", names=['a'])
    # video_names = list(video_names['a'])
    # dataset = datasetjian(video_names, videos_dir, scores)

    # for i in range(len(dataset)):
    #     current_data = dataset[i]
    #     current_video_name = current_data['name']
    #     current_video = current_data['video']
    #     # current_score = current_data['score']
    #     current_video_name = current_video_name.rstrip()
    #     print('Video {}: length {}'.format(i, current_video.shape[0]), current_video_name)
    #     features = get_features(current_video, args.frame_batch_size, device)
    #     print(features.shape)
    #     np.save(features_dir + current_video_name + '_resnet-50_res5c.npy', features.to('cpu').numpy())

    dataset = datasetjian(video_names, videos_dir)
    for i in range(len(dataset)):
        current_data = dataset[i]
        current_video_name = current_data['name']
        current_video = current_data['video']
        current_video_name = current_video_name.rstrip()
        print('Video {}: length {}'.format(i, current_video.shape[0]), current_video_name)
        features = get_features(current_video, args.frame_batch_size, device)
        print(features.shape)
        np.save(features_dir + current_video_name + '_resnet-50_res5c.npy', features.to('cpu').numpy())
