import argparse
import json
import os

import numpy as np
import torch
# from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt

from jelly.face.base.landmarks import draw_landmarks
from places2 import Places2
#
#
# class InfiniteSampler(data.sampler.Sampler):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples
#
#     def __iter__(self):
#         return iter(self.loop())
#
#     def __len__(self):
#         return 2 ** 31
#
#     def loop(self):
#         i = 0
#         order = np.random.permutation(self.num_samples)
#         while True:
#             yield order[i]
#             i += 1
#             if i >= self.num_samples:
#                 np.random.seed()
#                 order = np.random.permutation(self.num_samples)
#                 i = 0


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
parser.add_argument('--data_folder', type=str, default='val_data')
parser.add_argument('--segmentation_mask_folder', type=str, default='val_mask')
parser.add_argument('--landmarks', type=str, default='')
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()


def get_landmarks(landmark_type=None):
    landmarks_path = os.path.join(args.root, args.landmarks)
    with open(landmarks_path) as f:
        landmarks = json.loads(f.read())
    if landmark_type is None:
        return landmarks
    else:
        return {id_: np.array(points['face0'][landmark_type], dtype=np.uint)
                for id_, points in landmarks.items()}



torch.backends.cudnn.benchmark = True
device = torch.device('cuda')


size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])


contour_landmarks = get_landmarks('contour')
nose_landmarks = get_landmarks('nose')
mouth_landmarks = get_landmarks('mouth')

landmarks_path = os.path.join(args.root, args.landmarks)

dataset = Places2(
    os.path.join(args.root, args.data_folder),
    os.path.join(args.root, args.segmentation_mask_folder),
    landmarks_path, img_tf, mask_tf, 'train')


for i in tqdm(range(len(dataset))):
    _ = dataset[i]
