import argparse
import torch
from torch.utils import data
from torchvision import transforms

import opt
import os
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--data_folder', type=str, default='data_large')
parser.add_argument('--mask_folder', type=str, default='data_large_segmented')
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Places2(
    os.path.join(args.root, args.data_folder),
    os.path.join(args.root, args.mask_folder),
    img_transform, mask_transform, 'test')

iterator_val = iter(data.DataLoader(
    dataset_val, batch_size=20,
    shuffle=False, num_workers=args.n_threads))
# dataset_val = Places2(args.root, args.mask_root, img_transform, mask_transform, 'test')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate(model, iterator_val, device, args.save_dir)
# evaluate(model, dataset_val, device, 'result_.jpg')
