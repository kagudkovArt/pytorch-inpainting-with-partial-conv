import logging
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageOps


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.mode = split
        self.mask_root = mask_root

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.png'.format(img_root),
                              recursive=True)
            self.paths += glob('{:s}/data_large/**/*.jpg'.format(img_root),
                               recursive=True)
        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))

        self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.mask_paths += glob('{:s}/*.jpg'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __intersect_with_random_mask(self, index, gt_img_source):
        im_name = os.path.splitext(os.path.basename(self.paths[index]))[0] + '.png'
        mask_current_image = cv2.imread(os.path.join(self.mask_root, im_name))
        for i in range(20):
            try:
                mask = np.ones_like(mask_current_image) * 255
                random_mask = cv2.imread(self.mask_paths[random.randint(0, self.N_mask - 1)])
                mask[random_mask[:, :, 0] == 1] = 0
                mask[mask_current_image == 1] = 255
                mask[mask_current_image != 0] = 255
                mask[gt_img_source[:, :, 0] == 255] = 255
                random_mask[random_mask != 1] = 0
                #not more then 20% of area
                if 0 < len(mask[mask== 0]) < mask.shape[0] * mask.shape[1] * 0.2:
                    # cv2.imwrite(os.path.join('mask_ex', im_name), np.hstack((random_mask * 255, gt_img_source, mask)))
                    return mask
            except TypeError as re:
                logging.error(f"{str(re)}")
        return mask

    def __getitem__(self, index):
        try:
            need_horizontal_flip = np.random.random() < 0.5
            gt_img = Image.open(self.paths[index])
            if need_horizontal_flip:
                gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
            gt_img_source = np.array(gt_img)
            gt_img = self.img_transform(gt_img.convert('RGB'))
            if self.mode == 'train':
                mask = self.__intersect_with_random_mask(index, gt_img_source)
                mask = Image.fromarray(mask)
                if need_horizontal_flip:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                mask = Image.open(self.mask_paths[index])  # random.randint(0, self.N_mask - 1)])
                mask = ImageOps.invert(mask)
            mask = self.mask_transform(mask.convert('RGB'))

            return gt_img * mask, mask, gt_img
        except Exception as re:
            logging.error(f"{str(re)} on index index")
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.paths)
