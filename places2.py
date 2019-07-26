import json
import logging
import os
import random
from enum import IntEnum
from glob import glob

import cv2
import numpy as np
import torch
from torchvision.utils import save_image
from PIL import Image
from util.image import unnormalize_image


DEFAULT_DILATION_SIZE = 18
# DEFAULT_DILATION_SIZE = 24
DEFAULT_EROSION_SIZE = 12


class SegmentLabel(IntEnum):
  BACKGROUND = 0
  HAIRS = 1
  SKIN = 2
  LIPS = 3
  EYES = 4
  CLOTHES = 5
  GLASSES = 6
  TEETH = 7


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, landmarks_path, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        with open(landmarks_path) as f:
            self.landmarks = json.loads(f.read())
            print(len(self.landmarks))
        # self.landmarks = landmarks
        self.mode = split
        self.mask_root = mask_root

        self.paths = glob('{:s}/*.png'.format(img_root))
        self.paths += glob('{:s}/*.jpg'.format(img_root))

        # # use about 8M images in the challenge dataset
        # if split == 'train':
        #     self.paths = glob('{:s}/data_large/**/*.png'.format(img_root),
        #                       recursive=True)
        #     self.paths += glob('{:s}/data_large/**/*.jpg'.format(img_root),
        #                        recursive=True)
        # else:
        #     self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))
        # print(len(self.paths))

        self.mask_paths = glob('{:s}/*.png'.format(mask_root))
        self.mask_paths += glob('{:s}/*.jpg'.format(mask_root))

        # self.landmarks = [
        #     landmarks[os.path.splitext(os.path.basename(path))[0] + '.png']
        #     for path in self.paths]

        self.N_mask = len(self.mask_paths)

    def __find_contour_mask(self, index, gt_img_source):
                            # dilation_size=DEFAULT_DILATION_SIZE, erosion_size=DEFAULT_EROSION_SIZE):
        # dilation_size = np.random.randint(5, 16)
        # dilation_size = np.random.randint(1, 21)
        # erosion_size = np.random.randint(10, 31)
        # erosion_size = np.random.randint(10, 31)

        dilation_size = np.random.randint(1, 5)
        erosion_size = np.random.randint(5, 15)

        im_name = os.path.splitext(os.path.basename(self.paths[index]))[0] + '.png'
        segmentation_mask = cv2.imread(os.path.join(self.mask_root, im_name))

        mask = np.zeros_like(gt_img_source)
        mask[segmentation_mask == SegmentLabel.SKIN] = 255
        mask[segmentation_mask == SegmentLabel.EYES] = 255
        mask[segmentation_mask == SegmentLabel.LIPS] = 255
        mask[segmentation_mask == SegmentLabel.GLASSES] = 255
        mask[segmentation_mask == SegmentLabel.TEETH] = 255

        mask_dilated = cv2.dilate(mask, np.ones((dilation_size, dilation_size)), iterations=1)
        mask_erosed = cv2.erode(mask, np.ones((erosion_size, erosion_size)), iterations=1)

        mask = mask_dilated - mask_erosed

        if self.mode == 'train':
            cur_landmarks = np.array(self.landmarks[os.path.join('data_large', im_name)][0])
            mouth_landmarks = cur_landmarks[np.arange(48, 68)]
            contour_landmarks = cur_landmarks[np.arange(0, 17)]
            nose_landmarks = cur_landmarks[np.arange(27, 36)]
            eyebrows_landmarks = cur_landmarks[np.arange(17, 27)]
        else:
            cur_landmarks = self.landmarks[im_name]['face0']
            mouth_landmarks = cur_landmarks['mouth']
            contour_landmarks = cur_landmarks['contour']
            nose_landmarks = cur_landmarks['nose']
            eyebrows_landmarks = np.vstack((cur_landmarks['eyebrowRight'], cur_landmarks['eyebrowLeft']))

        # mouth_landmarks = cur_landmarks[[49, 52, 55, 58]]
        # contour_landmarks = cur_landmarks[[1, 17, 9]]
        # nose_landmarks = cur_landmarks[[34, 35]]


        # mouth_landmarks = cur_landmarks['face0']['mouth']
        # contour_landmarks = cur_landmarks['face0']['contour']
        # nose_landmarks = cur_landmarks['face0']['nose']
        # eyes_landmarks = cur_landmarks['face0']['eyeLeftSciera']

        left_bound = int(np.min(contour_landmarks, axis=0)[0]) - 2 * DEFAULT_DILATION_SIZE
        right_bound = int(np.max(contour_landmarks, axis=0)[0]) + 2 * DEFAULT_DILATION_SIZE

        contour_lower_bound = int(np.max(contour_landmarks, axis=0)[1])
        nose_lower_bound = int(np.max(nose_landmarks, axis=0)[1])
        lower_bound = contour_lower_bound + (contour_lower_bound - nose_lower_bound)

        eyebrows_lower_bound = int(np.max(eyebrows_landmarks, axis=0)[1])
        mouth_lower_bound = int(np.max(mouth_landmarks, axis=0)[1])
        upper_bound = eyebrows_lower_bound - (mouth_lower_bound - eyebrows_lower_bound)

        # trying to remove hands
        mask[:, :left_bound, :] = 0
        mask[:, right_bound:, :] = 0
        mask[lower_bound:, :, :] = 0
        mask[:upper_bound, :, :] = 0

        # # lines
        # mask[:, left_bound - 2:left_bound + 2, :] = 255
        # mask[:, right_bound - 2:right_bound + 2, :] = 255
        # mask[lower_bound - 2:lower_bound + 2, :, :] = 255

        # # mouth and central face rectangulars remove
        # left_bound_mouth, lower_bound_mouth = np.min(mouth_landmarks, axis=0).astype(np.uint)
        # right_bound_mouth, upper_bound_mouth = np.max(mouth_landmarks, axis=0).astype(np.uint)
        #
        # mask[lower_bound_mouth:upper_bound_mouth, left_bound_mouth:right_bound_mouth, :] = 0
        #
        # eyebrows_lower_bound = int(np.max(eyebrows_landmarks, axis=0)[1])
        # eyebrows_left_bound = int(np.min(eyebrows_landmarks, axis=0)[0]) + DEFAULT_EROSION_SIZE
        # eyebrows_right_bound = int(np.max(eyebrows_landmarks, axis=0)[0]) - DEFAULT_EROSION_SIZE
        #
        # mask[eyebrows_lower_bound:nose_lower_bound, eyebrows_left_bound:eyebrows_right_bound] = 0

        # white bounds
        mask[gt_img_source[:, :, 0] == (255, 255, 255)] = 0

        return 255 - mask

    def __intersect_with_random_mask(self, index, gt_img_source,
                                     inpainting_segments=[SegmentLabel.BACKGROUND]):
        im_name = os.path.splitext(os.path.basename(self.paths[index]))[0] + '.png'
        mask_current_image = cv2.imread(os.path.join(self.mask_root, im_name))
        for i in range(20):
            try:
                mask = np.ones_like(mask_current_image) * 255
                random_mask = cv2.imread(self.mask_paths[random.randint(0, self.N_mask - 1)])
                mask[random_mask[:, :, 0] == SegmentLabel.HAIRS] = 0
                # mask_with_hairs = mask.copy()

                not_inpainted_vec = np.vectorize(lambda x: x not in inpainting_segments)
                mask[not_inpainted_vec(mask_current_image)] = 255
                # mask_without_inpainted = mask.copy()

                mask[gt_img_source[:, :, 0] == (255, 255, 255)] = 255
                # mask_without_white = mask.copy()

                random_mask[random_mask != SegmentLabel.HAIRS] = 0
                #not more then 20% of area
                if 0 < len(mask[mask== 0]) < mask.shape[0] * mask.shape[1] * 0.2:
                    # cv2.imwrite(os.path.join('/mnt/sdb/results/inpainting/mask_ex', im_name),
                    #             np.hstack((random_mask * 255, mask_current_image, gt_img_source, mask_with_hairs,
                    #                        mask_without_inpainted, mask_without_white))
                    #             )
                    # cv2.imwrite(os.path.join('/mnt/sdb/datasets/pconv_inpainting_dataset/DataRoot/masks_clothes_ex', im_name),
                    #             np.hstack((random_mask * 255, Image.fromarray(gt_img_source), mask))
                    #             )
                    #
                    # cv2.imwrite(
                    #     os.path.join('/mnt/sdb/datasets/pconv_inpainting_dataset/DataRoot/val_masks_clothes', im_name),
                    #     mask)
                    return mask
            except TypeError as re:
                logging.error(f"{str(re)}")
        return mask

    def __getitem__(self, index):
        try:
            need_horizontal_flip = np.random.random() < 0.5
            im_name = os.path.splitext(os.path.basename(self.paths[index]))[0] + '.png'
            gt_img = Image.open(self.paths[index])
            gt_img_source = np.array(gt_img)

            # if self.mode == 'train':
            #     inpainting_segments = [SegmentLabel.CLOTHES]
            #     # mask = self.__intersect_with_random_mask(index, gt_img_source, inpainting_segments)
            #     mask = self.__find_contour_mask(index, gt_img_source)
            #     mask = Image.fromarray(mask)
            #     # if need_horizontal_flip:
            #     #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # else:
            #     mask = Image.open(self.mask_paths[index])  # random.randint(0, self.N_mask - 1)])
            #     # mask = ImageOps.invert(mask)

            mask = self.__find_contour_mask(index, gt_img_source)
            mask = Image.fromarray(mask)

            # log_path = '/mnt/sdb/results/inpainting/mask_contour_ex'
            #
            # segmentation_mask = cv2.imread(os.path.join(self.mask_root, im_name)) * 20
            # segmentation_mask = Image.fromarray(segmentation_mask)

            if need_horizontal_flip:
                gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                # segmentation_mask = segmentation_mask.transpose(Image.FLIP_LEFT_RIGHT)
            # cv2.imwrite(os.path.join(log_path, im_name),
            #             np.hstack((np.array(gt_img) * np.array(mask), mask, gt_img)))

            gt_img = self.img_transform(gt_img.convert('RGB'))
            mask = self.mask_transform(mask.convert('RGB'))
            # segmentation_mask = self.mask_transform(segmentation_mask.convert('RGB'))

            # if self.mode == 'train':
            #     save_image(torch.cat((unnormalize_image(gt_img * mask), mask, segmentation_mask, unnormalize_image(gt_img)), dim=-1),
            #                os.path.join(log_path, im_name))
            return gt_img * mask, mask, gt_img, im_name
        except Exception as re:
            logging.error(f"{str(re)} on index {str(index)}")
            return self.__getitem__(index + 1)
            # if index < len(self.paths) - 1:
            #     return self.__getitem__(index + 1)
            # else:
            #     return None

    def __len__(self):
        return len(self.paths)
