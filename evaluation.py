import cv2
import os
import torch
import numpy as np
from torch.utils import data
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm

from util.image import unnormalize


def evaluate(model, dataset, device, save_dir):
    dataset_iter = iter(data.DataLoader(
        dataset, batch_size=4,
        shuffle=False))

    for image, mask, gt, im_name in tqdm(dataset_iter):
        with torch.no_grad():
            output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu'))
        output_comp = mask * image + (1 - mask) * output

        image = unnormalize(image)
        output = unnormalize(output)
        output_comp = unnormalize(output_comp)
        gt = unnormalize(gt)

        batch_size = image.shape[0]
        for i in range(batch_size):
            save_image(torch.cat((image[i], mask[i], output[i], output_comp[i], gt[i]), dim=-1),
                       os.path.join(save_dir, im_name[i]))



# def evaluate(model, dataset, device, filename):
#     image, mask, gt = zip(*[dataset[i] for i in range(20)])
#     image = torch.stack(image)
#     mask = torch.stack(mask)
#     gt = torch.stack(gt)
#     with torch.no_grad():
#         output, _ = model(image.to(device), mask.to(device))
#     output = output.to(torch.device('cpu'))
#     output_comp = mask * image + (1 - mask) * output
#
#     grid = make_grid(
#         torch.cat((unnormalize(image), mask, unnormalize(output),
#                    unnormalize(output_comp), unnormalize(gt)), dim=0))
#     save_image(grid, filename)

# def evaluate(model, dataset, device, filename):
#     # start_index = 300
#     start_index = np.random.randint(0, len(dataset) - 8)
#     image, mask, gt = zip(*[dataset[i] for i in range(start_index, start_index + 8)])
# def evaluate(model, dataset_iter, device, save_dir):
    # image, mask, gt, im_name = zip(*[dataset[i] for i in range(len(dataset))])
    # image = torch.stack(image)
    # mask = torch.stack(mask)
    # gt = torch.stack(gt)
    # im_name = np.stack(im_name)
    # image, mask, gt, im_name = [x for x in next(dataset_iter)]
    # with torch.no_grad():
    #     output, _ = model(image.to(device), mask.to(device))
    # output = output.to(torch.device('cpu'))
    # output_comp = mask * image + (1 - mask) * output
    # print(len(dataset_iter))
    # for i in range(len(dataset_iter)):
    #     cv2.imwrite(os.path.join(save_dir, im_name[i]),
    #                 np.hstack((unnormalize(image[i]), mask[i], unnormalize(output[i]),
    #                            unnormalize(output_comp[i]), unnormalize(gt[i]))))

    # grid = make_grid(
    #     torch.cat((unnormalize(image), mask, unnormalize(output),
    #                unnormalize(output_comp), unnormalize(gt)), dim=0))
    # save_image(grid, filename)
