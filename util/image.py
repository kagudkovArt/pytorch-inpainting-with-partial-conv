import torch
import opt


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    return x

def unnormalize_image(x):
    x = x.transpose(0, 2)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(0, 2)
    return x