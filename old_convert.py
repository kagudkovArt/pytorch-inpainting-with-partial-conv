import argparse
import onnx
import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from net import weights_init, PConvUNet
from onnx_tf.backend import prepare
from time import time
from util.io import load_ckpt


parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='./snapshots/default/cpkt/1500000.pth')
parser.add_argument('--save_dir', type=str, default='./models')

args = parser.parse_args()

device = torch.device('cuda')


def get_input_and_mask(input_tensor):
    num_channels = input_tensor.shape[1] // 2
    input = input_tensor[:, :num_channels, :, :]
    mask = input_tensor[:, num_channels:, :, :]
    return input, mask


class PartialConvSingleInput(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input_tensor):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        input, mask = get_input_and_mask(input_tensor)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            # output_bias = torch.zeros(output.shape).cuda()
            output_bias = output.new(output.shape).zero_()
            # output_bias = torch.cuda.FloatTensor(output.shape).fill_(0)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        zero_tensor = output_mask.new(output_mask.shape).zero_()
        no_update_holes = (output_mask == zero_tensor).float()
        # no_update_holes = output_mask == 0
        # mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        # mask_sum = output_mask * (1.0 - no_update_holes) + no_update_holes
        ones_tensor = no_update_holes.new(no_update_holes.shape).zero_() + 1
        # torch.ones(no_update_holes.shape).cuda()
        update_holes = ones_tensor - no_update_holes
        mask_sum = output_mask * update_holes + no_update_holes

        output_pre = (output - output_bias) / mask_sum + output_bias
        # output = output_pre.masked_fill_(no_update_holes, 0.0)
        # output = output_pre * (1.0 - no_update_holes)
        output = output_pre * update_holes

        # new_mask = torch.ones_like(output)
        new_mask = output.new(output.shape).zero_() + 1
        # new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        # new_mask = new_mask * (1.0 - no_update_holes)
        new_mask = new_mask * update_holes

        return torch.cat((output, new_mask), dim=1)


class PCBActivSingleInput(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConvSingleInput(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConvSingleInput(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConvSingleInput(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConvSingleInput(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input_tensor):
        h_tensor = self.conv(input_tensor)
        h, h_mask = get_input_and_mask(h_tensor)
        # num_channels = int(input_tensor.shape[1] / 2)
        # h = h_tensor[:, :num_channels, :, :]
        # h_mask = h_tensor[:, num_channels:, :, :]

        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)

        return torch.cat((h, h_mask), dim=1)


class PConvUNetSingleInput(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActivSingleInput(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActivSingleInput(64, 128, sample='down-5')
        self.enc_3 = PCBActivSingleInput(128, 256, sample='down-5')
        self.enc_4 = PCBActivSingleInput(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActivSingleInput(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActivSingleInput(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActivSingleInput(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActivSingleInput(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActivSingleInput(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActivSingleInput(64 + input_channels, input_channels,
                              bn=False, activ=None, conv_bias=True)

    def forward(self, input_tensor):
        # input = input_tensor[:, :3, :, :]
        # input_mask = input_tensor[:, 3:, :, :]
        input, input_mask = get_input_and_mask(input_tensor)

        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h = getattr(self, l_key)(
                torch.cat((h_dict[h_key_prev], h_mask_dict[h_key_prev]), dim=1))
            h_dict[h_key], h_mask_dict[h_key] = get_input_and_mask(h)
            # h_dict[h_key] = h[:, :3, :, :]
            # h_mask_dict[h_key] = h[:, 3:, :, :]
            h_key_prev = h_key
            # h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
            #     h_dict[h_key_prev], h_mask_dict[h_key_prev])
            # h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(
                h_mask, scale_factor=2, mode='nearest')

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)

            h_tensor = getattr(self, dec_l_key)(
                torch.cat((h, h_mask), dim=1))
            h, h_mask = get_input_and_mask(h_tensor)

        return h_tensor
        # return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()

# class PartialConvSingleInput(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         super().__init__()
#         self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
#                                     stride, padding, dilation, groups, bias)
#         self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
#                                    stride, padding, dilation, groups, False)
#         self.input_conv.apply(weights_init('kaiming'))
#
#         torch.nn.init.constant_(self.mask_conv.weight, 1.0)
#
#         # mask is not updated
#         for param in self.mask_conv.parameters():
#             param.requires_grad = False
#
#     def forward(self, input_list):
#         # http://masc.cs.gmu.edu/wiki/partialconv
#         # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
#         # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
#         input, mask = input_list
#         output = self.input_conv(input * mask)
#         if self.input_conv.bias is not None:
#             output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
#                 output)
#         else:
#             # output_bias = torch.zeros_like(output)
#             output_bias = torch.zeros(output.shape).cuda()
#
#         with torch.no_grad():
#             output_mask = self.mask_conv(mask)
#
#         no_update_holes = (output_mask == torch.zeros(output_mask.shape).cuda()).float()
#         # no_update_holes = output_mask == 0
#         # mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
#         # mask_sum = output_mask * (1.0 - no_update_holes) + no_update_holes
#         update_holes = torch.ones(no_update_holes.shape).cuda() - no_update_holes
#         mask_sum = output_mask * update_holes + no_update_holes
#
#         output_pre = (output - output_bias) / mask_sum + output_bias
#         # output = output_pre.masked_fill_(no_update_holes, 0.0)
#         # output = output_pre * (1.0 - no_update_holes)
#         output = output_pre * update_holes
#
#         # new_mask = torch.ones_like(output)
#         new_mask = torch.ones(output.shape).cuda()
#         # new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
#         # new_mask = new_mask * (1.0 - no_update_holes)
#         new_mask = new_mask * update_holes
#
#         return (output, new_mask)
#
#
# class PCBActivSingleInput(nn.Module):
#     def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
#                  conv_bias=False):
#         super().__init__()
#         if sample == 'down-5':
#             self.conv = PartialConvSingleInput(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
#         elif sample == 'down-7':
#             self.conv = PartialConvSingleInput(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
#         elif sample == 'down-3':
#             self.conv = PartialConvSingleInput(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
#         else:
#             self.conv = PartialConvSingleInput(in_ch, out_ch, 3, 1, 1, bias=conv_bias)
#
#         if bn:
#             self.bn = nn.BatchNorm2d(out_ch)
#         if activ == 'relu':
#             self.activation = nn.ReLU()
#         elif activ == 'leaky':
#             self.activation = nn.LeakyReLU(negative_slope=0.2)
#
#     def forward(self, input_list):
#         input, input_mask = input_list
#         h, h_mask = self.conv([input, input_mask])
#         if hasattr(self, 'bn'):
#             h = self.bn(h)
#         if hasattr(self, 'activation'):
#             h = self.activation(h)
#         return (h, h_mask)
#
#
# class PConvUNetSingleInput(nn.Module):
#     def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
#         super().__init__()
#         self.freeze_enc_bn = False
#         self.upsampling_mode = upsampling_mode
#         self.layer_size = layer_size
#         self.enc_1 = PCBActivSingleInput(input_channels, 64, bn=False, sample='down-7')
#         self.enc_2 = PCBActivSingleInput(64, 128, sample='down-5')
#         self.enc_3 = PCBActivSingleInput(128, 256, sample='down-5')
#         self.enc_4 = PCBActivSingleInput(256, 512, sample='down-3')
#         for i in range(4, self.layer_size):
#             name = 'enc_{:d}'.format(i + 1)
#             setattr(self, name, PCBActivSingleInput(512, 512, sample='down-3'))
#
#         for i in range(4, self.layer_size):
#             name = 'dec_{:d}'.format(i + 1)
#             setattr(self, name, PCBActivSingleInput(512 + 512, 512, activ='leaky'))
#         self.dec_4 = PCBActivSingleInput(512 + 256, 256, activ='leaky')
#         self.dec_3 = PCBActivSingleInput(256 + 128, 128, activ='leaky')
#         self.dec_2 = PCBActivSingleInput(128 + 64, 64, activ='leaky')
#         self.dec_1 = PCBActivSingleInput(64 + input_channels, input_channels,
#                                          bn=False, activ=None, conv_bias=True)
#
#     def forward(self, input_list):
#         input, input_mask = input_list
#         h_dict = {}  # for the output of enc_N
#         h_mask_dict = {}  # for the output of enc_N
#
#         h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask
#
#         h_key_prev = 'h_0'
#         for i in range(1, self.layer_size + 1):
#             l_key = 'enc_{:d}'.format(i)
#             h_key = 'h_{:d}'.format(i)
#             h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
#                 [h_dict[h_key_prev], h_mask_dict[h_key_prev]])
#             h_key_prev = h_key
#
#         h_key = 'h_{:d}'.format(self.layer_size)
#         h, h_mask = h_dict[h_key], h_mask_dict[h_key]
#
#         # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
#         # (exception)
#         #                            input         dec_2            dec_1
#         #                            h_enc_7       h_enc_8          dec_8
#
#         for i in range(self.layer_size, 0, -1):
#             enc_h_key = 'h_{:d}'.format(i - 1)
#             dec_l_key = 'dec_{:d}'.format(i)
#
#             h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
#             h_mask = F.interpolate(
#                 h_mask, scale_factor=2, mode='nearest')
#
#             h = torch.cat([h, h_dict[enc_h_key]], dim=1)
#             h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
#             h, h_mask = getattr(self, dec_l_key)([h, h_mask])
#
#         return (h, h_mask)
#
#     def train(self, mode=True):
#         """
#         Override the default train() to freeze the BN parameters
#         """
#         super().train(mode)
#         if self.freeze_enc_bn:
#             for name, module in self.named_modules():
#                 if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
#                     module.eval()


def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


input_shape = (3, 512, 512)

input = np.random.randn(1, *input_shape).astype(np.float32)
mask = np.random.binomial(1, 0.5, np.prod(input_shape)) \
    .reshape((1, *input_shape)).astype(np.float32)
input_array = np.concatenate((input, mask), 1)


model_name = os.path.splitext(os.path.basename(args.model_path))[0]

old_model = PConvUNet().to(device)
_ = load_ckpt(args.model_path, [('model', old_model)])
old_model.eval()

model = PConvUNetSingleInput().to(device)
_ = load_ckpt(args.model_path, [('model', model)])
model.eval()


# Single pass of dummy variable required
input_tensor = torch.from_numpy(input).float().to(device)
mask_tensor = torch.from_numpy(mask).float().to(device)
dummy_input = torch.from_numpy(input_array).float().to(device)


with torch.no_grad():
    old_output, old_mask = old_model(input_tensor, mask_tensor)
    start_old = time()
    for i in range(100):
        old_output, old_mask = old_model(input_tensor, mask_tensor)
    print("Old model: {}".format(time() - start_old))

    dummy_output = model(dummy_input)
    start_new = time()
    for i in range(100):
        dummy_output = model(dummy_input)
    print("New model: {}".format(time() - start_new))

    new_output, new_mask = get_input_and_mask(dummy_output)


assert(new_output.allclose(old_output))
assert(new_mask.allclose(old_mask))

# Export to ONNX format
torch.onnx.export(model, dummy_input, '{:s}/{}.onnx'.format(args.save_dir, model_name),
                  input_names=['input'], output_names=['output'])


model_onnx = onnx.load('{:s}/{}.onnx'.format(args.save_dir, model_name))

# Export to tf format
tf_rep = prepare(model_onnx)
input_name = tf_rep.tensor_dict['input'].name
output_name = tf_rep.tensor_dict['output'].name

print("Input: {}, output: {}".format(input_name, output_name))

tf_rep.export_graph('{:s}/{}.pb'.format(args.save_dir, model_name))

# output_tf = tf_rep.run(input_array).output
# print(output_tf)

tf_graph = load_pb('{:s}/{}.pb'.format(args.save_dir, model_name))

sess = tf.Session(graph=tf_graph)

output_tf_tensor = tf_graph.get_tensor_by_name(output_name)
input_tf_tensor = tf_graph.get_tensor_by_name(input_name)

start = time()
output_tf = sess.run(output_tf_tensor,
                     feed_dict={input_tf_tensor: input_array})
print(time() - start)

output_torch = dummy_output.cpu().detach().numpy()

data_torch, mask_torch = get_input_and_mask(output_torch)
data_tf, mask_tf = get_input_and_mask(output_tf)

# print(np.max(np.abs(output_torch - output_tf)))

# print(data_torch, data_tf)
assert np.allclose(mask_torch, mask_tf), "Masks are not equal"
assert np.allclose(data_torch, data_tf, atol=1e-5), "Outputs are not equal"
