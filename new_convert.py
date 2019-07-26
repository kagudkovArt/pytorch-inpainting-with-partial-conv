"""
Exports a pytorch model to an ONNX format, and then converts from the
ONNX to a Tensorflow serving protobuf file.
Running example:
python3 pytorch_to_tf_serving.py \
 --onnx-file text.onnx \
 --meta-file text.meta \
 --export-dir serving_model/
"""

import logging
import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tensorflow as tf



from net import weights_init, PConvUNet
from onnx_tf.backend import prepare
from tensorflow.python import ops
from tensorflow.python.saved_model import utils as smutils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools.import_pb_to_tensorboard import import_to_tensorboard
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from time import time
from util.io import load_ckpt


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TAG_CONSTANTS = [tag_constants.TRAINING]
# TAG_CONSTANTS = [tag_constants.GPU]
# TAG_CONSTANTS = [tag_constants.EVAL]
# TAG_CONSTANTS = [tag_constants.SERVING]
# TAG_CONSTANTS = [tag_constants.EVAL, tag_constants.GPU]
# TAG_CONSTANTS = [tag_constants.TRAINING, tag_constants.GPU]
TAG_CONSTANTS = [tag_constants.SERVING, tag_constants.GPU]


class PartialConvSimplified(nn.Module):
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

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = output.new(output.shape).zero_()

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        zero_tensor = output_mask.new(output_mask.shape).zero_()
        no_update_holes = (output_mask == zero_tensor).float()
        ones_tensor = zero_tensor + 1

        update_holes = ones_tensor - no_update_holes
        mask_sum = output_mask * update_holes + no_update_holes

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre * update_holes

        return output, update_holes


class PCBActivSimplified(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConvSimplified(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConvSimplified(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConvSimplified(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConvSimplified(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class PConvUNetSimplified(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActivSimplified(input_channels, 64, bn=False, sample='down-7')
        self.enc_2 = PCBActivSimplified(64, 128, sample='down-5')
        self.enc_3 = PCBActivSimplified(128, 256, sample='down-5')
        self.enc_4 = PCBActivSimplified(256, 512, sample='down-3')
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActivSimplified(512, 512, sample='down-3'))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActivSimplified(512 + 512, 512, activ='leaky'))
        self.dec_4 = PCBActivSimplified(512 + 256, 256, activ='leaky')
        self.dec_3 = PCBActivSimplified(256 + 128, 128, activ='leaky')
        self.dec_2 = PCBActivSimplified(128 + 64, 64, activ='leaky')
        self.dec_1 = PCBActivSimplified(64 + input_channels, input_channels,
                                        bn=False, activ=None, conv_bias=True)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

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
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h, h_mask

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


def export_onnx(model, dummy_input, file, input_names, output_names,
                num_inputs):
    """
    Converts a Pytorch model to the ONNX format and saves the .onnx model file.
    The first dimension of the input nodes are of size N, where N is the
    minibatch size. This dimensions is here replaced by an arbitrary string
    which the ONNX -> TF library interprets as the '?' dimension in Tensorflow.
    This process is applied because the input minibatch size should be of an
    arbitrary size.
    :param model: Pytorch model instance with loaded weights
    :param dummy_input: tuple, dummy input numpy arrays that the model
        accepts in the inference time. E.g. for the Text+Image model, the
        tuple would be (np.float32 array of N x W x H x 3, np.int64 array of
        N x VocabDim). Actual numpy arrays values don't matter, only the shape
        and the type must match the model input shape and type. N represents
        the minibatch size and can be any positive integer. True batch size
        is later handled when exporting the model from the ONNX to TF format.
    :param file: string, Path to the exported .onnx model file
    :param input_names: list of strings, Names assigned to the input nodes
    :param output_names: list of strings, Names assigned to the output nodes
    :param num_inputs: int, Number of model inputs (e.g. 2 for Text and Image)
    """
    # List of onnx.export function arguments:
    # https://github.com/pytorch/pytorch/blob/master/torch/onnx/utils.py
    # ISSUE: https://github.com/pytorch/pytorch/issues/14698
    torch.onnx.export(model, args=dummy_input, input_names=input_names,
                      output_names=output_names, f=file)

    # Reload model to fix the batch size
    model = onnx.load(file)
    model = make_variable_batch_size(num_inputs, model)
    onnx.save(model, file)

    log.info("Exported ONNX model to {}".format(file))


def make_variable_batch_size(num_inputs, onnx_model):
    """
    Changes the input batch dimension to a string, which makes it variable.
    Tensorflow interpretes this as the "?" shape.
    `num_inputs` must be specified because `onnx_model.graph.input` is a list
    of inputs of all layers and not just model inputs.
    :param num_inputs: int, Number of model inputs (e.g. 2 for Text and Image)
    :param onnx_model: ONNX model instance
    :return: ONNX model instance with variable input batch size
    """
    for i in range(num_inputs):
        onnx_model.graph.input[i].type.tensor_type.\
                                shape.dim[0].dim_param = 'batch_size'
    return onnx_model


def export_tf_proto(onnx_file, meta_file):
    """
    Exports the ONNX model to a Tensorflow Proto file.
    The exported file will have a .meta extension.
    :param onnx_file: string, Path to the .onnx model file
    :param meta_file: string, Path to the exported Tensorflow .meta file
    :return: tuple, input and output tensor dictionaries. Dictionaries have a
        {tensor_name: TF_Tensor_op} structure.
    """
    model = onnx.load(onnx_file)

    # Convert the ONNX model to a Tensorflow graph
    tf_rep = prepare(model)
    output_keys = tf_rep.outputs
    input_keys = tf_rep.inputs

    tf_dict = tf_rep.tensor_dict
    input_tensor_names = {key: tf_dict[key] for key in input_keys}
    output_tensor_names = {key: tf_dict[key] for key in output_keys}

    tf_rep.export_graph(meta_file)
    log.info("Exported Tensorflow proto file to {}".format(meta_file))

    return input_tensor_names, output_tensor_names


# def freeze_meta_file(meta_path):
#     saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
#     graph = tf.get_default_graph()


def export_for_serving(meta_path, export_dir, input_tensors, output_tensors):
    """
    Exports the Tensorflow .meta model to a frozen .pb Tensorflow serving
       format.
    :param meta_path: string, Path to the .meta TF proto file.
    :param export_dir: string, Path to directory where the serving model will
        be exported.
    :param input_tensor: dict, Input tensors dictionary of
        {name: TF placeholder} structure.
    :param output_tensors: dict, Output tensors dictionary of {name: TF tensor}
        structure.
    """
    g = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_def = tf.GraphDef()

    with g.as_default():
        with open(meta_path, "rb") as f:
            graph_def.ParseFromString(f.read())

        # name argument must explicitly be set to an empty string, otherwise
        # TF will prepend an `import` scope name on all operations
        tf.import_graph_def(graph_def, name="")

        # input_saver_def_path = ""
        # input_binary = True
        # output_node_names = "output_node"
        # restore_op_name = "save/restore_all"
        # filename_tensor_name = "save/Const:0"
        # clear_devices = False
        # input_meta_graph = meta_path
        # checkpoint_path = ''
        # output_graph_filename = '/mnt/sdb/results/inpainting/models/graph_optimized.pb'
        #
        # freeze_graph(
        #     "", input_saver_def_path, input_binary, checkpoint_path,
        #     output_node_names, restore_op_name, filename_tensor_name,
        #     output_graph_filename, clear_devices, "", "", "", input_meta_graph)

        # input_graph_path = meta_path
        # input_saver_def_path = ""
        # input_binary = False
        # output_node_names = "output,output_mask"
        # restore_op_name = "save/restore_all"
        # filename_tensor_name = "save/Const:0"
        # output_graph_path = '/mnt/sdb/results/inpainting/models/graph_optimized.pb'
        # clear_devices = False
        #
        # freeze_graph.freeze_graph(
        #     input_graph_path,
        #     input_saver_def_path,
        #     input_binary,
        #     checkpoint_path,
        #     output_node_names,
        #     restore_op_name,
        #     filename_tensor_name,
        #     output_graph_path,
        #     clear_devices,
        #     "",
        #     "",
        #     "",
        #     checkpoint_version=saver_write_version)
        #
        tensor_info_inputs = {name: smutils.build_tensor_info(in_tensor)
                              for name, in_tensor in input_tensors.items()}

        tensor_info_outputs = {name: smutils.build_tensor_info(out_tensor)
                               for name, out_tensor in output_tensors.items()}

        prediction_signature = signature_def_utils.build_signature_def(
            inputs=tensor_info_inputs,
            outputs=tensor_info_outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        if tf.gfile.Exists(export_dir):
            tf.gfile.DeleteRecursively(export_dir)
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess, TAG_CONSTANTS,
            signature_def_map={"predict_images": prediction_signature})
        # builder.add_meta_graph_and_variables(
        #     sess, [tag_constants.SERVING],
        #     signature_def_map={"predict_images": prediction_signature})
        builder.save()

        log.info("Input info:\n{}".format(tensor_info_inputs))
        log.info("Output info:\n{}".format(tensor_info_outputs))


def freeze_saved_model(saved_model_dir, frozen_graph_path):
    input_saved_model_dir = saved_model_dir
    output_node_names = "Mul_56,Sub_26"
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = ','.join(TAG_CONSTANTS)

    freeze_graph(input_graph_filename, input_saver_def_path,
                 input_binary, checkpoint_path, output_node_names,
                 restore_op_name, filename_tensor_name,
                 frozen_graph_path, clear_devices, "", "", "",
                 input_meta_graph, input_saved_model_dir,
                 saved_model_tags)


def get_graph_def_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


# def optimize_graph(graph_path, model_dir, transforms, output_node):
def optimize_graph(graph_path, model_dir, transforms, input_nodes, output_nodes):
    # input_names = []
    # output_names = [output_node]
    # if graph_filename is None:
    #   graph_def = get_graph_def_from_saved_model(model_dir)
    # else:
    # graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_path))
    optimized_graph_def = TransformGraph(
      graph_def,
      input_nodes,
      output_nodes,
      transforms)
    tf.train.write_graph(optimized_graph_def,
                         logdir=model_dir,
                         as_text=False,
                         name='optimized_model.pb')
    print('Graph optimized!')


def convert_graph_def_to_saved_model(export_dir, graph_filepath):
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={
                node.name: session.graph.get_tensor_by_name(
                    '{}:0'.format(node.name))
                for node in graph_def.node if node.op=='Placeholder'},
            outputs={'class_ids': session.graph.get_tensor_by_name(
                'head/predictions/class_ids:0')}
        )
    print('Optimized graph converted to SavedModel!')


def describe_graph(graph_def, show_nodes=False):
    print('Input Feature Nodes: {}'.format(
        [node.name for node in graph_def.node if node.op=='Placeholder']))
    print('')
    print('Unused Nodes: {}'.format(
        [node.name for node in graph_def.node if 'unused'  in node.name]))
    print('')
    print('Output Nodes: {}'.format(
        [node.name for node in graph_def.node if (
                'predictions' in node.name or 'softmax' in node.name)]))
    print('')
    print('Quantization Nodes: {}'.format(
        [node.name for node in graph_def.node if 'quant' in node.name]))
    print('')
    print('Constant Count: {}'.format(
        len([node for node in graph_def.node if node.op=='Const'])))
    print('')
    print('Variable Count: {}'.format(
        len([node for node in graph_def.node if 'Variable' in node.op])))
    print('')
    print('Identity Count: {}'.format(
        len([node for node in graph_def.node if node.op=='Identity'])))
    print('', 'Total nodes: {}'.format(len(graph_def.node)), '')

    if show_nodes==True:
        for node in graph_def.node:
            print('Op:{} - Name: {}'.format(node.op, node.name))


# def save_optimized_graph(optimized_graph, save_dir, input_tensors, output_tensors):
#     # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#     # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=optimized_graph)
#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#     tf.import_graph_def(optimized_graph, name="")
#
#     tensor_info_inputs = {name: smutils.build_tensor_info(in_tensor)
#                           for name, in_tensor in input_tensors.items()}
#
#     tensor_info_outputs = {name: smutils.build_tensor_info(out_tensor)
#                            for name, out_tensor in output_tensors.items()}
#
#     prediction_signature = signature_def_utils.build_signature_def(
#         inputs=tensor_info_inputs,
#         outputs=tensor_info_outputs,
#         method_name=signature_constants.PREDICT_METHOD_NAME)
#
#     builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
#     builder.add_meta_graph_and_variables(
#         sess, [tag_constants.SERVING, tag_constants.GPU],
#         signature_def_map={"predict_images": prediction_signature})
#     builder.save()
#     sess.close()
#     import_to_tensorboard(os.path.join(save_dir, 'saved_model.pb'), '/mnt/sdb/results/inpainting/graph_new_logs')
#
#
# from tensorflow.tools.graph_transforms import TransformGraph
#
#
# def get_graph_def_from_file(graph_filepath):
#   tf.reset_default_graph()
#   with ops.Graph().as_default():
#     with tf.gfile.GFile(graph_filepath, 'rb') as f:
#       graph_def = tf.GraphDef()
#       graph_def.ParseFromString(f.read())
#       return graph_def
#
#
# def optimize_graph(model_dir, graph_filename, transforms, output_names, outname='optimized_model.pb'):
#     input_names = ['input_image', ]  # change this as per how you have saved the model
#
#     graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
#
#     optimized_graph_def = TransformGraph(
#
#         graph_def,
#         input_names,
#         output_names,
#
#         transforms)
#     tf.train.write_graph(
#         optimized_graph_def,
#         logdir=model_dir,
#         as_text=False,
#         name=outname)
#     print('Graph optimized!')
#
# def find_optimized_graph(export_dir):
#     # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#     # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#
#     tf.saved_model.loader.load(sess, ['serve', 'gpu'], export_dir)
#     optimized_graph = optimize_for_inference(
#         sess.graph_def,
#         ['input', 'mask'],  # an array of the input node(s)
#         ['Mul_56', 'Sub_26'],  # an array of output nodes
#         tf.float32.as_datatype_enum)
#     sess.close()
#     return optimized_graph


    # # Save the optimized graph'test.pb'
    # f = tf.gfile.FastGFile(optimized_graph_path, "w")
    # f.write(output_graph.SerializeToString())


def main(args, device):
    input_shape = (3, 512, 512)

    # input = np.random.randn(1, *input_shape).astype(np.float32)
    # mask = np.random.binomial(1, 0.5, np.prod(input_shape)) \
    #     .reshape((1, *input_shape)).astype(np.float32)

    input = torch.randn(1, *input_shape).float().to(device)
    # input = torch.randint(0, 256, (1, *input_shape)).float().to(device)
    mask = torch.bernoulli(torch.ones_like(input) * 0.5).float()

    model = PConvUNetSimplified().to(device)
    _ = load_ckpt(args.model_path, [('model', model)])
    model.eval()

    old_model = PConvUNet().to(device)
    _ = load_ckpt(args.model_path, [('model', old_model)])
    old_model.eval()

    with torch.no_grad():
        old_output, old_mask = old_model(input, mask)
        start_old = time()
        for i in range(100):
            old_output, old_mask = old_model(input, mask)
        print("Old model: {}".format(time() - start_old))

        new_output, new_mask = model(input, mask)
        start_new = time()
        for i in range(100):
            new_output, new_mask = model(input, mask)
        print("New model: {}".format(time() - start_new))

    assert (new_output.allclose(old_output))
    assert (new_mask.allclose(old_mask))

    input_names = ['input', 'mask']
    output_names = ['output', 'output_mask']

    # Use a tuple if there are multiple model inputs
    dummy_inputs = (input, mask)

    export_onnx(model, dummy_inputs, args.onnx_file,
                input_names=input_names,
                output_names=output_names,
                num_inputs=len(dummy_inputs))
    input_tensors, output_tensors = export_tf_proto(args.onnx_file,
                                                    args.meta_file)
    export_for_serving(args.meta_file, args.export_dir, input_tensors,
                       output_tensors)
    freeze_saved_model(args.export_dir, args.frozen_graph_path)
    transforms = [
        'remove_nodes(op=Identity)',
        'merge_duplicate_nodes',
        'strip_unused_nodes',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms',
    ]
    input_nodes = ['input', 'mask']
    output_nodes = ['Mul_56', 'Sub_26']
    optimize_graph(args.frozen_graph_path, args.save_dir, transforms, input_nodes, output_nodes)

    optimized_graph_path = os.path.join(args.save_dir, 'optimized_model.pb')
    optimized_graph_def = get_graph_def_from_file(optimized_graph_path)
    describe_graph(optimized_graph_def)
    # convert_graph_def_to_saved_model(args.optimized_saved_model, optimized_graph_path)
    export_for_serving(optimized_graph_path, args.optimized_saved_model, input_tensors, output_tensors)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # tf.saved_model.loader.load(sess, TAG_CONSTANTS, args.export_dir)
        tf.saved_model.loader.load(sess, TAG_CONSTANTS, args.optimized_saved_model)
        tf_output = sess.run(
            output_nodes[0] + ':0',
            feed_dict={input_nodes[0] + ':0': input.cpu().numpy(), input_nodes[1] + ':0': mask.cpu().numpy()}
        )
        assert (np.allclose(old_output.cpu().numpy(), tf_output, atol=1e-5))



    # optimized_graph_dir = '/mnt/sdb/results/inpainting/models_optimized_serve_gpu'
    # optimized_graph = find_optimized_graph(args.export_dir)
    # save_optimized_graph(optimized_graph, optimized_graph_dir, input_tensors, output_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./snapshots/default/cpkt/1500000.pth')
    parser.add_argument(
        '--onnx_file', help="File where to export the ONNX file", type=str,
        required=True)
    parser.add_argument(
        '--meta_file', help="File where to export the Tensorflow meta file",
        type=str, required=True)
    parser.add_argument(
        '--export_dir',
        help="Folder where to export proto models for TF serving",
        type=str, required=True)
    parser.add_argument(
        '--save_dir',
        help="Folder where models will be saved",
        type=str, required=True)
    parser.add_argument(
        '--frozen_graph_path', help="File where to export frozen graph", type=str,
        required=True
    )
    parser.add_argument(
        '--optimized_saved_model',
        help="Folder where save model for optimized graph will be saved", type=str,
        required=True
    )
    args = parser.parse_args()
    device = torch.device('cuda')
    main(args, device)
