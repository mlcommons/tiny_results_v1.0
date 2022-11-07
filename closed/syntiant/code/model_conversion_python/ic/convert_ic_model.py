import tensorflow as tf

import syntiant_networks
from syntiant_networks import layers
from syntiant_networks import activations

from syntiant_networks.architectures import NDP

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
import sys

sys.path.insert(1, '..')
from perf_utils import generate_package, unpickle, img_from_binfile, img_to_binfile, load_cifar_10_data, show_range

write_package = True
perf_dir = '/home/jeremy/dev/TinyMLPerf/tiny/benchmark/training/image_classification/perf_samples'

ref_model = tf.keras.models.load_model(model_path)
ref_model.compile(metrics="accuracy")

cifar_10_dir = "/data/alexander/tinyml/cifar-10-batches-py"
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir, negatives=False)

def get_synt_eembc_resnet_model(weights_path: str, 
                                use_fake_quant: bool = True, 
                                clip_dict: dict = None,
                                verbose: bool = False):
    input_shape = (3, 32, 32)
    input_type = "SIGNED16"  # signed16 => [1 sign bit, 6 int bits, 9 frac bits] => [-64,63.9xx]
    num_classes=10
    num_filters = 16
    qparams = {"bits": 8, "method": "uniform"}
    
    if clip_dict is None:
      clip_dict = {
        "conv2d_blk0":1.0,
        "batchnorm_blk0":1.0,
        "maxpool_blk0":1.0,
        "conv2d_blk1_0":1.0,
        "bn_blk1_0":1.0,
        "conv2d_blk1_1":1.0,
        "bn_blk1_1":1.0,
        "add_blk1":1.0,
        "conv2d_blk2_0":1.0,
        "bn_blk2_0":1.0,
        "conv2d_blk2_1":1.0,
        "conv2d_skip_blk2":1.0,
        "bn_blk2_1":1.0,
        "add_blk2":1.0,
        "conv2d_blk3_0":1.0,
        "bn_blk3_0":1.0,
        "conv2d_blk3_1":1.0,
        "conv2d_skip_blk3":1.0,
        "bn_blk3_1":1.0,
        "add_blk3":1.0,
        "avg_pool":1.0,
        "flatten":1.0,
        "dense":1.0
      }
    
    inputs = layers.SyntiantInput(
        shape=input_shape,
        input_type=input_type,
        # bitmatch_ops = [] # jhdbg REMOVE THIS
    )
    # conv2d
    x = layers.SyntiantConv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_blk0"], fake_quantize=use_fake_quant), 
        quantization_params=qparams,
        name="conv2d_blk0"
    )(inputs)
    # batch_normalization
    x = layers.SyntiantBatchNorm(
        axis=1,
        activation=activations.ReluClipped(virtual_max_clip=clip_dict["batchnorm_blk0"], fake_quantize=use_fake_quant),
        name="batchnorm_blk0"
    )(x)
    
    # This MaxPool is just to convert the ReLU from batchnorm_blk0 into a Linear, to match
    # with bn_blk1_1 at the inputs to add_blk1
    x = layers.SyntiantMaxPooling2D(
        pool_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        activation=activations.LinearClipped(virtual_max_clip=clip_dict["maxpool_blk0"], fake_quantize=use_fake_quant), # Linear16Clipped
        name="maxpool_blk0"
    )(x)

    # First stack
    # conv2d_1
    y = layers.SyntiantConv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_blk1_0"], fake_quantize=use_fake_quant),
        quantization_params=qparams,
        name="conv2d_blk1_0"
    )(x)
    # batch_normalization_1
    y = layers.SyntiantBatchNorm(
        axis=1,
        activation=activations.ReluClipped(virtual_max_clip=clip_dict["bn_blk1_0"], fake_quantize=use_fake_quant),
        name="bn_blk1_0"
    )(y)
    # conv2d_2
    y = layers.SyntiantConv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_blk1_1"], fake_quantize=use_fake_quant),
        quantization_params=qparams,
        name="conv2d_blk1_1"
    )(y)
    # batch_normalization_2
    y = layers.SyntiantBatchNorm(
        axis=1,
        activation=activations.LinearClipped(virtual_max_clip=clip_dict["bn_blk1_1"], fake_quantize=use_fake_quant), # Linear16Clipped
        name="bn_blk1_1"
    )(y)
    # add
    x = layers.SyntiantAdd(
        activation=activations.ReluClipped(virtual_max_clip=clip_dict["add_blk1"], fake_quantize=use_fake_quant),
        name="add_blk1"
    )([x, y]) # bn_blk1_1 + maxpool_blk0

    # Second stack
    num_filters = 32
    # conv2d_3
    y = layers.SyntiantConv2D(
        num_filters,
        kernel_size=3,
        strides=2,
        padding='same',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_blk2_0"], fake_quantize=use_fake_quant),
        quantization_params=qparams,
        name="conv2d_blk2_0"
    )(x)
    # batch_normalization_3
    y = layers.SyntiantBatchNorm(
        axis=1,
        activation=activations.ReluClipped(virtual_max_clip=clip_dict["bn_blk2_0"], fake_quantize=use_fake_quant),
        name="bn_blk2_0"
    )(y)
    # conv2d_4
    y = layers.SyntiantConv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_blk2_1"], fake_quantize=use_fake_quant),
        quantization_params=qparams,
        name="conv2d_blk2_1"
    )(y)
    # batch_normalization_4
    y = layers.SyntiantBatchNorm(
        axis=1,
        activation=activations.LinearClipped(virtual_max_clip=clip_dict["bn_blk2_1"], fake_quantize=use_fake_quant),
        name="bn_blk2_1"  
    )(y)

    # conv2d_5
    x = layers.SyntiantConv2D(
        num_filters,
        kernel_size=1,
        strides=2,
        padding='same',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_skip_blk2"], fake_quantize=use_fake_quant),
        quantization_params=qparams,
        name="conv2d_skip_blk2"
    )(x)
    # add_1
    x = layers.SyntiantAdd(
        activation=activations.ReluClipped(virtual_max_clip=clip_dict["add_blk2"], fake_quantize=use_fake_quant),
        name="add_blk2"
    )([x, y])

    # Third stack 
    num_filters = 64
    # conv2d_6
    y = layers.SyntiantConv2D(
        num_filters,
        kernel_size=3,
        strides=2,
        padding='same',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_blk3_0"], fake_quantize=use_fake_quant), # Linear16Clipped
        quantization_params=qparams,
        name="conv2d_blk3_0"
    )(x)
    # batch_normalization_5
    y = layers.SyntiantBatchNorm(
        axis=1,
        activation=activations.ReluClipped(virtual_max_clip=clip_dict["bn_blk3_0"], fake_quantize=use_fake_quant),
        name="bn_blk3_0"
    )(y)
    # conv2d_7
    y = layers.SyntiantConv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_blk3_1"], fake_quantize=use_fake_quant),
        quantization_params=qparams,
        name="conv2d_blk3_1"
    )(y)
    # batch_normalization_6
    y = layers.SyntiantBatchNorm(
        axis=1,
        activation=activations.LinearClipped(virtual_max_clip=clip_dict["bn_blk3_1"], fake_quantize=use_fake_quant), # Linear16Clipped
        name="bn_blk3_1"
    )(y)

    # conv2d_8
    x = layers.SyntiantConv2D(
        num_filters,
        kernel_size=1,
        strides=2,
        padding='same',
        activation= activations.LinearClipped(virtual_max_clip=clip_dict["conv2d_skip_blk3"], fake_quantize=use_fake_quant), # Linear16Clipped
        quantization_params=qparams,
        name="conv2d_skip_blk3"
    )(x)
    # add_2
    x = layers.SyntiantAdd(
        activation=activations.ReluClipped(virtual_max_clip=clip_dict["add_blk3"], fake_quantize=use_fake_quant),
        name="add_blk3"
    )([x, y])

    pool_size = int(np.amin(x.shape[1:3]))
    x = layers.SyntiantAveragePooling2D(
        pool_size=pool_size,
        activation=activations.LinearClipped(virtual_max_clip=clip_dict["avg_pool"], fake_quantize=use_fake_quant), # Linear16Clipped
        name="avg_pool"
    )(x)
    x = layers.SyntiantFlatten(name="flatten")(x)
    x = layers.SyntiantDense(
        num_classes,
        activation=activations.LinearClipped(virtual_max_clip=clip_dict["dense"], fake_quantize=use_fake_quant),
        quantization_params=qparams,
        name="dense"
    )(x)

    outputs = layers.SyntiantSoftmax(axis=-1, scale_factor=1.0, name="softmax")(x)

    model = NDP(inputs=inputs, outputs=outputs)
    if verbose: model.summary()
    model.load_weights(weights_path)

    conv_layer = model.get_layer(name="conv2d_blk0")
    w, b = conv_layer.get_weights()
    conv_layer.set_weights((w, b / 8))

    bn_layer = model.get_layer(name="batchnorm_blk0")
    g, b, m, v = bn_layer.get_weights()
    m = m / 8
    g = g * 8
    bn_layer.set_weights((g, b, m, v))
    q_model = model.get_bitmatch_architecture("NDP120_B0")
    if verbose: q_model.summary()

    return q_model, model

## end of get_synt_eembc_resnet_model()

clip_dict = {
  "conv2d_blk0":32.0,
  "batchnorm_blk0":12.0, 
  "conv2d_blk1_0":12.0,
  "bn_blk1_0":20.0,
  "conv2d_blk1_1":19.0,
  "maxpool_blk0":12.0, # needs to match with bn_blk1_1
  "bn_blk1_1":12.0,  # needs to match with maxpool_blk0
  "add_blk1":18.0,
  "conv2d_blk2_0":16.0,
  "bn_blk2_0":16.0,
  "conv2d_blk2_1":16.0,
  "conv2d_skip_blk2":16.0,  # must match w/ conv2d_skip_blk2
  "bn_blk2_1":16.0,         # must match w/ bn_blk2_1
  "add_blk2":16.0,
  "conv2d_blk3_0":20.0,
  "bn_blk3_0":10.0,
  "conv2d_blk3_1":8.0,
  "conv2d_skip_blk3":36.0, # conv2d_skip_blk3 and bn_blk3_1 must match
  "bn_blk3_1":36.0,
  "add_blk3":36.0,
  "avg_pool":8.0,
  "flatten":8.0,
  "dense":36.0
}


w_path = "./pretrainedResnet.h5"
q_model, fp_model = get_synt_eembc_resnet_model(w_path, verbose=False, clip_dict=clip_dict, use_fake_quant=False)
fp_model.compile(metrics="accuracy")

ref_eval = ref_model.evaluate(x=test_data, y=test_labels, return_dict=True)

q_eval = q_model.evaluate(
    x=test_data.transpose(0, 3, 1, 2)/8, y=test_labels, return_dict=True
)


generate_package(q_model,  pkg_filename="resnet_eembc_new.synpkg", verbose=True)
