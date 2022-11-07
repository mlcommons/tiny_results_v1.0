"""
AUTODOC_IGNORE:

SYNTIANT CONFIDENTIAL
_____________________

Copyright (c) 2021 Syntiant Corporation
All Rights Reserved.

NOTICE:  All information contained herein is, and remains the property of
Syntiant Corporation and its suppliers, if any.  The intellectual and
technical concepts contained herein are proprietary to Syntiant Corporation
and its suppliers and may be covered by U.S. and Foreign Patents, patents in
process, and are protected by trade secret or copyright law.  Dissemination
of this information or reproduction of this material is strictly forbidden
unless prior written permission is obtained from Syntiant Corporation.


AUTODOC_IGNORE

**ABOUT THIS FILE**

This file will load the tinyml perf keras model. You have the option of retraining
the model or load the pretrained weights. Once loaded, we transfer the weights into a 
model that is defined in Syntiant Float version. Afterwards, we create the 
bitmatch version of the model. The dataset for this model is Google's speech script.
If you don't have the data locally, it will automatically download the data and save
it on a folder named data next to train.py. At the end we compare model performance and
will see how does the quantization change the performance of the model. 
"""

import sys
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import softmax
from collections import defaultdict
from tensorflow import keras
from tensorflow.keras import backend as K
from syntiant_packager.packages import SyntiantPackageFactory
from syntiant_networks.activations import (
    ReluClipped, LinearClipped,
    Linear16Clipped)

from syntiant_networks.architectures.ndp12x.ndp12X import NDP12X
from tinyml_model import TinyMlPerf



def float_to_fixed(x, signed, int_bits, frac_bits):
    """converting a floating point tensor into a fix point one

    Args:
        x(numpy array): Input tensor
        signed(Boolean): Whether the output is signed or not
        int_bits(int): Number of bits used to represent the integer part
        frac_bits(int): Number of bits used to represent the fractional part
            
    Returns:
        x(numpy array): Fixed point representation of the input tensor
    """

    if signed:
        min_val = (-2 ** int_bits) * (2 ** frac_bits)
        max_val = ((2 ** int_bits) - (1 / (2 ** frac_bits))) * (2 ** frac_bits)
    else:
        min_val = 0
        max_val = ((2 ** int_bits) - (1 / (2 ** frac_bits))) * (2 ** frac_bits)
    x = np.asarray(x)
    x *= (2 ** frac_bits)
    x = np.floor(x + 0.5)
    np.clip(x, min_val, max_val, out=x)
    return x


def data_preprocessing(data_q_8bit):
    """Converting an 8bit array into 16bit

    Args:
        data_q_8bit(numpy array): Input tensor
            
    Returns:
        flattened_input(numpy array): Flattened 16bit version of the input
    """

    # reshaping the input data
    reshaped_input = data_q_8bit.reshape((1,49,10))
    # dequantize the data
    dequantized_input = 0.5847029089927673 * (reshaped_input - 83)
    # change the dequantized data into signed16
    input_sign16 = float_to_fixed(np.copy(dequantized_input), signed=True, \
        int_bits=6, frac_bits=9 )
    # flatten the data so that our databank can read
    flattened_input = input_sign16.flatten("F")
    return flattened_input


def test_flattened_input(flattened_input):
    """Reshaping the input tensor

    Args:
        flattened_input(numpy array): Input tensor
            
    Returns:
        reshaped_input(numpy array): reshaped version of the input tensor
            with different ordering
    """

    reshaped_input = flattened_input.reshape((1,1,49,10), order="F")
    return reshaped_input


def test_data_processing(bitmatch_model, test_data):
    """Checking how will the model perform when we convert 8bit input
     into 16bit.

    Args:
        bitmatch_model(syntiant model): bitmatch version of the syntiant
            float model
        test_data(python iterator): Test data to be used for checking the
            performance of the bitmatch model
            
    Returns:
        prints the accuracy of the model to the console
    """

    eval_data = test_data.unbatch().batch(1).take(100).as_numpy_iterator()
    input_scale, input_zero_point =  0.5847029089927673, 83
    total_samples = 0
    match = 0
    for dat, label in eval_data:
        dat_q = np.array(dat/input_scale + input_zero_point, dtype=np.int8)
        dat_q = dat_q.flatten()
        label = label[0]
        flattened_signed16 = data_preprocessing(dat_q)
        reshaped_input_signed16 = test_flattened_input(flattened_signed16)
        y_pred_prob = bitmatch_model.predict(reshaped_input_signed16)
        y_pred = np.argmax(y_pred_prob, axis=1)
        total_samples += 1
        if y_pred == label:
            match += 1
    print("Accuracy = ", match/total_samples)


def custom_argmax(input_array, threshold=0.5):
    """Creating a customized softmax where we have the ability
     to change the threshold of the probability.

    Args:
        input_array(numpy array): Input numpy array which consists of the
            output probabilities
        threshold(float): A float value between 0.0 and 1.0 where we specify the
            the threshold of output probability predictions.
            
    Returns:
        custom_arg_out(numpy array): returning the index of the array
            corresponding to the max value that can exceed the threshold.
    """

    sorted_array = [x.argsort()[-2:] for x in input_array]
    custom_arg_out = [np.argmax(x) if np.max(x)>=threshold else sorted_array[i][0]  for i, x in enumerate(input_array)]
    return custom_arg_out


def rmse(A, B):
    """Calculating the root mean square error between two tensors

    Args:
        A(numpy array/tensor): Input tensor in any shape but same shape as B
        B(numpy array/tensor): Input tensor in any shape but same shape as A
            
    Returns:
        root mean square error between tensors A and B
    """
    return np.sqrt(np.mean((A-B)*(A-B)))


def extract_intermediate(model, acceptable_layers):
    """Extracting the intermediate layers of a keras/syntiant model

    Args:
        model(syntiant/keras model): Input model defined in Keras/Syntiant format
        acceptable_layers(list of strings): List of strings that corresponds
            to the beginning of the name of layers that we would like to extract.
            
    Returns:
        intermediate(List of keras Intermediate layers): List of keras model layers
            where one can pass in input values and get the corresponding output
        model_layers(list of strings): List of the strings that corresponds to all the
            model layers that match the naming in acceptable_layers

    """

    model_layers = []
    for layer in model.layers:
        layer_check = list(filter(layer.name.startswith, acceptable_layers)) != []
        if layer_check:
            model_layers.append(layer.name)
    inp = model.input
    intermediate_out = []
    for item in model_layers:
        if item == 'dense':
            intermediate_out.append(model.output.op.inputs[0])
        else:
            intermediate_out.append(model.get_layer(item).output)
    intermediate = K.function([inp], intermediate_out)
    return intermediate, model_layers


def intermeidate_layer_rmse(model_1, model_2, model_3, input_, layer_index):
    """Calculate the rmse(root mean square error) between the intermediate layers
     outputs of keras vs bitmatch models

    Args:
        model_1(syntiant float model): Model definition in Syntiant float
        model_2(syntiant bitmatch model): Model definition is Syntiant Bitmatch
        model_3(keras model): Model definition in Keras
        input_(numpy array or tensor): Sample input tensor to be used in calculating
            the rmse for.
        layer_index(integer): Index of the syntiant float model that we would like to
            calculate the rmse for
            
    Returns:
        rmse(float):

    """

    # model_1 float, model_2 bitmatch, model_3 keras
    # Layer info for the float model
    acceptable_layers_1 = ['BatchNorm', 'Dw_conv', 'Conv', 'AvgPool', 'Dense']
    intermediate1, model_1_layers = extract_intermediate(model_1, acceptable_layers_1)

    # Layer info for the bitmatch model
    acceptable_layers_2 = ['nd_p12x_b0conv2d', 'syntiant_batch_norm', 'nd_p12x_b0depthwise', 'nd_p12x_b0average_pool2d', 'nd_p12x_b0dense_bitmatch']
    intermediate2, model_2_layers = extract_intermediate(model_2, acceptable_layers_2)

    # Layer info for the keras model
    acceptable_layers_3 = ['conv2d', 'activation', 'depthwise_conv2d', 'average_pooling2d', 'dense']
    intermediate3, model_3_layers = extract_intermediate(model_3, acceptable_layers_3)

    # input_ = input_/0.574836373329162687 + 87
    intermediate_layer_out_1 = intermediate1(input_)
    intermediate_layer_out_2 = intermediate2(input_)
    intermediate_layer_out_3 = intermediate3(np.transpose(input_, axes=[0, 2, 3, 1]))
    layer_name = model_1_layers[layer_index]
    if layer_name.startswith("Conv") or layer_name.startswith("Dw_conv"):
        activation = model_1.get_layer(layer_name).activation
        rescaling = 1.0/activation.get_rescale()
        # if activation.name.startswith("linear16_clipped"):
        converting_factor = rescaling / 512 # for linear16
    elif layer_name.startswith("BatchNorm"):
        activation = model_1.get_layer(layer_name).activation
        rescaling = 1.0/activation.get_rescale()
        converting_factor = rescaling / 256 
    elif layer_name.startswith('AvgPool'):
        activation = model_1.get_layer(layer_name).activation
        rescaling = 1.0/activation.get_rescale()
        converting_factor = rescaling / 512 # for linear16
    elif layer_name.startswith('Dense'):
        activation = model_1.get_layer(layer_name).activation
        rescaling = 1.0/activation.get_rescale()
        converting_factor = rescaling / 512 # for linear16
    else:
        print("Unrecognized layer")
        exit()
    
    values_model_2 = intermediate_layer_out_2[layer_index]*converting_factor
    values_model_3 = intermediate_layer_out_3[layer_index]
    if not layer_name.startswith('Dense'):
        values_model_3 = np.transpose(values_model_3, axes=[0, 3, 1, 2])
    return rmse(values_model_2, values_model_3)


def intermeidate_layer_hist(model_1, model_2, model_3, input_):
    """Plots the histogram of the values of the intermediate layers
     and save them in .png format

    Args:
        model_1(syntiant float model): Model definition in Syntiant float
        model_2(syntiant bitmatch model): Model definition is Syntiant Bitmatch
        model_3(keras model): Model definition in Keras
        input_(numpy array or tensor): Sample input tensor to be used in calculating
            the rmse for.
            
    Returns:
        None: Saves the histogram plot of the distribution of the intermediate layer
            output values

    """
    # Layer info for the float model
    acceptable_layers_1 = ['Input_0', 'BatchNorm', 'Dw_conv', 'Conv', 'AvgPool', 'Dense', 'syntiant_softmax']
    intermediate1, model_1_layers = extract_intermediate(model_1, acceptable_layers_1)

    # Layer info for the bitmatch model
    acceptable_layers_2 = ['syntiant_input_ops','nd_p120b0_conv2d', 'syntiant_batch_norm', 'nd_p120b0_depthwise', 'nd_p120b0_average_pool', 'nd_p120b0_dense', 'nd_p120b0_softmax']
    intermediate2, model_2_layers = extract_intermediate(model_2, acceptable_layers_2)

    # Layer info for the keras model
    acceptable_layers_3 = ['input_1', 'conv2d', 'activation', 'depthwise_conv2d', 'average_pooling2d', 'flatten', 'dense']
    intermediate3, model_3_layers = extract_intermediate(model_3, acceptable_layers_3)

    # input_ = input_/0.574836373329162687 + 87
    input_2 = input_
    # intermediate_layer_out_1 = intermediate1(input_)
    intermediate_layer_out_2 = intermediate2(input_2)
    # intermediate_layer_out_3 = intermediate3(np.transpose(input_, axes=[0, 2, 3, 1]))

    for layer_index, layer_name in enumerate(model_1_layers):
        png_file_name = layer_name
        if png_file_name.startswith("Conv") or png_file_name.startswith("Dw_conv"):
            activation = model_1.get_layer(layer_name).activation
            rescaling = 1.0/activation.get_rescale()
            # if activation.name.startswith("linear16_clipped"):
            converting_factor = rescaling / 512 # for linear16
        elif png_file_name.startswith("BatchNorm"):
            activation = model_1.get_layer(layer_name).activation
            rescaling = 1.0/activation.get_rescale()
            converting_factor = rescaling / 256 
        elif png_file_name.startswith('Dense'):
            activation = model_1.get_layer(layer_name).activation
            rescaling = 1.0/activation.get_rescale()
            converting_factor = rescaling / 512 # for linear16
            print(model_2_layers[layer_index])
            print(intermediate_layer_out_2[layer_index])
        elif png_file_name.startswith('AvgPool'):
            activation = model_1.get_layer(layer_name).activation
            rescaling = 1.0/activation.get_rescale()
            converting_factor = rescaling / 512 # for linear16
        elif png_file_name.startswith('Input_0'):
            # activation = model_1.get_layer(layer_name).activation
            # rescaling = 1.0/activation.get_rescale()
            converting_factor = 1  # for signed16 input
        elif png_file_name.startswith("syntiant_flatten"):
            converting_factor = 1/512
        else:
            converting_factor = 1
            print(model_2_layers[layer_index])
            print(intermediate_layer_out_2[layer_index])
        
        
        png_file_name = str(layer_index) + '_' + png_file_name + ".png"

        fig = pl.figure()
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        values_model_1 = intermediate_layer_out_1[layer_index]
    
        n, bins, patches = ax1.hist(values_model_1.ravel(), bins = 40, color='r', label='synt fl')
        ax1.set_xlabel('value')
        ax1.set_ylabel('Frequency')
        ax1.legend(prop={'size': 10})
        
        values_model_2 = intermediate_layer_out_2[layer_index]*converting_factor
        n, bins, patches = ax2.hist(values_model_2.ravel(), bins = 40, color='b', label='synt bm')
        ax2.set_xlabel('value')
        ax2.set_ylabel('Frequency')
        ax2.legend(prop={'size': 10})

        values_model_3 = intermediate_layer_out_3[layer_index]
        n, bins, patches = ax3.hist(values_model_3.ravel(), bins = 40, color='g', label='keras')
        ax3.set_xlabel('value')
        ax3.set_ylabel('Frequency')
        ax3.legend(prop={'size': 10})

        image_path = os.path.join(os.getenv('HOME'), "tiny-mlperf/images_hist/")
        if not os.path.isdir(image_path):
            os.mkdir(image_path)
        pl.savefig('./images_hist/'+png_file_name)
        pl.close('all')
        
        print("converting_factor = ", converting_factor)
        print("RMSE = ", rmse(values_model_1, values_model_2))
        print("---------------------------------------------------")


def plot_weight_hist(syntiant_float_model, syntiant_bitmatch_model):
    '''Plotting the distribution of weights from syntiant bitmatch model vs float model
        Args:
            syntiant_float_model(Keras Model): compiled syntiant float model
                that accepts input images and predicts their corresponding label
            syntiant_bitmatch_model(Keras Model): compiled syntiant bitmatch model
                that accepts input images and predicts their corresponding label
        Returns:
            None
    '''

    acceptable_layers_synt = ['Conv', 'Dw_conv', 'Dense']
    acceptable_layers_bm = ['nd_p12x_b0conv2d', 'nd_p12x_b0depthwise', 'nd_p12x_b0dense']
    layer_counter = 0
    partial_weights = []
    bm_layer_indexes = defaultdict(list)
    for indx, layer in enumerate(syntiant_bitmatch_model.layers):
        layer_check = list(filter(layer.name.startswith, acceptable_layers_bm))
        if layer_check:
            bm_layer_indexes[layer_check[0]].append(indx)

    for idx, layer in enumerate(syntiant_float_model.layers):
        layer_check = list(filter(layer.name.startswith, acceptable_layers_synt))
        if layer_check:
            png_file_name = layer.name
            name_index = acceptable_layers_synt.index(layer_check[0])
            bm_layer_indx = bm_layer_indexes[acceptable_layers_bm[name_index]].pop(0)
            for i, weight in enumerate(layer.weights):
                png_file_name += '_' + str(i) + '.png'
                fl_weight = layer.weights[i].numpy()
                bm_weight = syntiant_bitmatch_model.layers[bm_layer_indx].weights[i].numpy()

                bm_weight = bm_weight/(2**7)
                fig = pl.figure()
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
                

                n, bins, patches = ax1.hist(fl_weight.ravel(), bins = 60, color='r', label='synt fl')
                # ax1.set_xlabel('Weight value')
                ax1.set_ylabel('Frequency')
                
                ax1.legend(prop={'size': 10})
                if i==0:
                    ax1.set_title(layer.name+"_weight")
                else:
                    ax1.set_title(layer.name+"_bias")
                
                n, bins, patches = ax2.hist(bm_weight.ravel(), bins = 60, color='b', label='synt bm')
                ax2.set_xlabel('Weight value')
                ax2.set_ylabel('Frequency')
                ax2.legend(prop={'size': 10})

                # first create the folder if it doesn't exist
                image_path = os.path.join(os.getenv('HOME'), "tiny-mlperf/images_weights/")
                if not os.path.isdir(image_path):
                    os.mkdir(image_path)

                pl.savefig('./images_weights/'+png_file_name)
                pl.close('all')


def assign_biases(syntiant_float_model, syntiant_bitmatch_model):
    """Tranferring bias from syntiant float model into syntiant bitmatch

    Args:
        syntiant_float_model(Syntiant Keras Model): compiled syntiant float model
            that accepts input images and predicts their corresponding label
        syntiant_bitmatch_model(Syntiant bitmatch Model): compiled syntiant bitmatch model
            that accepts input images and predicts their corresponding label
                
    Returns:
        None
    """

    acceptable_layers_synt = ['Conv', 'Dw_conv', 'Dense']
    acceptable_layers_bm = ['nd_p120b0_conv2d', 'nd_p120b0_depthwise', 'nd_p120b0_dense']
    layer_counter = 0
    partial_weights = []
    bm_layer_indexes = defaultdict(list)
    for indx, layer in enumerate(syntiant_bitmatch_model.layers):
      layer_check = list(filter(layer.name.startswith, acceptable_layers_bm))
      if layer_check:
        bm_layer_indexes[layer_check[0]].append(indx)
    for idx, layer in enumerate(syntiant_float_model.layers):
      layer_check = list(filter(layer.name.startswith, acceptable_layers_synt))
      if layer_check:
        name_index = acceptable_layers_synt.index(layer_check[0])
        keras_layer_indx = bm_layer_indexes[acceptable_layers_bm[name_index]].pop(0)
        bias_index = len(layer.weights)-1
        temp_bias = layer.weights[bias_index].numpy().astype(np.float16)
        syntiant_bitmatch_model.layers[keras_layer_indx].weights[bias_index].assign(temp_bias)


def compare_model_performances(model, syntiant_float, bitmatch_model, ds_test, ds_test_keras):
    """Comparing syntiant float, bitmatch, and keras models performances

    Args:
        model(Keras Model): compiled keras float model
            that we import the pretrained weights
        syntiant_float_model(Syntiant Keras Model): compiled syntiant float model
            that accepts input images and predicts their corresponding label
        bitmatch_model(Syntiant bitmatch Model): compiled syntiant bitmatch model
            that accepts input images and predicts their corresponding label
        ds_test: Data(channel First) in the form of iterator or numpy array
        ds_test_keras: Data(channel last) in the form of iterator or numpy array.
            Same as the ds_test but its channel last
            
    Returns:
        None
    """
    
    print("Performing inference on Syntiant Float Model ...")
    pred_prob_fl = syntiant_float.predict(ds_test)

    print("Performing inference on Syntiant Bitmatch Model ...")
    pred_prob_bm = bitmatch_model.predict(ds_test)

    print("Performing inference on Keras Imported Model ...")
    pred_prob_keras = model.predict(ds_test_keras)

    y_pred_fl = np.argmax(pred_prob_fl, axis=1)
    y_pred_bm = np.argmax(pred_prob_bm, axis=1)
    y_pred_keras = np.argmax(pred_prob_keras, axis=1)

    match = 0
    for i, item in enumerate(y_pred_bm):
        if y_pred_keras[i] == y_pred_bm[i]:
            match += 1
    print("Syntiant Bitmatch matches the Keras model = %0.2f percent" %(100*match/len(y_pred_keras)))

    match = 0
    for i, item in enumerate(y_pred_fl):
        if y_pred_fl[i] == y_pred_bm[i]:
            match += 1
    print("Syntiant Bitmatch matches the Syntiant Float = %0.2f percent" %(100*match/len(y_pred_fl)))

    match = 0
    for i, item in enumerate(y_pred_fl):
        if y_pred_fl[i] == y_pred_keras[i]:
            match += 1
    print("Syntiant Float matches the Keras model = %0.2f percent" %(100*match/len(y_pred_fl)))

    # Keras model Performance
    test_scores_keras = model.evaluate(ds_test_keras)
    print('Keras performance')
    print("Loss = ", test_scores_keras[0])
    print("Accuracy = ", test_scores_keras[1])
    
    # Syntiant Bitmatch performance
    test_scores_bm = bitmatch_model.evaluate(ds_test)
    print('Bitmatch performance')
    print("Loss = ", test_scores_bm[0])
    print("Accuracy = ", test_scores_bm[1])
  
    # Syntiant Float Performance
    test_scores_fl = syntiant_float.evaluate(ds_test)
    print('Float performance')
    print("Loss = ", test_scores_fl[0])
    print("Accuracy = ", test_scores_fl[1])

def update_layer_activation(layer, activation_value):
    if isinstance(layer.activation, Linear16Clipped):
        new_activation = Linear16Clipped(virtual_max_clip=activation_value, fake_quantize=True)
    elif isinstance(layer.activation, Relu16Clipped):
        new_activation = Relu16Clipped(virtual_max_clip=activation_value, fake_quantize=True)
    elif isinstance(layer.activation, LinearClipped):
        new_activation = LinearClipped(virtual_max_clip=activation_value, fake_quantize=True)
    elif isinstance(layer.activation, ReluClipped):
        new_activation = ReluClipped(virtual_max_clip=activation_value, fake_quantize=True)
  
    layer.activation = new_activation


def search_virtual_max_clip(keras_model, input_data):
    """Iterating through a list of values for virtual_max_clip and Comparing
     the intermediate layer performance of keras model vs the bitmatch version
     of the model. At the returning a model with the best virtual_max_clip
     numbers.

    Args:
        keras_model(Keras Model): compiled keras float model
            that we import the pretrained weights
        input_(numpy array or tensor): Sample input tensor to be used in calculating
            the rmse for.
            
    Returns:
        syntiant_float(Syntiant Keras Model): compiled syntiant float model
            that accepts input images and predicts their corresponding label
        bitmatch_model(Syntiant bitmatch Model): compiled syntiant bitmatch model
            that accepts input images and predicts their corresponding label
    """

    activation_values = [1, 15, 60, 70, 1024]
    # Building the Syntiant Float model
    syntiant_float = TinyMlPerf()
    syntiant_float.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
    # Transferring the weights from Keras to Syntiant Float
    transfer_weights_biases(syntiant_float, model, zeros_bias=True)
    activation_layer_counter = -1
    for layer_index, layer in enumerate(syntiant_float.layers):
        layer_rmses = []
        best_rmse = math.inf
        best_value = None
        if hasattr(layer, 'activation'):
            print("calculating rmse in %s" %layer.name)
            activation_layer_counter += 1
            print(activation_layer_counter)
            for value in activation_values:
                print("Checking the activation value of %s"%value)
                update_layer_activation(layer, value)
                bitmatch_model = get_bitmatch_model(syntiant_float)
                rmse_value = intermeidate_layer_rmse(syntiant_float, bitmatch_model, keras_model, input_data, activation_layer_counter)
                if rmse_value < best_rmse:
                    best_rmse = rmse_value
                    best_value = value
                print("rmse value is %s"%rmse_value)
                layer_rmses.append(rmse_value)
        if best_value:
            print("Best rmse value is %s"%best_rmse)
            print("Corresponding activation value is %s"%best_value)
            # updating the activation of the layer to the best value
            print("Saving the plot ...")
            update_layer_activation(layer, best_value)
            # plot the rmses
            plt.plot(activation_values, layer_rmses)
            plt.title(layer.name)
            plt.xlabel("Virtual Max Clip")
            plt.ylabel("RMSE")
            plt.savefig('./images_rmse/'+ str(layer_index) +"_"+layer.name+'.png')
            plt.close('all')
            print("-------------------------------------------------------------")
    bitmatch_model = get_bitmatch_model(syntiant_float)

    return syntiant_float, bitmatch_model

def transfer_weights_biases(syntiant_float_model, keras_model, zeros_bias=False):
    """Transferring the weights from keras model into syntiant float model

    Args:
        syntiant_float_model(Syntiant Keras Model): compiled syntiant float model
            that accepts input images and predicts their corresponding label
        keras_model(Keras Model): compiled keras float model
            that we import the pretrained weights
        zeros_bias (boolean): Whether to zero the biases or not.
            
    Returns:
        None
    """
    # Zeroing the bias in the following three layers leads to performance increase
    zero_layers = ['Dw_conv', 'Conv', 'Dense']
    # Layers for which we transfer weights from keras into syntiant network
    acceptable_layers_synt = ['BatchNorm', 'Dw_conv', 'Conv', 'Dense']
    acceptable_layers_keras = ['batch_normalization', 'depthwise_conv2d', 'conv2d', 'dense']
    layer_counter = 0
    partial_weights = []
    keras_layer_indexes = defaultdict(list)
    for indx, layer in enumerate(keras_model.layers):
        layer_check = list(filter(layer.name.startswith, acceptable_layers_keras))
        if layer_check:
            keras_layer_indexes[layer_check[0]].append(indx)
    for idx, layer in enumerate(syntiant_float_model.layers):
        layer_check = list(filter(layer.name.startswith, acceptable_layers_synt))
        if layer_check:
            name_index = acceptable_layers_synt.index(layer_check[0])
            keras_layer_indx = keras_layer_indexes[acceptable_layers_keras[name_index]].pop(0)
            for i, weight in enumerate(layer.weights):
                if zeros_bias and (layer_check[0] in zero_layers) and i!=0:
                    temp_arr = keras_model.layers[keras_layer_indx].weights[i].numpy()
                    temp_arr = np.zeros_like(temp_arr)
                    layer.weights[i].assign(temp_arr)
                    # keras_model.layers[keras_layer_indx].weights[i].assign(temp_arr)
                else:
                    layer.weights[i].assign(keras_model.layers[keras_layer_indx].weights[i])

def get_bitmatch_model(float_model):
    """Getting the bitmatch architecture from float architecture

    Args:
        float_model(Keras Model): compiled syntiant float model
            that accepts input images and predicts their corresponding label

    Returns:
        Syntiant Bitamtch architecture
    """
    # The following configuration leads to the least amount of performance loss
    # uniform: quantization scheme
    # 8: number of quantization bits, 
    # no_clip: no clipping of the values happen
    (q_scheme, q_bits, q_range) = ('uniform', 8, 'no_clip')
    for layer in float_model.layers:
        if layer.is_quantizeable():
            layer.set_layer_quantization_args(q_scheme=q_scheme,
                                              q_bits=q_bits,
                                              q_range=q_range)
    bitmatch_model = float_model.get_bitmatch_architecture("NDP12X_B0")
    bitmatch_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),\
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    return bitmatch_model

def create_synpkg(bitmatch_model):
    """This function will get the bitmatch_model as input and generate the 
     corresponding synpkg.

    Args:
        bitmatch_model(Syntiant bitmatch Model): compiled syntiant bitmatch model
            that accepts input images and predicts their corresponding label

    Returns:
        None: Saves the synpkg of the bitmatch model
    """

    chip = "NDP120_B0"
    package_filename = "tinyMLperf.synpkg"
    ph_file_name = "posterior.json"
    model_labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', \
        'yes', 'silence', 'unknown']
    flattened_configs = bitmatch_model.get_flattened_serialization()
    bitmatch_model.get_runtime_summary()
    package = SyntiantPackageFactory.build_package(chip=chip,
                                        version_string="TinyML April 2021",
                                        config_data=[flattened_configs],
                                        class_labels=[model_labels])
    package.copy_from_package(file_name=ph_file_name)
    package.to_file(filename=package_filename)

def input_flattening(features, melbin_name):
    """This function receives the input feature vector, quantizes it to 16bit, 
     flattens it, and then saves the 16bit version in the .melbin format.

    Args:
        features(Numpy tensor): Input tensor consisting of float features.

    Returns:
        None: Saves the melbin format of the input feature
    """

    input_shape = (1, 49, 10)
    q_features = float_to_fixed(
        features, signed=True, int_bits=6, frac_bits=9)
    if melbin_name == 'down':
        print(q_features)
    fq_features = q_features.flatten(order='F')
    barray = fq_features.astype(np.int16).tobytes()
    with open(melbin_name+'.melbins', 'wb') as fp:
        fp.write(barray)

def create_databin(ds_test):
    """This function creates a melbin file for each of the labels. This is a 
     helper function for the testing purposes.

    Args:
        ds_test(numpy iterator): Data(channel First) in the form of iterator
            or numpy array

    Returns:
        all_data(list): List of numpy tensors that is the flattened reshaped 
            version of one batch of input data
    """
    all_data = []
    # pick one sample data
    model_labels = {0:'down', 1:'go', 2:'left', 3:'no', 4:'off', 5:'on', \
        6:'right', 7:'stop', 8:'up', 9:'yes', 10:'silence', 11:'unknown'}
    ds_test_iter = iter(ds_test)
    ds_test_batch = ds_test_iter.next()
    ds_test_x_batch = ds_test_batch[0].numpy()
    ds_test_y_batch = ds_test_batch[1].numpy()
    data_location = {}
    for i, item in enumerate(ds_test_y_batch):
        if item in data_location:
            continue
        else:
            data_location[item] = i
    sample_data = {}
    for i in range(len(model_labels)):
        sample_data[model_labels[i]] = np.copy(ds_test_x_batch[data_location[i]])
    for key in sample_data:
        all_data.append(np.copy(sample_data[key]))
        input_flattening(np.copy(sample_data[key]), key)
    return all_data


def zero_pad_input(ds_test):
    """This function receives the input tensor and pads it with zero.

    Args:
        ds_test(numpy iterator): Data(channel First) in the form of iterator
            or numpy array

    Returns:
        ds_test(numpy iterator): Same as the input but with padded zeros
    """

    paddings = tf.constant([[0, 0], [0, 0], [1, 0],[0, 0]])
    ds_test = ds_test.map(lambda x, y : (tf.pad(x, paddings, "SYMMETRIC"), y))
    return ds_test

def test_bin_files(bitmatch_model):
    """This function receives the bitmatch model and runs the inference on the 
     8bit input bin files that are located inside kws_data folder.

    Args:
        bitmatch_model(Syntiant bitmatch Model): compiled syntiant bitmatch model
            that accepts input images and predicts their corresponding label

    Returns:
        None: Printing the Accuracy to the console.
    """
    print("""
########################################################################
Attention! Make sure the bitmatch ops is turned off when you defined
the syntiant float model, otherwise you'll receive incorrect results.
########################################################################
""")
    input_shape = (1, 49, 10)
    data_path = os.path.join(os.getenv('HOME'), "tiny-mlperf/kws_data/")
    bin_file_names = glob.glob(data_path + "/*.bin")
    test_data = []
    labels = []
    print("Reshaping and reformatting the signed8 input data to signed16")
    for fname in bin_file_names:
        with open(fname, 'rb') as fp:
            q_data = np.frombuffer(fp.read(), dtype=np.int8)
        data = 0.5847029089927673 * (q_data - 83)
        data = data.reshape((1,) + input_shape, order='C')
        converted16 = float_to_fixed(data, signed=True, int_bits=6, frac_bits=9)
        test_data.append(converted16)
        label = fname.split("_")[-1].split(".")[0]
        labels.append(int(label))

    total_samples = len(labels)
    match = 0
    print("Running inference on %s sample data" %total_samples)
    for i, reshaped_input_signed16 in enumerate(test_data):
        y_pred_prob = bitmatch_model.predict(reshaped_input_signed16)
        y_pred = np.argmax(y_pred_prob, axis=1)
        if y_pred == labels[i]:
            match += 1
    print("Accuracy = ", match/total_samples)
