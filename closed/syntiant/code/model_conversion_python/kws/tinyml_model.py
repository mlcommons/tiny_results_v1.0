import copy
from tensorflow.keras.constraints import min_max_norm
from tensorflow import Tensor
from syntiant_networks.architectures.ndp import NDP
from syntiant_networks.layers import (
    SyntiantInput, SyntiantSoftmax, SyntiantFlatten, SyntiantDropout,
    SyntiantDense, SyntiantBatchNorm,
    SyntiantConv2D, SyntiantDepthwiseConv2D, SyntiantConv2DTranspose,
    SyntiantMaxPooling2D, SyntiantAveragePooling2D)
from syntiant_networks.activations import (
    ReluClipped, 
    TanhClipped, TanhScaledClipped,
    LinearClipped, Linear16Clipped,
    SigmoidClipped, SigmoidScaledClipped)

from syntiant_networks.architectures.ndp12x.ndp12X import NDP12X

class TinyMlPerf(NDP12X):
    """Network specification for TinyMlPert running on NDP12x. Original implementation can 
    be found here https://github.com/mlcommons/tiny
    """

    weights_min = -1.0
    weights_max = 1.0
    VIRTUAL_MAX_LINEAR = 1024.0
    num_filters = 64
    VIRTUAL_MAX_ReLU = 15.0

    min_max_norm_params = {"min_value": weights_min,
                           "max_value": weights_max,
                           "rate": 1.0,
                           "axis": 0}

    dense_params = {
        "kernel_initializer": 'glorot_uniform',
        "bias_initializer": 'zeros',
        "kernel_constraint": min_max_norm(**min_max_norm_params),
        "bias_constraint": min_max_norm(**min_max_norm_params)
    }

    conv_params = {
        "kernel_initializer": 'glorot_uniform',
        "bias_initializer": 'zeros',
        "kernel_constraint": min_max_norm(**min_max_norm_params),
        "bias_constraint": min_max_norm(**min_max_norm_params)
    }

    depthwise_conv_params = {
        "depthwise_initializer": 'glorot_uniform',
        "bias_initializer": 'zeros',
        "depthwise_constraint": min_max_norm(**min_max_norm_params),
        "bias_constraint": min_max_norm(**min_max_norm_params)
    }

    default_hyper = {
        'layers': [
            {'layer_type': 'SyntiantInput', 'name': 'Input_0', 'shape': (1, 49, 10),
                'input_type':'SIGNED16',
                'bitmatch_ops': [('quantize', {'input_type': 'SIGNED16'})]
           #      'bitmatch_ops': None
                },
            {'layer_type': 'SyntiantConv', 'name': 'Conv', 'filters': num_filters,
             'kernel_size': (10, 4), 'strides':(2, 2), 'padding': 'same',
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16', 
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR, 
                            }, **conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm', 'axis':1,
             'activation': {'activation_type': 'relu', 
                            'virtual_max_clip': VIRTUAL_MAX_ReLU,
                            'fake_quantize':True}},
            {'layer_type': 'SyntiantDepthwiseConv', 'name': 'Dw_conv',
             'kernel_size': (3, 3), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **depthwise_conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_1', 'axis':1,
             'activation': {'activation_type': 'relu',
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantConv', 'name': 'Conv_1', 'filters': num_filters,
             'kernel_size': (1, 1), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_2', 'axis':1,
             'activation': {'activation_type': 'relu',
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantDepthwiseConv', 'name': 'Dw_conv_1',
             'kernel_size': (3, 3), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **depthwise_conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_3', 'axis':1,
             'activation': {'activation_type': 'relu',
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantConv', 'name': 'Conv_2', 'filters': num_filters,
             'kernel_size': (1, 1), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_4', 'axis':1,
             'activation': {'activation_type': 'relu', 
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantDepthwiseConv', 'name': 'Dw_conv_2',
             'kernel_size': (3, 3), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **depthwise_conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_5', 'axis':1,
             'activation': {'activation_type': 'relu',
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantConv', 'name': 'Conv_3', 'filters': num_filters,
             'kernel_size': (1, 1), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_6', 'axis':1,
             'activation': {'activation_type': 'relu',
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantDepthwiseConv', 'name': 'Dw_conv_3',
             'kernel_size': (3, 3), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **depthwise_conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_7', 'axis':1,
             'activation': {'activation_type': 'relu',
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantConv', 'name': 'Conv_4', 'filters': num_filters,
             'kernel_size': (1, 1), 'padding': 'same', 'strides':(1, 1),
             'bias_type':'MB', 'sf_type':'MS', 'dilation_rate':(1, 1),
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            # 'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }, **conv_params},
            {'layer_type': 'SyntiantBatchNorm', 'name': 'BatchNorm_8', 'axis':1,
             'activation': {'activation_type': 'relu',
                            'virtual_max_clip': VIRTUAL_MAX_ReLU, 
                             'fake_quantize':True}},
            {'layer_type': 'SyntiantAveragePool', 'name': 'AvgPool', 'pool_size': (25, 5),
             'strides':(1,1), 'padding': 'valid', 
             'activation': {'activation_type': 'linear16',
                            'fake_quantize':True,
                            'virtual_max_clip': VIRTUAL_MAX_LINEAR,
                            }
             },
            {'layer_type': 'SyntiantFlatten', 'name': 'Flatten_29'},
            {'layer_type': 'SyntiantDense', 'name': 'Dense', 'units': 12,
             'compute_on_float':False,
             **NDP12X.layer_quantization_defaults_settings,
             'activation': {'activation_type': 'linear', 
                            'fake_quantize':True,
                            'virtual_max_clip': 8,
                            # 'native_max_clip':VIRTUAL_MAX_LINEAR,
                            },
                             **dense_params},
            {'layer_type': 'SyntiantSoftMax'}
        ],
        'training_parameters': {**NDP12X.training_default_hyper}
    }

    def __init__(self, hyperparameters={}):
        """Initialize the class.

        Args:
            hyperparameters (dict): The hyperparameters to apply to the model (e.g., l4 size).
                These hyperparameters supercede the hyperparameters defaults specified within the
                class and the NDP superclass.
        """

        self.hyper = self._merge_hyperparam_dictionaries(self.default_hyper, hyperparameters)
        super().__init__(arch_dict=self.hyper)

#### end of class TinyMlPerf(NDP12X):

