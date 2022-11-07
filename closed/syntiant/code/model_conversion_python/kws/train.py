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

#!/usr/bin/env python
import os
import math
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import keras_model as models
import get_dataset as aww_data
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import aww_util
from utility import compare_model_performances, transfer_weights_biases, \
    get_bitmatch_model, test_bin_files

from syntiant_networks.architectures.ndp12x.ndp12X import NDP12X
from tinyml_model import TinyMlPerf



if __name__ == '__main__':

    # Importing the keras model
    Flags, unparsed = aww_util.parse_command()

    print('We will download data to {:} directory'.format(Flags.data_dir))
    print("""
########################################################################
Depending on your internet speed, it might take a while for the data to
be downloaded.
########################################################################
""")
    ds_train, ds_test, ds_val = aww_data.get_training_data(Flags)
    print("Done getting data")

    if Flags.model_init_path == 'None':
        print("Starting with untrained model")
        model = models.get_model(args=Flags)
        callbacks = aww_util.get_callbacks(args=Flags)
        train_hist = model.fit(ds_train, validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks)
        aww_util.plot_training(Flags.plot_dir,train_hist)
        model.save(Flags.saved_model_path)
    else:
        print(f"Starting with pre-trained model from {Flags.model_init_path}")
        model = keras.models.load_model(Flags.model_init_path)

    # Constructing the tiny ml perf model
    syntiant_float = TinyMlPerf()
    syntiant_float.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),\
        metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # Transferring weights from the imported keras model into Syntiant network
    print("""
########################################################################
Transferring weights from the imported keras model into Syntiant network. 
########################################################################
""")
    transfer_weights_biases(syntiant_float, model, zeros_bias=False)

    # Building the bitmatch architecture
    bitmatch_model = get_bitmatch_model(syntiant_float)

    # Moving the channel to the last index since mlperf accept data as channel last 
    ds_test_keras = ds_test.map(lambda x, y : (tf.transpose(x, perm=[0, 2, 3, 1]), y))

    # Comparing syntiant float, bitmatch, and keras models on test data
    compare_model_performances(model, syntiant_float, bitmatch_model, ds_test, ds_test_keras)

    # bitmatch_model.save('syntiant_bitmatch.h5')
    syntiant_float.save('syntiant_float.h5')
    syntiant_float.save_weights('syntiant_float_wts.h5')
