import tensorflow as tf
import numpy as np
import train
import os
import struct

#from test import model_name

#tfmodel_path = 'trained_models/' + model_name
#tfmodel = tf.keras.models.load_model(tfmodel_path)
cifar_10_dir = 'cifar-10-batches-py'
#model_name = model_name[:-3]

cal_samples_dir = 'calibration_samples'

    

if not os.path.exists(cal_samples_dir):
    os.makedirs(cal_samples_dir)

train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
    train.load_cifar_10_data(cifar_10_dir)
_idx = np.load('source/calibration_samples_idxs.npy')
j = 0
for i in _idx:
    #print(j)
    sample_img = np.expand_dims(np.array(test_data[i], dtype=np.uint8), axis=0)
    #yield [sample_img]
    #rprint(sample_img)
    f = open(cal_samples_dir + "/image_%s.bin" %j, "wb") 
    sample_img = sample_img.flatten()
    mydata = sample_img
    myfmt = 'B' * len(mydata)
    bin = struct.pack(myfmt, *mydata)
    f.write(sample_img)
    f.close()
    j+=1

print('Done')
