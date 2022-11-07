import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
assert tf.__version__.startswith('2')
from tensorflow.keras.utils import to_categorical

import syntiant_networks
from syntiant_networks import layers
from syntiant_networks import activations

from syntiant_networks.architectures import NDP
from syntiant_networks import experimental as syntnet_exp
from syntiant_networks.experimental.model_analyzer import ModelAnalyzer

import PIL
import numpy as np
import matplotlib.pyplot as plt

import pickle
import sys
from copy import copy

from vww_model import mobilenet_v1

sys.path.insert(1, '..') # perf_utils.py is one directory up, shared across the benchmarks
from perf_utils import generate_package, unpickle, img_from_binfile,   img_to_binfile, load_cifar_10_data, show_range


IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

BASE_DIR = os.path.join(os.getcwd(), 'vw_coco2014_96')
ref_model_path = './trained_models/vww_96.h5'

validation_split = 0.1
batch_size = 50


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=.1,
    horizontal_flip=True,
    validation_split=validation_split,
    rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training',
    color_mode='rgb')
val_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation',
    color_mode='rgb')

# Syntiant models are channels first, instead of channels-last, so transpose 
syn_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=.1,
    horizontal_flip=True,
    validation_split=validation_split,
    rescale=1. / 255,
    data_format="channels_first")
syn_val_generator = syn_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=1,
    subset='validation',
    color_mode='rgb')



def take_samples_from_datagen(gen, num_samples):
  for i, (x_batch, y_batch) in enumerate(gen):
    if i == 0: 
      x_out = copy(x_batch)
      y_out = copy(y_batch)
    else:
      x_out = np.concatenate((x_out, x_batch), axis=0)
      y_out = np.concatenate((y_out, y_batch), axis=0)
    if x_out.shape[0] >= num_samples:
      x_out = x_out[:num_samples, ...]
      y_out = y_out[:num_samples, ...]
      break
  return x_out, y_out



def build_cal_set():
  dataset_dir = os.path.join(BASE_DIR, "person") # cal set is built only of person images
  img_set = np.zeros((0, 96, 96, 3))
  for image_file in open("calibration_data.txt"):
    image_file = image_file.rstrip()
    if image_file[-3:] != "jpg": 
      continue
    full_path = os.path.join(dataset_dir, image_file)
    if os.path.isfile(full_path):
      img = tf.keras.preprocessing.image.load_img(
          full_path, color_mode='rgb').resize((96, 96))
      arr = tf.keras.preprocessing.image.img_to_array(img)
      # Scale input to [0, 1.0] like in training.
      arr = arr.reshape((1, 96, 96, 3)) / 255
      img_set = np.concatenate((img_set, arr))
    else:
      print(f"File {full_path} not found. Skipping.")
  return img_set

x_cal = build_cal_set()

ref_model = tf.keras.models.load_model(ref_model_path)
ref_model.compile(
      optimizer=tf.keras.optimizers.Adam(.001),
      loss='categorical_crossentropy',
      metrics=['accuracy'])

ref_model.evaluate(val_generator)

converter = syntnet_exp.Core2KerasToSyntConverter(ref_model, "NDP120_B0", verbose=False)
syn_model_q, syn_model_fp = converter.get_converted_model(x_cal);
syn_model_fp.compile(metrics="accuracy")

syn_model_q.evaluate(syn_val_generator)

generate_package(syn_model_q,  pkg_filename="mobilenet_vww.synpkg", verbose=True)
