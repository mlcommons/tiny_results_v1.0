#import tensorflow as tf
import os
import numpy as np
#import matplotlib.pyplot as plt
import argparse

import get_dataset as kws_data
import kws_util
#import keras_model as models

if __name__ == '__main__':
  Flags, unparsed = kws_util.parse_command()

  num_test_files = Flags.num_bin_files
  test_file_path = Flags.bin_file_path
  
  print(f"Extracting {num_test_files} to {test_file_path}")
  word_labels = ["Down", "Go", "Left", "No", "Off", "On", "Right",
                 "Stop", "Up", "Yes", "Silence", "Unknown"]

  num_labels = len(word_labels)
  ds_train, ds_test, ds_val = kws_data.get_training_data(Flags, val_cal_subset=False)
  eval_data = ds_test
  labels = []  
  file_names = []
  count = 0
  # make the target directory and all directories above it if it doesn't exist
  os.makedirs(test_file_path, exist_ok = True) 

  eval_data = eval_data.unbatch().batch(1).take(num_test_files).as_numpy_iterator()
  for dat, label in eval_data:
    dat_q = np.array(dat/0.5847029089927673 + 83, dtype=np.int8) # should match input type in quantize.py

    label_str = word_labels[label[0]]
    fname = f"tst_{count:06d}_{label_str}_{label[0]}.bin"
    with open(os.path.join(test_file_path, fname), "wb") as fpo:
      fpo.write(dat_q.flatten())

    labels.append(label[0])
    file_names.append(fname)
    count += 1
    
  # with open(os.path.join(test_file_path, "y_labels.csv"), "w") as fpo_true_labels:
  #   for (fname, lbl) in zip(file_names, labels):
  #     fpo_true_labels.write(f"{fname}, {num_labels}, {lbl}\n")

  print("SUCCESS")


