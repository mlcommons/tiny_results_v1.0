import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from syntiant_networks.architectures.ndp import NDP
from syntiant_networks import experimental as syntnet_exp
import syntiant_networks

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

def generate_package(q_model, pkg_filename="resnet_eembc_new.synpkg", 
                     pkg_version_str="perf_pkg", verbose=True):
    from syntiant_packager.packages import SyntiantPackageFactory

    syn_pkg = SyntiantPackageFactory.build_package(
        'NDP120_B0',
        pkg_version_str,
        configs=[q_model.get_packager_config()],
        class_labels=[["temp"] * 10],
        package_type="IMAGE"
    )
    print("Done building package")
    if verbose:
      print(syn_pkg)
    syn_pkg.to_file(filename=pkg_filename)

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, to_categorical(cifar_train_labels), \
        cifar_test_data, cifar_test_filenames, to_categorical(cifar_test_labels), cifar_label_names

  
def find_file(target_file, search_set='both', verbose=True):
  # search_set = 'test', 'train', or 'both'
  if search_set == 'test':
    search_sets = {'test':test_filenames}
  elif search_set == 'train':
    search_sets = {'train':train_filenames}
  if search_set == 'both':
    search_sets = {'test':test_filenames, 'train':train_filenames}

  found_match = False
  for set_name, file_list in search_sets.items():
    for i, fn in enumerate(file_list):
      if fn.decode('utf8').split('.')[0] == target_file.split('.')[0]: # strip extension
        if verbose: print(f"Found match at index {i} of set {set_name}.")
        return i
    if verbose: print(f"Not found in {set_name}")
  return -1


def img_from_binfile(ifname):
  """
  returns image in width, height, channel array.  
  not suitable for a syntiant channels-first model
  suitable for display in imshow.
  to run in a syntiant channels-first model, transpose  with 
  img.transpose((2,0,1))
  then you can convert back with
  img.transpose((1,2,0))
  """

  with open(ifname, "rb") as fpi:
    buff = fpi.read()
  img = np.frombuffer(buff, dtype=np.uint8).reshape(32,32,3)
  img = img / 255
  # for the files in test_images, uploaded from my eembc directory, 
  # the call should be reshape(3,32,32)
  # Those images are also coming out heigh,width, so they display sideways.  Fix with:
  # img_cwh = img_cwh.transpose((0,2,1))
  return img

def img_to_binfile(img, ofname):
  img_q = np.array(img*255, dtype=np.uint8)
  buff = img_q.tobytes()
  with open(ofname, "wb") as fpo:
    fpo.write(buff)
    

def show_range(x, places=4):
  """
  show_range(x, places=4):
  prints the mininum, mean, and maximum of x, to places decimal places
  """
  if 'int' in str(x.dtype):
    print(f"{np.min(x):4} : {np.mean(x):4} : {np.max(x):4}")
  else:
    print(f"{np.min(x):4.{places}} : {np.mean(x):4.{places}} : {np.max(x):4.{places}}")

    
def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, to_categorical(cifar_train_labels), \
        cifar_test_data, cifar_test_filenames, to_categorical(cifar_test_labels), cifar_label_names
