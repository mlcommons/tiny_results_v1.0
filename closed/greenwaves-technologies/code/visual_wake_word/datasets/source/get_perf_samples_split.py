import os

from absl import app
import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

BASE_DIR = os.path.join(os.getcwd(), "vw_coco2014_96")

if __name__ == '__main__':
	dataset_dir = os.path.join(BASE_DIR, "person")

	if not os.path.exists("vw_coco2014_96_bin/person/"):
		os.makedirs("vw_coco2014_96_bin/person/")

	i = 0
	for idx, image_file in enumerate(os.listdir(dataset_dir)):
		#print(idx)
			# 10 representative images should be enough for calibration.
		#if idx > 10:
			#break
		if i > 499:
			break

		full_path = os.path.join(dataset_dir, image_file)
		#print(full_path)
		if os.path.isfile(full_path):
			if image_file[5:8] == "val":
				i+=1
				img = tf.keras.preprocessing.image.load_img(
					full_path, color_mode='rgb').resize((96, 96))
				arr = tf.keras.preprocessing.image.img_to_array(img)
				# Scale input to [0, 1.0] like in training.
				arr = arr.reshape([1, 96, 96, 3])
				#arr = arr / 255.
				#print(arr)
				#print(np.shape(arr))
				f = open("vw_coco2014_96_bin/person/" + image_file[0:-3] + "bin", "wb")
				#print("calibration_samples/" + image_file[0:-3] + ".bin)
				f.write(arr)
				f.close()

	dataset_dir = os.path.join(BASE_DIR, "non_person")


	if not os.path.exists("vw_coco2014_96_bin/non_person/"):
		os.makedirs("vw_coco2014_96_bin/non_person/")
	i = 0
	for idx, image_file in enumerate(os.listdir(dataset_dir)):
		#print(idx)
			# 10 representative images should be enough for calibration.
		#if idx > 10:
		if i > 499:
			exit()
			#break
		full_path = os.path.join(dataset_dir, image_file)
		#print(full_path)
		if os.path.isfile(full_path):
			if image_file[5:8] == "val":
				i+=1
				img = tf.keras.preprocessing.image.load_img(
					full_path, color_mode='rgb').resize((96, 96))
				arr = tf.keras.preprocessing.image.img_to_array(img)
				# Scale input to [0, 1.0] like in training.
				arr = arr.reshape([1, 96, 96, 3])
			#arr = arr / 255.
			#print(arr)
			#print(np.shape(arr))
				f = open("vw_coco2014_96_bin/non_person/" + image_file[0:-3] + "bin", "wb")
			#print("calibration_samples/" + image_file[0:-3] + ".bin)
				f.write(arr)
				f.close()