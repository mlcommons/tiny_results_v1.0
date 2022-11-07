import os

from absl import app
import numpy as np

import csv
from tqdm import tqdm
import tensorflow as tf
assert tf.__version__.startswith('2')

BASE_DIR = os.path.join(os.getcwd(), "vw_coco2014_96")

if __name__ == '__main__':
	dataset_dir = os.path.join(BASE_DIR, "person")

	if not os.path.exists("perf_samples/"):
		os.makedirs("perf_samples/")

	ANNOTATION_FILE_PATH = "y_labels.csv"
	n_files = 0
	for row in open(ANNOTATION_FILE_PATH):
		n_files += 1

	with open(ANNOTATION_FILE_PATH) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for index, row in tqdm(enumerate(csv_reader), total=n_files):
			entry = row[0][:-3] + 'jpg'
			target = int(row[2])
			dataset_dir = os.path.join(BASE_DIR, "person")
			full_path = os.path.join(dataset_dir, entry)
			if not os.path.isfile(full_path):
				dataset_dir = os.path.join(BASE_DIR, "non_person")
				full_path = os.path.join(dataset_dir, entry)
				if not os.path.isfile(full_path):
					print(f"Error with image: {entry} cannot find in person and non_person")
					continue

			if entry[5:8] == "val":
				img = tf.keras.preprocessing.image.load_img(
					full_path, color_mode='rgb').resize((96, 96))
				arr = tf.keras.preprocessing.image.img_to_array(img)
				# Scale input to [0, 1.0] like in training.
				arr = arr.reshape([1, 96, 96, 3])
				arr = arr.astype(np.uint8)
	
				f = open("perf_samples/" + entry[0:-3] + "bin", "wb")
				f.write(arr)
				f.close()


## OLD

	# i = 0
	# for idx, image_file in enumerate(os.listdir(dataset_dir)):
	# 	#print(idx)
	# 		# 10 representative images should be enough for calibration.
	# 	if i > 499:
	# 		break
	# 	full_path = os.path.join(dataset_dir, image_file)
	# 	#print(full_path)
	# 	if os.path.isfile(full_path):
	# 		if image_file[5:8] == "val":
	# 			i+=1
	# 			img = tf.keras.preprocessing.image.load_img(
	# 				full_path, color_mode='rgb').resize((96, 96))
	# 			arr = tf.keras.preprocessing.image.img_to_array(img)
	# 			# Scale input to [0, 1.0] like in training.
	# 			arr = arr.reshape([1, 96, 96, 3])
	# 			arr = arr.astype(np.uint8)
	
	# 		#arr = arr / 255.
	# 		#print(arr)
	# 		#print(np.shape(arr))
	# 			f = open("perf_samples/" + image_file[0:-3] + "bin", "wb")
	# 		#print("calibration_samples/" + image_file[0:-3] + ".bin)
	# 			f.write(arr)
	# 			f.close()

	# dataset_dir = os.path.join(BASE_DIR, "non_person")

	# i = 0
	# for idx, image_file in enumerate(os.listdir(dataset_dir)):
	# 	#print(idx)
	# 		# 10 representative images should be enough for calibration.
	# 	#if idx > 10:
	# 	if i > 499:
	# 		exit()
	# 		#break
	# 	full_path = os.path.join(dataset_dir, image_file)
	# 	if os.path.isfile(full_path):
	# 		if image_file[5:8] == "val":
	# 			i+=1
	# 			img = tf.keras.preprocessing.image.load_img(
	# 				full_path, color_mode='rgb').resize((96, 96))
	# 			arr = tf.keras.preprocessing.image.img_to_array(img)
	# 			# Scale input to [0, 1.0] like in training.
	# 			arr = arr.reshape([1, 96, 96, 3])
	# 			arr = arr.astype(np.uint8)
	# 			#print(np.shape(arr))
	# 			#arr = arr / 255.
	# 			#print(arr)
	# 			f = open("perf_samples/" + image_file[0:-3] + "bin", "wb")
	# 			#print("calibration_samples/" + image_file[0:-3] + ".bin)
	# 			f.write(arr)
	# 			f.close()