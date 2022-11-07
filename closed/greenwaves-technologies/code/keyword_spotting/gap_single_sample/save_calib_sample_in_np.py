import glob
import numpy as np
import os
import sys


class MyDataLoader():
	def __init__(self, image_files, max_idx=120, transpose_to_chw=False):
		self._file_list = image_files
		self._idx = 0
		self._max_idx = max_idx if max_idx is not None else len(image_files)
		self._transpose_to_chw = transpose_to_chw

	def __iter__(self):
		self._idx = 0
		return self

	def __next__(self):
		if self._idx > self._max_idx:
			raise StopIteration()
		filename = self._file_list[self._idx]

		# Here we read the image and make it a numpy array
		img_array = np.fromfile(filename, dtype=np.float32)
		img_array = img_array.reshape([ 1, 49, 10,  1])


		self._idx += 1
		return img_array


#image_files = "../calibration_samples/*"



if not os.path.exists("calibration_samples_np/"):
	#print("bonjour")
	os.makedirs("calibration_samples_np/")

data_loader = MyDataLoader(glob.glob("../calibration_samples/*"))

for i in range(120):
	img = next(data_loader)
	np.save("calibration_samples_np/calibration_samples%s.npy" %i,img)
	print(i)
	 # f = open("calibration_samples/" + "image_%s.bin" %i, "wb") 



#img = next(data_loader)
#img = next(data_loader)
#np.save("calibration_samples_np/calibration_samples" + %i +".npy",img)
#np.save("calibration_samples_np/calibration_samples.npy",img)