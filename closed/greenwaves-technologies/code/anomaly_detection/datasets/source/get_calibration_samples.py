"""
 @file   02_convert.py
 @brief  Script to convert model to tflite
 @author Csaba Kiraly
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import common as com
import numpy 
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# main 02_convert.py
########################################################################
if __name__ == "__main__":
    assert len(sys.argv) > 1
    cal_dir = sys.argv[1]


    mode = True
    # load base directory
    dirs = com.select_dirs(param=param, mode=True)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("============== DATASET_GENERATOR ==============")
        files = com.file_list_generator(target_dir)
        train_data = com.list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"])
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        i = 0
        for sample in train_data[::5]:
            #print(i)
            sample = numpy.expand_dims(sample.astype(numpy.float32), axis=0)
            f = open(cal_dir + "/sample_%s.bin" %i, "wb") 
            f.write(sample)
            f.close()
            #print(numpy.shape(sample))
            i+=1    
        print("nb of samples = ",i)
