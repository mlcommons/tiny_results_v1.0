#!/bin/sh


python3 test_accuracy_v2.py --tflite_model "../model/kws_ref_model.tflite" --dataset_path "../kws_bin_files/"  --model_type "int8"
