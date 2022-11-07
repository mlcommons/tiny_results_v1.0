#!/bin/bash

sh download_dataset.sh
python3 get_perf_samples_split.py
rm -rf vw_coco2014_96/
rm -rf vw_coco2014_96_bin/