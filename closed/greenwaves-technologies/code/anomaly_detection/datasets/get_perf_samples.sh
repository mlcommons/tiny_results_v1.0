#!/bin/sh

python3 source/convert_dataset.py

cp -r dev_data/ToyCar/test/ "perf_samples/"
