# tinyMLPerf Deep Learning Benchmarks for Embedded Devices


## Running the code

Running the code is as simple as `python train.py`

## Scripts and Dependencies

The main script that you'll be working with is `train.py`. This file will load the tinyml perf keras model. You have the option of retraining the model or load the pretrained weights. Once loaded, we transfer the weights into a model that is defined in Syntiant Float version. Afterwards, we create the bitmatch version of the model. The dataset for this model is Google's speech script. If you don't have the data locally, it will automatically download the data and save it on a folder named data next to train.py. The dataset contains the following classes `up`, `down`, `left`, `right`, `on`, `off`, `no`, `yes`, `stop`, `go`, `silence`, `unknown`. At the end we compare model performance and will see how does the quantization change the performance of the model. 



<br />

`aww_utils.py` includes the default model and data settings for running `train.py`. You don't have to change this.

<br />

`get_dataset.py` includes the necessary functions for downloading the dataset and processing the data. You don't have to change this.

<br />

`keras_model.py` is tinyml-perf model definition in keras.

