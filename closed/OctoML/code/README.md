# MLPerf™ Tiny Benchmarking with microTVM

microTVM uses TVM to run models on microcontrollers. microTVM supports both bare-metal devices (i.e. STM microcontrollers) and real-time operating systems such as Zephyr and Arduino.
In addition, any other firmware/operating system could be easily integrated into microTVM by following minimal steps.

The goal of this submission is to provide benchmark examples deploying models with microTVM. One of the advantages of microTVM is providing flexibility for compilation and deployment. Using microTVM you can import models from ONNX or TFLite into Relay representation, optimize based on your MCU, include custom compiler path and finally deploy easily.

In this submission we focused on using Zephyr OS with microTVM to show the compatibility and ease of deployment. microTVM provides two deployment scenarios:

- Host-Driven Execution: In this case you can import your model and build it for the target MCU. microTVM would automatically do all the steps and finally it will provide a communication mechanism using RPC (over UART) where one can send input samples, infer the model and receive the output of the model. This case is mostly useful where one is still developing the model and optimizing it.

- Standalone Execution: In this case, microTVM generates an artifact called Model Library Format (MLF). An MLF file includes the model generated code with the parameters and
all metadata that is required to execute the model. In this mode, we use ahead-of-time (AOT) compilation where microTVM would compile model and generate model in C language using TVM `C` code generator. Using this file with a minimal API calls, one can integrate microTVM in their project.

In this benchmark, we used standalone mode since the communication with host was already designed in the reference submission. This shows a perfect example of a real world application.

## Before You Start
One of the advantages of Model Library Format is that it would generate the full package of all you need to run a model and from there you do not need to have TVM installed in your environment anymore. Therefore, to rebuild our submission you only need to install Zephyr in your environment and then you can build this submission package.

### Build and Run using CM
[Collective Mind (CM) ](https://github.com/mlcommons/ck) is the second generation of the CK meta-framework being developed by the [open workgroup](https://github.com/mlcommons/ck/blob/master/docs/mlperf-education-workgroup.md) to modularize complex AI systems and automate their co-design, benchmarking, optimization and deployment across continuously changing software, hardware and data. CM makes it easier to reproduce our experiments and we have a [CM script](https://github.com/mlcommons/ck/blob/master/cm/docs/tutorial-scripts.md) located [here](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/generate-mlperf-tiny-submission) to reproduce our results. The following instructions are tested on Ubuntu 20.04, Ubuntu 22.04 and RHEL 9.
#### Setup CM
```
python3 -m pip install cmind
cm pull repo --url=https://github.com/mlcommons/ck
```
#### Install
```
cm run script --tags=generate,tiny,mlperf,octoml,submission
```

The main CM scripts which automatically gets called from the above command are given below.

1. [Build Tiny Models](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/reproduce-mlperf-octoml-tinyml-results)
2. [Flash Tiny Models](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/flash-tinyml-binary)
3. [Get Zephyr](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-zephyr)
4. [Get Zephyr SDK](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-zephyr-sdk)
5. [Get MictoTVM](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-microtvm)
6. [GET CMSIS_5](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/get-cmsis_5)

The above command should produce five elf binaries which can be located inside the respective cache entries given by the below command
```
cm show cache --tags=reproduce,tiny,octoml,mlperf
```

#### Flash
To flash each submission, follow the command bellow. Make sure to replace `VARIANT` by either `cmsis_nn` or `native`. You need to specify the model by replacing `MODEL` with a value from (`ad`, `kws`, `ic`, `vww`). Finally, you need to choose `_NUCLEO` or `_NRF` to specify the target board to flash.  

``` 
cm run script --tags=flash,tiny,_VARIANT,_MODEL,_BOARD
```

Example:

``` 
cm run script --tags=flash,tiny,_cmsis_nn,_ic,_NRF
```

### Native Setup
Note: These instruction are for an Ubuntu system and has been tested on Ubuntu 18.04 and 20.04.

- Install cmake: 
```bash
sudo apt install cmake
```
- Install Zephyr. You can find the instruction [here](https://docs.zephyrproject.org/2.7.0/getting_started/index.html) or follow these steps.
```bash
# install west
pip3 install west

# init Zephyr project with correct version
west init --mr v2.7-branch "${HOME}/zephyrproject" # this could take a while since it downloads a large repository

# setup Zephyr environment
cd ${HOME}/zephyrproject
west update # this could take a while since it will update the submodules to the correct version based on v2.7-branch
west zephyr-export
pip3 install -r ${HOME}/zephyrproject/zephyr/scripts/requirements.txt

# install Zephyr SDK
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.13.2/zephyr-sdk-0.13.2-linux-x86_64-setup.run
chmod zephyr-sdk-0.13.2-linux-x86_64-setup.run
./zephyr-sdk-0.13.2-linux-x86_64-setup.run -- -d $HOME/zephyr-sdk

# export required environment variables
export ZEPHYR_BASE="${HOME}/zephyrproject/zephyr"
export PATH="${PATH}:${HOME}/zephyr-sdk/sysroots/x86_64-pokysdk-linux/usr/bin"
```

- Install CMSIS: You need to download [CMSIS package](https://github.com/ARM-software/CMSIS_5.git) and checkout the correct version/SHA based on the information provided in [system files](../systems).
Then, you should export `CMSIS_PATH` to downloaded CMSIS path as an environment variable.


The environment is ready to build each project.

#### Build and Run
- Build the project
```bash
cd NUCLEO_L4R5ZI/'MODEL_NAME'
mkdir build
cd build
cmake .. #If you want to build in Energy mode: `cmake -DENERGY_MODE=1 ..`
make -j2
```

- Flash the project
```bash
cd NUCLEO_L4R5ZI/'MODEL_NAME'
cd build
west flash
```

Device is ready to connect with EEMBC EnergyRunner software.
  
