# Qualcomm MLPerf™ Tiny benchmark v1.0

## Preparation
DUT: Qualcomm Next Generation Snapdragon Mobile Platform HDK<br>
Hexagon SDK: Hexagon SDK 5.0.0.0 for Linux<br>
<br>
>### Flash meta build 
```bash
adb reboot fastboot
python ${META_PATH}/common/build/fastboot_complete.py
```
>### Set up environment in Linux
To set up environment in Linux, you must first to switch to a bash shell. To switch from any unknown shell to a bash shell in Linux, enter bash in the terminal. This step is required because the setup script works in the bash environment.<br>
Make sure you can find conda in your PATH and create a new conda environment (using the eai.yml):
```bash
conda env create -f code/eai.yml
conda activate eai
# Install the C++ Multilib
sudo apt-get install g++-multilib
```
Hexagon SDK is necessary for eAI build environement. Please install the Hexagon SDK locally and change the path of Hexagon SDK in build script to the local path.<br>
The path to build.sh: ${EAI_SDK}/build.sh. You need to modify line 125 to: <br>
```bash
line 118:if [ "$hexagon_sdk_series" == "3" ]; then
line 119:    HEXAGON_SDK_LOC="/prj/qct/yyzsw_vol8/tools/hexagon_sdk/3.5.1"
line 120:elif [ "$hexagon_sdk_series" == "4.3" ]; then
line 121:    HEXAGON_SDK_LOC="/prj/qct/yyzsw_vol8/tools/hexagon_sdk/4.3.0.0"
line 122:elif [ "$hexagon_sdk_series" == "4" ]; then
line 123:    HEXAGON_SDK_LOC="/prj/qct/yyzsw_vol8/tools/hexagon_sdk/4.2.0.2"
line 124:else
line 125:    HEXAGON_SDK_LOC=${HEXAGON_SDK}
line 126:fi

## Model conversion
Please use the tool “eai_builder” in eAI SDK to convert the models. <br>
>### Conversion Commands:
```bash
${EAI_SDK}/eai_runtime/tools/model_builder/eai_builder --tflite ad01_int8.tflite --enable_enpu_ver v3 --output ad_model.eai --enable_legacy_quant 0 --enable_channel_align 1eai  --enable_legacy_quant 0 --enable_channel_align 1
```
After the model conversion is successful, you will get the Qualcomm AI models under the current working director，please do not rename them.<br>
```bash
tree $PWD
$PWD
├── ad_model.eai
```

## Build libs
>### Copy source code to eAI SDK
```bash
cp -r code/anomaly_detection ${EAI_SDK}/eai_runtime/
```
>### Copy CMakeLists to eAI SDK
```bash
cp code/CMakeLists.txt ${EAI_SDK}/
```
>### build binaries
```bash
cd ${EAI_SDK}
./build.sh --build_adsp --clean_build --build_enpu_ver 3
```
If the build is successful you can find the generated files in the ${EAI_SDK}/build-fixed32/eai_runtime.
## Push resources
Connect devices with USB Type-C
```bash
adb root
adb wait-for-device remount
# push test resources to DUT
adb push ad_model.eai /data/local/tmp/
adb push ${EAI_SDK}/build-fixed32/eai_runtime/anomaly_detection/libeai_tiny_ad.so /vendor/lib/rfsa/adsp/
adb push ${EAI_SDK}/binaries/enpu_launcher/enpu_launcher /vendor/bin/
adb push ${EAI_SDK}/binaries/enpu_launcher/libenpu_launcher_skel.so /vendor/lib/rfsa/adsp/
```
## Launch test app
Connect DUT with UART port with UART communication tool(e.g. using [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)).UART config:
> baudRate: 115200<br>
> dataBits: 8<br>
> parity: None<br>
> stopBits: 1<br>

After successfully connecting with the DUT, please send the following commands through UART
```bash
su
stty raw
cd /data/local/tmp
export ADSP_LIBRARY_PATH=/vendor/lib/rfsa/adsp
#Launch the test app for specified model. e.g. anomaly_detection: 
enpu_launcher libeai_tiny_ad.so 
```
## Start the EEMBC test
Close the UART connection and launch [EEMBC runner](https://github.com/eembc/energyrunner/) app. Please follow the [EEMBC runner](https://github.com/eembc/energyrunner/) instructions for testing process.

## Close the thread of test app 
```bash
# send the following command to stop the test app
%%
# ifstart testing other models, launch the test app for specified model.
# CMD: enpu_launcher libeai_tiny_${MODEL}.so 
# ${MODEL} must be one of {vww, ic, kws, ad}
```
## The results
The logs of performance and accuracy can be found in the results folder. The are the outputs if EEMBC runner. 
|model|accuracy|latency in ms|
|:-----:|:----:|:----:|
| AD | AUC: 0.86 | 0.098 |

