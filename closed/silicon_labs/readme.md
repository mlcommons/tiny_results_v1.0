# Silicon Labs MLPerf Tiny benchmark v1 - Closed Division

This document provides an overview of Silicon Labs submission to the MLPerf Tiny benchmark v1. The benchmarks were recorded on the EFR32xG24 Dev Kit (xG24-DK2601B), using the [TensorFlow Lite for Microcontrollers integration in the Silicon Labs Gecko SDK](https://docs.silabs.com/gecko-platform/4.1/machine-learning/tensorflow/overview). The submission contains both energy and performance results.

## Generating and building the Firmware

The ```code/``` folder contains the implemented benchmark firmware API along with a [Silicon Labs Configurator Project](https://docs.silabs.com/simplicity-studio-5-users-guide/latest/ss-5-users-guide-developing-with-project-configurator/) (.slcp file) file for each benchmark. The project files contain information of the dependencies of a project, and can be used to generate the benchmark firmware together with the Silicon Labs Gecko SDK.

The firmware binaries can be generated by running the ```build_firmware.sh``` script, but first you need to install the requirements as described below.

### Silicon Labs GSDK

Clone the Silicon Labs Gecko SDK from GitHub:

```
git clone https://github.com/SiliconLabs/gecko_sdk.git
```

Ensure you are using the correct version:

```
cd gecko_sdk
git checkout v4.1.1
```

### GNU Arm embedded toolchain

Download and install v10.2 of [GNU Arm Embedded Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads) for your operating system. You also need to have Make installed and on your path. 

### Silicon Labs Configurator (SLC) Command Line Interface (CLI)

In order to generate the projects you need to install the Silicon Labs Configurator Command Line Interface (SLC-CLI). Install SLC-CLI as described in [this user guide](https://www.silabs.com/documents/public/user-guides/ug520-software-project-generation-configuration-with-slc-cli.pdf). Add the downloaded SLC-CLI to your path and take care to follow the steps in chapter 3.1 to ensure that you configure an SDK and GCC location:

```
slc configuration --sdk="\path\to\gecko_sdk\gecko_sdk.slcs"
slc configuration --gcc-toolchain "\path\to\your\GNU\ARM\embedded\toolchain"
```

## Generation

Once SLC-CLI is installed and configured with and SDK and GCC location, the ```build_firmware.sh``` script can be run to generate and compile the benchmark firmware for all the models. Firmware binaries will be built for both energy mode and performance mode. The generated binaries can be found in the ```build/``` folder, under ```performance``` and ```energy```. The generated firmwares have been compiled with ```EE_CFG_ENERGY_MODE``` set to ```0``` and ```1``` respectively.

## Flashing the firmware

To flash the firmware binary onto device you need to install Simplicity Commander. Simplicity Commander can be installed from [this page](https://www.silabs.com/developers/mcu-programming-options) for your operating system.

To flash the energy mode firmware for the keyword spotting model, use the following command

```<path_to_commander> flash build/energy/mlperf_tiny_kws.s37```

# Hardware Setup
## Performance Mode measurement
After generating the firmware binaries as described above, flash the device with one of the performance mode benchmark firmwares in ```build/performance/```.

The xG24-DK2601B provides a virtual COM port (VCOM) interface that allows for easy serial connection between the PC and the kit. To connect to the EEMBC EnergyRunner in performance mode, simply connect the kit to the host PC with a USB cable.


## Energy Mode measurement
After generating the firmware binaries as described above, flash the device with one of the energy mode benchmark firmwares in ```build/energy/```.

See [Energy Mode Connection](systems/energy_hookup.md) for information on how to connect the EFR32xG24 Dev Kit to the EEMBC EnergyRunner in Energy Mode.
