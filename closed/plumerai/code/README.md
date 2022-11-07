# Plumerai inference engine on microcontrollers

This code is based on the reference submission example and has been slightly extended to use Plumerai's inference engine instead of the example TFLite back-end. It uses [MBED](https://os.mbed.com/) in a similar way to the example. In short, installing MBED on Linux involves the [steps detailed on the official website](https://os.mbed.com/docs/mbed-os/v6.15/build-tools/install-and-set-up.html) (i.e. `pip install mbed-cli`), followed by `cd`-ing into this folder and running `mbed add mbed-os`. Make sure there is no `.mbed` file in the root of the repository.

It is tested on these devices (listing the MBED names below):
* `NUCLEO_L4R5ZI`
* `cy8cproto_062_4343w`
* `DISCO_F746NG`
* `B_U585I_IOT02A`

To compile and link the code you need to store the proprietary Plumerai inference engine for one of the four benchmarks as `libPlumerai.a` in this folder. If you have that file, then you can compile, link and program the example code as follows using MBED (e.g. for VWW on STM32L4R5 on Linux):
```
mbed new . --scm none
mbed compile -m NUCLEO_L4R5ZI -t GCC_ARM -D BENCHMARK_VWW
cp ./BUILD/NUCLEO_L4R5ZI/GCC_ARM/code.bin /media/$USER/NODE_L4R5ZI
```

You can then open a TTY connection (e.g. using [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)) and after initialization type `help%` to get information on available commands. To run inference, type `infer%`. Or you can use the official [EEMBC runner](https://github.com/eembc/energyrunner/) (get it from [the eembc website](https://www.eembc.org/download/) after logging in) to get official latency numbers and/or automatically run an entire validation set to get accuracy results.

Note: For the `cy8cproto_062_4343w`, first remove `WHD` from the `components_add` list under the device name in `mbed-os/targets/targets.json`, then add `#include "blockdevice/BlockDevice.h"` to the top of `mbed-os/targets/TARGET_Cypress/TARGET_PSOC6/reserved-region-bd/CyReservedRegionBlockDevice.h` to solve compilation bugs in MBED-OS.
