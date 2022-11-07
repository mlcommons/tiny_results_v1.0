# Plumerai inference engine on microcontrollers

[Plumerai](https://plumerai.com/) is a company making deep learning tiny. Although specialized in Binarized Neural Networks (BNNs), Plumerai also delivers [state-of-the-art software](https://blog.plumerai.com/2021/10/cortex-m-inference-software/) to run 8-bit integer quantized neural networks on micro-controllers. The results of this submission are based on that software: the Plumerai inference engine. The Plumerai inference engine is commercially available to be used in the cloud or on developer's machines for Linux, Windows, and macOS. It is optimized for ARM Cortex-M, ARC EM4, and RISC-V micro-controllers.

The Plumerai inference engine does no pruning, quantization, or binarization. Model accuracy stays the same, but compared to other inference engines inference speed goes up, memory usage goes down, and code size is reduced.


## The systems under test

We ran the Plumerai inference engine on the following microcontrollers:
* STMicroelectronics STM32L4R5 (`NUCLEO_L4R5ZI`) with Cortex-M4
* Infineon Cypress CY8CPROTO-062-4343W (`cy8cproto_062_4343w`) with Cortex-M4
* STMicroelectronics STM32F746 (`DISCO_F746NG`) with Cortex-M7
* STMicroelectronics STM32U585 (`B_U585I_IOT02A`) with Cortex-M33

More details about the devices can be found under `systems/<device_name>.json`.

Note that the same code runs on many other devices. The example code submitted here is based on [MBED](https://os.mbed.com/), meaning only microcontrollers with MBED support will work. However, Plumerai's inference engine does not have this limitation and can work on other microcontrollers as well.


## The code

See [code/README.md](code/README.md) for details.


## The results

The logs for accuracy and performance can be found in the `results` subdirectory grouped by device and benchmark. They are the (unmodified) outputs of the official [EEMBC runner](https://github.com/eembc/energyrunner/) for accuracy and performance measurements. In summary the results are:

```
    |         latency in ms         |              |
    |  L4R5 |  CY8C |  F746 |  U585 |     accuracy |
----x-------x-------x-------x-------x--------------x
VWW | 208.6 | 192.5 |  57.0 | 107.0 | Top-1: 84.9% |
IC  | 173.2 | 193.1 |  64.8 | 107.1 | Top-1: 88.0% |
KWS |  71.7 |  61.4 |  19.1 |  35.4 | Top-1: 90.2% |
AD  |   5.6 |   6.7 |   2.3 |   4.9 |    AUC: 0.86 |
```
