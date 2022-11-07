# GAP TinyMLbenchmark

## Repository

The repository contains:

- an API folder that should not be modified
- 4 folders for the 4 benchmarks 
- main.c file shared with the 4 models
- submitted_implemented.c file shared with the 4 models

## Setup HW

The hardware setup contains:

	- DUT - GAP9_EVK
	- Energy Monitor - X-NUCLEO-LPM01A
	- IO Manager - Arduino Uno
	- Level Shifter - BSS138

See document "Energy-Hookup"

## Setup SW

The complete software setup contains:

- Energy Runner application framework
- This repository

## Run the TinyMLperf benchamrks on GAP9

### Run on GAP9

To run the benchmark on GAP9 and reproduce submitted results of TinyML Perf V1.0:

~~~~~shell
cd <benchmark_folder>/submission
make clean all run
~~~~~

In each makefile you can use one of the following flag to enable test of accuracy, performance, energy at 0.65 Volt (240 MhZ) and 0.8 Volt (370):

For Accuracy and Performance at 0.8V:
~~~~~shell
make clean all run PERF_ACCURACY_0800=1
~~~~~


For Accuracy and Performance at 0.65V:
~~~~~shell
make clean all run PERF_ACCURACY_0650=1
~~~~~


For Energy at 0.8V:
~~~~~shell
make clean all run ENERGY_0800=1
~~~~~


For Energy at 0.65V:
~~~~~shell
make clean all run ENERGY_0650=1
~~~~~

The other compilation, quantization and run flags are set for the TinyML V1.0 submission. 
