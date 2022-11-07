# Tiny MLPerf™ v1.0 `hls4ml` Xilinx PYNQ-Z2 Open Submission

<a href="https://fastmachinelearning.org/"><img src="https://fastmachinelearning.org/hls4ml/_images/hls4ml_logo.png" alt="hls4ml logo" width="300"/></a>

By the the `hls4ml` team.

* Contacts:
  * Ben Hawks, email: <bhawks@fnal.gov>, GitHub: [@ben-hawks](https://github.com/ben-hawks)
  * Nhan Tran, email: <ntran@fnal.gov>, GitHub: [@nhanvtran](https://github.com/nhanvtran)
  * Javier Duarte, email: <jduarte@ucsd.edu>, GitHub: [@jmduarte](https://github.com/jmduarte)
  * Giuseppe DiGuglielmo, email: <gdg@fnal.gov>, GitHub: [@GiuseppeDiGuglielmo](https://github.com/GiuseppeDiGuglielmo)
* Team members:
  * Nicolò Ghielmetti, CERN
  * Jules Muhizi, Fermilab/Harvard
  * Ryan Kastner, Jason Liang, Andy Meza, Tai Nguyen, Rushil Roy, Olivia Weng, UC San Diego
  * Hendrik Borras, Ruprecht-Karls-Universität Heidelberg
  * Scott Hauck, Shih-Chieh Hsu, Aidan Yokuda, University of Washington

This is a minor revision to the submission from last round, MLPerf Tiny v0.7

To view a larger set of benchmarks on hls4ml, please refer to the [official MLPerf Tiny v0.7 Results repository.](https://github.com/mlcommons/tiny_results_v0.7/tree/main/open/hls4ml-finn)

## Hardware
* The board is a TUL PYNQ-Z2 based on Xilinx Zynq SoC (See https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html for more information).

<img src="https://user-images.githubusercontent.com/4932543/120665525-b47d6580-c440-11eb-9e74-fb3d86673683.jpg" alt="PYNQ-Z2" width="400"/>

## Code structure
The code/results are structured as follows:
```
hls4ml
├── code
│   └── ic
│       └── RN08
│           ├── img
│           ├── inference
│           │   └── pynq-z2
│           │       └── vivado_project
│           │           ├── hdf
│           │           └── sdk
│           │               └── common
│           │                   └── harness
│           │                       └── api
│           └── training
│               └── trained_model
├── results
│   └── pynq-z2
│       └── ic
│           └── RN08
│               ├── accuracy
│               ├── performance
│               └── power
└── systems
```
* For both the different tasks/models, there are `training` and `inference` subdirectories.
* Under `training`, there are scripts/directions to train the models as well as a pretrained model.
* Under `inference`, the Xilinx HLS, Vivado, and SDK projects will be automatically created by following the corresponding READMEs.
