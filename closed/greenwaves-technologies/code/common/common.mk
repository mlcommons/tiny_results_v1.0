NNTOOL=nntool

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

ifeq ($(INT8), 1)
  $(info INT8 (Load quantization ranges from NNTool inference done previously through accuracy/calibrate_nntool.sh) Quantization Type selected)
  NNTOOL_SCRIPT=nntool_script/nntool_script_int8
  APP_CFLAGS += -DINT8 -DIN_OUT_INT8

  MODEL_SQ8=1
  MODEL_SUFFIX = _SQ8
else
ifeq ($(INT8_QUANT), 1)
  $(info INT8_QUANT (Load quantization ranges from TFLite) Quantization Type selected)
  NNTOOL_SCRIPT=nntool_script/nntool_script_int8_quant
  APP_CFLAGS += -DINT8_QUANT -DIN_OUT_INT8

  MODEL_QUANTIZED=1
  MODEL_SQ8=1
  MODEL_SUFFIX = _SQ8_QUANT
else
ifeq ($(FP16), 1)
  $(info FP16 (No calibration required) Quantization Type selected)
  NNTOOL_SCRIPT=nntool_script/nntool_script_fp16
  APP_CFLAGS += -DFP16 -DIN_OUT_FP16

  MODEL_FP16=1
  MODEL_SUFFIX = _FP16
else
ifeq ($(NE16), 1)
  $(info NE16 (Load quantization ranges from NNTool inference done previously through accuracy/calibrate_nntool.sh) Quantization Type selected)
  NNTOOL_SCRIPT=nntool_script/nntool_script_ne16
  APP_CFLAGS += -DNE16 -DIN_OUT_NE16

  MODEL_NE16=1
  MODEL_SQ8=1
  MODEL_SUFFIX = _NE16
else
ifeq ($(NE16_QUANT), 1)
  $(info NE16_QUANT (Load quantization ranges from TFLite) Quantization Type selected)
  NNTOOL_SCRIPT=nntool_script/nntool_script_ne16_quant
  APP_CFLAGS += -DNE16_QUANT -DIN_OUT_NE16

  MODEL_QUANTIZED=1
  MODEL_NE16=1
  MODEL_SQ8=1
  MODEL_SUFFIX = _NE16_QUANT
else
  $(error quantization type not defined, select one of: INT8, INT8_QUANT, FP16, NE16, NE16_QUANT)
endif
endif
endif
endif
endif

PERFORMANCE_MODE         ?= 1
ENERGY_MODE              ?= 0
TEST_WITHOUT_EEMBC_SETUP ?= 0
NORMAL_PRINTF            ?= 0
TEST_IMAGE               ?= 0
ENERGY                   ?= 0

ifeq ($(ENERGY_MODE),1)
    FREQ_PE=5
    FREQ_FC=240
    FREQ_CL=240
    VOLTAGE=650
    ENERGY=1
else
ifeq ($(PERFORMANCE_MODE),1)
    FREQ_PE=5
    FREQ_FC=370
    FREQ_CL=370
    VOLTAGE=800
else
ifeq ($(PERFORMANCE_MODE_065),1)
    FREQ_PE=5
    FREQ_FC=240
    FREQ_CL=240
    VOLTAGE=650
endif
endif
endif

ifeq ($(ENERGY),1)
    APP_CFLAGS += -DEE_CFG_ENERGY_MODE=1
endif

ifeq ($(TEST_WITHOUT_EEMBC_SETUP), 1)
    APP_CFLAGS += -DTEST_WITHOUT_EEMBC_SETUP
    ifeq ($(NORMAL_PRINTF), 1)
        APP_CFLAGS += -DNORMAL_PRINTF
    endif
    ifeq ($(PERF),1)
        APP_CFLAGS += -DPERF
    endif
    ifeq ($(TEST_IMAGE), 1)
        APP_CFLAGS += -DTEST_IMAGE
    endif
    ifeq ($(PRINT_OUT),1)
        APP_CFLAGS += -DPRINT_OUT
    endif
endif

MODEL_SUFFIX?=
MODEL_PYTHON=python3
MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)


MODEL_EXPRESSIONS = $(MODEL_BUILD)/Expression_Kernels.c

NNTOOL_EXTRA_FLAGS += 
#MODEL_QUANTIZED=1

# Options for the memory settings: will require
# set l3_flash_device $(MODEL_L3_FLASH)
# set l3_ram_device $(MODEL_L3_RAM)
# in the nntool_script
# FLASH and RAM type
FLASH_TYPE = MRAM
RAM_TYPE   = DEFAULT

ifeq '$(FLASH_TYPE)' 'HYPER'
    MODEL_L3_FLASH=AT_MEM_L3_HFLASH
else ifeq '$(FLASH_TYPE)' 'MRAM'
    MODEL_L3_FLASH=AT_MEM_L3_MRAMFLASH
    READFS_FLASH = target/chip/soc/mram
else ifeq '$(FLASH_TYPE)' 'QSPI'
    MODEL_L3_FLASH=AT_MEM_L3_QSPIFLASH
    READFS_FLASH = target/board/devices/spiflash
else ifeq '$(FLASH_TYPE)' 'OSPI'
    MODEL_L3_FLASH=AT_MEM_L3_OSPIFLASH
else ifeq '$(FLASH_TYPE)' 'DEFAULT'
    MODEL_L3_FLASH=AT_MEM_L3_DEFAULTFLASH
endif

ifeq '$(RAM_TYPE)' 'HYPER'
    MODEL_L3_RAM=AT_MEM_L3_HRAM
else ifeq '$(RAM_TYPE)' 'QSPI'
    MODEL_L3_RAM=AT_MEM_L3_QSPIRAM
else ifeq '$(RAM_TYPE)' 'OSPI'
    MODEL_L3_RAM=AT_MEM_L3_OSPIRAM
else ifeq '$(RAM_TYPE)' 'DEFAULT'
    MODEL_L3_RAM=AT_MEM_L3_DEFAULTRAM
endif

ifeq '$(TARGET_CHIP_FAMILY)' 'GAP9'
    FREQ_CL?=240
    FREQ_FC?=240
    FREQ_PE?=240
    VOLTAGE?=650
else
    ifeq '$(TARGET_CHIP)' 'GAP8_V3'
    FREQ_CL?=175
    else
    FREQ_CL?=50
    endif
    FREQ_FC?=250
    FREQ_PE?=250
endif

ifdef VOLTAGE
    APP_CFLAGS += -DVOLTAGE=$(VOLTAGE)
endif

# Memory sizes for cluster L1, SoC L2 and Flash
ifeq '$(TARGET_CHIP_FAMILY)' 'GAP9'
	TARGET_L1_SIZE = 128000
	TARGET_L2_SIZE = 1400000
	TARGET_L3_SIZE = 8000000
else
	TARGET_L1_SIZE = 64000
	TARGET_L2_SIZE = 400000
	TARGET_L3_SIZE = 8000000
endif

# Cluster stack size for master core and other cores
CLUSTER_STACK_SIZE=4096
CLUSTER_SLAVE_STACK_SIZE=1024

nntool_test_predict:
	python ../accuracy/test_nntool_single_sample.py --input_path=$(IMAGE) --model_path=$(MODEL_STATE)

$(info GEN ... $(CNN_GEN))
