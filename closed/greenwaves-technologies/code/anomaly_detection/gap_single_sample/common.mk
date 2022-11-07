NNTOOL=nntool
#MODEL_SQ8=1
# MODEL_POW2=1
# MODEL_FP16=1
# MODEL_NE16=1


MODEL_SUFFIX?=
MODEL_PREFIX?=ad01_int8
MODEL_PYTHON=python3
MODEL_BUILD=BUILD_MODEL$(MODEL_SUFFIX)

#TRAINED_MODEL = pretrainedResnet_quant.tflite

MODEL_EXPRESSIONS = $(MODEL_BUILD)/Expression_Kernels.c

#NNTOOL_EXTRA_FLAGS += 
#MODEL_QUANTIZED=1

# Options for the memory settings: will require
# set l3_flash_device $(MODEL_L3_FLASH)
# set l3_ram_device $(MODEL_L3_RAM)
# in the nntool_script
# FLASH and RAM type
FLASH_TYPE = HYPER
RAM_TYPE   = HYPER

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
    FREQ_CL?=370
    FREQ_FC?=370
    FREQ_PE?=370
else
    ifeq '$(TARGET_CHIP)' 'GAP8_V3'
    FREQ_CL?=175
    else
    FREQ_CL?=50
    endif
    FREQ_FC?=250
    FREQ_PE?=250
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

#NNTOOL_SCRIPT = nntool_script
# define STD_FLOAT if float16 in use

# load math library if float expressions in use

$(info GEN ... $(CNN_GEN))
