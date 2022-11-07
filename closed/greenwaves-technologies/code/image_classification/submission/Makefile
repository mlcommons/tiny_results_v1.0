# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifdef FP16
  TRAINED_MODEL = ../model/pretrainedResnet.tflite
else
  TRAINED_MODEL = ../model/pretrainedResnet_quant.tflite
endif
#CALIBRATION_PICKLE = ../model/nntool_calibration_fp32tflite.pickle
# Best configuration
W_BITS = 7
NE16_QUANT = 1


# Decomment one line at time to select mode, or add it in the shell when calling makefile:
# i.e.: make clean all run PERF_ACCURACY_0800=1
#PERF_ACCURACY_0800=1
#PERF_ACCURACY_0650=1
#ENERGY_0800=1
#ENERGY_0650=1

## For Performance at 0.8 and accuracy
ifeq ($(PERF_ACCURACY_0800), 1)
  PERFORMANCE_MODE = 1
  ENERGY_MODE = 0
else

## For Performance at 0.65 and accuracy
ifeq ($(PERF_ACCURACY_0650), 1)
  PERFORMANCE_MODE_065 = 1
  PERFORMANCE_MODE = 0
  ENERGY_MODE = 0
else

## For Energy at 0.65
ifeq ($(ENERGY_0650), 1)
  PERFORMANCE_MODE = 0
  ENERGY_MODE = 1
  CONFIG_BOOT_DEVICE=mram 
else

## For Energy at 0.8
ifeq ($(ENERGY_0800), 1)
  ENERGY=1
  PERFORMANCE_MODE = 1
  ENERGY_MODE = 0
  CONFIG_BOOT_DEVICE=mram 
else 
  $(error Mode not selected! Please select between PERF_ACCURACY_0800, PERF_ACCURACY_0650, ENERGY_0800 and ENERGY_0650)
endif
endif
endif
endif

MODEL_PREFIX?=pretrainedResnet
include ../../common/common.mk
# Cluster stack size for master core and other cores
CLUSTER_STACK_SIZE=1024
CLUSTER_SLAVE_STACK_SIZE=512
include $(RULES_DIR)/at_common_decl.mk

io?=host

include ../../common/model_decl.mk

IMAGE ?= $(CURDIR)/../datasets/perf_samples/taxi_s_001241.bin

AT_INPUT_WIDTH=32
AT_INPUT_HEIGHT=32
AT_INPUT_COLORS=3


MAIN=main
APP_SRCS += ../../$(MAIN).c 
APP_SRCS += ../../submitter_implemented.c 
APP_SRCS += ../../api/internally_implemented.c
APP_SRCS += $(MODEL_GEN_C) $(MODEL_EXPRESSIONS) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -g -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL) -DFREQ_PE=$(FREQ_PE)
APP_CFLAGS += -DAT_CONSTRUCT=$(AT_CONSTRUCT) -DAT_CONSTRUCT_CL=$(AT_CONSTRUCT)Cluster -DAT_DESTRUCT=$(AT_DESTRUCT) -DAT_CNN=$(AT_CNN) -DAT_L3_ADDR=$(AT_L3_ADDR)

APP_CFLAGS += -DPROMOTED_L1

READFS_FILES=$(abspath $(MODEL_TENSORS))
    $(info  READFS_FILES ...$(READFS_FILES))

# all depends on the model
all:: model

clean:: clean_model

include ../../common/model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

