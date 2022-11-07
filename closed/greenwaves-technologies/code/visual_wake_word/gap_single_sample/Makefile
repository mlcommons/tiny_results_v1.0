  # Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif


#MODEL_SQ8=1
# MODEL_POW2=1
# MODEL_FP16=1
# MODEL_NE16=1

ifdef INT8
  NNTOOL_SCRIPT=nntool_script/nntool_script_int8
  TRAINED_MODEL = ../model/vww_96_int8.tflite
  APP_CFLAGS += -DINT8 -DIN_OUT_INT8
  IMAGE = $(CURDIR)/../datasets/perf_samples_chw/COCO_val2014_000000000974_chw.bin
  MODEL_SQ8=1


else
  ifdef INT8_QUANT
    NNTOOL_SCRIPT=nntool_script/nntool_script_int8_quant
    TRAINED_MODEL = ../model/vww_96_int8.tflite
    MODEL_QUANTIZED=1
    APP_CFLAGS += -DINT8_QUANT -DIN_OUT_INT8
    IMAGE = $(CURDIR)/../datasets/perf_samples_chw/COCO_val2014_000000000974_chw.bin
    MODEL_SQ8=1

else
  ifdef FP16
    NNTOOL_SCRIPT=nntool_script/nntool_script_fp16
    TRAINED_MODEL = ../model/vww_96_float.tflite
    APP_CFLAGS += -DFP16 -DIN_OUT_FP16
    IMAGE = $(CURDIR)/../datasets/perf_samples_chw/COCO_val2014_000000000974_chw.bin
    MODEL_FP16=1

else
  ifdef NE16
    NNTOOL_SCRIPT=nntool_script/nntool_script_ne16
    TRAINED_MODEL = ../model/vww_96_int8.tflite
    APP_CFLAGS += -DNE16 -DIN_OUT_NE16
    IMAGE = $(CURDIR)/../datasets/perf_samples/COCO_val2014_000000000974.bin
    MODEL_NE16=1
    MODEL_SQ8=1



else
  ifdef NE16_QUANT
    NNTOOL_SCRIPT=nntool_script/nntool_script_ne16_quant
    TRAINED_MODEL = ../model/vww_96_int8.tflite
    MODEL_QUANTIZED=1
    APP_CFLAGS += -DNE16_QUANT -DIN_OUT_NE16
    IMAGE = $(CURDIR)/../datasets/perf_samples/COCO_val2014_000000000974.bin
    MODEL_NE16=1
    MODEL_SQ8=1



  else
    $(error quantization type not defined)
endif
endif
endif
endif
endif


include common.mk
include $(RULES_DIR)/at_common_decl.mk

io?=host

$(info Building NNTOOL model)
NNTOOL_EXTRA_FLAGS ?= 

include common/model_decl.mk
#IMAGE = $(CURDIR)/../perf_samples/taxi_s_001241.bin

  

# pulpChip = GAP
# PULP_APP = $(MODEL_PREFIX)

APP = $(MODEL_PREFIX)
APP_SRCS += $(MODEL_PREFIX).c $(MODEL_GEN_C) $(MODEL_EXPRESSIONS) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -g -O2 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL) -DFREQ_PE=$(FREQ_PE)

READFS_FILES=$(abspath $(MODEL_TENSORS))

# all depends on the model
all:: model

clean:: clean_model

include common/model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

