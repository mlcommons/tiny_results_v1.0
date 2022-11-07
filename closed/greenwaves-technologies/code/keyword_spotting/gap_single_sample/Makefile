# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

ifdef INT8
  NNTOOL_SCRIPT=nntool_script/nntool_script_int8
  TRAINED_MODEL = kws_ref_model.tflite
  APP_CFLAGS += -DINT8 -DIN_OUT_INT8 

else
  ifdef INT8_QUANT
    NNTOOL_SCRIPT=nntool_script/nntool_script_int8_quant
    TRAINED_MODEL = ../model/kws_ref_model.tflite
    MODEL_QUANTIZED=1
    APP_CFLAGS += -DINT8_QUANT -DIN_OUT_INT8
else
  ifdef FP16
    NNTOOL_SCRIPT=nntool_script/nntool_script_fp16
    TRAINED_MODEL = ../model/kws_ref_model_float32.tflite
    APP_CFLAGS += -DFP16 -DIN_OUT_FP16
else
  ifdef FLOAT32
    NNTOOL_SCRIPT=nntool_script/nntool_script_float32
    TRAINED_MODEL = ../model/kws_ref_model_float32.tflite
   # MODEL_QUANTIZED=1
    APP_CFLAGS += -DFLOAT32 -DIN_OUT_FLOAT32


else
  ifdef NE16
    NNTOOL_SCRIPT=nntool_script/nntool_script_ne16
    TRAINED_MODEL = ../model/kws_ref_model.tflite
    APP_CFLAGS += -DNE16 -DIN_OUT_INT8

else
  ifdef NE16_QUANT
    NNTOOL_SCRIPT=nntool_script/nntool_script_ne16_quant
    TRAINED_MODEL = ../model/kws_ref_model.tflite
    MODEL_QUANTIZED=1
    APP_CFLAGS += -DNE16_QUANT -DIN_OUT_INT8

  else
    $(error quantization type not defined)
endif
endif
endif
endif
endif
endif


#NNTOOL_SCRIPT = nntool_script





include common.mk
include $(RULES_DIR)/at_common_decl.mk

io?=host

$(info Building NNTOOL model)
NNTOOL_EXTRA_FLAGS ?= 


include common/model_decl.mk
#IMAGE = $(CURDIR)/../kws_bin_files/tst_000000_Stop_7.bin
#IMAGE = $(CURDIR)/../datasets/perf_samples/tst_000001_Left_2.bin
IMAGE = $(CURDIR)/../datasets/perf_samples/tst_000000_Stop_7.bin

#IMAGE = $(CURDIR)/../kws_bin_files/tst_000004_On_5.bin

#IMAGE = $(CURDIR)/../kws_bin_files/tst_000004_On_5.bin
#IMAGE = $(CURDIR)/../kws_bin_files/tst_000010_Unknown_11.bin



# pulpChip = GAP
# PULP_APP = $(MODEL_PREFIX)

APP = $(MODEL_PREFIX)
APP_SRCS += $(MODEL_PREFIX).c $(MODEL_GEN_C) $(MODEL_EXPRESSIONS) $(MODEL_COMMON_SRCS) $(CNN_LIB)

APP_CFLAGS += -g -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DFREQ_FC=$(FREQ_FC) -DFREQ_CL=$(FREQ_CL) -DFREQ_PE=$(FREQ_PE)



READFS_FILES=$(abspath $(MODEL_TENSORS))
READFS_FLASH = target/chip/soc/mram


# all depends on the model
all:: model

clean:: clean_model

include common/model_rules.mk
$(info APP_SRCS... $(APP_SRCS))
$(info APP_CFLAGS... $(APP_CFLAGS))
include $(RULES_DIR)/pmsis_rules.mk

