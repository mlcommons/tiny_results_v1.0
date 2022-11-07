// tvm target: cmsis-nn 
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlpack/dlpack.h>
#include <arm_nnfunctions.h>
#include <arm_nn_types.h>
#include <arm_nn_math_types.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_0(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  void* context_buffer_0_let = (&(global_workspace_3_var[16384]));
  cmsis_nn_context context= {context_buffer_0_let,108};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,32,32,3};
  cmsis_nn_dims filter_dims = {16,3,3,3};
  cmsis_nn_dims bias_dims = {1,1,1,16};
  cmsis_nn_dims output_dims = {1,32,32,16};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_1(int8_t* input_0_, int8_t* input_1_, int8_t* output_, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  arm_elementwise_add_s8(input_0_, input_1_, 128, 1623821475, -2, -4, 1073741824, 0, 20, output_, -128, 1098017566, -17, -128, 127, 16384);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_2(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* context_buffer_1_let = (&(global_workspace_5_var[65536]));
  cmsis_nn_context context= {context_buffer_1_let,576};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,32,32,16};
  cmsis_nn_dims filter_dims = {16,3,3,16};
  cmsis_nn_dims bias_dims = {1,1,1,16};
  cmsis_nn_dims output_dims = {1,32,32,16};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_3(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* context_buffer_2_let = (&(global_workspace_7_var[65536]));
  cmsis_nn_context context= {context_buffer_2_let,576};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, 4, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,32,32,16};
  cmsis_nn_dims filter_dims = {16,3,3,16};
  cmsis_nn_dims bias_dims = {1,1,1,16};
  cmsis_nn_dims output_dims = {1,32,32,16};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_5(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  void* context_buffer_3_let = (&(global_workspace_11_var[40960]));
  cmsis_nn_context context= {context_buffer_3_let,64};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -17, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,32,32,16};
  cmsis_nn_dims filter_dims = {32,1,1,16};
  cmsis_nn_dims bias_dims = {1,1,1,32};
  cmsis_nn_dims output_dims = {1,16,16,32};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_6(int8_t* input_0_, int8_t* input_1_, int8_t* output_, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  arm_elementwise_add_s8(input_0_, input_1_, 17, 1699529983, -2, -4, 1073741824, 0, 20, output_, -128, 1140768826, -17, -128, 127, 8192);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_7(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  void* context_buffer_4_let = (&(global_workspace_13_var[49152]));
  cmsis_nn_context context= {context_buffer_4_let,576};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,32,32,16};
  cmsis_nn_dims filter_dims = {32,3,3,16};
  cmsis_nn_dims bias_dims = {1,1,1,32};
  cmsis_nn_dims output_dims = {1,16,16,32};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_8(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* context_buffer_5_let = (&(global_workspace_15_var[57344]));
  cmsis_nn_context context= {context_buffer_5_let,1152};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, 4, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,16,16,32};
  cmsis_nn_dims filter_dims = {32,3,3,32};
  cmsis_nn_dims bias_dims = {1,1,1,32};
  cmsis_nn_dims output_dims = {1,16,16,32};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_10(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* context_buffer_6_let = (&(global_workspace_19_var[53248]));
  cmsis_nn_context context= {context_buffer_6_let,128};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, 38, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,16,16,32};
  cmsis_nn_dims filter_dims = {64,1,1,32};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,8,8,64};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_11(int8_t* input_0_, int8_t* input_1_, int8_t* output_, uint8_t* global_const_workspace_24_var, uint8_t* global_workspace_25_var) {
  arm_elementwise_add_s8(input_0_, input_1_, -38, 1657902019, -2, 2, 1073741824, 0, 20, output_, -128, 1835721671, -18, -128, 127, 4096);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_12(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_20_var, uint8_t* global_workspace_21_var) {
  void* context_buffer_7_let = (&(global_workspace_21_var[57344]));
  cmsis_nn_context context= {context_buffer_7_let,1152};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,16,16,32};
  cmsis_nn_dims filter_dims = {64,3,3,32};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,8,8,64};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_13(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_, uint8_t* global_const_workspace_22_var, uint8_t* global_workspace_23_var) {
  void* context_buffer_8_let = (&(global_workspace_23_var[61440]));
  cmsis_nn_context context= {context_buffer_8_let,2304};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -2, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,8,8,64};
  cmsis_nn_dims filter_dims = {64,3,3,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,8,8,64};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_15(int8_t* input_, int8_t* output_, uint8_t* global_const_workspace_26_var, uint8_t* global_workspace_27_var) {
  void* context_buffer_9_let = (&(global_workspace_27_var[57344]));
  cmsis_nn_context context= {context_buffer_9_let,256};
  cmsis_nn_tile stride = {8,8};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_pool_params pool_params = {stride, padding, activation};
  cmsis_nn_dims input_dims = {1,8,8,64};
  cmsis_nn_dims filter_dims = {1,8,8,1};
  cmsis_nn_dims output_dims = {1,1,1,64};
  arm_cmsis_nn_status status = arm_avgpool_s8(&context, &pool_params, &input_dims, input_, &filter_dims, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_16(int8_t* input_, int8_t* filter_, int32_t* bias_, int8_t* output_, uint8_t* global_const_workspace_28_var, uint8_t* global_workspace_29_var) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_fc_params fc_params = {128, 0, 24, activation};
  cmsis_nn_per_tensor_quant_params quant_params = {1552512742, -5};
  cmsis_nn_dims input_dims = {1,1,1,64};
  cmsis_nn_dims filter_dims = {64,1,1,10};
  cmsis_nn_dims bias_dims = {1,1,1,10};
  cmsis_nn_dims output_dims = {1,1,1,10};
  arm_cmsis_nn_status status = arm_fully_connected_s8(&context, &fc_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_17(int8_t* input_, int8_t* output_, uint8_t* global_const_workspace_30_var, uint8_t* global_workspace_31_var) {
  arm_softmax_s8(input_, 1, 10, 1476210432, 24, -124, output_);
  return 0;
}

