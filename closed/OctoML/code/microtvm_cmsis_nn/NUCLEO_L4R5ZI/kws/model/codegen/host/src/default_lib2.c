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
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_0(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {2,2};
  cmsis_nn_tile padding = {1,4};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_dw_conv_params conv_params = {-83, -128, 64, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,49,10,1};
  cmsis_nn_dims filter_dims = {1,10,4,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_1(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  void* context_buffer_0 = TVMBackendAllocWorkspace(1, 0, (uint64_t)1152, 0, 8);
  if (context_buffer_0 == NULL) {
    return -1;
  }
  cmsis_nn_context context= {context_buffer_0,1152};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_dw_conv_params conv_params = {128, -128, 1, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {1,3,3,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, context_buffer_0) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_2(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {64,1,1,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_3(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  void* context_buffer_1 = TVMBackendAllocWorkspace(1, 0, (uint64_t)1152, 0, 8);
  if (context_buffer_1 == NULL) {
    return -1;
  }
  cmsis_nn_context context= {context_buffer_1,1152};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_dw_conv_params conv_params = {128, -128, 1, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {1,3,3,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, context_buffer_1) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_4(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {64,1,1,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_5(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  void* context_buffer_2 = TVMBackendAllocWorkspace(1, 0, (uint64_t)1152, 0, 8);
  if (context_buffer_2 == NULL) {
    return -1;
  }
  cmsis_nn_context context= {context_buffer_2,1152};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_dw_conv_params conv_params = {128, -128, 1, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {1,3,3,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, context_buffer_2) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_6(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {64,1,1,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_7(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  void* context_buffer_3 = TVMBackendAllocWorkspace(1, 0, (uint64_t)1152, 0, 8);
  if (context_buffer_3 == NULL) {
    return -1;
  }
  cmsis_nn_context context= {context_buffer_3,1152};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {1,1};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_dw_conv_params conv_params = {128, -128, 1, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {1,3,3,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_depthwise_conv_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, context_buffer_3) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_8(int8_t* input_, int8_t* filter_, int32_t* multiplier_, int32_t* bias_, int32_t* shift_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_tile stride = {1,1};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_tile dilation = {1,1};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_conv_params conv_params = {128, -128, stride, padding, dilation, activation};
  cmsis_nn_per_channel_quant_params quant_params = {multiplier_, shift_};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {64,1,1,64};
  cmsis_nn_dims bias_dims = {1,1,1,64};
  cmsis_nn_dims output_dims = {1,25,5,64};
  arm_cmsis_nn_status status = arm_convolve_wrapper_s8(&context, &conv_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_9(int8_t* input_, int8_t* output_) {
  void* context_buffer_4 = TVMBackendAllocWorkspace(1, 0, (uint64_t)256, 0, 8);
  if (context_buffer_4 == NULL) {
    return -1;
  }
  cmsis_nn_context context= {context_buffer_4,256};
  cmsis_nn_tile stride = {5,25};
  cmsis_nn_tile padding = {0,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_pool_params pool_params = {stride, padding, activation};
  cmsis_nn_dims input_dims = {1,25,5,64};
  cmsis_nn_dims filter_dims = {1,25,5,1};
  cmsis_nn_dims output_dims = {1,1,1,64};
  arm_cmsis_nn_status status = arm_avgpool_s8(&context, &pool_params, &input_dims, input_, &filter_dims, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, 0, context_buffer_4) != 0) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_10(int8_t* input_, int8_t* filter_, int32_t* bias_, int8_t* output_) {
  cmsis_nn_context context= {NULL,0};
  cmsis_nn_activation activation = {-128,127};
  cmsis_nn_fc_params fc_params = {128, 0, 14, activation};
  cmsis_nn_per_tensor_quant_params quant_params = {1278221461, -7};
  cmsis_nn_dims input_dims = {1,1,1,64};
  cmsis_nn_dims filter_dims = {64,1,1,12};
  cmsis_nn_dims bias_dims = {1,1,1,12};
  cmsis_nn_dims output_dims = {1,1,1,12};
  arm_cmsis_nn_status status = arm_fully_connected_s8(&context, &fc_params, &quant_params, &input_dims, input_, &filter_dims, filter_, &bias_dims, bias_, &output_dims, output_);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    return -1;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_cmsis_nn_main_11(int8_t* input_, int8_t* output_) {
  arm_softmax_s8(input_, 1, 12, 1242899200, 24, -124, output_);
  return 0;
}

