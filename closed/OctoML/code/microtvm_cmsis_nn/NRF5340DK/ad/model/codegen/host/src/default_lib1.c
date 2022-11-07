// tvm target: c -keys=arm_cpu,cpu -device=arm_cpu -mcpu=cortex-m33 -model=nrf5340dk
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_cast_subtract_cast_multiply(int8_t* p0, float* T_multiply, uint8_t* global_const_workspace_24_var, uint8_t* global_workspace_25_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 160; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
      int32_t cse_var_1 = ((ax1_outer * 4) + ax1_inner);
      T_multiply[cse_var_1] = (((float)(((int32_t)p0[cse_var_1]) - 89)) * 3.760228e-01f);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_divide_round_add_clip_cast_reshape(float* p0, int8_t* T_reshape, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 40; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 16; ++ax1_inner) {
      int32_t cse_var_1 = ((ax1_outer * 16) + ax1_inner);
      float __1 = roundf((p0[cse_var_1] * 2.470071e+00f)) + 8.100000e+01f;
      float __2 = (__1) < (1.270000e+02f) ? (__1) : (1.270000e+02f);
      T_reshape[cse_var_1] = ((int8_t)((__2) > (-1.280000e+02f) ? (__2) : (-1.280000e+02f)));
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* input_1_buffer_var, float* Identity_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* constant_12_let = (&(global_const_workspace_0_var[245760]));
  void* constant_2_let = (&(global_const_workspace_0_var[196608]));
  void* constant_11_let = (&(global_const_workspace_0_var[269824]));
  void* constant_8_let = (&(global_const_workspace_0_var[264704]));
  void* constant_0_let = (&(global_const_workspace_0_var[81920]));
  void* constant_10_let = (&(global_const_workspace_0_var[265728]));
  void* constant_9_let = (&(global_const_workspace_0_var[270848]));
  void* constant_5_let = (&(global_const_workspace_0_var[267264]));
  void* constant_4_let = (&(global_const_workspace_0_var[180224]));
  void* constant_3_let = (&(global_const_workspace_0_var[267776]));
  void* constant_7_let = (&(global_const_workspace_0_var[266752]));
  void* constant_19_let = (&(global_const_workspace_0_var[262144]));
  void* constant_6_let = (&(global_const_workspace_0_var[163840]));
  void* constant_1_let = (&(global_const_workspace_0_var[270336]));
  void* constant_13_let = (&(global_const_workspace_0_var[269312]));
  void* constant_14_let = (&(global_const_workspace_0_var[229376]));
  void* constant_15_let = (&(global_const_workspace_0_var[268800]));
  void* constant_16_let = (&(global_const_workspace_0_var[212992]));
  void* constant_17_let = (&(global_const_workspace_0_var[268288]));
  void* constant_18_let = (&(global_const_workspace_0_var[0]));
  void* sid_19_let = (&(global_workspace_1_var[1024]));
  void* sid_13_let = (&(global_workspace_1_var[0]));
  void* sid_25_let = (&(global_workspace_1_var[768]));
  void* sid_4_let = (&(global_workspace_1_var[640]));
  void* sid_16_let = (&(global_workspace_1_var[1152]));
  void* sid_1_let = (&(global_workspace_1_var[0]));
  void* sid_31_let = (&(global_workspace_1_var[0]));
  void* sid_7_let = (&(global_workspace_1_var[0]));
  void* sid_10_let = (&(global_workspace_1_var[128]));
  void* sid_22_let = (&(global_workspace_1_var[896]));
  void* sid_28_let = (&(global_workspace_1_var[640]));
  if (tvmgen_default_fused_divide_round_add_clip_cast_reshape(input_1_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_0(sid_1_let, constant_0_let, constant_1_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_1(sid_4_let, constant_2_let, constant_3_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_2(sid_7_let, constant_4_let, constant_5_let, sid_10_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_3(sid_10_let, constant_6_let, constant_7_let, sid_13_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_4(sid_13_let, constant_8_let, constant_9_let, sid_16_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_5(sid_16_let, constant_10_let, constant_11_let, sid_19_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_6(sid_19_let, constant_12_let, constant_13_let, sid_22_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_7(sid_22_let, constant_14_let, constant_15_let, sid_25_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_8(sid_25_let, constant_16_let, constant_17_let, sid_28_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_9(sid_28_let, constant_18_let, constant_19_let, sid_31_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_cast_subtract_cast_multiply(sid_31_let, Identity_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

