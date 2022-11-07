// tvm target: c -keys=arm_cpu,cpu -device=arm_cpu -mcpu=cortex-m4 -model=stm32l4r5zi
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* input_1_int8_buffer_var, int8_t* Identity_int8_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* constant_27_let = (&(global_const_workspace_0_var[82096]));
  void* constant_16_let = (&(global_const_workspace_0_var[83632]));
  void* constant_23_let = (&(global_const_workspace_0_var[82480]));
  void* constant_47_let = (&(global_const_workspace_0_var[78640]));
  void* constant_20_let = (&(global_const_workspace_0_var[82864]));
  void* constant_26_let = (&(global_const_workspace_0_var[82224]));
  void* constant_55_let = (&(global_const_workspace_0_var[84080]));
  void* constant_19_let = (&(global_const_workspace_0_var[82992]));
  void* constant_49_let = (&(global_const_workspace_0_var[78384]));
  void* constant_17_let = (&(global_const_workspace_0_var[83568]));
  void* constant_54_let = (&(global_const_workspace_0_var[75776]));
  void* constant_28_let = (&(global_const_workspace_0_var[81968]));
  void* constant_21_let = (&(global_const_workspace_0_var[82736]));
  void* constant_29_let = (&(global_const_workspace_0_var[81840]));
  void* constant_31_let = (&(global_const_workspace_0_var[81712]));
  void* constant_14_let = (&(global_const_workspace_0_var[83760]));
  void* constant_9_let = (&(global_const_workspace_0_var[83120]));
  void* constant_7_let = (&(global_const_workspace_0_var[83248]));
  void* constant_13_let = (&(global_const_workspace_0_var[83824]));
  void* constant_0_let = (&(global_const_workspace_0_var[76928]));
  void* constant_5_let = (&(global_const_workspace_0_var[83312]));
  void* constant_1_let = (&(global_const_workspace_0_var[84016]));
  void* constant_25_let = (&(global_const_workspace_0_var[82352]));
  void* constant_22_let = (&(global_const_workspace_0_var[82608]));
  void* constant_2_let = (&(global_const_workspace_0_var[83504]));
  void* constant_6_let = (&(global_const_workspace_0_var[69120]));
  void* constant_34_let = (&(global_const_workspace_0_var[81328]));
  void* constant_3_let = (&(global_const_workspace_0_var[83440]));
  void* constant_18_let = (&(global_const_workspace_0_var[76416]));
  void* constant_51_let = (&(global_const_workspace_0_var[77872]));
  void* constant_4_let = (&(global_const_workspace_0_var[83376]));
  void* constant_15_let = (&(global_const_workspace_0_var[83696]));
  void* constant_52_let = (&(global_const_workspace_0_var[77616]));
  void* constant_24_let = (&(global_const_workspace_0_var[64512]));
  void* constant_43_let = (&(global_const_workspace_0_var[79664]));
  void* constant_10_let = (&(global_const_workspace_0_var[83952]));
  void* constant_8_let = (&(global_const_workspace_0_var[83184]));
  void* constant_53_let = (&(global_const_workspace_0_var[77360]));
  void* constant_11_let = (&(global_const_workspace_0_var[83888]));
  void* constant_12_let = (&(global_const_workspace_0_var[71424]));
  void* constant_30_let = (&(global_const_workspace_0_var[55296]));
  void* constant_32_let = (&(global_const_workspace_0_var[81584]));
  void* constant_33_let = (&(global_const_workspace_0_var[81456]));
  void* constant_35_let = (&(global_const_workspace_0_var[81200]));
  void* constant_36_let = (&(global_const_workspace_0_var[73728]));
  void* constant_37_let = (&(global_const_workspace_0_var[80944]));
  void* constant_38_let = (&(global_const_workspace_0_var[80688]));
  void* constant_39_let = (&(global_const_workspace_0_var[80432]));
  void* constant_42_let = (&(global_const_workspace_0_var[36864]));
  void* constant_40_let = (&(global_const_workspace_0_var[80176]));
  void* constant_48_let = (&(global_const_workspace_0_var[0]));
  void* constant_41_let = (&(global_const_workspace_0_var[79920]));
  void* constant_44_let = (&(global_const_workspace_0_var[79408]));
  void* constant_45_let = (&(global_const_workspace_0_var[79152]));
  void* constant_46_let = (&(global_const_workspace_0_var[78896]));
  void* constant_50_let = (&(global_const_workspace_0_var[78128]));
  void* sid_43_let = (&(global_workspace_1_var[49152]));
  void* sid_36_let = (&(global_workspace_1_var[40960]));
  void* sid_7_let = (&(global_workspace_1_var[0]));
  void* sid_58_let = (&(global_workspace_1_var[53248]));
  void* sid_22_let = (&(global_workspace_1_var[16384]));
  void* sid_44_let = (&(global_workspace_1_var[40960]));
  void* sid_66_let = (&(global_workspace_1_var[53248]));
  void* sid_21_let = (&(global_workspace_1_var[32768]));
  void* sid_14_let = (&(global_workspace_1_var[49152]));
  void* sid_51_let = (&(global_workspace_1_var[49152]));
  void* sid_29_let = (&(global_workspace_1_var[32768]));
  void* sid_65_let = (&(global_workspace_1_var[57344]));
  void* sid_67_let = (&(global_workspace_1_var[57600]));
  void* sid_70_let = (&(global_workspace_1_var[57664]));
  if (tvmgen_default_cmsis_nn_main_0(input_1_int8_buffer_var, constant_0_let, constant_1_let, constant_3_let, constant_5_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_2(sid_7_let, constant_6_let, constant_7_let, constant_9_let, constant_11_let, sid_14_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_3(sid_14_let, constant_12_let, constant_13_let, constant_15_let, constant_17_let, sid_21_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_1(sid_7_let, sid_21_let, sid_22_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_5(sid_22_let, constant_18_let, constant_19_let, constant_21_let, constant_23_let, sid_29_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_7(sid_22_let, constant_24_let, constant_25_let, constant_27_let, constant_29_let, sid_36_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_8(sid_36_let, constant_30_let, constant_31_let, constant_33_let, constant_35_let, sid_43_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_6(sid_29_let, sid_43_let, sid_44_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_10(sid_44_let, constant_36_let, constant_37_let, constant_39_let, constant_41_let, sid_51_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_12(sid_44_let, constant_42_let, constant_43_let, constant_45_let, constant_47_let, sid_58_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_13(sid_58_let, constant_48_let, constant_49_let, constant_51_let, constant_53_let, sid_65_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_11(sid_51_let, sid_65_let, sid_66_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_15(sid_66_let, sid_67_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_16(sid_67_let, constant_54_let, constant_55_let, sid_70_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_cmsis_nn_main_17(sid_70_let, Identity_int8_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

