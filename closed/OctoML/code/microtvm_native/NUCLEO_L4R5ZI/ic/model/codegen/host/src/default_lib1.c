// tvm target: c -keys=arm_cpu,cpu -device=arm_cpu -mcpu=cortex-m4 -model=stm32l4r5zi
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>


#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_UGHLTAKV(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 2; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 2; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 2; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] =   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                               + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                               + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x1x2_body_loop_UGHLTAKV(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x1x2_body_UGHLTAKV(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[2];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x1x2_body_loop_UGHLTAKV(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x2_body_rest_UGHLTAKV(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_UGHLTAKV(
    int K,
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 4) * 4;
  switch ( K % 4 ) {
  case 1:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 2; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
      }
    }
    break;
  case 2:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 2; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1];
      }
    }
    break;
  case 3:
    for (int i = 0; i < 1; i++) {
      for (int j = 0; j < 2; j++) {
        int8_t *a_ptr = &aa[i * A_stride + k_base];
        int8_t *b_ptr = &bb[j * B_stride + k_base];
        cc[i * C_stride + j] +=   (int32_t) a_ptr[0] * (int32_t) b_ptr[0]
                                + (int32_t) a_ptr[1] * (int32_t) b_ptr[1]
                                + (int32_t) a_ptr[2] * (int32_t) b_ptr[2];
      }
    }
    break;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x1x2_update_loop_UGHLTAKV(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x1x2_update_UGHLTAKV(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[2];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x1x2_update_loop_UGHLTAKV(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 1 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*1 + j*4], (int32_t*) &bb_pad[i*1 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[1];
    for (int l = 0; l < 1 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*1];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (1 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 4 != 0 )
    gemm_1x2_update_rest_UGHLTAKV(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_UGHLTAKV(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] = (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x1x2_body_loop_UGHLTAKV(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x1x2_body_UGHLTAKV(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x1x2_body_loop_UGHLTAKV(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  if(((uint32_t)aa & 0x3) != 0 || ((uint32_t)bb & 0x3) != 0){
    retcode = kTvmErrorFunctionCallInvalidArg;
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x2_body_rest_UGHLTAKV(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_UGHLTAKV(
    int K,
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int k_base = (K / 2) * 2;
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int16_t *a_ptr = &aa[i * A_stride + k_base];
      int16_t *b_ptr = &bb[j * B_stride + k_base];
      cc[i * C_stride + j] += (int32_t) a_ptr[0] * (int32_t) b_ptr[0];
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x1x2_update_loop_UGHLTAKV(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 1; l++) {
        sum += (int32_t) aa[i*A_stride + l] * (int32_t) bb[j*B_stride + l];
      }
      cc[i*C_stride + j] += sum;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x1x2_update_UGHLTAKV(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x1x2_update_loop_UGHLTAKV(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 1 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 1 % 2 != 0 )
    gemm16_1x2_update_rest_UGHLTAKV(1, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x1x2_reset_UGHLTAKV(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_cast_subtract(int8_t* p0, int16_t* T_subtract, uint8_t* global_const_workspace_2_var, uint8_t* global_workspace_3_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3_inner = 0; ax3_inner < 3; ++ax3_inner) {
      int32_t cse_var_1 = ((ax0_ax1_fused_ax2_fused * 3) + ax3_inner);
      T_subtract[cse_var_1] = (((int16_t)p0[cse_var_1]) - (int16_t)-128);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_cast_subtract_1(int8_t* p0, int16_t* T_subtract, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 1024; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
        int32_t cse_var_1 = (((ax0_ax1_fused_ax2_fused * 16) + (ax3_outer * 8)) + ax3_inner);
        T_subtract[cse_var_1] = (((int16_t)p0[cse_var_1]) - (int16_t)-128);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_rig_5494088d2bce6f3f_(int8_t* p0, int16_t* p1, int16_t* T_subtract, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  void* fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let = (&(global_const_workspace_10_var[164240]));
  void* fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let = (&(global_const_workspace_10_var[163680]));
  void* fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let = (&(global_const_workspace_10_var[163552]));
  void* fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let = (&(global_const_workspace_10_var[163808]));
  void* fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let = (&(global_const_workspace_10_var[164064]));
  void* fused_cast_constant_10_let = (&(global_const_workspace_10_var[164256]));
  void* fused_cast_subtract_fixed_point_multiply_add_constant_11_let = (&(global_const_workspace_10_var[142848]));
  void* PadInput_let = (&(global_workspace_11_var[126976]));
  void* data_vec_let = (&(global_workspace_11_var[65536]));
  void* conv_let = (&(global_workspace_11_var[0]));
  for (int32_t i1 = 0; i1 < 34; ++i1) {
    for (int32_t i2 = 0; i2 < 34; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        int32_t cse_var_1 = (i2 * 16);
        ((int16_t*)PadInput_let)[(((i1 * 544) + cse_var_1) + i3)] = (((((1 <= i1) && (i1 < 33)) && (1 <= i2)) && (i2 < 33)) ? p1[((((i1 * 512) + cse_var_1) + i3) - 528)] : (int16_t)0);
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 8; ++n_oho_fused) {
    for (int32_t owo = 0; owo < 4; ++owo) {
      for (int32_t ohi = 0; ohi < 6; ++ohi) {
        for (int32_t ic = 0; ic < 16; ++ic) {
          ((int16_t*)data_vec_let)[((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic)];
        }
        for (int32_t ic_1 = 0; ic_1 < 16; ++ic_1) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_1) + 16)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_1) + 16)];
        }
        for (int32_t ic_2 = 0; ic_2 < 16; ++ic_2) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_2) + 32)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_2) + 32)];
        }
        for (int32_t ic_3 = 0; ic_3 < 16; ++ic_3) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_3) + 48)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_3) + 48)];
        }
        for (int32_t ic_4 = 0; ic_4 < 16; ++ic_4) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_4) + 64)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_4) + 64)];
        }
        for (int32_t ic_5 = 0; ic_5 < 16; ++ic_5) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_5) + 80)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_5) + 80)];
        }
        for (int32_t ic_6 = 0; ic_6 < 16; ++ic_6) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_6) + 96)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_6) + 96)];
        }
        for (int32_t ic_7 = 0; ic_7 < 16; ++ic_7) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_7) + 112)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_7) + 112)];
        }
        for (int32_t ic_8 = 0; ic_8 < 16; ++ic_8) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_8) + 128)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_8) + 128)];
        }
        for (int32_t ic_9 = 0; ic_9 < 16; ++ic_9) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_9) + 144)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_9) + 144)];
        }
      }
    }
  }
  for (int32_t oco = 0; oco < 8; ++oco) {
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t oci = 0; oci < 2; ++oci) {
          ((int16_t*)PadInput_let)[((((oco * 288) + (kh * 96)) + (kw * 32)) + oci)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[((((kh * 768) + (kw * 256)) + (oco * 2)) + oci)];
        }
        for (int32_t oci_1 = 0; oci_1 < 2; ++oci_1) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_1) + 2)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_1) + 16)];
        }
        for (int32_t oci_2 = 0; oci_2 < 2; ++oci_2) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_2) + 4)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_2) + 32)];
        }
        for (int32_t oci_3 = 0; oci_3 < 2; ++oci_3) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_3) + 6)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_3) + 48)];
        }
        for (int32_t oci_4 = 0; oci_4 < 2; ++oci_4) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_4) + 8)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_4) + 64)];
        }
        for (int32_t oci_5 = 0; oci_5 < 2; ++oci_5) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_5) + 10)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_5) + 80)];
        }
        for (int32_t oci_6 = 0; oci_6 < 2; ++oci_6) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_6) + 12)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_6) + 96)];
        }
        for (int32_t oci_7 = 0; oci_7 < 2; ++oci_7) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_7) + 14)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_7) + 112)];
        }
        for (int32_t oci_8 = 0; oci_8 < 2; ++oci_8) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_8) + 16)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_8) + 128)];
        }
        for (int32_t oci_9 = 0; oci_9 < 2; ++oci_9) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_9) + 18)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_9) + 144)];
        }
        for (int32_t oci_10 = 0; oci_10 < 2; ++oci_10) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_10) + 20)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_10) + 160)];
        }
        for (int32_t oci_11 = 0; oci_11 < 2; ++oci_11) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_11) + 22)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_11) + 176)];
        }
        for (int32_t oci_12 = 0; oci_12 < 2; ++oci_12) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_12) + 24)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_12) + 192)];
        }
        for (int32_t oci_13 = 0; oci_13 < 2; ++oci_13) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_13) + 26)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_13) + 208)];
        }
        for (int32_t oci_14 = 0; oci_14 < 2; ++oci_14) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_14) + 28)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_14) + 224)];
        }
        for (int32_t oci_15 = 0; oci_15 < 2; ++oci_15) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_15) + 30)] = ((int16_t*)fused_cast_subtract_fixed_point_multiply_add_constant_11_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_15) + 240)];
        }
      }
    }
  }
  for (int32_t oho = 0; oho < 8; ++oho) {
    for (int32_t owo_1 = 0; owo_1 < 4; ++owo_1) {
      for (int32_t oco_1 = 0; oco_1 < 8; ++oco_1) {
        for (int32_t owi = 0; owi < 8; ++owi) {
          int32_t cse_var_2 = ((((oho * 2048) + (owo_1 * 512)) + (oco_1 * 64)) + (owi * 2));
          ((int32_t*)conv_let)[cse_var_2] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 1)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 16)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 17)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 32)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 33)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 48)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 49)] = 0;
          for (int32_t ic_10 = 0; ic_10 < 16; ++ic_10) {
            int32_t cse_var_15 = (cse_var_2 + 49);
            int32_t cse_var_14 = (cse_var_2 + 48);
            int32_t cse_var_13 = (cse_var_2 + 33);
            int32_t cse_var_12 = (cse_var_2 + 32);
            int32_t cse_var_11 = (cse_var_2 + 17);
            int32_t cse_var_10 = (cse_var_2 + 16);
            int32_t cse_var_9 = (cse_var_2 + 1);
            int32_t cse_var_8 = ((oco_1 * 288) + (ic_10 * 2));
            int32_t cse_var_7 = (cse_var_8 + 1);
            int32_t cse_var_6 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_10);
            int32_t cse_var_5 = (cse_var_6 + 480);
            int32_t cse_var_4 = (cse_var_6 + 320);
            int32_t cse_var_3 = (cse_var_6 + 160);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_9] = (((int32_t*)conv_let)[cse_var_9] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_10] = (((int32_t*)conv_let)[cse_var_10] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_11] = (((int32_t*)conv_let)[cse_var_11] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_12] = (((int32_t*)conv_let)[cse_var_12] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_13] = (((int32_t*)conv_let)[cse_var_13] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_14] = (((int32_t*)conv_let)[cse_var_14] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_15] = (((int32_t*)conv_let)[cse_var_15] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
          }
          for (int32_t ic_11 = 0; ic_11 < 16; ++ic_11) {
            int32_t cse_var_30 = ((oco_1 * 288) + (ic_11 * 2));
            int32_t cse_var_29 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_11);
            int32_t cse_var_28 = (cse_var_2 + 49);
            int32_t cse_var_27 = (cse_var_2 + 48);
            int32_t cse_var_26 = (cse_var_2 + 33);
            int32_t cse_var_25 = (cse_var_2 + 32);
            int32_t cse_var_24 = (cse_var_2 + 17);
            int32_t cse_var_23 = (cse_var_2 + 16);
            int32_t cse_var_22 = (cse_var_2 + 1);
            int32_t cse_var_21 = (cse_var_30 + 33);
            int32_t cse_var_20 = (cse_var_30 + 32);
            int32_t cse_var_19 = (cse_var_29 + 496);
            int32_t cse_var_18 = (cse_var_29 + 336);
            int32_t cse_var_17 = (cse_var_29 + 176);
            int32_t cse_var_16 = (cse_var_29 + 16);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_22] = (((int32_t*)conv_let)[cse_var_22] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_23] = (((int32_t*)conv_let)[cse_var_23] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_24] = (((int32_t*)conv_let)[cse_var_24] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_25] = (((int32_t*)conv_let)[cse_var_25] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_26] = (((int32_t*)conv_let)[cse_var_26] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_27] = (((int32_t*)conv_let)[cse_var_27] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_28] = (((int32_t*)conv_let)[cse_var_28] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
          }
          for (int32_t ic_12 = 0; ic_12 < 16; ++ic_12) {
            int32_t cse_var_45 = ((oco_1 * 288) + (ic_12 * 2));
            int32_t cse_var_44 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_12);
            int32_t cse_var_43 = (cse_var_2 + 49);
            int32_t cse_var_42 = (cse_var_2 + 48);
            int32_t cse_var_41 = (cse_var_2 + 33);
            int32_t cse_var_40 = (cse_var_2 + 32);
            int32_t cse_var_39 = (cse_var_2 + 17);
            int32_t cse_var_38 = (cse_var_2 + 16);
            int32_t cse_var_37 = (cse_var_2 + 1);
            int32_t cse_var_36 = (cse_var_45 + 65);
            int32_t cse_var_35 = (cse_var_45 + 64);
            int32_t cse_var_34 = (cse_var_44 + 512);
            int32_t cse_var_33 = (cse_var_44 + 352);
            int32_t cse_var_32 = (cse_var_44 + 32);
            int32_t cse_var_31 = (cse_var_44 + 192);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_37] = (((int32_t*)conv_let)[cse_var_37] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_38] = (((int32_t*)conv_let)[cse_var_38] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_39] = (((int32_t*)conv_let)[cse_var_39] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_40] = (((int32_t*)conv_let)[cse_var_40] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_41] = (((int32_t*)conv_let)[cse_var_41] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_42] = (((int32_t*)conv_let)[cse_var_42] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_43] = (((int32_t*)conv_let)[cse_var_43] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
          }
          for (int32_t ic_13 = 0; ic_13 < 16; ++ic_13) {
            int32_t cse_var_60 = ((oco_1 * 288) + (ic_13 * 2));
            int32_t cse_var_59 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_13);
            int32_t cse_var_58 = (cse_var_2 + 49);
            int32_t cse_var_57 = (cse_var_2 + 48);
            int32_t cse_var_56 = (cse_var_2 + 33);
            int32_t cse_var_55 = (cse_var_2 + 32);
            int32_t cse_var_54 = (cse_var_2 + 17);
            int32_t cse_var_53 = (cse_var_2 + 16);
            int32_t cse_var_52 = (cse_var_2 + 1);
            int32_t cse_var_51 = (cse_var_60 + 97);
            int32_t cse_var_50 = (cse_var_60 + 96);
            int32_t cse_var_49 = (cse_var_59 + 640);
            int32_t cse_var_48 = (cse_var_59 + 480);
            int32_t cse_var_47 = (cse_var_59 + 320);
            int32_t cse_var_46 = (cse_var_59 + 160);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_52] = (((int32_t*)conv_let)[cse_var_52] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_53] = (((int32_t*)conv_let)[cse_var_53] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_54] = (((int32_t*)conv_let)[cse_var_54] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_55] = (((int32_t*)conv_let)[cse_var_55] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_56] = (((int32_t*)conv_let)[cse_var_56] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_57] = (((int32_t*)conv_let)[cse_var_57] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_58] = (((int32_t*)conv_let)[cse_var_58] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
          }
          for (int32_t ic_14 = 0; ic_14 < 16; ++ic_14) {
            int32_t cse_var_75 = ((oco_1 * 288) + (ic_14 * 2));
            int32_t cse_var_74 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_14);
            int32_t cse_var_73 = (cse_var_2 + 49);
            int32_t cse_var_72 = (cse_var_2 + 48);
            int32_t cse_var_71 = (cse_var_2 + 33);
            int32_t cse_var_70 = (cse_var_2 + 32);
            int32_t cse_var_69 = (cse_var_2 + 17);
            int32_t cse_var_68 = (cse_var_2 + 16);
            int32_t cse_var_67 = (cse_var_2 + 1);
            int32_t cse_var_66 = (cse_var_75 + 129);
            int32_t cse_var_65 = (cse_var_75 + 128);
            int32_t cse_var_64 = (cse_var_74 + 656);
            int32_t cse_var_63 = (cse_var_74 + 496);
            int32_t cse_var_62 = (cse_var_74 + 336);
            int32_t cse_var_61 = (cse_var_74 + 176);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_67] = (((int32_t*)conv_let)[cse_var_67] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_68] = (((int32_t*)conv_let)[cse_var_68] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_69] = (((int32_t*)conv_let)[cse_var_69] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_70] = (((int32_t*)conv_let)[cse_var_70] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_71] = (((int32_t*)conv_let)[cse_var_71] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_72] = (((int32_t*)conv_let)[cse_var_72] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_73] = (((int32_t*)conv_let)[cse_var_73] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
          }
          for (int32_t ic_15 = 0; ic_15 < 16; ++ic_15) {
            int32_t cse_var_90 = ((oco_1 * 288) + (ic_15 * 2));
            int32_t cse_var_89 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_15);
            int32_t cse_var_88 = (cse_var_2 + 49);
            int32_t cse_var_87 = (cse_var_2 + 48);
            int32_t cse_var_86 = (cse_var_2 + 33);
            int32_t cse_var_85 = (cse_var_2 + 32);
            int32_t cse_var_84 = (cse_var_2 + 17);
            int32_t cse_var_83 = (cse_var_2 + 16);
            int32_t cse_var_82 = (cse_var_2 + 1);
            int32_t cse_var_81 = (cse_var_90 + 161);
            int32_t cse_var_80 = (cse_var_90 + 160);
            int32_t cse_var_79 = (cse_var_89 + 672);
            int32_t cse_var_78 = (cse_var_89 + 512);
            int32_t cse_var_77 = (cse_var_89 + 352);
            int32_t cse_var_76 = (cse_var_89 + 192);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_82] = (((int32_t*)conv_let)[cse_var_82] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_83] = (((int32_t*)conv_let)[cse_var_83] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_84] = (((int32_t*)conv_let)[cse_var_84] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_85] = (((int32_t*)conv_let)[cse_var_85] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_86] = (((int32_t*)conv_let)[cse_var_86] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_87] = (((int32_t*)conv_let)[cse_var_87] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_88] = (((int32_t*)conv_let)[cse_var_88] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
          }
          for (int32_t ic_16 = 0; ic_16 < 16; ++ic_16) {
            int32_t cse_var_105 = ((oco_1 * 288) + (ic_16 * 2));
            int32_t cse_var_104 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_16);
            int32_t cse_var_103 = (cse_var_2 + 49);
            int32_t cse_var_102 = (cse_var_2 + 48);
            int32_t cse_var_101 = (cse_var_2 + 33);
            int32_t cse_var_100 = (cse_var_2 + 32);
            int32_t cse_var_99 = (cse_var_2 + 17);
            int32_t cse_var_98 = (cse_var_2 + 16);
            int32_t cse_var_97 = (cse_var_2 + 1);
            int32_t cse_var_96 = (cse_var_105 + 193);
            int32_t cse_var_95 = (cse_var_105 + 192);
            int32_t cse_var_94 = (cse_var_104 + 800);
            int32_t cse_var_93 = (cse_var_104 + 640);
            int32_t cse_var_92 = (cse_var_104 + 480);
            int32_t cse_var_91 = (cse_var_104 + 320);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_97] = (((int32_t*)conv_let)[cse_var_97] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_98] = (((int32_t*)conv_let)[cse_var_98] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_99] = (((int32_t*)conv_let)[cse_var_99] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_100] = (((int32_t*)conv_let)[cse_var_100] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_101] = (((int32_t*)conv_let)[cse_var_101] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_102] = (((int32_t*)conv_let)[cse_var_102] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_103] = (((int32_t*)conv_let)[cse_var_103] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
          }
          for (int32_t ic_17 = 0; ic_17 < 16; ++ic_17) {
            int32_t cse_var_120 = ((oco_1 * 288) + (ic_17 * 2));
            int32_t cse_var_119 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_17);
            int32_t cse_var_118 = (cse_var_2 + 49);
            int32_t cse_var_117 = (cse_var_2 + 48);
            int32_t cse_var_116 = (cse_var_2 + 33);
            int32_t cse_var_115 = (cse_var_2 + 32);
            int32_t cse_var_114 = (cse_var_2 + 17);
            int32_t cse_var_113 = (cse_var_2 + 16);
            int32_t cse_var_112 = (cse_var_2 + 1);
            int32_t cse_var_111 = (cse_var_120 + 225);
            int32_t cse_var_110 = (cse_var_120 + 224);
            int32_t cse_var_109 = (cse_var_119 + 816);
            int32_t cse_var_108 = (cse_var_119 + 656);
            int32_t cse_var_107 = (cse_var_119 + 496);
            int32_t cse_var_106 = (cse_var_119 + 336);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_112] = (((int32_t*)conv_let)[cse_var_112] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_113] = (((int32_t*)conv_let)[cse_var_113] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_114] = (((int32_t*)conv_let)[cse_var_114] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_115] = (((int32_t*)conv_let)[cse_var_115] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_116] = (((int32_t*)conv_let)[cse_var_116] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_117] = (((int32_t*)conv_let)[cse_var_117] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_118] = (((int32_t*)conv_let)[cse_var_118] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
          }
          for (int32_t ic_18 = 0; ic_18 < 16; ++ic_18) {
            int32_t cse_var_135 = ((oco_1 * 288) + (ic_18 * 2));
            int32_t cse_var_134 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_18);
            int32_t cse_var_133 = (cse_var_2 + 49);
            int32_t cse_var_132 = (cse_var_2 + 48);
            int32_t cse_var_131 = (cse_var_2 + 33);
            int32_t cse_var_130 = (cse_var_2 + 32);
            int32_t cse_var_129 = (cse_var_2 + 17);
            int32_t cse_var_128 = (cse_var_2 + 16);
            int32_t cse_var_127 = (cse_var_2 + 1);
            int32_t cse_var_126 = (cse_var_135 + 257);
            int32_t cse_var_125 = (cse_var_135 + 256);
            int32_t cse_var_124 = (cse_var_134 + 832);
            int32_t cse_var_123 = (cse_var_134 + 672);
            int32_t cse_var_122 = (cse_var_134 + 512);
            int32_t cse_var_121 = (cse_var_134 + 352);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_127] = (((int32_t*)conv_let)[cse_var_127] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_128] = (((int32_t*)conv_let)[cse_var_128] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_129] = (((int32_t*)conv_let)[cse_var_129] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_130] = (((int32_t*)conv_let)[cse_var_130] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_131] = (((int32_t*)conv_let)[cse_var_131] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_132] = (((int32_t*)conv_let)[cse_var_132] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_133] = (((int32_t*)conv_let)[cse_var_133] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
          }
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 8; ++ax0_ax1_outer_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
        for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
          int32_t cse_var_141 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_140 = (ax3_outer * 2);
          int32_t cse_var_139 = (cse_var_140 + 1);
          int32_t cse_var_138 = (((cse_var_141 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner * 2));
          int32_t cse_var_137 = (((cse_var_141 + (ax2_outer * 128)) + (ax2_inner * 16)) + cse_var_140);
          int32_t cse_var_136 = (cse_var_137 + 1);
          int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_138]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_140])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_140]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_140]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_140])) + 4;
          int32_t __2 = (__1) < (127) ? (__1) : (127);
          int32_t __3 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_137]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_137]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __4 = (__3) < (127) ? (__3) : (127);
          int8_t __5 = (int8_t)((__4) > (-128) ? (__4) : (-128));
          int8_t __6 = (int8_t)127;
          int8_t __7 = (__5) < (__6) ? (__5) : (__6);
          int8_t __8 = (int8_t)-128;
          T_subtract[cse_var_137] = (((int16_t)((__7) > (__8) ? (__7) : (__8))) - (int16_t)-128);
          int32_t __9 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_138 + 1)]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_139])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_139]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_139]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_139])) + 4;
          int32_t __10 = (__9) < (127) ? (__9) : (127);
          int32_t __11 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_136]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_136]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __12 = (__11) < (127) ? (__11) : (127);
          int8_t __13 = (int8_t)((__12) > (-128) ? (__12) : (-128));
          int8_t __14 = (__13) < (__6) ? (__13) : (__6);
          T_subtract[cse_var_136] = (((int16_t)((__14) > (__8) ? (__14) : (__8))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 8; ++ax2_inner_1) {
          int32_t cse_var_147 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_146 = (ax3_outer * 2);
          int32_t cse_var_148 = (((cse_var_147 + (ax2_outer * 128)) + (ax2_inner_1 * 16)) + cse_var_146);
          int32_t cse_var_145 = (cse_var_146 + 1);
          int32_t cse_var_144 = (((cse_var_147 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner_1 * 2));
          int32_t cse_var_143 = (cse_var_148 + 513);
          int32_t cse_var_142 = (cse_var_148 + 512);
          int32_t __15 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_144 + 16)]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_146])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_146]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_146]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_146])) + 4;
          int32_t __16 = (__15) < (127) ? (__15) : (127);
          int32_t __17 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_142]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_142]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __18 = (__17) < (127) ? (__17) : (127);
          int8_t __19 = (int8_t)((__18) > (-128) ? (__18) : (-128));
          int8_t __20 = (int8_t)127;
          int8_t __21 = (__19) < (__20) ? (__19) : (__20);
          int8_t __22 = (int8_t)-128;
          T_subtract[cse_var_142] = (((int16_t)((__21) > (__22) ? (__21) : (__22))) - (int16_t)-128);
          int32_t __23 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_144 + 17)]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_145])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_145]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_145]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_145])) + 4;
          int32_t __24 = (__23) < (127) ? (__23) : (127);
          int32_t __25 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_143]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_143]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__24) > (-128) ? (__24) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__24) > (-128) ? (__24) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __26 = (__25) < (127) ? (__25) : (127);
          int8_t __27 = (int8_t)((__26) > (-128) ? (__26) : (-128));
          int8_t __28 = (__27) < (__20) ? (__27) : (__20);
          T_subtract[cse_var_143] = (((int16_t)((__28) > (__22) ? (__28) : (__22))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_2 = 0; ax2_inner_2 < 8; ++ax2_inner_2) {
          int32_t cse_var_154 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_153 = (ax3_outer * 2);
          int32_t cse_var_155 = (((cse_var_154 + (ax2_outer * 128)) + (ax2_inner_2 * 16)) + cse_var_153);
          int32_t cse_var_152 = (cse_var_153 + 1);
          int32_t cse_var_151 = (((cse_var_154 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner_2 * 2));
          int32_t cse_var_150 = (cse_var_155 + 1025);
          int32_t cse_var_149 = (cse_var_155 + 1024);
          int32_t __29 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_151 + 32)]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_153])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_153]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_153]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_153])) + 4;
          int32_t __30 = (__29) < (127) ? (__29) : (127);
          int32_t __31 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_149]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_149]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__30) > (-128) ? (__30) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__30) > (-128) ? (__30) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __32 = (__31) < (127) ? (__31) : (127);
          int8_t __33 = (int8_t)((__32) > (-128) ? (__32) : (-128));
          int8_t __34 = (int8_t)127;
          int8_t __35 = (__33) < (__34) ? (__33) : (__34);
          int8_t __36 = (int8_t)-128;
          T_subtract[cse_var_149] = (((int16_t)((__35) > (__36) ? (__35) : (__36))) - (int16_t)-128);
          int32_t __37 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_151 + 33)]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_152])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_152]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_152]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_152])) + 4;
          int32_t __38 = (__37) < (127) ? (__37) : (127);
          int32_t __39 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_150]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_150]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__38) > (-128) ? (__38) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__38) > (-128) ? (__38) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __40 = (__39) < (127) ? (__39) : (127);
          int8_t __41 = (int8_t)((__40) > (-128) ? (__40) : (-128));
          int8_t __42 = (__41) < (__34) ? (__41) : (__34);
          T_subtract[cse_var_150] = (((int16_t)((__42) > (__36) ? (__42) : (__36))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_3 = 0; ax2_inner_3 < 8; ++ax2_inner_3) {
          int32_t cse_var_161 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_160 = (ax3_outer * 2);
          int32_t cse_var_162 = (((cse_var_161 + (ax2_outer * 128)) + (ax2_inner_3 * 16)) + cse_var_160);
          int32_t cse_var_159 = (cse_var_160 + 1);
          int32_t cse_var_158 = (((cse_var_161 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner_3 * 2));
          int32_t cse_var_157 = (cse_var_162 + 1537);
          int32_t cse_var_156 = (cse_var_162 + 1536);
          int32_t __43 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_158 + 48)]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_160])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_160]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_160]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_160])) + 4;
          int32_t __44 = (__43) < (127) ? (__43) : (127);
          int32_t __45 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_156]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_156]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__44) > (-128) ? (__44) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__44) > (-128) ? (__44) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __46 = (__45) < (127) ? (__45) : (127);
          int8_t __47 = (int8_t)((__46) > (-128) ? (__46) : (-128));
          int8_t __48 = (int8_t)127;
          int8_t __49 = (__47) < (__48) ? (__47) : (__48);
          int8_t __50 = (int8_t)-128;
          T_subtract[cse_var_156] = (((int16_t)((__49) > (__50) ? (__49) : (__50))) - (int16_t)-128);
          int32_t __51 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_158 + 49)]) + ((int64_t)((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_constant_12_let)[cse_var_159])) * ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_constant_13_let)[cse_var_159]) + ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_constant_14_let)[cse_var_159]) >> ((int64_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_constant_15_let)[cse_var_159])) + 4;
          int32_t __52 = (__51) < (127) ? (__51) : (127);
          int32_t __53 = (((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)p0[cse_var_157]) - ((int32_t*)fused_cast_constant_10_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)p0[cse_var_157]) - ((int32_t*)fused_cast_constant_10_let)[0]))) * (int64_t)1660533717) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + ((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__52) > (-128) ? (__52) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__52) > (-128) ? (__52) : (-128)))) - ((int32_t*)fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_16_let)[0]))) * (int64_t)1098017566) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31))))) - 128;
          int32_t __54 = (__53) < (127) ? (__53) : (127);
          int8_t __55 = (int8_t)((__54) > (-128) ? (__54) : (-128));
          int8_t __56 = (__55) < (__48) ? (__55) : (__48);
          T_subtract[cse_var_157] = (((int16_t)((__56) > (__50) ? (__56) : (__50))) - (int16_t)-128);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_avg_pool2d_cast(int32_t* p0, int8_t* T_cast, uint8_t* global_const_workspace_24_var, uint8_t* global_workspace_25_var) {
  void* tensor_let = (&(global_workspace_25_var[237856]));
  for (int32_t ax3 = 0; ax3 < 64; ++ax3) {
    ((int32_t*)tensor_let)[ax3] = 0;
    for (int32_t rv0 = 0; rv0 < 8; ++rv0) {
      for (int32_t rv1 = 0; rv1 < 8; ++rv1) {
        ((int32_t*)tensor_let)[ax3] = (((int32_t*)tensor_let)[ax3] + p0[(((rv0 * 512) + (rv1 * 64)) + ax3)]);
      }
    }
  }
  for (int32_t ax3_1 = 0; ax3_1 < 64; ++ax3_1) {
    ((int32_t*)tensor_let)[ax3_1] = (((int32_t*)tensor_let)[ax3_1] / 64);
  }
  for (int32_t ax3_2 = 0; ax3_2 < 64; ++ax3_2) {
    T_cast[ax3_2] = ((int8_t)((int32_t*)tensor_let)[ax3_2]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_866ca172ecfe2cfb_(int16_t* p0, int32_t* p1, int16_t* T_subtract, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let = (&(global_const_workspace_16_var[164208]));
  void* fused_nn_conv2d_add_cast_multiply_add_constant_32_let = (&(global_const_workspace_16_var[160864]));
  void* fused_nn_conv2d_add_cast_multiply_constant_31_let = (&(global_const_workspace_16_var[160096]));
  void* fused_nn_conv2d_add_cast_constant_30_let = (&(global_const_workspace_16_var[161632]));
  void* fused_nn_conv2d_constant_29_let = (&(global_const_workspace_16_var[162400]));
  void* fused_constant_28_let = (&(global_const_workspace_16_var[152832]));
  void* PadInput_let = (&(global_workspace_17_var[139552]));
  void* data_vec_let = (&(global_workspace_17_var[172320]));
  void* kernel_vec_let = (&(global_workspace_17_var[195360]));
  void* conv_let = (&(global_workspace_17_var[139552]));
  for (int32_t i1 = 0; i1 < 31; ++i1) {
    for (int32_t i2 = 0; i2 < 31; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        int32_t cse_var_1 = (i2 * 16);
        ((int16_t*)PadInput_let)[(((i1 * 496) + cse_var_1) + i3)] = p0[(((i1 * 512) + cse_var_1) + i3)];
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 8; ++n_oho_fused) {
    for (int32_t owo = 0; owo < 2; ++owo) {
      for (int32_t ohi = 0; ohi < 3; ++ohi) {
        for (int32_t ic = 0; ic < 16; ++ic) {
          ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic)];
        }
        for (int32_t ic_1 = 0; ic_1 < 16; ++ic_1) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_1) + 16)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_1) + 16)];
        }
        for (int32_t ic_2 = 0; ic_2 < 16; ++ic_2) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_2) + 32)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_2) + 32)];
        }
        for (int32_t ic_3 = 0; ic_3 < 16; ++ic_3) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_3) + 48)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_3) + 48)];
        }
        for (int32_t ic_4 = 0; ic_4 < 16; ++ic_4) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_4) + 64)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_4) + 64)];
        }
        for (int32_t ic_5 = 0; ic_5 < 16; ++ic_5) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_5) + 80)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_5) + 80)];
        }
        for (int32_t ic_6 = 0; ic_6 < 16; ++ic_6) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_6) + 96)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_6) + 96)];
        }
        for (int32_t ic_7 = 0; ic_7 < 16; ++ic_7) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_7) + 112)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_7) + 112)];
        }
        for (int32_t ic_8 = 0; ic_8 < 16; ++ic_8) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_8) + 128)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_8) + 128)];
        }
        for (int32_t ic_9 = 0; ic_9 < 16; ++ic_9) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_9) + 144)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_9) + 144)];
        }
        for (int32_t ic_10 = 0; ic_10 < 16; ++ic_10) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_10) + 160)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_10) + 160)];
        }
        for (int32_t ic_11 = 0; ic_11 < 16; ++ic_11) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_11) + 176)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_11) + 176)];
        }
        for (int32_t ic_12 = 0; ic_12 < 16; ++ic_12) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_12) + 192)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_12) + 192)];
        }
        for (int32_t ic_13 = 0; ic_13 < 16; ++ic_13) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_13) + 208)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_13) + 208)];
        }
        for (int32_t ic_14 = 0; ic_14 < 16; ++ic_14) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 1440) + (owo * 720)) + (ohi * 240)) + ic_14) + 224)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 1984) + (ohi * 496)) + (owo * 256)) + ic_14) + 224)];
        }
      }
    }
  }
  for (int32_t oco = 0; oco < 8; ++oco) {
    for (int32_t oci = 0; oci < 4; ++oci) {
      ((int16_t*)kernel_vec_let)[((oco * 64) + oci)] = ((int16_t*)fused_constant_28_let)[((oco * 4) + oci)];
    }
    for (int32_t oci_1 = 0; oci_1 < 4; ++oci_1) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_1) + 4)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_1) + 32)];
    }
    for (int32_t oci_2 = 0; oci_2 < 4; ++oci_2) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_2) + 8)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_2) + 64)];
    }
    for (int32_t oci_3 = 0; oci_3 < 4; ++oci_3) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_3) + 12)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_3) + 96)];
    }
    for (int32_t oci_4 = 0; oci_4 < 4; ++oci_4) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_4) + 16)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_4) + 128)];
    }
    for (int32_t oci_5 = 0; oci_5 < 4; ++oci_5) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_5) + 20)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_5) + 160)];
    }
    for (int32_t oci_6 = 0; oci_6 < 4; ++oci_6) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_6) + 24)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_6) + 192)];
    }
    for (int32_t oci_7 = 0; oci_7 < 4; ++oci_7) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_7) + 28)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_7) + 224)];
    }
    for (int32_t oci_8 = 0; oci_8 < 4; ++oci_8) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_8) + 32)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_8) + 256)];
    }
    for (int32_t oci_9 = 0; oci_9 < 4; ++oci_9) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_9) + 36)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_9) + 288)];
    }
    for (int32_t oci_10 = 0; oci_10 < 4; ++oci_10) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_10) + 40)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_10) + 320)];
    }
    for (int32_t oci_11 = 0; oci_11 < 4; ++oci_11) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_11) + 44)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_11) + 352)];
    }
    for (int32_t oci_12 = 0; oci_12 < 4; ++oci_12) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_12) + 48)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_12) + 384)];
    }
    for (int32_t oci_13 = 0; oci_13 < 4; ++oci_13) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_13) + 52)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_13) + 416)];
    }
    for (int32_t oci_14 = 0; oci_14 < 4; ++oci_14) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_14) + 56)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_14) + 448)];
    }
    for (int32_t oci_15 = 0; oci_15 < 4; ++oci_15) {
      ((int16_t*)kernel_vec_let)[(((oco * 64) + oci_15) + 60)] = ((int16_t*)fused_constant_28_let)[(((oco * 4) + oci_15) + 480)];
    }
  }
  for (int32_t oho = 0; oho < 8; ++oho) {
    for (int32_t owo_1 = 0; owo_1 < 2; ++owo_1) {
      for (int32_t oco_1 = 0; oco_1 < 8; ++oco_1) {
        for (int32_t owi = 0; owi < 8; ++owi) {
          int32_t cse_var_2 = ((((oho * 1024) + (owo_1 * 512)) + (oco_1 * 64)) + (owi * 4));
          ((int32_t*)conv_let)[cse_var_2] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 1)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 2)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 3)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 32)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 33)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 34)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 35)] = 0;
          for (int32_t ic_15 = 0; ic_15 < 16; ++ic_15) {
            int32_t cse_var_15 = (cse_var_2 + 35);
            int32_t cse_var_14 = (cse_var_2 + 34);
            int32_t cse_var_13 = (cse_var_2 + 33);
            int32_t cse_var_12 = (cse_var_2 + 32);
            int32_t cse_var_11 = (cse_var_2 + 3);
            int32_t cse_var_10 = (cse_var_2 + 2);
            int32_t cse_var_9 = (cse_var_2 + 1);
            int32_t cse_var_8 = ((oco_1 * 64) + (ic_15 * 4));
            int32_t cse_var_7 = (cse_var_8 + 3);
            int32_t cse_var_6 = (cse_var_8 + 2);
            int32_t cse_var_5 = (cse_var_8 + 1);
            int32_t cse_var_4 = ((((oho * 1440) + (owo_1 * 720)) + (owi * 32)) + ic_15);
            int32_t cse_var_3 = (cse_var_4 + 480);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_9] = (((int32_t*)conv_let)[cse_var_9] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_5])));
            ((int32_t*)conv_let)[cse_var_10] = (((int32_t*)conv_let)[cse_var_10] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_6])));
            ((int32_t*)conv_let)[cse_var_11] = (((int32_t*)conv_let)[cse_var_11] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_12] = (((int32_t*)conv_let)[cse_var_12] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_13] = (((int32_t*)conv_let)[cse_var_13] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_5])));
            ((int32_t*)conv_let)[cse_var_14] = (((int32_t*)conv_let)[cse_var_14] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_6])));
            ((int32_t*)conv_let)[cse_var_15] = (((int32_t*)conv_let)[cse_var_15] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)kernel_vec_let)[cse_var_7])));
          }
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 8; ++ax0_ax1_outer_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 2; ++ax2_outer) {
      for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
        for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
          int32_t cse_var_25 = (ax0_ax1_outer_fused * 1024);
          int32_t cse_var_24 = (ax3_outer * 4);
          int32_t cse_var_23 = (cse_var_24 + 3);
          int32_t cse_var_22 = (cse_var_24 + 2);
          int32_t cse_var_21 = (cse_var_24 + 1);
          int32_t cse_var_20 = (((cse_var_25 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner * 4));
          int32_t cse_var_19 = (((cse_var_25 + (ax2_outer * 256)) + (ax2_inner * 32)) + cse_var_24);
          int32_t cse_var_18 = (cse_var_19 + 3);
          int32_t cse_var_17 = (cse_var_19 + 2);
          int32_t cse_var_16 = (cse_var_19 + 1);
          int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_20]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_24])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_24]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_24]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_24])) - 17;
          int32_t __2 = (__1) < (127) ? (__1) : (127);
          int32_t __3 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_19];
          int32_t __4 = (__3) < (127) ? (__3) : (127);
          int8_t __5 = (int8_t)((__4) > (-128) ? (__4) : (-128));
          int8_t __6 = (int8_t)127;
          int8_t __7 = (__5) < (__6) ? (__5) : (__6);
          int8_t __8 = (int8_t)-128;
          T_subtract[cse_var_19] = (((int16_t)((__7) > (__8) ? (__7) : (__8))) - (int16_t)-128);
          int32_t __9 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_20 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_21])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_21]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_21]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_21])) - 17;
          int32_t __10 = (__9) < (127) ? (__9) : (127);
          int32_t __11 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_16];
          int32_t __12 = (__11) < (127) ? (__11) : (127);
          int8_t __13 = (int8_t)((__12) > (-128) ? (__12) : (-128));
          int8_t __14 = (__13) < (__6) ? (__13) : (__6);
          T_subtract[cse_var_16] = (((int16_t)((__14) > (__8) ? (__14) : (__8))) - (int16_t)-128);
          int32_t __15 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_20 + 2)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_22])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_22]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_22]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_22])) - 17;
          int32_t __16 = (__15) < (127) ? (__15) : (127);
          int32_t __17 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_17];
          int32_t __18 = (__17) < (127) ? (__17) : (127);
          int8_t __19 = (int8_t)((__18) > (-128) ? (__18) : (-128));
          int8_t __20 = (__19) < (__6) ? (__19) : (__6);
          T_subtract[cse_var_17] = (((int16_t)((__20) > (__8) ? (__20) : (__8))) - (int16_t)-128);
          int32_t __21 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_20 + 3)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_23])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_23]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_23]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_23])) - 17;
          int32_t __22 = (__21) < (127) ? (__21) : (127);
          int32_t __23 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__22) > (-128) ? (__22) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__22) > (-128) ? (__22) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_18];
          int32_t __24 = (__23) < (127) ? (__23) : (127);
          int8_t __25 = (int8_t)((__24) > (-128) ? (__24) : (-128));
          int8_t __26 = (__25) < (__6) ? (__25) : (__6);
          T_subtract[cse_var_18] = (((int16_t)((__26) > (__8) ? (__26) : (__8))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 8; ++ax2_inner_1) {
          int32_t cse_var_35 = (ax0_ax1_outer_fused * 1024);
          int32_t cse_var_34 = (ax3_outer * 4);
          int32_t cse_var_36 = (((cse_var_35 + (ax2_outer * 256)) + (ax2_inner_1 * 32)) + cse_var_34);
          int32_t cse_var_33 = (cse_var_34 + 3);
          int32_t cse_var_32 = (cse_var_34 + 2);
          int32_t cse_var_31 = (cse_var_34 + 1);
          int32_t cse_var_30 = (((cse_var_35 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner_1 * 4));
          int32_t cse_var_29 = (cse_var_36 + 515);
          int32_t cse_var_28 = (cse_var_36 + 514);
          int32_t cse_var_27 = (cse_var_36 + 513);
          int32_t cse_var_26 = (cse_var_36 + 512);
          int32_t __27 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_30 + 32)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_34])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_34]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_34]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_34])) - 17;
          int32_t __28 = (__27) < (127) ? (__27) : (127);
          int32_t __29 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__28) > (-128) ? (__28) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__28) > (-128) ? (__28) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_26];
          int32_t __30 = (__29) < (127) ? (__29) : (127);
          int8_t __31 = (int8_t)((__30) > (-128) ? (__30) : (-128));
          int8_t __32 = (int8_t)127;
          int8_t __33 = (__31) < (__32) ? (__31) : (__32);
          int8_t __34 = (int8_t)-128;
          T_subtract[cse_var_26] = (((int16_t)((__33) > (__34) ? (__33) : (__34))) - (int16_t)-128);
          int32_t __35 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_30 + 33)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_31])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_31]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_31]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_31])) - 17;
          int32_t __36 = (__35) < (127) ? (__35) : (127);
          int32_t __37 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__36) > (-128) ? (__36) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__36) > (-128) ? (__36) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_27];
          int32_t __38 = (__37) < (127) ? (__37) : (127);
          int8_t __39 = (int8_t)((__38) > (-128) ? (__38) : (-128));
          int8_t __40 = (__39) < (__32) ? (__39) : (__32);
          T_subtract[cse_var_27] = (((int16_t)((__40) > (__34) ? (__40) : (__34))) - (int16_t)-128);
          int32_t __41 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_30 + 34)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_32])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_32]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_32]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_32])) - 17;
          int32_t __42 = (__41) < (127) ? (__41) : (127);
          int32_t __43 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__42) > (-128) ? (__42) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__42) > (-128) ? (__42) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_28];
          int32_t __44 = (__43) < (127) ? (__43) : (127);
          int8_t __45 = (int8_t)((__44) > (-128) ? (__44) : (-128));
          int8_t __46 = (__45) < (__32) ? (__45) : (__32);
          T_subtract[cse_var_28] = (((int16_t)((__46) > (__34) ? (__46) : (__34))) - (int16_t)-128);
          int32_t __47 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_30 + 35)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_29_let)[cse_var_33])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_30_let)[cse_var_33]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_31_let)[cse_var_33]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_32_let)[cse_var_33])) - 17;
          int32_t __48 = (__47) < (127) ? (__47) : (127);
          int32_t __49 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__48) > (-128) ? (__48) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__48) > (-128) ? (__48) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_33_let)[0]))) * (int64_t)1805621035) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_29];
          int32_t __50 = (__49) < (127) ? (__49) : (127);
          int8_t __51 = (int8_t)((__50) > (-128) ? (__50) : (-128));
          int8_t __52 = (__51) < (__32) ? (__51) : (__32);
          T_subtract[cse_var_29] = (((int16_t)((__52) > (__34) ? (__52) : (__34))) - (int16_t)-128);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_8b760d480c798df_(int16_t* p0, int32_t* p1, int32_t* T_cast, uint8_t* global_const_workspace_22_var, uint8_t* global_workspace_23_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let = (&(global_const_workspace_22_var[164176]));
  void* fused_nn_conv2d_add_cast_multiply_add_constant_49_let = (&(global_const_workspace_22_var[156256]));
  void* fused_nn_conv2d_add_cast_multiply_constant_48_let = (&(global_const_workspace_22_var[154720]));
  void* fused_nn_conv2d_add_cast_constant_47_let = (&(global_const_workspace_22_var[157792]));
  void* fused_nn_conv2d_constant_46_let = (&(global_const_workspace_22_var[159328]));
  void* fused_constant_45_let = (&(global_const_workspace_22_var[147456]));
  void* PadInput_let = (&(global_workspace_23_var[221472]));
  void* data_vec_let = (&(global_workspace_23_var[235872]));
  void* conv_let = (&(global_workspace_23_var[205088]));
  for (int32_t i1 = 0; i1 < 15; ++i1) {
    for (int32_t i2 = 0; i2 < 15; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        int32_t cse_var_1 = (i2 * 32);
        ((int16_t*)PadInput_let)[(((i1 * 480) + cse_var_1) + i3)] = p0[(((i1 * 512) + cse_var_1) + i3)];
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 4; ++n_oho_fused) {
    for (int32_t ohi = 0; ohi < 3; ++ohi) {
      for (int32_t ic = 0; ic < 32; ++ic) {
        int32_t cse_var_2 = (ohi * 480);
        ((int16_t*)data_vec_let)[(((n_oho_fused * 1440) + cse_var_2) + ic)] = ((int16_t*)PadInput_let)[(((n_oho_fused * 1920) + cse_var_2) + ic)];
      }
      for (int32_t ic_1 = 0; ic_1 < 32; ++ic_1) {
        int32_t cse_var_3 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_3) + ic_1) + 32)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_3) + ic_1) + 32)];
      }
      for (int32_t ic_2 = 0; ic_2 < 32; ++ic_2) {
        int32_t cse_var_4 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_4) + ic_2) + 64)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_4) + ic_2) + 64)];
      }
      for (int32_t ic_3 = 0; ic_3 < 32; ++ic_3) {
        int32_t cse_var_5 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_5) + ic_3) + 96)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_5) + ic_3) + 96)];
      }
      for (int32_t ic_4 = 0; ic_4 < 32; ++ic_4) {
        int32_t cse_var_6 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_6) + ic_4) + 128)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_6) + ic_4) + 128)];
      }
      for (int32_t ic_5 = 0; ic_5 < 32; ++ic_5) {
        int32_t cse_var_7 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_7) + ic_5) + 160)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_7) + ic_5) + 160)];
      }
      for (int32_t ic_6 = 0; ic_6 < 32; ++ic_6) {
        int32_t cse_var_8 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_8) + ic_6) + 192)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_8) + ic_6) + 192)];
      }
      for (int32_t ic_7 = 0; ic_7 < 32; ++ic_7) {
        int32_t cse_var_9 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_9) + ic_7) + 224)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_9) + ic_7) + 224)];
      }
      for (int32_t ic_8 = 0; ic_8 < 32; ++ic_8) {
        int32_t cse_var_10 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_10) + ic_8) + 256)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_10) + ic_8) + 256)];
      }
      for (int32_t ic_9 = 0; ic_9 < 32; ++ic_9) {
        int32_t cse_var_11 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_11) + ic_9) + 288)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_11) + ic_9) + 288)];
      }
      for (int32_t ic_10 = 0; ic_10 < 32; ++ic_10) {
        int32_t cse_var_12 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_12) + ic_10) + 320)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_12) + ic_10) + 320)];
      }
      for (int32_t ic_11 = 0; ic_11 < 32; ++ic_11) {
        int32_t cse_var_13 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_13) + ic_11) + 352)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_13) + ic_11) + 352)];
      }
      for (int32_t ic_12 = 0; ic_12 < 32; ++ic_12) {
        int32_t cse_var_14 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_14) + ic_12) + 384)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_14) + ic_12) + 384)];
      }
      for (int32_t ic_13 = 0; ic_13 < 32; ++ic_13) {
        int32_t cse_var_15 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_15) + ic_13) + 416)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_15) + ic_13) + 416)];
      }
      for (int32_t ic_14 = 0; ic_14 < 32; ++ic_14) {
        int32_t cse_var_16 = (ohi * 480);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 1440) + cse_var_16) + ic_14) + 448)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 1920) + cse_var_16) + ic_14) + 448)];
      }
    }
  }
  for (int32_t oco = 0; oco < 16; ++oco) {
    for (int32_t oci = 0; oci < 4; ++oci) {
      ((int16_t*)PadInput_let)[((oco * 128) + oci)] = ((int16_t*)fused_constant_45_let)[((oco * 4) + oci)];
    }
    for (int32_t oci_1 = 0; oci_1 < 4; ++oci_1) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_1) + 4)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_1) + 64)];
    }
    for (int32_t oci_2 = 0; oci_2 < 4; ++oci_2) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_2) + 8)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_2) + 128)];
    }
    for (int32_t oci_3 = 0; oci_3 < 4; ++oci_3) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_3) + 12)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_3) + 192)];
    }
    for (int32_t oci_4 = 0; oci_4 < 4; ++oci_4) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_4) + 16)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_4) + 256)];
    }
    for (int32_t oci_5 = 0; oci_5 < 4; ++oci_5) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_5) + 20)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_5) + 320)];
    }
    for (int32_t oci_6 = 0; oci_6 < 4; ++oci_6) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_6) + 24)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_6) + 384)];
    }
    for (int32_t oci_7 = 0; oci_7 < 4; ++oci_7) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_7) + 28)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_7) + 448)];
    }
    for (int32_t oci_8 = 0; oci_8 < 4; ++oci_8) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_8) + 32)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_8) + 512)];
    }
    for (int32_t oci_9 = 0; oci_9 < 4; ++oci_9) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_9) + 36)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_9) + 576)];
    }
    for (int32_t oci_10 = 0; oci_10 < 4; ++oci_10) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_10) + 40)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_10) + 640)];
    }
    for (int32_t oci_11 = 0; oci_11 < 4; ++oci_11) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_11) + 44)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_11) + 704)];
    }
    for (int32_t oci_12 = 0; oci_12 < 4; ++oci_12) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_12) + 48)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_12) + 768)];
    }
    for (int32_t oci_13 = 0; oci_13 < 4; ++oci_13) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_13) + 52)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_13) + 832)];
    }
    for (int32_t oci_14 = 0; oci_14 < 4; ++oci_14) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_14) + 56)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_14) + 896)];
    }
    for (int32_t oci_15 = 0; oci_15 < 4; ++oci_15) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_15) + 60)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_15) + 960)];
    }
    for (int32_t oci_16 = 0; oci_16 < 4; ++oci_16) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_16) + 64)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_16) + 1024)];
    }
    for (int32_t oci_17 = 0; oci_17 < 4; ++oci_17) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_17) + 68)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_17) + 1088)];
    }
    for (int32_t oci_18 = 0; oci_18 < 4; ++oci_18) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_18) + 72)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_18) + 1152)];
    }
    for (int32_t oci_19 = 0; oci_19 < 4; ++oci_19) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_19) + 76)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_19) + 1216)];
    }
    for (int32_t oci_20 = 0; oci_20 < 4; ++oci_20) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_20) + 80)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_20) + 1280)];
    }
    for (int32_t oci_21 = 0; oci_21 < 4; ++oci_21) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_21) + 84)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_21) + 1344)];
    }
    for (int32_t oci_22 = 0; oci_22 < 4; ++oci_22) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_22) + 88)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_22) + 1408)];
    }
    for (int32_t oci_23 = 0; oci_23 < 4; ++oci_23) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_23) + 92)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_23) + 1472)];
    }
    for (int32_t oci_24 = 0; oci_24 < 4; ++oci_24) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_24) + 96)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_24) + 1536)];
    }
    for (int32_t oci_25 = 0; oci_25 < 4; ++oci_25) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_25) + 100)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_25) + 1600)];
    }
    for (int32_t oci_26 = 0; oci_26 < 4; ++oci_26) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_26) + 104)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_26) + 1664)];
    }
    for (int32_t oci_27 = 0; oci_27 < 4; ++oci_27) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_27) + 108)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_27) + 1728)];
    }
    for (int32_t oci_28 = 0; oci_28 < 4; ++oci_28) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_28) + 112)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_28) + 1792)];
    }
    for (int32_t oci_29 = 0; oci_29 < 4; ++oci_29) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_29) + 116)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_29) + 1856)];
    }
    for (int32_t oci_30 = 0; oci_30 < 4; ++oci_30) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_30) + 120)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_30) + 1920)];
    }
    for (int32_t oci_31 = 0; oci_31 < 4; ++oci_31) {
      ((int16_t*)PadInput_let)[(((oco * 128) + oci_31) + 124)] = ((int16_t*)fused_constant_45_let)[(((oco * 4) + oci_31) + 1984)];
    }
  }
  for (int32_t oho = 0; oho < 4; ++oho) {
    for (int32_t oco_1 = 0; oco_1 < 16; ++oco_1) {
      for (int32_t owi = 0; owi < 8; ++owi) {
        int32_t cse_var_17 = (((oho * 1024) + (oco_1 * 64)) + (owi * 4));
        ((int32_t*)conv_let)[cse_var_17] = 0;
        ((int32_t*)conv_let)[(cse_var_17 + 1)] = 0;
        ((int32_t*)conv_let)[(cse_var_17 + 2)] = 0;
        ((int32_t*)conv_let)[(cse_var_17 + 3)] = 0;
        ((int32_t*)conv_let)[(cse_var_17 + 32)] = 0;
        ((int32_t*)conv_let)[(cse_var_17 + 33)] = 0;
        ((int32_t*)conv_let)[(cse_var_17 + 34)] = 0;
        ((int32_t*)conv_let)[(cse_var_17 + 35)] = 0;
        for (int32_t ic_15 = 0; ic_15 < 32; ++ic_15) {
          int32_t cse_var_30 = (cse_var_17 + 35);
          int32_t cse_var_29 = (cse_var_17 + 34);
          int32_t cse_var_28 = (cse_var_17 + 33);
          int32_t cse_var_27 = (cse_var_17 + 32);
          int32_t cse_var_26 = (cse_var_17 + 3);
          int32_t cse_var_25 = (cse_var_17 + 2);
          int32_t cse_var_24 = (cse_var_17 + 1);
          int32_t cse_var_23 = ((oco_1 * 128) + (ic_15 * 4));
          int32_t cse_var_22 = (((oho * 1440) + (owi * 64)) + ic_15);
          int32_t cse_var_21 = (cse_var_23 + 3);
          int32_t cse_var_20 = (cse_var_23 + 2);
          int32_t cse_var_19 = (cse_var_23 + 1);
          int32_t cse_var_18 = (cse_var_22 + 960);
          ((int32_t*)conv_let)[cse_var_17] = (((int32_t*)conv_let)[cse_var_17] + (((int32_t)((int16_t*)data_vec_let)[cse_var_22]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_23])));
          ((int32_t*)conv_let)[cse_var_24] = (((int32_t*)conv_let)[cse_var_24] + (((int32_t)((int16_t*)data_vec_let)[cse_var_22]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_19])));
          ((int32_t*)conv_let)[cse_var_25] = (((int32_t*)conv_let)[cse_var_25] + (((int32_t)((int16_t*)data_vec_let)[cse_var_22]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
          ((int32_t*)conv_let)[cse_var_26] = (((int32_t*)conv_let)[cse_var_26] + (((int32_t)((int16_t*)data_vec_let)[cse_var_22]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
          ((int32_t*)conv_let)[cse_var_27] = (((int32_t*)conv_let)[cse_var_27] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_23])));
          ((int32_t*)conv_let)[cse_var_28] = (((int32_t*)conv_let)[cse_var_28] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_19])));
          ((int32_t*)conv_let)[cse_var_29] = (((int32_t*)conv_let)[cse_var_29] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
          ((int32_t*)conv_let)[cse_var_30] = (((int32_t*)conv_let)[cse_var_30] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 4; ++ax0_ax1_outer_fused) {
    for (int32_t ax3_outer = 0; ax3_outer < 16; ++ax3_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
        int32_t cse_var_40 = (ax0_ax1_outer_fused * 1024);
        int32_t cse_var_39 = (ax3_outer * 4);
        int32_t cse_var_38 = (cse_var_39 + 3);
        int32_t cse_var_37 = (cse_var_39 + 2);
        int32_t cse_var_36 = (cse_var_39 + 1);
        int32_t cse_var_35 = ((cse_var_40 + (ax3_outer * 64)) + (ax2_inner * 4));
        int32_t cse_var_34 = ((cse_var_40 + (ax2_inner * 64)) + cse_var_39);
        int32_t cse_var_33 = (cse_var_34 + 3);
        int32_t cse_var_32 = (cse_var_34 + 2);
        int32_t cse_var_31 = (cse_var_34 + 1);
        int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_35]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_39])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_39]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_39]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_39])) + 38;
        int32_t __2 = (__1) < (127) ? (__1) : (127);
        int32_t __3 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_34];
        int32_t __4 = (__3) < (127) ? (__3) : (127);
        int8_t __5 = (int8_t)((__4) > (-128) ? (__4) : (-128));
        int8_t __6 = (int8_t)127;
        int8_t __7 = (__5) < (__6) ? (__5) : (__6);
        int8_t __8 = (int8_t)-128;
        T_cast[cse_var_34] = ((int32_t)((__7) > (__8) ? (__7) : (__8)));
        int32_t __9 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_35 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_36])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_36]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_36]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_36])) + 38;
        int32_t __10 = (__9) < (127) ? (__9) : (127);
        int32_t __11 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_31];
        int32_t __12 = (__11) < (127) ? (__11) : (127);
        int8_t __13 = (int8_t)((__12) > (-128) ? (__12) : (-128));
        int8_t __14 = (__13) < (__6) ? (__13) : (__6);
        T_cast[cse_var_31] = ((int32_t)((__14) > (__8) ? (__14) : (__8)));
        int32_t __15 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_35 + 2)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_37])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_37]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_37]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_37])) + 38;
        int32_t __16 = (__15) < (127) ? (__15) : (127);
        int32_t __17 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_32];
        int32_t __18 = (__17) < (127) ? (__17) : (127);
        int8_t __19 = (int8_t)((__18) > (-128) ? (__18) : (-128));
        int8_t __20 = (__19) < (__6) ? (__19) : (__6);
        T_cast[cse_var_32] = ((int32_t)((__20) > (__8) ? (__20) : (__8)));
        int32_t __21 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_35 + 3)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_38])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_38]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_38]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_38])) + 38;
        int32_t __22 = (__21) < (127) ? (__21) : (127);
        int32_t __23 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__22) > (-128) ? (__22) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__22) > (-128) ? (__22) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_33];
        int32_t __24 = (__23) < (127) ? (__23) : (127);
        int8_t __25 = (int8_t)((__24) > (-128) ? (__24) : (-128));
        int8_t __26 = (__25) < (__6) ? (__25) : (__6);
        T_cast[cse_var_33] = ((int32_t)((__26) > (__8) ? (__26) : (__8)));
      }
      for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 8; ++ax2_inner_1) {
        int32_t cse_var_50 = (ax0_ax1_outer_fused * 1024);
        int32_t cse_var_49 = (ax3_outer * 4);
        int32_t cse_var_51 = ((cse_var_50 + (ax2_inner_1 * 64)) + cse_var_49);
        int32_t cse_var_48 = (cse_var_49 + 3);
        int32_t cse_var_47 = (cse_var_49 + 2);
        int32_t cse_var_46 = (cse_var_49 + 1);
        int32_t cse_var_45 = ((cse_var_50 + (ax3_outer * 64)) + (ax2_inner_1 * 4));
        int32_t cse_var_44 = (cse_var_51 + 515);
        int32_t cse_var_43 = (cse_var_51 + 514);
        int32_t cse_var_42 = (cse_var_51 + 513);
        int32_t cse_var_41 = (cse_var_51 + 512);
        int32_t __27 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_45 + 32)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_49])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_49]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_49]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_49])) + 38;
        int32_t __28 = (__27) < (127) ? (__27) : (127);
        int32_t __29 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__28) > (-128) ? (__28) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__28) > (-128) ? (__28) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_41];
        int32_t __30 = (__29) < (127) ? (__29) : (127);
        int8_t __31 = (int8_t)((__30) > (-128) ? (__30) : (-128));
        int8_t __32 = (int8_t)127;
        int8_t __33 = (__31) < (__32) ? (__31) : (__32);
        int8_t __34 = (int8_t)-128;
        T_cast[cse_var_41] = ((int32_t)((__33) > (__34) ? (__33) : (__34)));
        int32_t __35 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_45 + 33)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_46])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_46]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_46]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_46])) + 38;
        int32_t __36 = (__35) < (127) ? (__35) : (127);
        int32_t __37 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__36) > (-128) ? (__36) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__36) > (-128) ? (__36) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_42];
        int32_t __38 = (__37) < (127) ? (__37) : (127);
        int8_t __39 = (int8_t)((__38) > (-128) ? (__38) : (-128));
        int8_t __40 = (__39) < (__32) ? (__39) : (__32);
        T_cast[cse_var_42] = ((int32_t)((__40) > (__34) ? (__40) : (__34)));
        int32_t __41 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_45 + 34)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_47])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_47]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_47]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_47])) + 38;
        int32_t __42 = (__41) < (127) ? (__41) : (127);
        int32_t __43 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__42) > (-128) ? (__42) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__42) > (-128) ? (__42) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_43];
        int32_t __44 = (__43) < (127) ? (__43) : (127);
        int8_t __45 = (int8_t)((__44) > (-128) ? (__44) : (-128));
        int8_t __46 = (__45) < (__32) ? (__45) : (__32);
        T_cast[cse_var_43] = ((int32_t)((__46) > (__34) ? (__46) : (__34)));
        int32_t __47 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_45 + 35)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_46_let)[cse_var_48])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_47_let)[cse_var_48]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_48_let)[cse_var_48]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_49_let)[cse_var_48])) + 38;
        int32_t __48 = (__47) < (127) ? (__47) : (127);
        int32_t __49 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t)((int8_t)((__48) > (-128) ? (__48) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0])) << ((int64_t)0)) : ((int64_t)(((int32_t)((int8_t)((__48) > (-128) ? (__48) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_50_let)[0]))) * (int64_t)1417215292) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) + p1[cse_var_44];
        int32_t __50 = (__49) < (127) ? (__49) : (127);
        int8_t __51 = (int8_t)((__50) > (-128) ? (__50) : (-128));
        int8_t __52 = (__51) < (__32) ? (__51) : (__32);
        T_cast[cse_var_44] = ((int32_t)((__52) > (__34) ? (__52) : (__34)));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_9b1cea826623845_(int16_t* p0, int32_t* T_add, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let = (&(global_const_workspace_14_var[164224]));
  void* fused_nn_conv2d_add_cast_multiply_add_constant_26_let = (&(global_const_workspace_14_var[161120]));
  void* fused_nn_conv2d_add_cast_multiply_constant_25_let = (&(global_const_workspace_14_var[160352]));
  void* fused_nn_conv2d_add_cast_constant_24_let = (&(global_const_workspace_14_var[161888]));
  void* fused_nn_conv2d_constant_23_let = (&(global_const_workspace_14_var[162528]));
  void* fused_constant_22_let = (&(global_const_workspace_14_var[110592]));
  void* PadInput_let = (&(global_workspace_15_var[172320]));
  void* data_vec_let = (&(global_workspace_15_var[0]));
  void* conv_let = (&(global_workspace_15_var[139552]));
  for (int32_t i1 = 0; i1 < 18; ++i1) {
    for (int32_t i2 = 0; i2 < 18; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        int32_t cse_var_1 = (i2 * 32);
        ((int16_t*)PadInput_let)[(((i1 * 576) + cse_var_1) + i3)] = (((((1 <= i1) && (i1 < 17)) && (1 <= i2)) && (i2 < 17)) ? p0[((((i1 * 512) + cse_var_1) + i3) - 544)] : (int16_t)0);
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 4; ++n_oho_fused) {
    for (int32_t owo = 0; owo < 8; ++owo) {
      for (int32_t ohi = 0; ohi < 6; ++ohi) {
        for (int32_t ic = 0; ic < 32; ++ic) {
          ((int16_t*)data_vec_let)[((((n_oho_fused * 6144) + (owo * 768)) + (ohi * 128)) + ic)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2304) + (ohi * 576)) + (owo * 64)) + ic)];
        }
        for (int32_t ic_1 = 0; ic_1 < 32; ++ic_1) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 6144) + (owo * 768)) + (ohi * 128)) + ic_1) + 32)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2304) + (ohi * 576)) + (owo * 64)) + ic_1) + 32)];
        }
        for (int32_t ic_2 = 0; ic_2 < 32; ++ic_2) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 6144) + (owo * 768)) + (ohi * 128)) + ic_2) + 64)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2304) + (ohi * 576)) + (owo * 64)) + ic_2) + 64)];
        }
        for (int32_t ic_3 = 0; ic_3 < 32; ++ic_3) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 6144) + (owo * 768)) + (ohi * 128)) + ic_3) + 96)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2304) + (ohi * 576)) + (owo * 64)) + ic_3) + 96)];
        }
      }
    }
  }
  for (int32_t oco = 0; oco < 16; ++oco) {
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t oci = 0; oci < 2; ++oci) {
          ((int16_t*)PadInput_let)[((((oco * 576) + (kh * 192)) + (kw * 64)) + oci)] = ((int16_t*)fused_constant_22_let)[((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci)];
        }
        for (int32_t oci_1 = 0; oci_1 < 2; ++oci_1) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_1) + 2)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_1) + 32)];
        }
        for (int32_t oci_2 = 0; oci_2 < 2; ++oci_2) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_2) + 4)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_2) + 64)];
        }
        for (int32_t oci_3 = 0; oci_3 < 2; ++oci_3) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_3) + 6)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_3) + 96)];
        }
        for (int32_t oci_4 = 0; oci_4 < 2; ++oci_4) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_4) + 8)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_4) + 128)];
        }
        for (int32_t oci_5 = 0; oci_5 < 2; ++oci_5) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_5) + 10)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_5) + 160)];
        }
        for (int32_t oci_6 = 0; oci_6 < 2; ++oci_6) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_6) + 12)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_6) + 192)];
        }
        for (int32_t oci_7 = 0; oci_7 < 2; ++oci_7) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_7) + 14)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_7) + 224)];
        }
        for (int32_t oci_8 = 0; oci_8 < 2; ++oci_8) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_8) + 16)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_8) + 256)];
        }
        for (int32_t oci_9 = 0; oci_9 < 2; ++oci_9) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_9) + 18)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_9) + 288)];
        }
        for (int32_t oci_10 = 0; oci_10 < 2; ++oci_10) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_10) + 20)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_10) + 320)];
        }
        for (int32_t oci_11 = 0; oci_11 < 2; ++oci_11) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_11) + 22)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_11) + 352)];
        }
        for (int32_t oci_12 = 0; oci_12 < 2; ++oci_12) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_12) + 24)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_12) + 384)];
        }
        for (int32_t oci_13 = 0; oci_13 < 2; ++oci_13) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_13) + 26)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_13) + 416)];
        }
        for (int32_t oci_14 = 0; oci_14 < 2; ++oci_14) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_14) + 28)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_14) + 448)];
        }
        for (int32_t oci_15 = 0; oci_15 < 2; ++oci_15) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_15) + 30)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_15) + 480)];
        }
        for (int32_t oci_16 = 0; oci_16 < 2; ++oci_16) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_16) + 32)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_16) + 512)];
        }
        for (int32_t oci_17 = 0; oci_17 < 2; ++oci_17) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_17) + 34)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_17) + 544)];
        }
        for (int32_t oci_18 = 0; oci_18 < 2; ++oci_18) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_18) + 36)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_18) + 576)];
        }
        for (int32_t oci_19 = 0; oci_19 < 2; ++oci_19) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_19) + 38)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_19) + 608)];
        }
        for (int32_t oci_20 = 0; oci_20 < 2; ++oci_20) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_20) + 40)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_20) + 640)];
        }
        for (int32_t oci_21 = 0; oci_21 < 2; ++oci_21) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_21) + 42)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_21) + 672)];
        }
        for (int32_t oci_22 = 0; oci_22 < 2; ++oci_22) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_22) + 44)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_22) + 704)];
        }
        for (int32_t oci_23 = 0; oci_23 < 2; ++oci_23) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_23) + 46)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_23) + 736)];
        }
        for (int32_t oci_24 = 0; oci_24 < 2; ++oci_24) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_24) + 48)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_24) + 768)];
        }
        for (int32_t oci_25 = 0; oci_25 < 2; ++oci_25) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_25) + 50)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_25) + 800)];
        }
        for (int32_t oci_26 = 0; oci_26 < 2; ++oci_26) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_26) + 52)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_26) + 832)];
        }
        for (int32_t oci_27 = 0; oci_27 < 2; ++oci_27) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_27) + 54)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_27) + 864)];
        }
        for (int32_t oci_28 = 0; oci_28 < 2; ++oci_28) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_28) + 56)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_28) + 896)];
        }
        for (int32_t oci_29 = 0; oci_29 < 2; ++oci_29) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_29) + 58)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_29) + 928)];
        }
        for (int32_t oci_30 = 0; oci_30 < 2; ++oci_30) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_30) + 60)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_30) + 960)];
        }
        for (int32_t oci_31 = 0; oci_31 < 2; ++oci_31) {
          ((int16_t*)PadInput_let)[(((((oco * 576) + (kh * 192)) + (kw * 64)) + oci_31) + 62)] = ((int16_t*)fused_constant_22_let)[(((((kh * 3072) + (kw * 1024)) + (oco * 2)) + oci_31) + 992)];
        }
      }
    }
  }
  for (int32_t oho = 0; oho < 4; ++oho) {
    for (int32_t owo_1 = 0; owo_1 < 8; ++owo_1) {
      for (int32_t owi = 0; owi < 2; ++owi) {
        for (int32_t oco_1 = 0; oco_1 < 16; ++oco_1) {
          int32_t cse_var_2 = ((((oho * 2048) + (owo_1 * 256)) + (oco_1 * 16)) + (owi * 2));
          ((int32_t*)conv_let)[cse_var_2] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 1)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 4)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 5)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 8)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 9)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 12)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 13)] = 0;
          for (int32_t kh_1 = 0; kh_1 < 3; ++kh_1) {
            for (int32_t kw_1 = 0; kw_1 < 3; ++kw_1) {
              for (int32_t ic_4 = 0; ic_4 < 32; ++ic_4) {
                int32_t cse_var_15 = (cse_var_2 + 9);
                int32_t cse_var_14 = (cse_var_2 + 8);
                int32_t cse_var_13 = (cse_var_2 + 5);
                int32_t cse_var_12 = (cse_var_2 + 4);
                int32_t cse_var_11 = (cse_var_2 + 13);
                int32_t cse_var_10 = (cse_var_2 + 12);
                int32_t cse_var_9 = (cse_var_2 + 1);
                int32_t cse_var_8 = ((((oco_1 * 576) + (kh_1 * 192)) + (kw_1 * 64)) + (ic_4 * 2));
                int32_t cse_var_7 = (cse_var_8 + 1);
                int32_t cse_var_6 = ((((((oho * 6144) + (owo_1 * 768)) + (kh_1 * 128)) + (owi * 32)) + (kw_1 * 32)) + ic_4);
                int32_t cse_var_5 = (cse_var_6 + 384);
                int32_t cse_var_4 = (cse_var_6 + 256);
                int32_t cse_var_3 = (cse_var_6 + 128);
                ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_9] = (((int32_t*)conv_let)[cse_var_9] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
                ((int32_t*)conv_let)[cse_var_12] = (((int32_t*)conv_let)[cse_var_12] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_13] = (((int32_t*)conv_let)[cse_var_13] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
                ((int32_t*)conv_let)[cse_var_14] = (((int32_t*)conv_let)[cse_var_14] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_15] = (((int32_t*)conv_let)[cse_var_15] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
                ((int32_t*)conv_let)[cse_var_10] = (((int32_t*)conv_let)[cse_var_10] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_11] = (((int32_t*)conv_let)[cse_var_11] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
              }
            }
          }
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 4; ++ax0_ax1_outer_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 8; ++ax2_outer) {
      for (int32_t ax3_outer = 0; ax3_outer < 16; ++ax3_outer) {
        for (int32_t ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
          int32_t cse_var_20 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_19 = (ax3_outer * 2);
          int32_t cse_var_18 = (cse_var_19 + 1);
          int32_t cse_var_17 = (((cse_var_20 + (ax2_outer * 64)) + (ax2_inner * 32)) + cse_var_19);
          int32_t cse_var_16 = (((cse_var_20 + (ax2_outer * 256)) + (ax3_outer * 16)) + (ax2_inner * 2));
          int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_16]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_19])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_19]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_19]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_19])) + 4;
          int32_t __2 = (__1) < (127) ? (__1) : (127);
          T_add[cse_var_17] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __3 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_16 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_18])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_18]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_18]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_18])) + 4;
          int32_t __4 = (__3) < (127) ? (__3) : (127);
          T_add[(cse_var_17 + 1)] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__4) > (-128) ? (__4) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__4) > (-128) ? (__4) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
        for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 2; ++ax2_inner_1) {
          int32_t cse_var_25 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_24 = (ax3_outer * 2);
          int32_t cse_var_23 = (cse_var_24 + 1);
          int32_t cse_var_22 = (((cse_var_25 + (ax2_outer * 64)) + (ax2_inner_1 * 32)) + cse_var_24);
          int32_t cse_var_21 = (((cse_var_25 + (ax2_outer * 256)) + (ax3_outer * 16)) + (ax2_inner_1 * 2));
          int32_t __5 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_21 + 4)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_24])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_24]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_24]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_24])) + 4;
          int32_t __6 = (__5) < (127) ? (__5) : (127);
          T_add[(cse_var_22 + 512)] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__6) > (-128) ? (__6) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__6) > (-128) ? (__6) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __7 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_21 + 5)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_23])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_23]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_23]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_23])) + 4;
          int32_t __8 = (__7) < (127) ? (__7) : (127);
          T_add[(cse_var_22 + 513)] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__8) > (-128) ? (__8) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__8) > (-128) ? (__8) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
        for (int32_t ax2_inner_2 = 0; ax2_inner_2 < 2; ++ax2_inner_2) {
          int32_t cse_var_30 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_29 = (ax3_outer * 2);
          int32_t cse_var_28 = (cse_var_29 + 1);
          int32_t cse_var_27 = (((cse_var_30 + (ax2_outer * 64)) + (ax2_inner_2 * 32)) + cse_var_29);
          int32_t cse_var_26 = (((cse_var_30 + (ax2_outer * 256)) + (ax3_outer * 16)) + (ax2_inner_2 * 2));
          int32_t __9 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_26 + 8)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_29])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_29]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_29]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_29])) + 4;
          int32_t __10 = (__9) < (127) ? (__9) : (127);
          T_add[(cse_var_27 + 1024)] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __11 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_26 + 9)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_28])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_28]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_28]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_28])) + 4;
          int32_t __12 = (__11) < (127) ? (__11) : (127);
          T_add[(cse_var_27 + 1025)] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__12) > (-128) ? (__12) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__12) > (-128) ? (__12) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
        for (int32_t ax2_inner_3 = 0; ax2_inner_3 < 2; ++ax2_inner_3) {
          int32_t cse_var_35 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_34 = (ax3_outer * 2);
          int32_t cse_var_33 = (cse_var_34 + 1);
          int32_t cse_var_32 = (((cse_var_35 + (ax2_outer * 64)) + (ax2_inner_3 * 32)) + cse_var_34);
          int32_t cse_var_31 = (((cse_var_35 + (ax2_outer * 256)) + (ax3_outer * 16)) + (ax2_inner_3 * 2));
          int32_t __13 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_31 + 12)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_34])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_34]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_34]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_34])) + 4;
          int32_t __14 = (__13) < (127) ? (__13) : (127);
          T_add[(cse_var_32 + 1536)] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__14) > (-128) ? (__14) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__14) > (-128) ? (__14) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __15 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_31 + 13)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_23_let)[cse_var_33])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_24_let)[cse_var_33]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_25_let)[cse_var_33]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_26_let)[cse_var_33])) + 4;
          int32_t __16 = (__15) < (127) ? (__15) : (127);
          T_add[(cse_var_32 + 1537)] = (((int32_t)(((((2 != 0) ? (((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0])) << ((int64_t)2)) : ((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_27_let)[0]))) * (int64_t)1140768826) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_9b1cea826623845__1(int16_t* p0, int32_t* T_add, uint8_t* global_const_workspace_20_var, uint8_t* global_workspace_21_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let = (&(global_const_workspace_20_var[164192]));
  void* fused_nn_conv2d_add_cast_multiply_add_constant_43_let = (&(global_const_workspace_20_var[156768]));
  void* fused_nn_conv2d_add_cast_multiply_constant_42_let = (&(global_const_workspace_20_var[155232]));
  void* fused_nn_conv2d_add_cast_constant_41_let = (&(global_const_workspace_20_var[158304]));
  void* fused_nn_conv2d_constant_40_let = (&(global_const_workspace_20_var[159584]));
  void* fused_constant_39_let = (&(global_const_workspace_20_var[0]));
  void* PadInput_let = (&(global_workspace_21_var[0]));
  void* data_vec_let = (&(global_workspace_21_var[73728]));
  void* conv_let = (&(global_workspace_21_var[205088]));
  for (int32_t i1 = 0; i1 < 10; ++i1) {
    for (int32_t i2 = 0; i2 < 10; ++i2) {
      for (int32_t i3 = 0; i3 < 64; ++i3) {
        int32_t cse_var_1 = (i2 * 64);
        ((int16_t*)PadInput_let)[(((i1 * 640) + cse_var_1) + i3)] = (((((1 <= i1) && (i1 < 9)) && (1 <= i2)) && (i2 < 9)) ? p0[((((i1 * 512) + cse_var_1) + i3) - 576)] : (int16_t)0);
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 2; ++n_oho_fused) {
    for (int32_t owo = 0; owo < 4; ++owo) {
      for (int32_t ohi = 0; ohi < 6; ++ohi) {
        for (int32_t ic = 0; ic < 64; ++ic) {
          ((int16_t*)data_vec_let)[((((n_oho_fused * 6144) + (owo * 1536)) + (ohi * 256)) + ic)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2560) + (ohi * 640)) + (owo * 128)) + ic)];
        }
        for (int32_t ic_1 = 0; ic_1 < 64; ++ic_1) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 6144) + (owo * 1536)) + (ohi * 256)) + ic_1) + 64)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2560) + (ohi * 640)) + (owo * 128)) + ic_1) + 64)];
        }
        for (int32_t ic_2 = 0; ic_2 < 64; ++ic_2) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 6144) + (owo * 1536)) + (ohi * 256)) + ic_2) + 128)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2560) + (ohi * 640)) + (owo * 128)) + ic_2) + 128)];
        }
        for (int32_t ic_3 = 0; ic_3 < 64; ++ic_3) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 6144) + (owo * 1536)) + (ohi * 256)) + ic_3) + 192)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2560) + (ohi * 640)) + (owo * 128)) + ic_3) + 192)];
        }
      }
    }
  }
  for (int32_t oco = 0; oco < 32; ++oco) {
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t oci = 0; oci < 2; ++oci) {
          ((int16_t*)PadInput_let)[((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci)] = ((int16_t*)fused_constant_39_let)[((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci)];
        }
        for (int32_t oci_1 = 0; oci_1 < 2; ++oci_1) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_1) + 2)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_1) + 64)];
        }
        for (int32_t oci_2 = 0; oci_2 < 2; ++oci_2) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_2) + 4)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_2) + 128)];
        }
        for (int32_t oci_3 = 0; oci_3 < 2; ++oci_3) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_3) + 6)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_3) + 192)];
        }
        for (int32_t oci_4 = 0; oci_4 < 2; ++oci_4) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_4) + 8)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_4) + 256)];
        }
        for (int32_t oci_5 = 0; oci_5 < 2; ++oci_5) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_5) + 10)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_5) + 320)];
        }
        for (int32_t oci_6 = 0; oci_6 < 2; ++oci_6) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_6) + 12)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_6) + 384)];
        }
        for (int32_t oci_7 = 0; oci_7 < 2; ++oci_7) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_7) + 14)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_7) + 448)];
        }
        for (int32_t oci_8 = 0; oci_8 < 2; ++oci_8) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_8) + 16)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_8) + 512)];
        }
        for (int32_t oci_9 = 0; oci_9 < 2; ++oci_9) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_9) + 18)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_9) + 576)];
        }
        for (int32_t oci_10 = 0; oci_10 < 2; ++oci_10) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_10) + 20)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_10) + 640)];
        }
        for (int32_t oci_11 = 0; oci_11 < 2; ++oci_11) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_11) + 22)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_11) + 704)];
        }
        for (int32_t oci_12 = 0; oci_12 < 2; ++oci_12) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_12) + 24)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_12) + 768)];
        }
        for (int32_t oci_13 = 0; oci_13 < 2; ++oci_13) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_13) + 26)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_13) + 832)];
        }
        for (int32_t oci_14 = 0; oci_14 < 2; ++oci_14) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_14) + 28)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_14) + 896)];
        }
        for (int32_t oci_15 = 0; oci_15 < 2; ++oci_15) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_15) + 30)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_15) + 960)];
        }
        for (int32_t oci_16 = 0; oci_16 < 2; ++oci_16) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_16) + 32)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_16) + 1024)];
        }
        for (int32_t oci_17 = 0; oci_17 < 2; ++oci_17) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_17) + 34)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_17) + 1088)];
        }
        for (int32_t oci_18 = 0; oci_18 < 2; ++oci_18) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_18) + 36)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_18) + 1152)];
        }
        for (int32_t oci_19 = 0; oci_19 < 2; ++oci_19) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_19) + 38)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_19) + 1216)];
        }
        for (int32_t oci_20 = 0; oci_20 < 2; ++oci_20) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_20) + 40)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_20) + 1280)];
        }
        for (int32_t oci_21 = 0; oci_21 < 2; ++oci_21) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_21) + 42)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_21) + 1344)];
        }
        for (int32_t oci_22 = 0; oci_22 < 2; ++oci_22) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_22) + 44)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_22) + 1408)];
        }
        for (int32_t oci_23 = 0; oci_23 < 2; ++oci_23) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_23) + 46)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_23) + 1472)];
        }
        for (int32_t oci_24 = 0; oci_24 < 2; ++oci_24) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_24) + 48)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_24) + 1536)];
        }
        for (int32_t oci_25 = 0; oci_25 < 2; ++oci_25) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_25) + 50)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_25) + 1600)];
        }
        for (int32_t oci_26 = 0; oci_26 < 2; ++oci_26) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_26) + 52)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_26) + 1664)];
        }
        for (int32_t oci_27 = 0; oci_27 < 2; ++oci_27) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_27) + 54)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_27) + 1728)];
        }
        for (int32_t oci_28 = 0; oci_28 < 2; ++oci_28) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_28) + 56)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_28) + 1792)];
        }
        for (int32_t oci_29 = 0; oci_29 < 2; ++oci_29) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_29) + 58)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_29) + 1856)];
        }
        for (int32_t oci_30 = 0; oci_30 < 2; ++oci_30) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_30) + 60)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_30) + 1920)];
        }
        for (int32_t oci_31 = 0; oci_31 < 2; ++oci_31) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_31) + 62)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_31) + 1984)];
        }
        for (int32_t oci_32 = 0; oci_32 < 2; ++oci_32) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_32) + 64)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_32) + 2048)];
        }
        for (int32_t oci_33 = 0; oci_33 < 2; ++oci_33) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_33) + 66)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_33) + 2112)];
        }
        for (int32_t oci_34 = 0; oci_34 < 2; ++oci_34) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_34) + 68)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_34) + 2176)];
        }
        for (int32_t oci_35 = 0; oci_35 < 2; ++oci_35) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_35) + 70)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_35) + 2240)];
        }
        for (int32_t oci_36 = 0; oci_36 < 2; ++oci_36) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_36) + 72)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_36) + 2304)];
        }
        for (int32_t oci_37 = 0; oci_37 < 2; ++oci_37) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_37) + 74)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_37) + 2368)];
        }
        for (int32_t oci_38 = 0; oci_38 < 2; ++oci_38) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_38) + 76)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_38) + 2432)];
        }
        for (int32_t oci_39 = 0; oci_39 < 2; ++oci_39) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_39) + 78)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_39) + 2496)];
        }
        for (int32_t oci_40 = 0; oci_40 < 2; ++oci_40) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_40) + 80)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_40) + 2560)];
        }
        for (int32_t oci_41 = 0; oci_41 < 2; ++oci_41) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_41) + 82)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_41) + 2624)];
        }
        for (int32_t oci_42 = 0; oci_42 < 2; ++oci_42) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_42) + 84)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_42) + 2688)];
        }
        for (int32_t oci_43 = 0; oci_43 < 2; ++oci_43) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_43) + 86)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_43) + 2752)];
        }
        for (int32_t oci_44 = 0; oci_44 < 2; ++oci_44) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_44) + 88)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_44) + 2816)];
        }
        for (int32_t oci_45 = 0; oci_45 < 2; ++oci_45) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_45) + 90)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_45) + 2880)];
        }
        for (int32_t oci_46 = 0; oci_46 < 2; ++oci_46) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_46) + 92)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_46) + 2944)];
        }
        for (int32_t oci_47 = 0; oci_47 < 2; ++oci_47) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_47) + 94)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_47) + 3008)];
        }
        for (int32_t oci_48 = 0; oci_48 < 2; ++oci_48) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_48) + 96)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_48) + 3072)];
        }
        for (int32_t oci_49 = 0; oci_49 < 2; ++oci_49) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_49) + 98)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_49) + 3136)];
        }
        for (int32_t oci_50 = 0; oci_50 < 2; ++oci_50) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_50) + 100)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_50) + 3200)];
        }
        for (int32_t oci_51 = 0; oci_51 < 2; ++oci_51) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_51) + 102)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_51) + 3264)];
        }
        for (int32_t oci_52 = 0; oci_52 < 2; ++oci_52) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_52) + 104)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_52) + 3328)];
        }
        for (int32_t oci_53 = 0; oci_53 < 2; ++oci_53) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_53) + 106)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_53) + 3392)];
        }
        for (int32_t oci_54 = 0; oci_54 < 2; ++oci_54) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_54) + 108)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_54) + 3456)];
        }
        for (int32_t oci_55 = 0; oci_55 < 2; ++oci_55) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_55) + 110)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_55) + 3520)];
        }
        for (int32_t oci_56 = 0; oci_56 < 2; ++oci_56) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_56) + 112)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_56) + 3584)];
        }
        for (int32_t oci_57 = 0; oci_57 < 2; ++oci_57) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_57) + 114)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_57) + 3648)];
        }
        for (int32_t oci_58 = 0; oci_58 < 2; ++oci_58) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_58) + 116)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_58) + 3712)];
        }
        for (int32_t oci_59 = 0; oci_59 < 2; ++oci_59) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_59) + 118)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_59) + 3776)];
        }
        for (int32_t oci_60 = 0; oci_60 < 2; ++oci_60) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_60) + 120)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_60) + 3840)];
        }
        for (int32_t oci_61 = 0; oci_61 < 2; ++oci_61) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_61) + 122)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_61) + 3904)];
        }
        for (int32_t oci_62 = 0; oci_62 < 2; ++oci_62) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_62) + 124)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_62) + 3968)];
        }
        for (int32_t oci_63 = 0; oci_63 < 2; ++oci_63) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_63) + 126)] = ((int16_t*)fused_constant_39_let)[(((((kh * 12288) + (kw * 4096)) + (oco * 2)) + oci_63) + 4032)];
        }
      }
    }
  }
  for (int32_t oho = 0; oho < 2; ++oho) {
    for (int32_t owo_1 = 0; owo_1 < 4; ++owo_1) {
      for (int32_t owi = 0; owi < 2; ++owi) {
        for (int32_t oco_1 = 0; oco_1 < 32; ++oco_1) {
          int32_t cse_var_2 = ((((oho * 2048) + (owo_1 * 512)) + (oco_1 * 16)) + (owi * 2));
          ((int32_t*)conv_let)[cse_var_2] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 1)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 4)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 5)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 8)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 9)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 12)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 13)] = 0;
          for (int32_t kh_1 = 0; kh_1 < 3; ++kh_1) {
            for (int32_t kw_1 = 0; kw_1 < 3; ++kw_1) {
              for (int32_t ic_4 = 0; ic_4 < 64; ++ic_4) {
                int32_t cse_var_15 = (cse_var_2 + 9);
                int32_t cse_var_14 = (cse_var_2 + 8);
                int32_t cse_var_13 = (cse_var_2 + 5);
                int32_t cse_var_12 = (cse_var_2 + 4);
                int32_t cse_var_11 = (cse_var_2 + 13);
                int32_t cse_var_10 = (cse_var_2 + 12);
                int32_t cse_var_9 = (cse_var_2 + 1);
                int32_t cse_var_8 = ((((oco_1 * 1152) + (kh_1 * 384)) + (kw_1 * 128)) + (ic_4 * 2));
                int32_t cse_var_7 = (cse_var_8 + 1);
                int32_t cse_var_6 = ((((((oho * 6144) + (owo_1 * 1536)) + (kh_1 * 256)) + (owi * 64)) + (kw_1 * 64)) + ic_4);
                int32_t cse_var_5 = (cse_var_6 + 768);
                int32_t cse_var_4 = (cse_var_6 + 512);
                int32_t cse_var_3 = (cse_var_6 + 256);
                ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_9] = (((int32_t*)conv_let)[cse_var_9] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
                ((int32_t*)conv_let)[cse_var_12] = (((int32_t*)conv_let)[cse_var_12] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_13] = (((int32_t*)conv_let)[cse_var_13] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
                ((int32_t*)conv_let)[cse_var_14] = (((int32_t*)conv_let)[cse_var_14] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_15] = (((int32_t*)conv_let)[cse_var_15] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
                ((int32_t*)conv_let)[cse_var_10] = (((int32_t*)conv_let)[cse_var_10] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
                ((int32_t*)conv_let)[cse_var_11] = (((int32_t*)conv_let)[cse_var_11] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
              }
            }
          }
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 2; ++ax0_ax1_outer_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax3_outer = 0; ax3_outer < 32; ++ax3_outer) {
        for (int32_t ax2_inner = 0; ax2_inner < 2; ++ax2_inner) {
          int32_t cse_var_20 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_19 = (ax3_outer * 2);
          int32_t cse_var_18 = (cse_var_19 + 1);
          int32_t cse_var_17 = (((cse_var_20 + (ax2_outer * 512)) + (ax3_outer * 16)) + (ax2_inner * 2));
          int32_t cse_var_16 = (((cse_var_20 + (ax2_outer * 128)) + (ax2_inner * 64)) + cse_var_19);
          int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_17]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_19])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_19]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_19]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_19])) - 2;
          int32_t __2 = (__1) < (127) ? (__1) : (127);
          T_add[cse_var_16] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __3 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_17 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_18])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_18]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_18]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_18])) - 2;
          int32_t __4 = (__3) < (127) ? (__3) : (127);
          T_add[(cse_var_16 + 1)] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__4) > (-128) ? (__4) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__4) > (-128) ? (__4) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
        for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 2; ++ax2_inner_1) {
          int32_t cse_var_25 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_24 = (ax3_outer * 2);
          int32_t cse_var_23 = (cse_var_24 + 1);
          int32_t cse_var_22 = (((cse_var_25 + (ax2_outer * 512)) + (ax3_outer * 16)) + (ax2_inner_1 * 2));
          int32_t cse_var_21 = (((cse_var_25 + (ax2_outer * 128)) + (ax2_inner_1 * 64)) + cse_var_24);
          int32_t __5 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_22 + 4)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_24])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_24]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_24]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_24])) - 2;
          int32_t __6 = (__5) < (127) ? (__5) : (127);
          T_add[(cse_var_21 + 512)] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__6) > (-128) ? (__6) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__6) > (-128) ? (__6) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __7 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_22 + 5)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_23])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_23]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_23]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_23])) - 2;
          int32_t __8 = (__7) < (127) ? (__7) : (127);
          T_add[(cse_var_21 + 513)] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__8) > (-128) ? (__8) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__8) > (-128) ? (__8) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
        for (int32_t ax2_inner_2 = 0; ax2_inner_2 < 2; ++ax2_inner_2) {
          int32_t cse_var_30 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_29 = (ax3_outer * 2);
          int32_t cse_var_28 = (cse_var_29 + 1);
          int32_t cse_var_27 = (((cse_var_30 + (ax2_outer * 512)) + (ax3_outer * 16)) + (ax2_inner_2 * 2));
          int32_t cse_var_26 = (((cse_var_30 + (ax2_outer * 128)) + (ax2_inner_2 * 64)) + cse_var_29);
          int32_t __9 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_27 + 8)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_29])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_29]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_29]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_29])) - 2;
          int32_t __10 = (__9) < (127) ? (__9) : (127);
          T_add[(cse_var_26 + 1024)] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__10) > (-128) ? (__10) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __11 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_27 + 9)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_28])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_28]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_28]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_28])) - 2;
          int32_t __12 = (__11) < (127) ? (__11) : (127);
          T_add[(cse_var_26 + 1025)] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__12) > (-128) ? (__12) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__12) > (-128) ? (__12) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
        for (int32_t ax2_inner_3 = 0; ax2_inner_3 < 2; ++ax2_inner_3) {
          int32_t cse_var_35 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_34 = (ax3_outer * 2);
          int32_t cse_var_33 = (cse_var_34 + 1);
          int32_t cse_var_32 = (((cse_var_35 + (ax2_outer * 512)) + (ax3_outer * 16)) + (ax2_inner_3 * 2));
          int32_t cse_var_31 = (((cse_var_35 + (ax2_outer * 128)) + (ax2_inner_3 * 64)) + cse_var_34);
          int32_t __13 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_32 + 12)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_34])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_34]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_34]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_34])) - 2;
          int32_t __14 = (__13) < (127) ? (__13) : (127);
          T_add[(cse_var_31 + 1536)] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__14) > (-128) ? (__14) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__14) > (-128) ? (__14) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
          int32_t __15 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_32 + 13)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_40_let)[cse_var_33])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_41_let)[cse_var_33]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_42_let)[cse_var_33]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_43_let)[cse_var_33])) - 2;
          int32_t __16 = (__15) < (127) ? (__15) : (127);
          T_add[(cse_var_31 + 1537)] = (((int32_t)(((((1 != 0) ? (((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0])) << ((int64_t)1)) : ((int64_t)(((int32_t)((int8_t)((__16) > (-128) ? (__16) : (-128)))) - ((int32_t*)fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_constant_44_let)[0]))) * (int64_t)1835721671) + ((int64_t)1 << ((int64_t)((0 + 31) - 1)))) >> ((int64_t)(0 + 31)))) - 128);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip(int16_t* p0, int8_t* compute, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_constant_4_let = (&(global_const_workspace_4_var[163168]));
  void* fused_nn_conv2d_add_cast_multiply_constant_3_let = (&(global_const_workspace_4_var[162912]));
  void* fused_nn_conv2d_add_cast_constant_2_let = (&(global_const_workspace_4_var[163424]));
  void* fused_nn_conv2d_constant_1_let = (&(global_const_workspace_4_var[164000]));
  void* fused_constant_0_let = (&(global_const_workspace_4_var[153856]));
  void* PadInput_let = (&(global_workspace_5_var[77056]));
  void* data_vec_let = (&(global_workspace_5_var[65536]));
  void* conv_let = (&(global_workspace_5_var[0]));
  for (int32_t i1 = 0; i1 < 34; ++i1) {
    for (int32_t i2 = 0; i2 < 34; ++i2) {
      for (int32_t i3 = 0; i3 < 3; ++i3) {
        int32_t cse_var_1 = (i2 * 3);
        ((int16_t*)PadInput_let)[(((i1 * 102) + cse_var_1) + i3)] = (((((1 <= i1) && (i1 < 33)) && (1 <= i2)) && (i2 < 33)) ? p0[((((i1 * 96) + cse_var_1) + i3) - 99)] : (int16_t)0);
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 8; ++n_oho_fused) {
    for (int32_t owo = 0; owo < 4; ++owo) {
      for (int32_t ohi = 0; ohi < 6; ++ohi) {
        for (int32_t ic = 0; ic < 3; ++ic) {
          ((int16_t*)data_vec_let)[((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic)];
        }
        for (int32_t ic_1 = 0; ic_1 < 3; ++ic_1) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_1) + 3)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_1) + 3)];
        }
        for (int32_t ic_2 = 0; ic_2 < 3; ++ic_2) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_2) + 6)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_2) + 6)];
        }
        for (int32_t ic_3 = 0; ic_3 < 3; ++ic_3) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_3) + 9)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_3) + 9)];
        }
        for (int32_t ic_4 = 0; ic_4 < 3; ++ic_4) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_4) + 12)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_4) + 12)];
        }
        for (int32_t ic_5 = 0; ic_5 < 3; ++ic_5) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_5) + 15)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_5) + 15)];
        }
        for (int32_t ic_6 = 0; ic_6 < 3; ++ic_6) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_6) + 18)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_6) + 18)];
        }
        for (int32_t ic_7 = 0; ic_7 < 3; ++ic_7) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_7) + 21)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_7) + 21)];
        }
        for (int32_t ic_8 = 0; ic_8 < 3; ++ic_8) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_8) + 24)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_8) + 24)];
        }
        for (int32_t ic_9 = 0; ic_9 < 3; ++ic_9) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 720) + (owo * 180)) + (ohi * 30)) + ic_9) + 27)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 408) + (ohi * 102)) + (owo * 24)) + ic_9) + 27)];
        }
      }
    }
  }
  for (int32_t oco = 0; oco < 8; ++oco) {
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t oci = 0; oci < 2; ++oci) {
          ((int16_t*)PadInput_let)[((((oco * 54) + (kh * 18)) + (kw * 6)) + oci)] = ((int16_t*)fused_constant_0_let)[((((kh * 144) + (kw * 48)) + (oco * 2)) + oci)];
        }
        for (int32_t oci_1 = 0; oci_1 < 2; ++oci_1) {
          ((int16_t*)PadInput_let)[(((((oco * 54) + (kh * 18)) + (kw * 6)) + oci_1) + 2)] = ((int16_t*)fused_constant_0_let)[(((((kh * 144) + (kw * 48)) + (oco * 2)) + oci_1) + 16)];
        }
        for (int32_t oci_2 = 0; oci_2 < 2; ++oci_2) {
          ((int16_t*)PadInput_let)[(((((oco * 54) + (kh * 18)) + (kw * 6)) + oci_2) + 4)] = ((int16_t*)fused_constant_0_let)[(((((kh * 144) + (kw * 48)) + (oco * 2)) + oci_2) + 32)];
        }
      }
    }
  }
  for (int32_t oho = 0; oho < 8; ++oho) {
    for (int32_t owo_1 = 0; owo_1 < 4; ++owo_1) {
      for (int32_t oco_1 = 0; oco_1 < 8; ++oco_1) {
        for (int32_t owi = 0; owi < 8; ++owi) {
          int32_t cse_var_2 = ((((oho * 2048) + (owo_1 * 512)) + (oco_1 * 64)) + (owi * 2));
          ((int32_t*)conv_let)[cse_var_2] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 1)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 16)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 17)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 32)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 33)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 48)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 49)] = 0;
          for (int32_t ic_10 = 0; ic_10 < 3; ++ic_10) {
            int32_t cse_var_15 = (cse_var_2 + 49);
            int32_t cse_var_14 = (cse_var_2 + 48);
            int32_t cse_var_13 = (cse_var_2 + 33);
            int32_t cse_var_12 = (cse_var_2 + 32);
            int32_t cse_var_11 = (cse_var_2 + 17);
            int32_t cse_var_10 = (cse_var_2 + 16);
            int32_t cse_var_9 = (cse_var_2 + 1);
            int32_t cse_var_8 = ((oco_1 * 54) + (ic_10 * 2));
            int32_t cse_var_7 = (cse_var_8 + 1);
            int32_t cse_var_6 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_10);
            int32_t cse_var_5 = (cse_var_6 + 90);
            int32_t cse_var_4 = (cse_var_6 + 60);
            int32_t cse_var_3 = (cse_var_6 + 30);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_9] = (((int32_t*)conv_let)[cse_var_9] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_10] = (((int32_t*)conv_let)[cse_var_10] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_11] = (((int32_t*)conv_let)[cse_var_11] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_12] = (((int32_t*)conv_let)[cse_var_12] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_13] = (((int32_t*)conv_let)[cse_var_13] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_14] = (((int32_t*)conv_let)[cse_var_14] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_15] = (((int32_t*)conv_let)[cse_var_15] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
          }
          for (int32_t ic_11 = 0; ic_11 < 3; ++ic_11) {
            int32_t cse_var_30 = ((oco_1 * 54) + (ic_11 * 2));
            int32_t cse_var_29 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_11);
            int32_t cse_var_28 = (cse_var_2 + 49);
            int32_t cse_var_27 = (cse_var_2 + 48);
            int32_t cse_var_26 = (cse_var_2 + 33);
            int32_t cse_var_25 = (cse_var_2 + 32);
            int32_t cse_var_24 = (cse_var_2 + 17);
            int32_t cse_var_23 = (cse_var_2 + 16);
            int32_t cse_var_22 = (cse_var_2 + 1);
            int32_t cse_var_21 = (cse_var_30 + 7);
            int32_t cse_var_20 = (cse_var_30 + 6);
            int32_t cse_var_19 = (cse_var_29 + 93);
            int32_t cse_var_18 = (cse_var_29 + 63);
            int32_t cse_var_17 = (cse_var_29 + 33);
            int32_t cse_var_16 = (cse_var_29 + 3);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_22] = (((int32_t*)conv_let)[cse_var_22] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_23] = (((int32_t*)conv_let)[cse_var_23] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_24] = (((int32_t*)conv_let)[cse_var_24] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_25] = (((int32_t*)conv_let)[cse_var_25] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_26] = (((int32_t*)conv_let)[cse_var_26] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_27] = (((int32_t*)conv_let)[cse_var_27] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_28] = (((int32_t*)conv_let)[cse_var_28] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
          }
          for (int32_t ic_12 = 0; ic_12 < 3; ++ic_12) {
            int32_t cse_var_45 = ((oco_1 * 54) + (ic_12 * 2));
            int32_t cse_var_44 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_12);
            int32_t cse_var_43 = (cse_var_2 + 49);
            int32_t cse_var_42 = (cse_var_2 + 48);
            int32_t cse_var_41 = (cse_var_2 + 33);
            int32_t cse_var_40 = (cse_var_2 + 32);
            int32_t cse_var_39 = (cse_var_2 + 17);
            int32_t cse_var_38 = (cse_var_2 + 16);
            int32_t cse_var_37 = (cse_var_2 + 1);
            int32_t cse_var_36 = (cse_var_45 + 13);
            int32_t cse_var_35 = (cse_var_45 + 12);
            int32_t cse_var_34 = (cse_var_44 + 96);
            int32_t cse_var_33 = (cse_var_44 + 66);
            int32_t cse_var_32 = (cse_var_44 + 6);
            int32_t cse_var_31 = (cse_var_44 + 36);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_37] = (((int32_t*)conv_let)[cse_var_37] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_38] = (((int32_t*)conv_let)[cse_var_38] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_39] = (((int32_t*)conv_let)[cse_var_39] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_40] = (((int32_t*)conv_let)[cse_var_40] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_41] = (((int32_t*)conv_let)[cse_var_41] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_42] = (((int32_t*)conv_let)[cse_var_42] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_43] = (((int32_t*)conv_let)[cse_var_43] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
          }
          for (int32_t ic_13 = 0; ic_13 < 3; ++ic_13) {
            int32_t cse_var_60 = ((oco_1 * 54) + (ic_13 * 2));
            int32_t cse_var_59 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_13);
            int32_t cse_var_58 = (cse_var_2 + 49);
            int32_t cse_var_57 = (cse_var_2 + 48);
            int32_t cse_var_56 = (cse_var_2 + 33);
            int32_t cse_var_55 = (cse_var_2 + 32);
            int32_t cse_var_54 = (cse_var_2 + 17);
            int32_t cse_var_53 = (cse_var_2 + 16);
            int32_t cse_var_52 = (cse_var_2 + 1);
            int32_t cse_var_51 = (cse_var_60 + 19);
            int32_t cse_var_50 = (cse_var_60 + 18);
            int32_t cse_var_49 = (cse_var_59 + 90);
            int32_t cse_var_48 = (cse_var_59 + 60);
            int32_t cse_var_47 = (cse_var_59 + 30);
            int32_t cse_var_46 = (cse_var_59 + 120);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_52] = (((int32_t*)conv_let)[cse_var_52] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_53] = (((int32_t*)conv_let)[cse_var_53] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_54] = (((int32_t*)conv_let)[cse_var_54] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_55] = (((int32_t*)conv_let)[cse_var_55] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_56] = (((int32_t*)conv_let)[cse_var_56] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_57] = (((int32_t*)conv_let)[cse_var_57] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_58] = (((int32_t*)conv_let)[cse_var_58] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
          }
          for (int32_t ic_14 = 0; ic_14 < 3; ++ic_14) {
            int32_t cse_var_75 = ((oco_1 * 54) + (ic_14 * 2));
            int32_t cse_var_74 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_14);
            int32_t cse_var_73 = (cse_var_2 + 49);
            int32_t cse_var_72 = (cse_var_2 + 48);
            int32_t cse_var_71 = (cse_var_2 + 33);
            int32_t cse_var_70 = (cse_var_2 + 32);
            int32_t cse_var_69 = (cse_var_2 + 17);
            int32_t cse_var_68 = (cse_var_2 + 16);
            int32_t cse_var_67 = (cse_var_2 + 1);
            int32_t cse_var_66 = (cse_var_75 + 25);
            int32_t cse_var_65 = (cse_var_75 + 24);
            int32_t cse_var_64 = (cse_var_74 + 93);
            int32_t cse_var_63 = (cse_var_74 + 63);
            int32_t cse_var_62 = (cse_var_74 + 33);
            int32_t cse_var_61 = (cse_var_74 + 123);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_67] = (((int32_t*)conv_let)[cse_var_67] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_68] = (((int32_t*)conv_let)[cse_var_68] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_69] = (((int32_t*)conv_let)[cse_var_69] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_70] = (((int32_t*)conv_let)[cse_var_70] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_71] = (((int32_t*)conv_let)[cse_var_71] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_72] = (((int32_t*)conv_let)[cse_var_72] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_73] = (((int32_t*)conv_let)[cse_var_73] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
          }
          for (int32_t ic_15 = 0; ic_15 < 3; ++ic_15) {
            int32_t cse_var_90 = ((oco_1 * 54) + (ic_15 * 2));
            int32_t cse_var_89 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_15);
            int32_t cse_var_88 = (cse_var_2 + 49);
            int32_t cse_var_87 = (cse_var_2 + 48);
            int32_t cse_var_86 = (cse_var_2 + 33);
            int32_t cse_var_85 = (cse_var_2 + 32);
            int32_t cse_var_84 = (cse_var_2 + 17);
            int32_t cse_var_83 = (cse_var_2 + 16);
            int32_t cse_var_82 = (cse_var_2 + 1);
            int32_t cse_var_81 = (cse_var_90 + 31);
            int32_t cse_var_80 = (cse_var_90 + 30);
            int32_t cse_var_79 = (cse_var_89 + 96);
            int32_t cse_var_78 = (cse_var_89 + 66);
            int32_t cse_var_77 = (cse_var_89 + 36);
            int32_t cse_var_76 = (cse_var_89 + 126);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_82] = (((int32_t*)conv_let)[cse_var_82] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_83] = (((int32_t*)conv_let)[cse_var_83] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_84] = (((int32_t*)conv_let)[cse_var_84] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_85] = (((int32_t*)conv_let)[cse_var_85] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_86] = (((int32_t*)conv_let)[cse_var_86] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_87] = (((int32_t*)conv_let)[cse_var_87] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_88] = (((int32_t*)conv_let)[cse_var_88] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
          }
          for (int32_t ic_16 = 0; ic_16 < 3; ++ic_16) {
            int32_t cse_var_105 = ((oco_1 * 54) + (ic_16 * 2));
            int32_t cse_var_104 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_16);
            int32_t cse_var_103 = (cse_var_2 + 49);
            int32_t cse_var_102 = (cse_var_2 + 48);
            int32_t cse_var_101 = (cse_var_2 + 33);
            int32_t cse_var_100 = (cse_var_2 + 32);
            int32_t cse_var_99 = (cse_var_2 + 17);
            int32_t cse_var_98 = (cse_var_2 + 16);
            int32_t cse_var_97 = (cse_var_2 + 1);
            int32_t cse_var_96 = (cse_var_105 + 37);
            int32_t cse_var_95 = (cse_var_105 + 36);
            int32_t cse_var_94 = (cse_var_104 + 90);
            int32_t cse_var_93 = (cse_var_104 + 60);
            int32_t cse_var_92 = (cse_var_104 + 150);
            int32_t cse_var_91 = (cse_var_104 + 120);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_97] = (((int32_t*)conv_let)[cse_var_97] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_98] = (((int32_t*)conv_let)[cse_var_98] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_99] = (((int32_t*)conv_let)[cse_var_99] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_100] = (((int32_t*)conv_let)[cse_var_100] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_101] = (((int32_t*)conv_let)[cse_var_101] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_102] = (((int32_t*)conv_let)[cse_var_102] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_103] = (((int32_t*)conv_let)[cse_var_103] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
          }
          for (int32_t ic_17 = 0; ic_17 < 3; ++ic_17) {
            int32_t cse_var_120 = ((oco_1 * 54) + (ic_17 * 2));
            int32_t cse_var_119 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_17);
            int32_t cse_var_118 = (cse_var_2 + 49);
            int32_t cse_var_117 = (cse_var_2 + 48);
            int32_t cse_var_116 = (cse_var_2 + 33);
            int32_t cse_var_115 = (cse_var_2 + 32);
            int32_t cse_var_114 = (cse_var_2 + 17);
            int32_t cse_var_113 = (cse_var_2 + 16);
            int32_t cse_var_112 = (cse_var_2 + 1);
            int32_t cse_var_111 = (cse_var_120 + 43);
            int32_t cse_var_110 = (cse_var_120 + 42);
            int32_t cse_var_109 = (cse_var_119 + 93);
            int32_t cse_var_108 = (cse_var_119 + 63);
            int32_t cse_var_107 = (cse_var_119 + 153);
            int32_t cse_var_106 = (cse_var_119 + 123);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_112] = (((int32_t*)conv_let)[cse_var_112] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_113] = (((int32_t*)conv_let)[cse_var_113] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_114] = (((int32_t*)conv_let)[cse_var_114] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_115] = (((int32_t*)conv_let)[cse_var_115] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_116] = (((int32_t*)conv_let)[cse_var_116] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_117] = (((int32_t*)conv_let)[cse_var_117] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_118] = (((int32_t*)conv_let)[cse_var_118] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
          }
          for (int32_t ic_18 = 0; ic_18 < 3; ++ic_18) {
            int32_t cse_var_135 = ((oco_1 * 54) + (ic_18 * 2));
            int32_t cse_var_134 = ((((oho * 720) + (owo_1 * 180)) + (owi * 3)) + ic_18);
            int32_t cse_var_133 = (cse_var_2 + 49);
            int32_t cse_var_132 = (cse_var_2 + 48);
            int32_t cse_var_131 = (cse_var_2 + 33);
            int32_t cse_var_130 = (cse_var_2 + 32);
            int32_t cse_var_129 = (cse_var_2 + 17);
            int32_t cse_var_128 = (cse_var_2 + 16);
            int32_t cse_var_127 = (cse_var_2 + 1);
            int32_t cse_var_126 = (cse_var_135 + 49);
            int32_t cse_var_125 = (cse_var_135 + 48);
            int32_t cse_var_124 = (cse_var_134 + 96);
            int32_t cse_var_123 = (cse_var_134 + 66);
            int32_t cse_var_122 = (cse_var_134 + 156);
            int32_t cse_var_121 = (cse_var_134 + 126);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_127] = (((int32_t*)conv_let)[cse_var_127] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_128] = (((int32_t*)conv_let)[cse_var_128] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_129] = (((int32_t*)conv_let)[cse_var_129] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_130] = (((int32_t*)conv_let)[cse_var_130] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_131] = (((int32_t*)conv_let)[cse_var_131] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_132] = (((int32_t*)conv_let)[cse_var_132] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_133] = (((int32_t*)conv_let)[cse_var_133] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
          }
        }
      }
    }
  }
  for (int32_t i0_i1_outer_fused = 0; i0_i1_outer_fused < 8; ++i0_i1_outer_fused) {
    for (int32_t i2_outer = 0; i2_outer < 4; ++i2_outer) {
      for (int32_t i3_outer = 0; i3_outer < 8; ++i3_outer) {
        for (int32_t i2_inner = 0; i2_inner < 8; ++i2_inner) {
          int32_t cse_var_140 = (i0_i1_outer_fused * 2048);
          int32_t cse_var_139 = (i3_outer * 2);
          int32_t cse_var_138 = (cse_var_139 + 1);
          int32_t cse_var_137 = (((cse_var_140 + (i2_outer * 512)) + (i3_outer * 64)) + (i2_inner * 2));
          int32_t cse_var_136 = (((cse_var_140 + (i2_outer * 128)) + (i2_inner * 16)) + cse_var_139);
          int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_137]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_139])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_139]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_139]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_139])) - 128;
          int32_t __2 = (__1) < (127) ? (__1) : (127);
          int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
          int8_t __4 = (int8_t)127;
          int8_t __5 = (__3) < (__4) ? (__3) : (__4);
          int8_t __6 = (int8_t)-128;
          compute[cse_var_136] = ((__5) > (__6) ? (__5) : (__6));
          int32_t __7 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_137 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_138])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_138]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_138]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_138])) - 128;
          int32_t __8 = (__7) < (127) ? (__7) : (127);
          int8_t __9 = (int8_t)((__8) > (-128) ? (__8) : (-128));
          int8_t __10 = (__9) < (__4) ? (__9) : (__4);
          compute[(cse_var_136 + 1)] = ((__10) > (__6) ? (__10) : (__6));
        }
        for (int32_t i2_inner_1 = 0; i2_inner_1 < 8; ++i2_inner_1) {
          int32_t cse_var_145 = (i0_i1_outer_fused * 2048);
          int32_t cse_var_144 = (i3_outer * 2);
          int32_t cse_var_143 = (cse_var_144 + 1);
          int32_t cse_var_142 = (((cse_var_145 + (i2_outer * 512)) + (i3_outer * 64)) + (i2_inner_1 * 2));
          int32_t cse_var_141 = (((cse_var_145 + (i2_outer * 128)) + (i2_inner_1 * 16)) + cse_var_144);
          int32_t __11 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_142 + 16)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_144])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_144]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_144]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_144])) - 128;
          int32_t __12 = (__11) < (127) ? (__11) : (127);
          int8_t __13 = (int8_t)((__12) > (-128) ? (__12) : (-128));
          int8_t __14 = (int8_t)127;
          int8_t __15 = (__13) < (__14) ? (__13) : (__14);
          int8_t __16 = (int8_t)-128;
          compute[(cse_var_141 + 512)] = ((__15) > (__16) ? (__15) : (__16));
          int32_t __17 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_142 + 17)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_143])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_143]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_143]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_143])) - 128;
          int32_t __18 = (__17) < (127) ? (__17) : (127);
          int8_t __19 = (int8_t)((__18) > (-128) ? (__18) : (-128));
          int8_t __20 = (__19) < (__14) ? (__19) : (__14);
          compute[(cse_var_141 + 513)] = ((__20) > (__16) ? (__20) : (__16));
        }
        for (int32_t i2_inner_2 = 0; i2_inner_2 < 8; ++i2_inner_2) {
          int32_t cse_var_150 = (i0_i1_outer_fused * 2048);
          int32_t cse_var_149 = (i3_outer * 2);
          int32_t cse_var_148 = (cse_var_149 + 1);
          int32_t cse_var_147 = (((cse_var_150 + (i2_outer * 512)) + (i3_outer * 64)) + (i2_inner_2 * 2));
          int32_t cse_var_146 = (((cse_var_150 + (i2_outer * 128)) + (i2_inner_2 * 16)) + cse_var_149);
          int32_t __21 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_147 + 32)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_149])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_149]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_149]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_149])) - 128;
          int32_t __22 = (__21) < (127) ? (__21) : (127);
          int8_t __23 = (int8_t)((__22) > (-128) ? (__22) : (-128));
          int8_t __24 = (int8_t)127;
          int8_t __25 = (__23) < (__24) ? (__23) : (__24);
          int8_t __26 = (int8_t)-128;
          compute[(cse_var_146 + 1024)] = ((__25) > (__26) ? (__25) : (__26));
          int32_t __27 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_147 + 33)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_148])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_148]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_148]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_148])) - 128;
          int32_t __28 = (__27) < (127) ? (__27) : (127);
          int8_t __29 = (int8_t)((__28) > (-128) ? (__28) : (-128));
          int8_t __30 = (__29) < (__24) ? (__29) : (__24);
          compute[(cse_var_146 + 1025)] = ((__30) > (__26) ? (__30) : (__26));
        }
        for (int32_t i2_inner_3 = 0; i2_inner_3 < 8; ++i2_inner_3) {
          int32_t cse_var_155 = (i0_i1_outer_fused * 2048);
          int32_t cse_var_154 = (i3_outer * 2);
          int32_t cse_var_153 = (cse_var_154 + 1);
          int32_t cse_var_152 = (((cse_var_155 + (i2_outer * 512)) + (i3_outer * 64)) + (i2_inner_3 * 2));
          int32_t cse_var_151 = (((cse_var_155 + (i2_outer * 128)) + (i2_inner_3 * 16)) + cse_var_154);
          int32_t __31 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_152 + 48)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_154])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_154]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_154]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_154])) - 128;
          int32_t __32 = (__31) < (127) ? (__31) : (127);
          int8_t __33 = (int8_t)((__32) > (-128) ? (__32) : (-128));
          int8_t __34 = (int8_t)127;
          int8_t __35 = (__33) < (__34) ? (__33) : (__34);
          int8_t __36 = (int8_t)-128;
          compute[(cse_var_151 + 1536)] = ((__35) > (__36) ? (__35) : (__36));
          int32_t __37 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_152 + 49)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_1_let)[cse_var_153])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_2_let)[cse_var_153]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_3_let)[cse_var_153]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_4_let)[cse_var_153])) - 128;
          int32_t __38 = (__37) < (127) ? (__37) : (127);
          int8_t __39 = (int8_t)((__38) > (-128) ? (__38) : (-128));
          int8_t __40 = (__39) < (__34) ? (__39) : (__34);
          compute[(cse_var_151 + 1537)] = ((__40) > (__36) ? (__40) : (__36));
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245_(int16_t* p0, int16_t* T_subtract, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_constant_9_let = (&(global_const_workspace_8_var[163040]));
  void* fused_nn_conv2d_add_cast_multiply_constant_8_let = (&(global_const_workspace_8_var[162784]));
  void* fused_nn_conv2d_add_cast_constant_7_let = (&(global_const_workspace_8_var[163296]));
  void* fused_nn_conv2d_constant_6_let = (&(global_const_workspace_8_var[163936]));
  void* fused_constant_5_let = (&(global_const_workspace_8_var[138240]));
  void* PadInput_let = (&(global_workspace_9_var[126976]));
  void* data_vec_let = (&(global_workspace_9_var[65536]));
  void* conv_let = (&(global_workspace_9_var[0]));
  for (int32_t i1 = 0; i1 < 34; ++i1) {
    for (int32_t i2 = 0; i2 < 34; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        int32_t cse_var_1 = (i2 * 16);
        ((int16_t*)PadInput_let)[(((i1 * 544) + cse_var_1) + i3)] = (((((1 <= i1) && (i1 < 33)) && (1 <= i2)) && (i2 < 33)) ? p0[((((i1 * 512) + cse_var_1) + i3) - 528)] : (int16_t)0);
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 8; ++n_oho_fused) {
    for (int32_t owo = 0; owo < 4; ++owo) {
      for (int32_t ohi = 0; ohi < 6; ++ohi) {
        for (int32_t ic = 0; ic < 16; ++ic) {
          ((int16_t*)data_vec_let)[((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic)];
        }
        for (int32_t ic_1 = 0; ic_1 < 16; ++ic_1) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_1) + 16)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_1) + 16)];
        }
        for (int32_t ic_2 = 0; ic_2 < 16; ++ic_2) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_2) + 32)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_2) + 32)];
        }
        for (int32_t ic_3 = 0; ic_3 < 16; ++ic_3) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_3) + 48)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_3) + 48)];
        }
        for (int32_t ic_4 = 0; ic_4 < 16; ++ic_4) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_4) + 64)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_4) + 64)];
        }
        for (int32_t ic_5 = 0; ic_5 < 16; ++ic_5) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_5) + 80)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_5) + 80)];
        }
        for (int32_t ic_6 = 0; ic_6 < 16; ++ic_6) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_6) + 96)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_6) + 96)];
        }
        for (int32_t ic_7 = 0; ic_7 < 16; ++ic_7) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_7) + 112)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_7) + 112)];
        }
        for (int32_t ic_8 = 0; ic_8 < 16; ++ic_8) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_8) + 128)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_8) + 128)];
        }
        for (int32_t ic_9 = 0; ic_9 < 16; ++ic_9) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 3840) + (owo * 960)) + (ohi * 160)) + ic_9) + 144)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 2176) + (ohi * 544)) + (owo * 128)) + ic_9) + 144)];
        }
      }
    }
  }
  for (int32_t oco = 0; oco < 8; ++oco) {
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t oci = 0; oci < 2; ++oci) {
          ((int16_t*)PadInput_let)[((((oco * 288) + (kh * 96)) + (kw * 32)) + oci)] = ((int16_t*)fused_constant_5_let)[((((kh * 768) + (kw * 256)) + (oco * 2)) + oci)];
        }
        for (int32_t oci_1 = 0; oci_1 < 2; ++oci_1) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_1) + 2)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_1) + 16)];
        }
        for (int32_t oci_2 = 0; oci_2 < 2; ++oci_2) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_2) + 4)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_2) + 32)];
        }
        for (int32_t oci_3 = 0; oci_3 < 2; ++oci_3) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_3) + 6)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_3) + 48)];
        }
        for (int32_t oci_4 = 0; oci_4 < 2; ++oci_4) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_4) + 8)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_4) + 64)];
        }
        for (int32_t oci_5 = 0; oci_5 < 2; ++oci_5) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_5) + 10)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_5) + 80)];
        }
        for (int32_t oci_6 = 0; oci_6 < 2; ++oci_6) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_6) + 12)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_6) + 96)];
        }
        for (int32_t oci_7 = 0; oci_7 < 2; ++oci_7) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_7) + 14)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_7) + 112)];
        }
        for (int32_t oci_8 = 0; oci_8 < 2; ++oci_8) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_8) + 16)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_8) + 128)];
        }
        for (int32_t oci_9 = 0; oci_9 < 2; ++oci_9) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_9) + 18)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_9) + 144)];
        }
        for (int32_t oci_10 = 0; oci_10 < 2; ++oci_10) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_10) + 20)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_10) + 160)];
        }
        for (int32_t oci_11 = 0; oci_11 < 2; ++oci_11) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_11) + 22)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_11) + 176)];
        }
        for (int32_t oci_12 = 0; oci_12 < 2; ++oci_12) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_12) + 24)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_12) + 192)];
        }
        for (int32_t oci_13 = 0; oci_13 < 2; ++oci_13) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_13) + 26)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_13) + 208)];
        }
        for (int32_t oci_14 = 0; oci_14 < 2; ++oci_14) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_14) + 28)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_14) + 224)];
        }
        for (int32_t oci_15 = 0; oci_15 < 2; ++oci_15) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_15) + 30)] = ((int16_t*)fused_constant_5_let)[(((((kh * 768) + (kw * 256)) + (oco * 2)) + oci_15) + 240)];
        }
      }
    }
  }
  for (int32_t oho = 0; oho < 8; ++oho) {
    for (int32_t owo_1 = 0; owo_1 < 4; ++owo_1) {
      for (int32_t oco_1 = 0; oco_1 < 8; ++oco_1) {
        for (int32_t owi = 0; owi < 8; ++owi) {
          int32_t cse_var_2 = ((((oho * 2048) + (owo_1 * 512)) + (oco_1 * 64)) + (owi * 2));
          ((int32_t*)conv_let)[cse_var_2] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 1)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 16)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 17)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 32)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 33)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 48)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 49)] = 0;
          for (int32_t ic_10 = 0; ic_10 < 16; ++ic_10) {
            int32_t cse_var_15 = (cse_var_2 + 49);
            int32_t cse_var_14 = (cse_var_2 + 48);
            int32_t cse_var_13 = (cse_var_2 + 33);
            int32_t cse_var_12 = (cse_var_2 + 32);
            int32_t cse_var_11 = (cse_var_2 + 17);
            int32_t cse_var_10 = (cse_var_2 + 16);
            int32_t cse_var_9 = (cse_var_2 + 1);
            int32_t cse_var_8 = ((oco_1 * 288) + (ic_10 * 2));
            int32_t cse_var_7 = (cse_var_8 + 1);
            int32_t cse_var_6 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_10);
            int32_t cse_var_5 = (cse_var_6 + 480);
            int32_t cse_var_4 = (cse_var_6 + 320);
            int32_t cse_var_3 = (cse_var_6 + 160);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_9] = (((int32_t*)conv_let)[cse_var_9] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_10] = (((int32_t*)conv_let)[cse_var_10] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_11] = (((int32_t*)conv_let)[cse_var_11] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_12] = (((int32_t*)conv_let)[cse_var_12] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_13] = (((int32_t*)conv_let)[cse_var_13] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_14] = (((int32_t*)conv_let)[cse_var_14] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_15] = (((int32_t*)conv_let)[cse_var_15] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
          }
          for (int32_t ic_11 = 0; ic_11 < 16; ++ic_11) {
            int32_t cse_var_30 = ((oco_1 * 288) + (ic_11 * 2));
            int32_t cse_var_29 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_11);
            int32_t cse_var_28 = (cse_var_2 + 49);
            int32_t cse_var_27 = (cse_var_2 + 48);
            int32_t cse_var_26 = (cse_var_2 + 33);
            int32_t cse_var_25 = (cse_var_2 + 32);
            int32_t cse_var_24 = (cse_var_2 + 17);
            int32_t cse_var_23 = (cse_var_2 + 16);
            int32_t cse_var_22 = (cse_var_2 + 1);
            int32_t cse_var_21 = (cse_var_30 + 33);
            int32_t cse_var_20 = (cse_var_30 + 32);
            int32_t cse_var_19 = (cse_var_29 + 496);
            int32_t cse_var_18 = (cse_var_29 + 336);
            int32_t cse_var_17 = (cse_var_29 + 176);
            int32_t cse_var_16 = (cse_var_29 + 16);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_22] = (((int32_t*)conv_let)[cse_var_22] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_23] = (((int32_t*)conv_let)[cse_var_23] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_24] = (((int32_t*)conv_let)[cse_var_24] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_25] = (((int32_t*)conv_let)[cse_var_25] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_26] = (((int32_t*)conv_let)[cse_var_26] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_27] = (((int32_t*)conv_let)[cse_var_27] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_28] = (((int32_t*)conv_let)[cse_var_28] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
          }
          for (int32_t ic_12 = 0; ic_12 < 16; ++ic_12) {
            int32_t cse_var_45 = ((oco_1 * 288) + (ic_12 * 2));
            int32_t cse_var_44 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_12);
            int32_t cse_var_43 = (cse_var_2 + 49);
            int32_t cse_var_42 = (cse_var_2 + 48);
            int32_t cse_var_41 = (cse_var_2 + 33);
            int32_t cse_var_40 = (cse_var_2 + 32);
            int32_t cse_var_39 = (cse_var_2 + 17);
            int32_t cse_var_38 = (cse_var_2 + 16);
            int32_t cse_var_37 = (cse_var_2 + 1);
            int32_t cse_var_36 = (cse_var_45 + 65);
            int32_t cse_var_35 = (cse_var_45 + 64);
            int32_t cse_var_34 = (cse_var_44 + 512);
            int32_t cse_var_33 = (cse_var_44 + 352);
            int32_t cse_var_32 = (cse_var_44 + 32);
            int32_t cse_var_31 = (cse_var_44 + 192);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_37] = (((int32_t*)conv_let)[cse_var_37] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_38] = (((int32_t*)conv_let)[cse_var_38] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_39] = (((int32_t*)conv_let)[cse_var_39] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_40] = (((int32_t*)conv_let)[cse_var_40] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_41] = (((int32_t*)conv_let)[cse_var_41] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_42] = (((int32_t*)conv_let)[cse_var_42] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_43] = (((int32_t*)conv_let)[cse_var_43] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
          }
          for (int32_t ic_13 = 0; ic_13 < 16; ++ic_13) {
            int32_t cse_var_60 = ((oco_1 * 288) + (ic_13 * 2));
            int32_t cse_var_59 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_13);
            int32_t cse_var_58 = (cse_var_2 + 49);
            int32_t cse_var_57 = (cse_var_2 + 48);
            int32_t cse_var_56 = (cse_var_2 + 33);
            int32_t cse_var_55 = (cse_var_2 + 32);
            int32_t cse_var_54 = (cse_var_2 + 17);
            int32_t cse_var_53 = (cse_var_2 + 16);
            int32_t cse_var_52 = (cse_var_2 + 1);
            int32_t cse_var_51 = (cse_var_60 + 97);
            int32_t cse_var_50 = (cse_var_60 + 96);
            int32_t cse_var_49 = (cse_var_59 + 640);
            int32_t cse_var_48 = (cse_var_59 + 480);
            int32_t cse_var_47 = (cse_var_59 + 320);
            int32_t cse_var_46 = (cse_var_59 + 160);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_52] = (((int32_t*)conv_let)[cse_var_52] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_53] = (((int32_t*)conv_let)[cse_var_53] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_54] = (((int32_t*)conv_let)[cse_var_54] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_55] = (((int32_t*)conv_let)[cse_var_55] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_56] = (((int32_t*)conv_let)[cse_var_56] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_57] = (((int32_t*)conv_let)[cse_var_57] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_58] = (((int32_t*)conv_let)[cse_var_58] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
          }
          for (int32_t ic_14 = 0; ic_14 < 16; ++ic_14) {
            int32_t cse_var_75 = ((oco_1 * 288) + (ic_14 * 2));
            int32_t cse_var_74 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_14);
            int32_t cse_var_73 = (cse_var_2 + 49);
            int32_t cse_var_72 = (cse_var_2 + 48);
            int32_t cse_var_71 = (cse_var_2 + 33);
            int32_t cse_var_70 = (cse_var_2 + 32);
            int32_t cse_var_69 = (cse_var_2 + 17);
            int32_t cse_var_68 = (cse_var_2 + 16);
            int32_t cse_var_67 = (cse_var_2 + 1);
            int32_t cse_var_66 = (cse_var_75 + 129);
            int32_t cse_var_65 = (cse_var_75 + 128);
            int32_t cse_var_64 = (cse_var_74 + 656);
            int32_t cse_var_63 = (cse_var_74 + 496);
            int32_t cse_var_62 = (cse_var_74 + 336);
            int32_t cse_var_61 = (cse_var_74 + 176);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_67] = (((int32_t*)conv_let)[cse_var_67] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_68] = (((int32_t*)conv_let)[cse_var_68] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_69] = (((int32_t*)conv_let)[cse_var_69] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_70] = (((int32_t*)conv_let)[cse_var_70] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_71] = (((int32_t*)conv_let)[cse_var_71] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_72] = (((int32_t*)conv_let)[cse_var_72] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_73] = (((int32_t*)conv_let)[cse_var_73] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
          }
          for (int32_t ic_15 = 0; ic_15 < 16; ++ic_15) {
            int32_t cse_var_90 = ((oco_1 * 288) + (ic_15 * 2));
            int32_t cse_var_89 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_15);
            int32_t cse_var_88 = (cse_var_2 + 49);
            int32_t cse_var_87 = (cse_var_2 + 48);
            int32_t cse_var_86 = (cse_var_2 + 33);
            int32_t cse_var_85 = (cse_var_2 + 32);
            int32_t cse_var_84 = (cse_var_2 + 17);
            int32_t cse_var_83 = (cse_var_2 + 16);
            int32_t cse_var_82 = (cse_var_2 + 1);
            int32_t cse_var_81 = (cse_var_90 + 161);
            int32_t cse_var_80 = (cse_var_90 + 160);
            int32_t cse_var_79 = (cse_var_89 + 672);
            int32_t cse_var_78 = (cse_var_89 + 512);
            int32_t cse_var_77 = (cse_var_89 + 352);
            int32_t cse_var_76 = (cse_var_89 + 192);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_82] = (((int32_t*)conv_let)[cse_var_82] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_83] = (((int32_t*)conv_let)[cse_var_83] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_84] = (((int32_t*)conv_let)[cse_var_84] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_85] = (((int32_t*)conv_let)[cse_var_85] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_86] = (((int32_t*)conv_let)[cse_var_86] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_87] = (((int32_t*)conv_let)[cse_var_87] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_88] = (((int32_t*)conv_let)[cse_var_88] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
          }
          for (int32_t ic_16 = 0; ic_16 < 16; ++ic_16) {
            int32_t cse_var_105 = ((oco_1 * 288) + (ic_16 * 2));
            int32_t cse_var_104 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_16);
            int32_t cse_var_103 = (cse_var_2 + 49);
            int32_t cse_var_102 = (cse_var_2 + 48);
            int32_t cse_var_101 = (cse_var_2 + 33);
            int32_t cse_var_100 = (cse_var_2 + 32);
            int32_t cse_var_99 = (cse_var_2 + 17);
            int32_t cse_var_98 = (cse_var_2 + 16);
            int32_t cse_var_97 = (cse_var_2 + 1);
            int32_t cse_var_96 = (cse_var_105 + 193);
            int32_t cse_var_95 = (cse_var_105 + 192);
            int32_t cse_var_94 = (cse_var_104 + 800);
            int32_t cse_var_93 = (cse_var_104 + 640);
            int32_t cse_var_92 = (cse_var_104 + 480);
            int32_t cse_var_91 = (cse_var_104 + 320);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_97] = (((int32_t*)conv_let)[cse_var_97] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_98] = (((int32_t*)conv_let)[cse_var_98] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_99] = (((int32_t*)conv_let)[cse_var_99] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_100] = (((int32_t*)conv_let)[cse_var_100] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_101] = (((int32_t*)conv_let)[cse_var_101] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_102] = (((int32_t*)conv_let)[cse_var_102] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_103] = (((int32_t*)conv_let)[cse_var_103] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
          }
          for (int32_t ic_17 = 0; ic_17 < 16; ++ic_17) {
            int32_t cse_var_120 = ((oco_1 * 288) + (ic_17 * 2));
            int32_t cse_var_119 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_17);
            int32_t cse_var_118 = (cse_var_2 + 49);
            int32_t cse_var_117 = (cse_var_2 + 48);
            int32_t cse_var_116 = (cse_var_2 + 33);
            int32_t cse_var_115 = (cse_var_2 + 32);
            int32_t cse_var_114 = (cse_var_2 + 17);
            int32_t cse_var_113 = (cse_var_2 + 16);
            int32_t cse_var_112 = (cse_var_2 + 1);
            int32_t cse_var_111 = (cse_var_120 + 225);
            int32_t cse_var_110 = (cse_var_120 + 224);
            int32_t cse_var_109 = (cse_var_119 + 816);
            int32_t cse_var_108 = (cse_var_119 + 656);
            int32_t cse_var_107 = (cse_var_119 + 496);
            int32_t cse_var_106 = (cse_var_119 + 336);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_112] = (((int32_t*)conv_let)[cse_var_112] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_113] = (((int32_t*)conv_let)[cse_var_113] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_114] = (((int32_t*)conv_let)[cse_var_114] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_115] = (((int32_t*)conv_let)[cse_var_115] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_116] = (((int32_t*)conv_let)[cse_var_116] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_117] = (((int32_t*)conv_let)[cse_var_117] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_118] = (((int32_t*)conv_let)[cse_var_118] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
          }
          for (int32_t ic_18 = 0; ic_18 < 16; ++ic_18) {
            int32_t cse_var_135 = ((oco_1 * 288) + (ic_18 * 2));
            int32_t cse_var_134 = ((((oho * 3840) + (owo_1 * 960)) + (owi * 16)) + ic_18);
            int32_t cse_var_133 = (cse_var_2 + 49);
            int32_t cse_var_132 = (cse_var_2 + 48);
            int32_t cse_var_131 = (cse_var_2 + 33);
            int32_t cse_var_130 = (cse_var_2 + 32);
            int32_t cse_var_129 = (cse_var_2 + 17);
            int32_t cse_var_128 = (cse_var_2 + 16);
            int32_t cse_var_127 = (cse_var_2 + 1);
            int32_t cse_var_126 = (cse_var_135 + 257);
            int32_t cse_var_125 = (cse_var_135 + 256);
            int32_t cse_var_124 = (cse_var_134 + 832);
            int32_t cse_var_123 = (cse_var_134 + 672);
            int32_t cse_var_122 = (cse_var_134 + 512);
            int32_t cse_var_121 = (cse_var_134 + 352);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_127] = (((int32_t*)conv_let)[cse_var_127] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_128] = (((int32_t*)conv_let)[cse_var_128] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_129] = (((int32_t*)conv_let)[cse_var_129] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_130] = (((int32_t*)conv_let)[cse_var_130] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_131] = (((int32_t*)conv_let)[cse_var_131] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_132] = (((int32_t*)conv_let)[cse_var_132] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_133] = (((int32_t*)conv_let)[cse_var_133] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
          }
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 8; ++ax0_ax1_outer_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 4; ++ax2_outer) {
      for (int32_t ax3_outer = 0; ax3_outer < 8; ++ax3_outer) {
        for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
          int32_t cse_var_140 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_139 = (ax3_outer * 2);
          int32_t cse_var_138 = (cse_var_139 + 1);
          int32_t cse_var_137 = (((cse_var_140 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner * 2));
          int32_t cse_var_136 = (((cse_var_140 + (ax2_outer * 128)) + (ax2_inner * 16)) + cse_var_139);
          int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_137]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_139])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_139]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_139]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_139])) - 128;
          int32_t __2 = (__1) < (127) ? (__1) : (127);
          int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
          int8_t __4 = (int8_t)127;
          int8_t __5 = (__3) < (__4) ? (__3) : (__4);
          int8_t __6 = (int8_t)-128;
          T_subtract[cse_var_136] = (((int16_t)((__5) > (__6) ? (__5) : (__6))) - (int16_t)-128);
          int32_t __7 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_137 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_138])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_138]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_138]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_138])) - 128;
          int32_t __8 = (__7) < (127) ? (__7) : (127);
          int8_t __9 = (int8_t)((__8) > (-128) ? (__8) : (-128));
          int8_t __10 = (__9) < (__4) ? (__9) : (__4);
          T_subtract[(cse_var_136 + 1)] = (((int16_t)((__10) > (__6) ? (__10) : (__6))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 8; ++ax2_inner_1) {
          int32_t cse_var_145 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_144 = (ax3_outer * 2);
          int32_t cse_var_143 = (cse_var_144 + 1);
          int32_t cse_var_142 = (((cse_var_145 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner_1 * 2));
          int32_t cse_var_141 = (((cse_var_145 + (ax2_outer * 128)) + (ax2_inner_1 * 16)) + cse_var_144);
          int32_t __11 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_142 + 16)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_144])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_144]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_144]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_144])) - 128;
          int32_t __12 = (__11) < (127) ? (__11) : (127);
          int8_t __13 = (int8_t)((__12) > (-128) ? (__12) : (-128));
          int8_t __14 = (int8_t)127;
          int8_t __15 = (__13) < (__14) ? (__13) : (__14);
          int8_t __16 = (int8_t)-128;
          T_subtract[(cse_var_141 + 512)] = (((int16_t)((__15) > (__16) ? (__15) : (__16))) - (int16_t)-128);
          int32_t __17 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_142 + 17)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_143])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_143]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_143]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_143])) - 128;
          int32_t __18 = (__17) < (127) ? (__17) : (127);
          int8_t __19 = (int8_t)((__18) > (-128) ? (__18) : (-128));
          int8_t __20 = (__19) < (__14) ? (__19) : (__14);
          T_subtract[(cse_var_141 + 513)] = (((int16_t)((__20) > (__16) ? (__20) : (__16))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_2 = 0; ax2_inner_2 < 8; ++ax2_inner_2) {
          int32_t cse_var_150 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_149 = (ax3_outer * 2);
          int32_t cse_var_148 = (cse_var_149 + 1);
          int32_t cse_var_147 = (((cse_var_150 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner_2 * 2));
          int32_t cse_var_146 = (((cse_var_150 + (ax2_outer * 128)) + (ax2_inner_2 * 16)) + cse_var_149);
          int32_t __21 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_147 + 32)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_149])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_149]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_149]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_149])) - 128;
          int32_t __22 = (__21) < (127) ? (__21) : (127);
          int8_t __23 = (int8_t)((__22) > (-128) ? (__22) : (-128));
          int8_t __24 = (int8_t)127;
          int8_t __25 = (__23) < (__24) ? (__23) : (__24);
          int8_t __26 = (int8_t)-128;
          T_subtract[(cse_var_146 + 1024)] = (((int16_t)((__25) > (__26) ? (__25) : (__26))) - (int16_t)-128);
          int32_t __27 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_147 + 33)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_148])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_148]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_148]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_148])) - 128;
          int32_t __28 = (__27) < (127) ? (__27) : (127);
          int8_t __29 = (int8_t)((__28) > (-128) ? (__28) : (-128));
          int8_t __30 = (__29) < (__24) ? (__29) : (__24);
          T_subtract[(cse_var_146 + 1025)] = (((int16_t)((__30) > (__26) ? (__30) : (__26))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_3 = 0; ax2_inner_3 < 8; ++ax2_inner_3) {
          int32_t cse_var_155 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_154 = (ax3_outer * 2);
          int32_t cse_var_153 = (cse_var_154 + 1);
          int32_t cse_var_152 = (((cse_var_155 + (ax2_outer * 512)) + (ax3_outer * 64)) + (ax2_inner_3 * 2));
          int32_t cse_var_151 = (((cse_var_155 + (ax2_outer * 128)) + (ax2_inner_3 * 16)) + cse_var_154);
          int32_t __31 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_152 + 48)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_154])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_154]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_154]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_154])) - 128;
          int32_t __32 = (__31) < (127) ? (__31) : (127);
          int8_t __33 = (int8_t)((__32) > (-128) ? (__32) : (-128));
          int8_t __34 = (int8_t)127;
          int8_t __35 = (__33) < (__34) ? (__33) : (__34);
          int8_t __36 = (int8_t)-128;
          T_subtract[(cse_var_151 + 1536)] = (((int16_t)((__35) > (__36) ? (__35) : (__36))) - (int16_t)-128);
          int32_t __37 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_152 + 49)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_6_let)[cse_var_153])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_7_let)[cse_var_153]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_8_let)[cse_var_153]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_9_let)[cse_var_153])) - 128;
          int32_t __38 = (__37) < (127) ? (__37) : (127);
          int8_t __39 = (int8_t)((__38) > (-128) ? (__38) : (-128));
          int8_t __40 = (__39) < (__34) ? (__39) : (__34);
          T_subtract[(cse_var_151 + 1537)] = (((int16_t)((__40) > (__36) ? (__40) : (__36))) - (int16_t)-128);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__1(int16_t* p0, int16_t* T_subtract, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_constant_21_let = (&(global_const_workspace_12_var[161376]));
  void* fused_nn_conv2d_add_cast_multiply_constant_20_let = (&(global_const_workspace_12_var[160608]));
  void* fused_nn_conv2d_add_cast_constant_19_let = (&(global_const_workspace_12_var[162144]));
  void* fused_nn_conv2d_constant_18_let = (&(global_const_workspace_12_var[162656]));
  void* fused_constant_17_let = (&(global_const_workspace_12_var[129024]));
  void* PadInput_let = (&(global_workspace_13_var[39168]));
  void* data_vec_let = (&(global_workspace_13_var[0]));
  void* conv_let = (&(global_workspace_13_var[106784]));
  for (int32_t i1 = 0; i1 < 33; ++i1) {
    for (int32_t i2 = 0; i2 < 33; ++i2) {
      for (int32_t i3 = 0; i3 < 16; ++i3) {
        int32_t cse_var_1 = (i2 * 16);
        ((int16_t*)PadInput_let)[(((i1 * 528) + cse_var_1) + i3)] = (((i1 < 32) && (i2 < 32)) ? p0[(((i1 * 512) + cse_var_1) + i3)] : (int16_t)0);
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 4; ++n_oho_fused) {
    for (int32_t owo = 0; owo < 2; ++owo) {
      for (int32_t ohi = 0; ohi < 9; ++ohi) {
        for (int32_t ic = 0; ic < 16; ++ic) {
          ((int16_t*)data_vec_let)[((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic)];
        }
        for (int32_t ic_1 = 0; ic_1 < 16; ++ic_1) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_1) + 16)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_1) + 16)];
        }
        for (int32_t ic_2 = 0; ic_2 < 16; ++ic_2) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_2) + 32)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_2) + 32)];
        }
        for (int32_t ic_3 = 0; ic_3 < 16; ++ic_3) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_3) + 48)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_3) + 48)];
        }
        for (int32_t ic_4 = 0; ic_4 < 16; ++ic_4) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_4) + 64)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_4) + 64)];
        }
        for (int32_t ic_5 = 0; ic_5 < 16; ++ic_5) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_5) + 80)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_5) + 80)];
        }
        for (int32_t ic_6 = 0; ic_6 < 16; ++ic_6) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_6) + 96)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_6) + 96)];
        }
        for (int32_t ic_7 = 0; ic_7 < 16; ++ic_7) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_7) + 112)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_7) + 112)];
        }
        for (int32_t ic_8 = 0; ic_8 < 16; ++ic_8) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_8) + 128)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_8) + 128)];
        }
        for (int32_t ic_9 = 0; ic_9 < 16; ++ic_9) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_9) + 144)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_9) + 144)];
        }
        for (int32_t ic_10 = 0; ic_10 < 16; ++ic_10) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_10) + 160)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_10) + 160)];
        }
        for (int32_t ic_11 = 0; ic_11 < 16; ++ic_11) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_11) + 176)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_11) + 176)];
        }
        for (int32_t ic_12 = 0; ic_12 < 16; ++ic_12) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_12) + 192)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_12) + 192)];
        }
        for (int32_t ic_13 = 0; ic_13 < 16; ++ic_13) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_13) + 208)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_13) + 208)];
        }
        for (int32_t ic_14 = 0; ic_14 < 16; ++ic_14) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_14) + 224)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_14) + 224)];
        }
        for (int32_t ic_15 = 0; ic_15 < 16; ++ic_15) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_15) + 240)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_15) + 240)];
        }
        for (int32_t ic_16 = 0; ic_16 < 16; ++ic_16) {
          ((int16_t*)data_vec_let)[(((((n_oho_fused * 4896) + (owo * 2448)) + (ohi * 272)) + ic_16) + 256)] = ((int16_t*)PadInput_let)[(((((n_oho_fused * 4224) + (ohi * 528)) + (owo * 256)) + ic_16) + 256)];
        }
      }
    }
  }
  for (int32_t oco = 0; oco < 16; ++oco) {
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t oci = 0; oci < 2; ++oci) {
          ((int16_t*)PadInput_let)[((((oco * 288) + (kh * 96)) + (kw * 32)) + oci)] = ((int16_t*)fused_constant_17_let)[((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci)];
        }
        for (int32_t oci_1 = 0; oci_1 < 2; ++oci_1) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_1) + 2)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_1) + 32)];
        }
        for (int32_t oci_2 = 0; oci_2 < 2; ++oci_2) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_2) + 4)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_2) + 64)];
        }
        for (int32_t oci_3 = 0; oci_3 < 2; ++oci_3) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_3) + 6)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_3) + 96)];
        }
        for (int32_t oci_4 = 0; oci_4 < 2; ++oci_4) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_4) + 8)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_4) + 128)];
        }
        for (int32_t oci_5 = 0; oci_5 < 2; ++oci_5) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_5) + 10)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_5) + 160)];
        }
        for (int32_t oci_6 = 0; oci_6 < 2; ++oci_6) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_6) + 12)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_6) + 192)];
        }
        for (int32_t oci_7 = 0; oci_7 < 2; ++oci_7) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_7) + 14)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_7) + 224)];
        }
        for (int32_t oci_8 = 0; oci_8 < 2; ++oci_8) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_8) + 16)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_8) + 256)];
        }
        for (int32_t oci_9 = 0; oci_9 < 2; ++oci_9) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_9) + 18)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_9) + 288)];
        }
        for (int32_t oci_10 = 0; oci_10 < 2; ++oci_10) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_10) + 20)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_10) + 320)];
        }
        for (int32_t oci_11 = 0; oci_11 < 2; ++oci_11) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_11) + 22)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_11) + 352)];
        }
        for (int32_t oci_12 = 0; oci_12 < 2; ++oci_12) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_12) + 24)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_12) + 384)];
        }
        for (int32_t oci_13 = 0; oci_13 < 2; ++oci_13) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_13) + 26)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_13) + 416)];
        }
        for (int32_t oci_14 = 0; oci_14 < 2; ++oci_14) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_14) + 28)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_14) + 448)];
        }
        for (int32_t oci_15 = 0; oci_15 < 2; ++oci_15) {
          ((int16_t*)PadInput_let)[(((((oco * 288) + (kh * 96)) + (kw * 32)) + oci_15) + 30)] = ((int16_t*)fused_constant_17_let)[(((((kh * 1536) + (kw * 512)) + (oco * 2)) + oci_15) + 480)];
        }
      }
    }
  }
  for (int32_t oho = 0; oho < 4; ++oho) {
    for (int32_t owo_1 = 0; owo_1 < 2; ++owo_1) {
      for (int32_t oco_1 = 0; oco_1 < 16; ++oco_1) {
        for (int32_t owi = 0; owi < 8; ++owi) {
          int32_t cse_var_2 = ((((oho * 2048) + (owo_1 * 1024)) + (oco_1 * 64)) + (owi * 2));
          ((int32_t*)conv_let)[cse_var_2] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 1)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 16)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 17)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 32)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 33)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 48)] = 0;
          ((int32_t*)conv_let)[(cse_var_2 + 49)] = 0;
          for (int32_t ic_17 = 0; ic_17 < 16; ++ic_17) {
            int32_t cse_var_15 = (cse_var_2 + 49);
            int32_t cse_var_14 = (cse_var_2 + 48);
            int32_t cse_var_13 = (cse_var_2 + 33);
            int32_t cse_var_12 = (cse_var_2 + 32);
            int32_t cse_var_11 = (cse_var_2 + 17);
            int32_t cse_var_10 = (cse_var_2 + 16);
            int32_t cse_var_9 = (cse_var_2 + 1);
            int32_t cse_var_8 = ((oco_1 * 288) + (ic_17 * 2));
            int32_t cse_var_7 = (cse_var_8 + 1);
            int32_t cse_var_6 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_17);
            int32_t cse_var_5 = (cse_var_6 + 544);
            int32_t cse_var_4 = (cse_var_6 + 1632);
            int32_t cse_var_3 = (cse_var_6 + 1088);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_9] = (((int32_t*)conv_let)[cse_var_9] + (((int32_t)((int16_t*)data_vec_let)[cse_var_6]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_10] = (((int32_t*)conv_let)[cse_var_10] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_11] = (((int32_t*)conv_let)[cse_var_11] + (((int32_t)((int16_t*)data_vec_let)[cse_var_5]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_12] = (((int32_t*)conv_let)[cse_var_12] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_13] = (((int32_t*)conv_let)[cse_var_13] + (((int32_t)((int16_t*)data_vec_let)[cse_var_3]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
            ((int32_t*)conv_let)[cse_var_14] = (((int32_t*)conv_let)[cse_var_14] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_8])));
            ((int32_t*)conv_let)[cse_var_15] = (((int32_t*)conv_let)[cse_var_15] + (((int32_t)((int16_t*)data_vec_let)[cse_var_4]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_7])));
          }
          for (int32_t ic_18 = 0; ic_18 < 16; ++ic_18) {
            int32_t cse_var_30 = ((oco_1 * 288) + (ic_18 * 2));
            int32_t cse_var_29 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_18);
            int32_t cse_var_28 = (cse_var_2 + 49);
            int32_t cse_var_27 = (cse_var_2 + 48);
            int32_t cse_var_26 = (cse_var_2 + 33);
            int32_t cse_var_25 = (cse_var_2 + 32);
            int32_t cse_var_24 = (cse_var_2 + 17);
            int32_t cse_var_23 = (cse_var_2 + 16);
            int32_t cse_var_22 = (cse_var_2 + 1);
            int32_t cse_var_21 = (cse_var_30 + 33);
            int32_t cse_var_20 = (cse_var_30 + 32);
            int32_t cse_var_19 = (cse_var_29 + 560);
            int32_t cse_var_18 = (cse_var_29 + 1648);
            int32_t cse_var_17 = (cse_var_29 + 16);
            int32_t cse_var_16 = (cse_var_29 + 1104);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_22] = (((int32_t*)conv_let)[cse_var_22] + (((int32_t)((int16_t*)data_vec_let)[cse_var_17]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_23] = (((int32_t*)conv_let)[cse_var_23] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_24] = (((int32_t*)conv_let)[cse_var_24] + (((int32_t)((int16_t*)data_vec_let)[cse_var_19]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_25] = (((int32_t*)conv_let)[cse_var_25] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_26] = (((int32_t*)conv_let)[cse_var_26] + (((int32_t)((int16_t*)data_vec_let)[cse_var_16]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
            ((int32_t*)conv_let)[cse_var_27] = (((int32_t*)conv_let)[cse_var_27] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_20])));
            ((int32_t*)conv_let)[cse_var_28] = (((int32_t*)conv_let)[cse_var_28] + (((int32_t)((int16_t*)data_vec_let)[cse_var_18]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
          }
          for (int32_t ic_19 = 0; ic_19 < 16; ++ic_19) {
            int32_t cse_var_45 = ((oco_1 * 288) + (ic_19 * 2));
            int32_t cse_var_44 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_19);
            int32_t cse_var_43 = (cse_var_2 + 49);
            int32_t cse_var_42 = (cse_var_2 + 48);
            int32_t cse_var_41 = (cse_var_2 + 33);
            int32_t cse_var_40 = (cse_var_2 + 32);
            int32_t cse_var_39 = (cse_var_2 + 17);
            int32_t cse_var_38 = (cse_var_2 + 16);
            int32_t cse_var_37 = (cse_var_2 + 1);
            int32_t cse_var_36 = (cse_var_45 + 65);
            int32_t cse_var_35 = (cse_var_45 + 64);
            int32_t cse_var_34 = (cse_var_44 + 576);
            int32_t cse_var_33 = (cse_var_44 + 32);
            int32_t cse_var_32 = (cse_var_44 + 1664);
            int32_t cse_var_31 = (cse_var_44 + 1120);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_37] = (((int32_t*)conv_let)[cse_var_37] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_38] = (((int32_t*)conv_let)[cse_var_38] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_39] = (((int32_t*)conv_let)[cse_var_39] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_40] = (((int32_t*)conv_let)[cse_var_40] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_41] = (((int32_t*)conv_let)[cse_var_41] + (((int32_t)((int16_t*)data_vec_let)[cse_var_31]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
            ((int32_t*)conv_let)[cse_var_42] = (((int32_t*)conv_let)[cse_var_42] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
            ((int32_t*)conv_let)[cse_var_43] = (((int32_t*)conv_let)[cse_var_43] + (((int32_t)((int16_t*)data_vec_let)[cse_var_32]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
          }
          for (int32_t ic_20 = 0; ic_20 < 16; ++ic_20) {
            int32_t cse_var_60 = ((oco_1 * 288) + (ic_20 * 2));
            int32_t cse_var_59 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_20);
            int32_t cse_var_58 = (cse_var_2 + 49);
            int32_t cse_var_57 = (cse_var_2 + 48);
            int32_t cse_var_56 = (cse_var_2 + 33);
            int32_t cse_var_55 = (cse_var_2 + 32);
            int32_t cse_var_54 = (cse_var_2 + 17);
            int32_t cse_var_53 = (cse_var_2 + 16);
            int32_t cse_var_52 = (cse_var_2 + 1);
            int32_t cse_var_51 = (cse_var_60 + 97);
            int32_t cse_var_50 = (cse_var_60 + 96);
            int32_t cse_var_49 = (cse_var_59 + 816);
            int32_t cse_var_48 = (cse_var_59 + 272);
            int32_t cse_var_47 = (cse_var_59 + 1904);
            int32_t cse_var_46 = (cse_var_59 + 1360);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_52] = (((int32_t*)conv_let)[cse_var_52] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_53] = (((int32_t*)conv_let)[cse_var_53] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_54] = (((int32_t*)conv_let)[cse_var_54] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_55] = (((int32_t*)conv_let)[cse_var_55] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_56] = (((int32_t*)conv_let)[cse_var_56] + (((int32_t)((int16_t*)data_vec_let)[cse_var_46]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
            ((int32_t*)conv_let)[cse_var_57] = (((int32_t*)conv_let)[cse_var_57] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
            ((int32_t*)conv_let)[cse_var_58] = (((int32_t*)conv_let)[cse_var_58] + (((int32_t)((int16_t*)data_vec_let)[cse_var_47]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
          }
          for (int32_t ic_21 = 0; ic_21 < 16; ++ic_21) {
            int32_t cse_var_75 = ((oco_1 * 288) + (ic_21 * 2));
            int32_t cse_var_74 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_21);
            int32_t cse_var_73 = (cse_var_2 + 49);
            int32_t cse_var_72 = (cse_var_2 + 48);
            int32_t cse_var_71 = (cse_var_2 + 33);
            int32_t cse_var_70 = (cse_var_2 + 32);
            int32_t cse_var_69 = (cse_var_2 + 17);
            int32_t cse_var_68 = (cse_var_2 + 16);
            int32_t cse_var_67 = (cse_var_2 + 1);
            int32_t cse_var_66 = (cse_var_75 + 129);
            int32_t cse_var_65 = (cse_var_75 + 128);
            int32_t cse_var_64 = (cse_var_74 + 832);
            int32_t cse_var_63 = (cse_var_74 + 288);
            int32_t cse_var_62 = (cse_var_74 + 1920);
            int32_t cse_var_61 = (cse_var_74 + 1376);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_67] = (((int32_t*)conv_let)[cse_var_67] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_68] = (((int32_t*)conv_let)[cse_var_68] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_69] = (((int32_t*)conv_let)[cse_var_69] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_70] = (((int32_t*)conv_let)[cse_var_70] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_71] = (((int32_t*)conv_let)[cse_var_71] + (((int32_t)((int16_t*)data_vec_let)[cse_var_61]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
            ((int32_t*)conv_let)[cse_var_72] = (((int32_t*)conv_let)[cse_var_72] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
            ((int32_t*)conv_let)[cse_var_73] = (((int32_t*)conv_let)[cse_var_73] + (((int32_t)((int16_t*)data_vec_let)[cse_var_62]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
          }
          for (int32_t ic_22 = 0; ic_22 < 16; ++ic_22) {
            int32_t cse_var_90 = ((oco_1 * 288) + (ic_22 * 2));
            int32_t cse_var_89 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_22);
            int32_t cse_var_88 = (cse_var_2 + 49);
            int32_t cse_var_87 = (cse_var_2 + 48);
            int32_t cse_var_86 = (cse_var_2 + 33);
            int32_t cse_var_85 = (cse_var_2 + 32);
            int32_t cse_var_84 = (cse_var_2 + 17);
            int32_t cse_var_83 = (cse_var_2 + 16);
            int32_t cse_var_82 = (cse_var_2 + 1);
            int32_t cse_var_81 = (cse_var_90 + 161);
            int32_t cse_var_80 = (cse_var_90 + 160);
            int32_t cse_var_79 = (cse_var_89 + 848);
            int32_t cse_var_78 = (cse_var_89 + 304);
            int32_t cse_var_77 = (cse_var_89 + 1936);
            int32_t cse_var_76 = (cse_var_89 + 1392);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_82] = (((int32_t*)conv_let)[cse_var_82] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_83] = (((int32_t*)conv_let)[cse_var_83] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_84] = (((int32_t*)conv_let)[cse_var_84] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_85] = (((int32_t*)conv_let)[cse_var_85] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_86] = (((int32_t*)conv_let)[cse_var_86] + (((int32_t)((int16_t*)data_vec_let)[cse_var_76]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
            ((int32_t*)conv_let)[cse_var_87] = (((int32_t*)conv_let)[cse_var_87] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
            ((int32_t*)conv_let)[cse_var_88] = (((int32_t*)conv_let)[cse_var_88] + (((int32_t)((int16_t*)data_vec_let)[cse_var_77]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
          }
          for (int32_t ic_23 = 0; ic_23 < 16; ++ic_23) {
            int32_t cse_var_105 = ((oco_1 * 288) + (ic_23 * 2));
            int32_t cse_var_104 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_23);
            int32_t cse_var_103 = (cse_var_2 + 49);
            int32_t cse_var_102 = (cse_var_2 + 48);
            int32_t cse_var_101 = (cse_var_2 + 33);
            int32_t cse_var_100 = (cse_var_2 + 32);
            int32_t cse_var_99 = (cse_var_2 + 17);
            int32_t cse_var_98 = (cse_var_2 + 16);
            int32_t cse_var_97 = (cse_var_2 + 1);
            int32_t cse_var_96 = (cse_var_105 + 193);
            int32_t cse_var_95 = (cse_var_105 + 192);
            int32_t cse_var_94 = (cse_var_104 + 544);
            int32_t cse_var_93 = (cse_var_104 + 2176);
            int32_t cse_var_92 = (cse_var_104 + 1632);
            int32_t cse_var_91 = (cse_var_104 + 1088);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_97] = (((int32_t*)conv_let)[cse_var_97] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_98] = (((int32_t*)conv_let)[cse_var_98] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_99] = (((int32_t*)conv_let)[cse_var_99] + (((int32_t)((int16_t*)data_vec_let)[cse_var_91]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_100] = (((int32_t*)conv_let)[cse_var_100] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_101] = (((int32_t*)conv_let)[cse_var_101] + (((int32_t)((int16_t*)data_vec_let)[cse_var_92]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
            ((int32_t*)conv_let)[cse_var_102] = (((int32_t*)conv_let)[cse_var_102] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
            ((int32_t*)conv_let)[cse_var_103] = (((int32_t*)conv_let)[cse_var_103] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
          }
          for (int32_t ic_24 = 0; ic_24 < 16; ++ic_24) {
            int32_t cse_var_120 = ((oco_1 * 288) + (ic_24 * 2));
            int32_t cse_var_119 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_24);
            int32_t cse_var_118 = (cse_var_2 + 49);
            int32_t cse_var_117 = (cse_var_2 + 48);
            int32_t cse_var_116 = (cse_var_2 + 33);
            int32_t cse_var_115 = (cse_var_2 + 32);
            int32_t cse_var_114 = (cse_var_2 + 17);
            int32_t cse_var_113 = (cse_var_2 + 16);
            int32_t cse_var_112 = (cse_var_2 + 1);
            int32_t cse_var_111 = (cse_var_120 + 225);
            int32_t cse_var_110 = (cse_var_120 + 224);
            int32_t cse_var_109 = (cse_var_119 + 560);
            int32_t cse_var_108 = (cse_var_119 + 2192);
            int32_t cse_var_107 = (cse_var_119 + 1648);
            int32_t cse_var_106 = (cse_var_119 + 1104);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_112] = (((int32_t*)conv_let)[cse_var_112] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_113] = (((int32_t*)conv_let)[cse_var_113] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_114] = (((int32_t*)conv_let)[cse_var_114] + (((int32_t)((int16_t*)data_vec_let)[cse_var_106]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_115] = (((int32_t*)conv_let)[cse_var_115] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_116] = (((int32_t*)conv_let)[cse_var_116] + (((int32_t)((int16_t*)data_vec_let)[cse_var_107]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
            ((int32_t*)conv_let)[cse_var_117] = (((int32_t*)conv_let)[cse_var_117] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
            ((int32_t*)conv_let)[cse_var_118] = (((int32_t*)conv_let)[cse_var_118] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
          }
          for (int32_t ic_25 = 0; ic_25 < 16; ++ic_25) {
            int32_t cse_var_135 = ((oco_1 * 288) + (ic_25 * 2));
            int32_t cse_var_134 = ((((oho * 4896) + (owo_1 * 2448)) + (owi * 32)) + ic_25);
            int32_t cse_var_133 = (cse_var_2 + 49);
            int32_t cse_var_132 = (cse_var_2 + 48);
            int32_t cse_var_131 = (cse_var_2 + 33);
            int32_t cse_var_130 = (cse_var_2 + 32);
            int32_t cse_var_129 = (cse_var_2 + 17);
            int32_t cse_var_128 = (cse_var_2 + 16);
            int32_t cse_var_127 = (cse_var_2 + 1);
            int32_t cse_var_126 = (cse_var_135 + 257);
            int32_t cse_var_125 = (cse_var_135 + 256);
            int32_t cse_var_124 = (cse_var_134 + 576);
            int32_t cse_var_123 = (cse_var_134 + 2208);
            int32_t cse_var_122 = (cse_var_134 + 1664);
            int32_t cse_var_121 = (cse_var_134 + 1120);
            ((int32_t*)conv_let)[cse_var_2] = (((int32_t*)conv_let)[cse_var_2] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_127] = (((int32_t*)conv_let)[cse_var_127] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_128] = (((int32_t*)conv_let)[cse_var_128] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_129] = (((int32_t*)conv_let)[cse_var_129] + (((int32_t)((int16_t*)data_vec_let)[cse_var_121]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_130] = (((int32_t*)conv_let)[cse_var_130] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_131] = (((int32_t*)conv_let)[cse_var_131] + (((int32_t)((int16_t*)data_vec_let)[cse_var_122]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
            ((int32_t*)conv_let)[cse_var_132] = (((int32_t*)conv_let)[cse_var_132] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
            ((int32_t*)conv_let)[cse_var_133] = (((int32_t*)conv_let)[cse_var_133] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
          }
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 4; ++ax0_ax1_outer_fused) {
    for (int32_t ax2_outer = 0; ax2_outer < 2; ++ax2_outer) {
      for (int32_t ax3_outer = 0; ax3_outer < 16; ++ax3_outer) {
        for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
          int32_t cse_var_140 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_139 = (ax3_outer * 2);
          int32_t cse_var_138 = (cse_var_139 + 1);
          int32_t cse_var_137 = (((cse_var_140 + (ax2_outer * 256)) + (ax2_inner * 32)) + cse_var_139);
          int32_t cse_var_136 = (((cse_var_140 + (ax2_outer * 1024)) + (ax3_outer * 64)) + (ax2_inner * 2));
          int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_136]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_139])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_139]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_139]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_139])) - 128;
          int32_t __2 = (__1) < (127) ? (__1) : (127);
          int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
          int8_t __4 = (int8_t)127;
          int8_t __5 = (__3) < (__4) ? (__3) : (__4);
          int8_t __6 = (int8_t)-128;
          T_subtract[cse_var_137] = (((int16_t)((__5) > (__6) ? (__5) : (__6))) - (int16_t)-128);
          int32_t __7 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_136 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_138])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_138]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_138]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_138])) - 128;
          int32_t __8 = (__7) < (127) ? (__7) : (127);
          int8_t __9 = (int8_t)((__8) > (-128) ? (__8) : (-128));
          int8_t __10 = (__9) < (__4) ? (__9) : (__4);
          T_subtract[(cse_var_137 + 1)] = (((int16_t)((__10) > (__6) ? (__10) : (__6))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 8; ++ax2_inner_1) {
          int32_t cse_var_145 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_144 = (ax3_outer * 2);
          int32_t cse_var_143 = (cse_var_144 + 1);
          int32_t cse_var_142 = (((cse_var_145 + (ax2_outer * 256)) + (ax2_inner_1 * 32)) + cse_var_144);
          int32_t cse_var_141 = (((cse_var_145 + (ax2_outer * 1024)) + (ax3_outer * 64)) + (ax2_inner_1 * 2));
          int32_t __11 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_141 + 16)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_144])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_144]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_144]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_144])) - 128;
          int32_t __12 = (__11) < (127) ? (__11) : (127);
          int8_t __13 = (int8_t)((__12) > (-128) ? (__12) : (-128));
          int8_t __14 = (int8_t)127;
          int8_t __15 = (__13) < (__14) ? (__13) : (__14);
          int8_t __16 = (int8_t)-128;
          T_subtract[(cse_var_142 + 512)] = (((int16_t)((__15) > (__16) ? (__15) : (__16))) - (int16_t)-128);
          int32_t __17 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_141 + 17)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_143])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_143]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_143]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_143])) - 128;
          int32_t __18 = (__17) < (127) ? (__17) : (127);
          int8_t __19 = (int8_t)((__18) > (-128) ? (__18) : (-128));
          int8_t __20 = (__19) < (__14) ? (__19) : (__14);
          T_subtract[(cse_var_142 + 513)] = (((int16_t)((__20) > (__16) ? (__20) : (__16))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_2 = 0; ax2_inner_2 < 8; ++ax2_inner_2) {
          int32_t cse_var_150 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_149 = (ax3_outer * 2);
          int32_t cse_var_148 = (cse_var_149 + 1);
          int32_t cse_var_147 = (((cse_var_150 + (ax2_outer * 256)) + (ax2_inner_2 * 32)) + cse_var_149);
          int32_t cse_var_146 = (((cse_var_150 + (ax2_outer * 1024)) + (ax3_outer * 64)) + (ax2_inner_2 * 2));
          int32_t __21 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_146 + 32)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_149])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_149]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_149]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_149])) - 128;
          int32_t __22 = (__21) < (127) ? (__21) : (127);
          int8_t __23 = (int8_t)((__22) > (-128) ? (__22) : (-128));
          int8_t __24 = (int8_t)127;
          int8_t __25 = (__23) < (__24) ? (__23) : (__24);
          int8_t __26 = (int8_t)-128;
          T_subtract[(cse_var_147 + 1024)] = (((int16_t)((__25) > (__26) ? (__25) : (__26))) - (int16_t)-128);
          int32_t __27 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_146 + 33)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_148])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_148]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_148]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_148])) - 128;
          int32_t __28 = (__27) < (127) ? (__27) : (127);
          int8_t __29 = (int8_t)((__28) > (-128) ? (__28) : (-128));
          int8_t __30 = (__29) < (__24) ? (__29) : (__24);
          T_subtract[(cse_var_147 + 1025)] = (((int16_t)((__30) > (__26) ? (__30) : (__26))) - (int16_t)-128);
        }
        for (int32_t ax2_inner_3 = 0; ax2_inner_3 < 8; ++ax2_inner_3) {
          int32_t cse_var_155 = (ax0_ax1_outer_fused * 2048);
          int32_t cse_var_154 = (ax3_outer * 2);
          int32_t cse_var_153 = (cse_var_154 + 1);
          int32_t cse_var_152 = (((cse_var_155 + (ax2_outer * 256)) + (ax2_inner_3 * 32)) + cse_var_154);
          int32_t cse_var_151 = (((cse_var_155 + (ax2_outer * 1024)) + (ax3_outer * 64)) + (ax2_inner_3 * 2));
          int32_t __31 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_151 + 48)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_154])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_154]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_154]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_154])) - 128;
          int32_t __32 = (__31) < (127) ? (__31) : (127);
          int8_t __33 = (int8_t)((__32) > (-128) ? (__32) : (-128));
          int8_t __34 = (int8_t)127;
          int8_t __35 = (__33) < (__34) ? (__33) : (__34);
          int8_t __36 = (int8_t)-128;
          T_subtract[(cse_var_152 + 1536)] = (((int16_t)((__35) > (__36) ? (__35) : (__36))) - (int16_t)-128);
          int32_t __37 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_151 + 49)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_18_let)[cse_var_153])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_19_let)[cse_var_153]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_20_let)[cse_var_153]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_21_let)[cse_var_153])) - 128;
          int32_t __38 = (__37) < (127) ? (__37) : (127);
          int8_t __39 = (int8_t)((__38) > (-128) ? (__38) : (-128));
          int8_t __40 = (__39) < (__34) ? (__39) : (__34);
          T_subtract[(cse_var_152 + 1537)] = (((int16_t)((__40) > (__36) ? (__40) : (__36))) - (int16_t)-128);
        }
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__2(int16_t* p0, int16_t* T_subtract, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* fused_nn_conv2d_add_cast_multiply_add_constant_38_let = (&(global_const_workspace_18_var[157280]));
  void* fused_nn_conv2d_add_cast_multiply_constant_37_let = (&(global_const_workspace_18_var[155744]));
  void* fused_nn_conv2d_add_cast_constant_36_let = (&(global_const_workspace_18_var[158816]));
  void* fused_nn_conv2d_constant_35_let = (&(global_const_workspace_18_var[159840]));
  void* fused_constant_34_let = (&(global_const_workspace_18_var[73728]));
  void* PadInput_let = (&(global_workspace_19_var[0]));
  void* data_vec_let = (&(global_workspace_19_var[36864]));
  void* conv_let = (&(global_workspace_19_var[188704]));
  for (int32_t i1 = 0; i1 < 17; ++i1) {
    for (int32_t i2 = 0; i2 < 17; ++i2) {
      for (int32_t i3 = 0; i3 < 32; ++i3) {
        int32_t cse_var_1 = (i2 * 32);
        ((int16_t*)PadInput_let)[(((i1 * 544) + cse_var_1) + i3)] = (((i1 < 16) && (i2 < 16)) ? p0[(((i1 * 512) + cse_var_1) + i3)] : (int16_t)0);
      }
    }
  }
  for (int32_t n_oho_fused = 0; n_oho_fused < 4; ++n_oho_fused) {
    for (int32_t ohi = 0; ohi < 5; ++ohi) {
      for (int32_t ic = 0; ic < 32; ++ic) {
        int32_t cse_var_2 = (ohi * 544);
        ((int16_t*)data_vec_let)[(((n_oho_fused * 2720) + cse_var_2) + ic)] = ((int16_t*)PadInput_let)[(((n_oho_fused * 2176) + cse_var_2) + ic)];
      }
      for (int32_t ic_1 = 0; ic_1 < 32; ++ic_1) {
        int32_t cse_var_3 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_3) + ic_1) + 32)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_3) + ic_1) + 32)];
      }
      for (int32_t ic_2 = 0; ic_2 < 32; ++ic_2) {
        int32_t cse_var_4 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_4) + ic_2) + 64)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_4) + ic_2) + 64)];
      }
      for (int32_t ic_3 = 0; ic_3 < 32; ++ic_3) {
        int32_t cse_var_5 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_5) + ic_3) + 96)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_5) + ic_3) + 96)];
      }
      for (int32_t ic_4 = 0; ic_4 < 32; ++ic_4) {
        int32_t cse_var_6 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_6) + ic_4) + 128)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_6) + ic_4) + 128)];
      }
      for (int32_t ic_5 = 0; ic_5 < 32; ++ic_5) {
        int32_t cse_var_7 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_7) + ic_5) + 160)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_7) + ic_5) + 160)];
      }
      for (int32_t ic_6 = 0; ic_6 < 32; ++ic_6) {
        int32_t cse_var_8 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_8) + ic_6) + 192)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_8) + ic_6) + 192)];
      }
      for (int32_t ic_7 = 0; ic_7 < 32; ++ic_7) {
        int32_t cse_var_9 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_9) + ic_7) + 224)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_9) + ic_7) + 224)];
      }
      for (int32_t ic_8 = 0; ic_8 < 32; ++ic_8) {
        int32_t cse_var_10 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_10) + ic_8) + 256)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_10) + ic_8) + 256)];
      }
      for (int32_t ic_9 = 0; ic_9 < 32; ++ic_9) {
        int32_t cse_var_11 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_11) + ic_9) + 288)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_11) + ic_9) + 288)];
      }
      for (int32_t ic_10 = 0; ic_10 < 32; ++ic_10) {
        int32_t cse_var_12 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_12) + ic_10) + 320)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_12) + ic_10) + 320)];
      }
      for (int32_t ic_11 = 0; ic_11 < 32; ++ic_11) {
        int32_t cse_var_13 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_13) + ic_11) + 352)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_13) + ic_11) + 352)];
      }
      for (int32_t ic_12 = 0; ic_12 < 32; ++ic_12) {
        int32_t cse_var_14 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_14) + ic_12) + 384)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_14) + ic_12) + 384)];
      }
      for (int32_t ic_13 = 0; ic_13 < 32; ++ic_13) {
        int32_t cse_var_15 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_15) + ic_13) + 416)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_15) + ic_13) + 416)];
      }
      for (int32_t ic_14 = 0; ic_14 < 32; ++ic_14) {
        int32_t cse_var_16 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_16) + ic_14) + 448)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_16) + ic_14) + 448)];
      }
      for (int32_t ic_15 = 0; ic_15 < 32; ++ic_15) {
        int32_t cse_var_17 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_17) + ic_15) + 480)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_17) + ic_15) + 480)];
      }
      for (int32_t ic_16 = 0; ic_16 < 32; ++ic_16) {
        int32_t cse_var_18 = (ohi * 544);
        ((int16_t*)data_vec_let)[((((n_oho_fused * 2720) + cse_var_18) + ic_16) + 512)] = ((int16_t*)PadInput_let)[((((n_oho_fused * 2176) + cse_var_18) + ic_16) + 512)];
      }
    }
  }
  for (int32_t oco = 0; oco < 16; ++oco) {
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t oci = 0; oci < 4; ++oci) {
          ((int16_t*)PadInput_let)[((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci)] = ((int16_t*)fused_constant_34_let)[((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci)];
        }
        for (int32_t oci_1 = 0; oci_1 < 4; ++oci_1) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_1) + 4)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_1) + 64)];
        }
        for (int32_t oci_2 = 0; oci_2 < 4; ++oci_2) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_2) + 8)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_2) + 128)];
        }
        for (int32_t oci_3 = 0; oci_3 < 4; ++oci_3) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_3) + 12)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_3) + 192)];
        }
        for (int32_t oci_4 = 0; oci_4 < 4; ++oci_4) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_4) + 16)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_4) + 256)];
        }
        for (int32_t oci_5 = 0; oci_5 < 4; ++oci_5) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_5) + 20)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_5) + 320)];
        }
        for (int32_t oci_6 = 0; oci_6 < 4; ++oci_6) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_6) + 24)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_6) + 384)];
        }
        for (int32_t oci_7 = 0; oci_7 < 4; ++oci_7) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_7) + 28)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_7) + 448)];
        }
        for (int32_t oci_8 = 0; oci_8 < 4; ++oci_8) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_8) + 32)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_8) + 512)];
        }
        for (int32_t oci_9 = 0; oci_9 < 4; ++oci_9) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_9) + 36)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_9) + 576)];
        }
        for (int32_t oci_10 = 0; oci_10 < 4; ++oci_10) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_10) + 40)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_10) + 640)];
        }
        for (int32_t oci_11 = 0; oci_11 < 4; ++oci_11) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_11) + 44)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_11) + 704)];
        }
        for (int32_t oci_12 = 0; oci_12 < 4; ++oci_12) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_12) + 48)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_12) + 768)];
        }
        for (int32_t oci_13 = 0; oci_13 < 4; ++oci_13) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_13) + 52)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_13) + 832)];
        }
        for (int32_t oci_14 = 0; oci_14 < 4; ++oci_14) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_14) + 56)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_14) + 896)];
        }
        for (int32_t oci_15 = 0; oci_15 < 4; ++oci_15) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_15) + 60)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_15) + 960)];
        }
        for (int32_t oci_16 = 0; oci_16 < 4; ++oci_16) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_16) + 64)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_16) + 1024)];
        }
        for (int32_t oci_17 = 0; oci_17 < 4; ++oci_17) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_17) + 68)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_17) + 1088)];
        }
        for (int32_t oci_18 = 0; oci_18 < 4; ++oci_18) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_18) + 72)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_18) + 1152)];
        }
        for (int32_t oci_19 = 0; oci_19 < 4; ++oci_19) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_19) + 76)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_19) + 1216)];
        }
        for (int32_t oci_20 = 0; oci_20 < 4; ++oci_20) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_20) + 80)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_20) + 1280)];
        }
        for (int32_t oci_21 = 0; oci_21 < 4; ++oci_21) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_21) + 84)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_21) + 1344)];
        }
        for (int32_t oci_22 = 0; oci_22 < 4; ++oci_22) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_22) + 88)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_22) + 1408)];
        }
        for (int32_t oci_23 = 0; oci_23 < 4; ++oci_23) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_23) + 92)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_23) + 1472)];
        }
        for (int32_t oci_24 = 0; oci_24 < 4; ++oci_24) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_24) + 96)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_24) + 1536)];
        }
        for (int32_t oci_25 = 0; oci_25 < 4; ++oci_25) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_25) + 100)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_25) + 1600)];
        }
        for (int32_t oci_26 = 0; oci_26 < 4; ++oci_26) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_26) + 104)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_26) + 1664)];
        }
        for (int32_t oci_27 = 0; oci_27 < 4; ++oci_27) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_27) + 108)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_27) + 1728)];
        }
        for (int32_t oci_28 = 0; oci_28 < 4; ++oci_28) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_28) + 112)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_28) + 1792)];
        }
        for (int32_t oci_29 = 0; oci_29 < 4; ++oci_29) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_29) + 116)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_29) + 1856)];
        }
        for (int32_t oci_30 = 0; oci_30 < 4; ++oci_30) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_30) + 120)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_30) + 1920)];
        }
        for (int32_t oci_31 = 0; oci_31 < 4; ++oci_31) {
          ((int16_t*)PadInput_let)[(((((oco * 1152) + (kh * 384)) + (kw * 128)) + oci_31) + 124)] = ((int16_t*)fused_constant_34_let)[(((((kh * 6144) + (kw * 2048)) + (oco * 4)) + oci_31) + 1984)];
        }
      }
    }
  }
  for (int32_t oho = 0; oho < 4; ++oho) {
    for (int32_t oco_1 = 0; oco_1 < 16; ++oco_1) {
      for (int32_t owi = 0; owi < 8; ++owi) {
        int32_t cse_var_19 = (((oho * 1024) + (oco_1 * 64)) + (owi * 4));
        ((int32_t*)conv_let)[cse_var_19] = 0;
        ((int32_t*)conv_let)[(cse_var_19 + 1)] = 0;
        ((int32_t*)conv_let)[(cse_var_19 + 2)] = 0;
        ((int32_t*)conv_let)[(cse_var_19 + 3)] = 0;
        ((int32_t*)conv_let)[(cse_var_19 + 32)] = 0;
        ((int32_t*)conv_let)[(cse_var_19 + 33)] = 0;
        ((int32_t*)conv_let)[(cse_var_19 + 34)] = 0;
        ((int32_t*)conv_let)[(cse_var_19 + 35)] = 0;
        for (int32_t ic_17 = 0; ic_17 < 32; ++ic_17) {
          int32_t cse_var_32 = (cse_var_19 + 35);
          int32_t cse_var_31 = (cse_var_19 + 34);
          int32_t cse_var_30 = (cse_var_19 + 33);
          int32_t cse_var_29 = (cse_var_19 + 32);
          int32_t cse_var_28 = (cse_var_19 + 3);
          int32_t cse_var_27 = (cse_var_19 + 2);
          int32_t cse_var_26 = (cse_var_19 + 1);
          int32_t cse_var_25 = ((oco_1 * 1152) + (ic_17 * 4));
          int32_t cse_var_24 = (((oho * 2720) + (owi * 64)) + ic_17);
          int32_t cse_var_23 = (cse_var_25 + 3);
          int32_t cse_var_22 = (cse_var_25 + 2);
          int32_t cse_var_21 = (cse_var_25 + 1);
          int32_t cse_var_20 = (cse_var_24 + 1088);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_24]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_25])));
          ((int32_t*)conv_let)[cse_var_26] = (((int32_t*)conv_let)[cse_var_26] + (((int32_t)((int16_t*)data_vec_let)[cse_var_24]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
          ((int32_t*)conv_let)[cse_var_27] = (((int32_t*)conv_let)[cse_var_27] + (((int32_t)((int16_t*)data_vec_let)[cse_var_24]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_22])));
          ((int32_t*)conv_let)[cse_var_28] = (((int32_t*)conv_let)[cse_var_28] + (((int32_t)((int16_t*)data_vec_let)[cse_var_24]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_23])));
          ((int32_t*)conv_let)[cse_var_29] = (((int32_t*)conv_let)[cse_var_29] + (((int32_t)((int16_t*)data_vec_let)[cse_var_20]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_25])));
          ((int32_t*)conv_let)[cse_var_30] = (((int32_t*)conv_let)[cse_var_30] + (((int32_t)((int16_t*)data_vec_let)[cse_var_20]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_21])));
          ((int32_t*)conv_let)[cse_var_31] = (((int32_t*)conv_let)[cse_var_31] + (((int32_t)((int16_t*)data_vec_let)[cse_var_20]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_22])));
          ((int32_t*)conv_let)[cse_var_32] = (((int32_t*)conv_let)[cse_var_32] + (((int32_t)((int16_t*)data_vec_let)[cse_var_20]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_23])));
        }
        for (int32_t ic_18 = 0; ic_18 < 32; ++ic_18) {
          int32_t cse_var_47 = ((oco_1 * 1152) + (ic_18 * 4));
          int32_t cse_var_46 = (((oho * 2720) + (owi * 64)) + ic_18);
          int32_t cse_var_45 = (cse_var_19 + 35);
          int32_t cse_var_44 = (cse_var_19 + 34);
          int32_t cse_var_43 = (cse_var_19 + 33);
          int32_t cse_var_42 = (cse_var_19 + 32);
          int32_t cse_var_41 = (cse_var_19 + 3);
          int32_t cse_var_40 = (cse_var_19 + 2);
          int32_t cse_var_39 = (cse_var_19 + 1);
          int32_t cse_var_38 = (cse_var_47 + 131);
          int32_t cse_var_37 = (cse_var_47 + 130);
          int32_t cse_var_36 = (cse_var_47 + 129);
          int32_t cse_var_35 = (cse_var_47 + 128);
          int32_t cse_var_34 = (cse_var_46 + 32);
          int32_t cse_var_33 = (cse_var_46 + 1120);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
          ((int32_t*)conv_let)[cse_var_39] = (((int32_t*)conv_let)[cse_var_39] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
          ((int32_t*)conv_let)[cse_var_40] = (((int32_t*)conv_let)[cse_var_40] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_37])));
          ((int32_t*)conv_let)[cse_var_41] = (((int32_t*)conv_let)[cse_var_41] + (((int32_t)((int16_t*)data_vec_let)[cse_var_34]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_38])));
          ((int32_t*)conv_let)[cse_var_42] = (((int32_t*)conv_let)[cse_var_42] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_35])));
          ((int32_t*)conv_let)[cse_var_43] = (((int32_t*)conv_let)[cse_var_43] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_36])));
          ((int32_t*)conv_let)[cse_var_44] = (((int32_t*)conv_let)[cse_var_44] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_37])));
          ((int32_t*)conv_let)[cse_var_45] = (((int32_t*)conv_let)[cse_var_45] + (((int32_t)((int16_t*)data_vec_let)[cse_var_33]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_38])));
        }
        for (int32_t ic_19 = 0; ic_19 < 32; ++ic_19) {
          int32_t cse_var_62 = ((oco_1 * 1152) + (ic_19 * 4));
          int32_t cse_var_61 = (((oho * 2720) + (owi * 64)) + ic_19);
          int32_t cse_var_60 = (cse_var_19 + 35);
          int32_t cse_var_59 = (cse_var_19 + 34);
          int32_t cse_var_58 = (cse_var_19 + 33);
          int32_t cse_var_57 = (cse_var_19 + 32);
          int32_t cse_var_56 = (cse_var_19 + 3);
          int32_t cse_var_55 = (cse_var_19 + 2);
          int32_t cse_var_54 = (cse_var_19 + 1);
          int32_t cse_var_53 = (cse_var_62 + 259);
          int32_t cse_var_52 = (cse_var_62 + 258);
          int32_t cse_var_51 = (cse_var_62 + 257);
          int32_t cse_var_50 = (cse_var_62 + 256);
          int32_t cse_var_49 = (cse_var_61 + 64);
          int32_t cse_var_48 = (cse_var_61 + 1152);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
          ((int32_t*)conv_let)[cse_var_54] = (((int32_t*)conv_let)[cse_var_54] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
          ((int32_t*)conv_let)[cse_var_55] = (((int32_t*)conv_let)[cse_var_55] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_52])));
          ((int32_t*)conv_let)[cse_var_56] = (((int32_t*)conv_let)[cse_var_56] + (((int32_t)((int16_t*)data_vec_let)[cse_var_49]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_53])));
          ((int32_t*)conv_let)[cse_var_57] = (((int32_t*)conv_let)[cse_var_57] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_50])));
          ((int32_t*)conv_let)[cse_var_58] = (((int32_t*)conv_let)[cse_var_58] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_51])));
          ((int32_t*)conv_let)[cse_var_59] = (((int32_t*)conv_let)[cse_var_59] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_52])));
          ((int32_t*)conv_let)[cse_var_60] = (((int32_t*)conv_let)[cse_var_60] + (((int32_t)((int16_t*)data_vec_let)[cse_var_48]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_53])));
        }
        for (int32_t ic_20 = 0; ic_20 < 32; ++ic_20) {
          int32_t cse_var_77 = ((oco_1 * 1152) + (ic_20 * 4));
          int32_t cse_var_76 = (((oho * 2720) + (owi * 64)) + ic_20);
          int32_t cse_var_75 = (cse_var_19 + 35);
          int32_t cse_var_74 = (cse_var_19 + 34);
          int32_t cse_var_73 = (cse_var_19 + 33);
          int32_t cse_var_72 = (cse_var_19 + 32);
          int32_t cse_var_71 = (cse_var_19 + 3);
          int32_t cse_var_70 = (cse_var_19 + 2);
          int32_t cse_var_69 = (cse_var_19 + 1);
          int32_t cse_var_68 = (cse_var_77 + 387);
          int32_t cse_var_67 = (cse_var_77 + 386);
          int32_t cse_var_66 = (cse_var_77 + 385);
          int32_t cse_var_65 = (cse_var_77 + 384);
          int32_t cse_var_64 = (cse_var_76 + 544);
          int32_t cse_var_63 = (cse_var_76 + 1632);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
          ((int32_t*)conv_let)[cse_var_69] = (((int32_t*)conv_let)[cse_var_69] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
          ((int32_t*)conv_let)[cse_var_70] = (((int32_t*)conv_let)[cse_var_70] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_67])));
          ((int32_t*)conv_let)[cse_var_71] = (((int32_t*)conv_let)[cse_var_71] + (((int32_t)((int16_t*)data_vec_let)[cse_var_64]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_68])));
          ((int32_t*)conv_let)[cse_var_72] = (((int32_t*)conv_let)[cse_var_72] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_65])));
          ((int32_t*)conv_let)[cse_var_73] = (((int32_t*)conv_let)[cse_var_73] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_66])));
          ((int32_t*)conv_let)[cse_var_74] = (((int32_t*)conv_let)[cse_var_74] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_67])));
          ((int32_t*)conv_let)[cse_var_75] = (((int32_t*)conv_let)[cse_var_75] + (((int32_t)((int16_t*)data_vec_let)[cse_var_63]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_68])));
        }
        for (int32_t ic_21 = 0; ic_21 < 32; ++ic_21) {
          int32_t cse_var_92 = ((oco_1 * 1152) + (ic_21 * 4));
          int32_t cse_var_91 = (((oho * 2720) + (owi * 64)) + ic_21);
          int32_t cse_var_90 = (cse_var_19 + 35);
          int32_t cse_var_89 = (cse_var_19 + 34);
          int32_t cse_var_88 = (cse_var_19 + 33);
          int32_t cse_var_87 = (cse_var_19 + 32);
          int32_t cse_var_86 = (cse_var_19 + 3);
          int32_t cse_var_85 = (cse_var_19 + 2);
          int32_t cse_var_84 = (cse_var_19 + 1);
          int32_t cse_var_83 = (cse_var_92 + 515);
          int32_t cse_var_82 = (cse_var_92 + 514);
          int32_t cse_var_81 = (cse_var_92 + 513);
          int32_t cse_var_80 = (cse_var_92 + 512);
          int32_t cse_var_79 = (cse_var_91 + 576);
          int32_t cse_var_78 = (cse_var_91 + 1664);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
          ((int32_t*)conv_let)[cse_var_84] = (((int32_t*)conv_let)[cse_var_84] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
          ((int32_t*)conv_let)[cse_var_85] = (((int32_t*)conv_let)[cse_var_85] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_82])));
          ((int32_t*)conv_let)[cse_var_86] = (((int32_t*)conv_let)[cse_var_86] + (((int32_t)((int16_t*)data_vec_let)[cse_var_79]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_83])));
          ((int32_t*)conv_let)[cse_var_87] = (((int32_t*)conv_let)[cse_var_87] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_80])));
          ((int32_t*)conv_let)[cse_var_88] = (((int32_t*)conv_let)[cse_var_88] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_81])));
          ((int32_t*)conv_let)[cse_var_89] = (((int32_t*)conv_let)[cse_var_89] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_82])));
          ((int32_t*)conv_let)[cse_var_90] = (((int32_t*)conv_let)[cse_var_90] + (((int32_t)((int16_t*)data_vec_let)[cse_var_78]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_83])));
        }
        for (int32_t ic_22 = 0; ic_22 < 32; ++ic_22) {
          int32_t cse_var_107 = ((oco_1 * 1152) + (ic_22 * 4));
          int32_t cse_var_106 = (((oho * 2720) + (owi * 64)) + ic_22);
          int32_t cse_var_105 = (cse_var_19 + 35);
          int32_t cse_var_104 = (cse_var_19 + 34);
          int32_t cse_var_103 = (cse_var_19 + 33);
          int32_t cse_var_102 = (cse_var_19 + 32);
          int32_t cse_var_101 = (cse_var_19 + 3);
          int32_t cse_var_100 = (cse_var_19 + 2);
          int32_t cse_var_99 = (cse_var_19 + 1);
          int32_t cse_var_98 = (cse_var_107 + 643);
          int32_t cse_var_97 = (cse_var_107 + 642);
          int32_t cse_var_96 = (cse_var_107 + 641);
          int32_t cse_var_95 = (cse_var_107 + 640);
          int32_t cse_var_94 = (cse_var_106 + 608);
          int32_t cse_var_93 = (cse_var_106 + 1696);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
          ((int32_t*)conv_let)[cse_var_99] = (((int32_t*)conv_let)[cse_var_99] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
          ((int32_t*)conv_let)[cse_var_100] = (((int32_t*)conv_let)[cse_var_100] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_97])));
          ((int32_t*)conv_let)[cse_var_101] = (((int32_t*)conv_let)[cse_var_101] + (((int32_t)((int16_t*)data_vec_let)[cse_var_94]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_98])));
          ((int32_t*)conv_let)[cse_var_102] = (((int32_t*)conv_let)[cse_var_102] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_95])));
          ((int32_t*)conv_let)[cse_var_103] = (((int32_t*)conv_let)[cse_var_103] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_96])));
          ((int32_t*)conv_let)[cse_var_104] = (((int32_t*)conv_let)[cse_var_104] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_97])));
          ((int32_t*)conv_let)[cse_var_105] = (((int32_t*)conv_let)[cse_var_105] + (((int32_t)((int16_t*)data_vec_let)[cse_var_93]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_98])));
        }
        for (int32_t ic_23 = 0; ic_23 < 32; ++ic_23) {
          int32_t cse_var_122 = ((oco_1 * 1152) + (ic_23 * 4));
          int32_t cse_var_121 = (((oho * 2720) + (owi * 64)) + ic_23);
          int32_t cse_var_120 = (cse_var_19 + 35);
          int32_t cse_var_119 = (cse_var_19 + 34);
          int32_t cse_var_118 = (cse_var_19 + 33);
          int32_t cse_var_117 = (cse_var_19 + 32);
          int32_t cse_var_116 = (cse_var_19 + 3);
          int32_t cse_var_115 = (cse_var_19 + 2);
          int32_t cse_var_114 = (cse_var_19 + 1);
          int32_t cse_var_113 = (cse_var_122 + 771);
          int32_t cse_var_112 = (cse_var_122 + 770);
          int32_t cse_var_111 = (cse_var_122 + 769);
          int32_t cse_var_110 = (cse_var_122 + 768);
          int32_t cse_var_109 = (cse_var_121 + 2176);
          int32_t cse_var_108 = (cse_var_121 + 1088);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
          ((int32_t*)conv_let)[cse_var_114] = (((int32_t*)conv_let)[cse_var_114] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
          ((int32_t*)conv_let)[cse_var_115] = (((int32_t*)conv_let)[cse_var_115] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_112])));
          ((int32_t*)conv_let)[cse_var_116] = (((int32_t*)conv_let)[cse_var_116] + (((int32_t)((int16_t*)data_vec_let)[cse_var_108]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_113])));
          ((int32_t*)conv_let)[cse_var_117] = (((int32_t*)conv_let)[cse_var_117] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_110])));
          ((int32_t*)conv_let)[cse_var_118] = (((int32_t*)conv_let)[cse_var_118] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_111])));
          ((int32_t*)conv_let)[cse_var_119] = (((int32_t*)conv_let)[cse_var_119] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_112])));
          ((int32_t*)conv_let)[cse_var_120] = (((int32_t*)conv_let)[cse_var_120] + (((int32_t)((int16_t*)data_vec_let)[cse_var_109]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_113])));
        }
        for (int32_t ic_24 = 0; ic_24 < 32; ++ic_24) {
          int32_t cse_var_137 = ((oco_1 * 1152) + (ic_24 * 4));
          int32_t cse_var_136 = (((oho * 2720) + (owi * 64)) + ic_24);
          int32_t cse_var_135 = (cse_var_19 + 35);
          int32_t cse_var_134 = (cse_var_19 + 34);
          int32_t cse_var_133 = (cse_var_19 + 33);
          int32_t cse_var_132 = (cse_var_19 + 32);
          int32_t cse_var_131 = (cse_var_19 + 3);
          int32_t cse_var_130 = (cse_var_19 + 2);
          int32_t cse_var_129 = (cse_var_19 + 1);
          int32_t cse_var_128 = (cse_var_137 + 899);
          int32_t cse_var_127 = (cse_var_137 + 898);
          int32_t cse_var_126 = (cse_var_137 + 897);
          int32_t cse_var_125 = (cse_var_137 + 896);
          int32_t cse_var_124 = (cse_var_136 + 2208);
          int32_t cse_var_123 = (cse_var_136 + 1120);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
          ((int32_t*)conv_let)[cse_var_129] = (((int32_t*)conv_let)[cse_var_129] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
          ((int32_t*)conv_let)[cse_var_130] = (((int32_t*)conv_let)[cse_var_130] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_127])));
          ((int32_t*)conv_let)[cse_var_131] = (((int32_t*)conv_let)[cse_var_131] + (((int32_t)((int16_t*)data_vec_let)[cse_var_123]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_128])));
          ((int32_t*)conv_let)[cse_var_132] = (((int32_t*)conv_let)[cse_var_132] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_125])));
          ((int32_t*)conv_let)[cse_var_133] = (((int32_t*)conv_let)[cse_var_133] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_126])));
          ((int32_t*)conv_let)[cse_var_134] = (((int32_t*)conv_let)[cse_var_134] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_127])));
          ((int32_t*)conv_let)[cse_var_135] = (((int32_t*)conv_let)[cse_var_135] + (((int32_t)((int16_t*)data_vec_let)[cse_var_124]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_128])));
        }
        for (int32_t ic_25 = 0; ic_25 < 32; ++ic_25) {
          int32_t cse_var_152 = ((oco_1 * 1152) + (ic_25 * 4));
          int32_t cse_var_151 = (((oho * 2720) + (owi * 64)) + ic_25);
          int32_t cse_var_150 = (cse_var_19 + 35);
          int32_t cse_var_149 = (cse_var_19 + 34);
          int32_t cse_var_148 = (cse_var_19 + 33);
          int32_t cse_var_147 = (cse_var_19 + 32);
          int32_t cse_var_146 = (cse_var_19 + 3);
          int32_t cse_var_145 = (cse_var_19 + 2);
          int32_t cse_var_144 = (cse_var_19 + 1);
          int32_t cse_var_143 = (cse_var_152 + 1027);
          int32_t cse_var_142 = (cse_var_152 + 1026);
          int32_t cse_var_141 = (cse_var_152 + 1025);
          int32_t cse_var_140 = (cse_var_152 + 1024);
          int32_t cse_var_139 = (cse_var_151 + 2240);
          int32_t cse_var_138 = (cse_var_151 + 1152);
          ((int32_t*)conv_let)[cse_var_19] = (((int32_t*)conv_let)[cse_var_19] + (((int32_t)((int16_t*)data_vec_let)[cse_var_138]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_140])));
          ((int32_t*)conv_let)[cse_var_144] = (((int32_t*)conv_let)[cse_var_144] + (((int32_t)((int16_t*)data_vec_let)[cse_var_138]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_141])));
          ((int32_t*)conv_let)[cse_var_145] = (((int32_t*)conv_let)[cse_var_145] + (((int32_t)((int16_t*)data_vec_let)[cse_var_138]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_142])));
          ((int32_t*)conv_let)[cse_var_146] = (((int32_t*)conv_let)[cse_var_146] + (((int32_t)((int16_t*)data_vec_let)[cse_var_138]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_143])));
          ((int32_t*)conv_let)[cse_var_147] = (((int32_t*)conv_let)[cse_var_147] + (((int32_t)((int16_t*)data_vec_let)[cse_var_139]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_140])));
          ((int32_t*)conv_let)[cse_var_148] = (((int32_t*)conv_let)[cse_var_148] + (((int32_t)((int16_t*)data_vec_let)[cse_var_139]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_141])));
          ((int32_t*)conv_let)[cse_var_149] = (((int32_t*)conv_let)[cse_var_149] + (((int32_t)((int16_t*)data_vec_let)[cse_var_139]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_142])));
          ((int32_t*)conv_let)[cse_var_150] = (((int32_t*)conv_let)[cse_var_150] + (((int32_t)((int16_t*)data_vec_let)[cse_var_139]) * ((int32_t)((int16_t*)PadInput_let)[cse_var_143])));
        }
      }
    }
  }
  for (int32_t ax0_ax1_outer_fused = 0; ax0_ax1_outer_fused < 4; ++ax0_ax1_outer_fused) {
    for (int32_t ax3_outer = 0; ax3_outer < 16; ++ax3_outer) {
      for (int32_t ax2_inner = 0; ax2_inner < 8; ++ax2_inner) {
        int32_t cse_var_159 = (ax0_ax1_outer_fused * 1024);
        int32_t cse_var_158 = (ax3_outer * 4);
        int32_t cse_var_157 = (cse_var_158 + 3);
        int32_t cse_var_156 = (cse_var_158 + 2);
        int32_t cse_var_155 = (cse_var_158 + 1);
        int32_t cse_var_154 = ((cse_var_159 + (ax3_outer * 64)) + (ax2_inner * 4));
        int32_t cse_var_153 = ((cse_var_159 + (ax2_inner * 64)) + cse_var_158);
        int32_t __1 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[cse_var_154]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_158])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_158]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_158]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_158])) - 128;
        int32_t __2 = (__1) < (127) ? (__1) : (127);
        int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
        int8_t __4 = (int8_t)127;
        int8_t __5 = (__3) < (__4) ? (__3) : (__4);
        int8_t __6 = (int8_t)-128;
        T_subtract[cse_var_153] = (((int16_t)((__5) > (__6) ? (__5) : (__6))) - (int16_t)-128);
        int32_t __7 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_154 + 1)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_155])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_155]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_155]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_155])) - 128;
        int32_t __8 = (__7) < (127) ? (__7) : (127);
        int8_t __9 = (int8_t)((__8) > (-128) ? (__8) : (-128));
        int8_t __10 = (__9) < (__4) ? (__9) : (__4);
        T_subtract[(cse_var_153 + 1)] = (((int16_t)((__10) > (__6) ? (__10) : (__6))) - (int16_t)-128);
        int32_t __11 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_154 + 2)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_156])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_156]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_156]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_156])) - 128;
        int32_t __12 = (__11) < (127) ? (__11) : (127);
        int8_t __13 = (int8_t)((__12) > (-128) ? (__12) : (-128));
        int8_t __14 = (__13) < (__4) ? (__13) : (__4);
        T_subtract[(cse_var_153 + 2)] = (((int16_t)((__14) > (__6) ? (__14) : (__6))) - (int16_t)-128);
        int32_t __15 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_154 + 3)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_157])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_157]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_157]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_157])) - 128;
        int32_t __16 = (__15) < (127) ? (__15) : (127);
        int8_t __17 = (int8_t)((__16) > (-128) ? (__16) : (-128));
        int8_t __18 = (__17) < (__4) ? (__17) : (__4);
        T_subtract[(cse_var_153 + 3)] = (((int16_t)((__18) > (__6) ? (__18) : (__6))) - (int16_t)-128);
      }
      for (int32_t ax2_inner_1 = 0; ax2_inner_1 < 8; ++ax2_inner_1) {
        int32_t cse_var_166 = (ax0_ax1_outer_fused * 1024);
        int32_t cse_var_165 = (ax3_outer * 4);
        int32_t cse_var_164 = (cse_var_165 + 3);
        int32_t cse_var_163 = (cse_var_165 + 2);
        int32_t cse_var_162 = (cse_var_165 + 1);
        int32_t cse_var_161 = ((cse_var_166 + (ax3_outer * 64)) + (ax2_inner_1 * 4));
        int32_t cse_var_160 = ((cse_var_166 + (ax2_inner_1 * 64)) + cse_var_165);
        int32_t __19 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_161 + 32)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_165])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_165]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_165]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_165])) - 128;
        int32_t __20 = (__19) < (127) ? (__19) : (127);
        int8_t __21 = (int8_t)((__20) > (-128) ? (__20) : (-128));
        int8_t __22 = (int8_t)127;
        int8_t __23 = (__21) < (__22) ? (__21) : (__22);
        int8_t __24 = (int8_t)-128;
        T_subtract[(cse_var_160 + 512)] = (((int16_t)((__23) > (__24) ? (__23) : (__24))) - (int16_t)-128);
        int32_t __25 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_161 + 33)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_162])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_162]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_162]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_162])) - 128;
        int32_t __26 = (__25) < (127) ? (__25) : (127);
        int8_t __27 = (int8_t)((__26) > (-128) ? (__26) : (-128));
        int8_t __28 = (__27) < (__22) ? (__27) : (__22);
        T_subtract[(cse_var_160 + 513)] = (((int16_t)((__28) > (__24) ? (__28) : (__24))) - (int16_t)-128);
        int32_t __29 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_161 + 34)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_163])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_163]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_163]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_163])) - 128;
        int32_t __30 = (__29) < (127) ? (__29) : (127);
        int8_t __31 = (int8_t)((__30) > (-128) ? (__30) : (-128));
        int8_t __32 = (__31) < (__22) ? (__31) : (__22);
        T_subtract[(cse_var_160 + 514)] = (((int16_t)((__32) > (__24) ? (__32) : (__24))) - (int16_t)-128);
        int32_t __33 = ((int32_t)((((((int64_t)((int32_t*)conv_let)[(cse_var_161 + 35)]) + ((int64_t)((int32_t*)fused_nn_conv2d_constant_35_let)[cse_var_164])) * ((int64_t*)fused_nn_conv2d_add_cast_constant_36_let)[cse_var_164]) + ((int64_t*)fused_nn_conv2d_add_cast_multiply_constant_37_let)[cse_var_164]) >> ((int64_t*)fused_nn_conv2d_add_cast_multiply_add_constant_38_let)[cse_var_164])) - 128;
        int32_t __34 = (__33) < (127) ? (__33) : (127);
        int8_t __35 = (int8_t)((__34) > (-128) ? (__34) : (-128));
        int8_t __36 = (__35) < (__22) ? (__35) : (__22);
        T_subtract[(cse_var_160 + 515)] = (((int16_t)((__36) > (__24) ? (__36) : (__24))) - (int16_t)-128);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_add_fixed_point_multiply_add_clip_cast_cast_subtract_cast_multipl_10ab34501c4d620e_(int16_t* p0, float* T_multiply, uint8_t* global_const_workspace_28_var, uint8_t* global_workspace_29_var) {
  void* fused_nn_dense_constant_52_let = (&(global_const_workspace_28_var[164128]));
  void* fused_constant_51_let = (&(global_const_workspace_28_var[151552]));
  void* dense_let = (&(global_workspace_29_var[128]));
  for (int32_t y_outer = 0; y_outer < 5; ++y_outer) {
    gemm_1x1x2_reset_UGHLTAKV((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 64; ++k_outer) {
      gemm16_1x1x2_update_UGHLTAKV((&(p0[k_outer])), (&(((int16_t*)fused_constant_51_let)[((y_outer * 128) + k_outer)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 1, 64, 2);
    }
  }
  for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)(((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_constant_52_let)[ax1])) << ((int64_t)0)) : ((int64_t)(((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_constant_52_let)[ax1]))) * (int64_t)1552512742) + ((int64_t)1 << ((int64_t)((5 + 31) - 1)))) >> ((int64_t)(5 + 31)))) + 24;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    T_multiply[ax1] = (((float)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - 24)) * 1.718535e-01f);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_softmax_divide_round_add_clip_cast(float* p0, int8_t* T_cast, uint8_t* global_const_workspace_30_var, uint8_t* global_workspace_31_var) {
  void* T_softmax_maxelem_let = (&(global_workspace_31_var[80]));
  void* T_softmax_exp_let = (&(global_workspace_31_var[40]));
  void* T_softmax_expsum_let = (&(global_workspace_31_var[80]));
  ((float*)T_softmax_maxelem_let)[0] = -3.402823e+38f;
  for (int32_t k = 0; k < 10; ++k) {
    float __1 = ((float*)T_softmax_maxelem_let)[0];
    float __2 = p0[k];
    ((float*)T_softmax_maxelem_let)[0] = ((__1) > (__2) ? (__1) : (__2));
  }
  for (int32_t i1 = 0; i1 < 10; ++i1) {
    ((float*)T_softmax_exp_let)[i1] = expf((p0[i1] - ((float*)T_softmax_maxelem_let)[0]));
  }
  ((float*)T_softmax_expsum_let)[0] = 0.000000e+00f;
  for (int32_t k_1 = 0; k_1 < 10; ++k_1) {
    ((float*)T_softmax_expsum_let)[0] = (((float*)T_softmax_expsum_let)[0] + ((float*)T_softmax_exp_let)[k_1]);
  }
  for (int32_t i1_1 = 0; i1_1 < 10; ++i1_1) {
    ((float*)T_softmax_exp_let)[i1_1] = (((float*)T_softmax_exp_let)[i1_1] / ((float*)T_softmax_expsum_let)[0]);
  }
  for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
    float __3 = roundf((((float*)T_softmax_exp_let)[ax1] * 2.560000e+02f)) + -1.280000e+02f;
    float __4 = (__3) < (1.270000e+02f) ? (__3) : (1.270000e+02f);
    T_cast[ax1] = ((int8_t)((__4) > (-1.280000e+02f) ? (__4) : (-1.280000e+02f)));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_reshape_cast_subtract(int8_t* p0, int16_t* T_subtract, uint8_t* global_const_workspace_26_var, uint8_t* global_workspace_27_var) {
  for (int32_t ax1_outer = 0; ax1_outer < 8; ++ax1_outer) {
    for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
      int32_t cse_var_1 = ((ax1_outer * 8) + ax1_inner);
      T_subtract[cse_var_1] = (((int16_t)p0[cse_var_1]) - (int16_t)-128);
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(int8_t* input_1_int8_buffer_var, int8_t* Identity_int8_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_14_let = (&(global_workspace_1_var[0]));
  void* sid_11_let = (&(global_workspace_1_var[221472]));
  void* sid_7_let = (&(global_workspace_1_var[106784]));
  void* sid_13_let = (&(global_workspace_1_var[0]));
  void* sid_10_let = (&(global_workspace_1_var[188704]));
  void* sid_9_let = (&(global_workspace_1_var[205088]));
  void* sid_3_let = (&(global_workspace_1_var[163968]));
  void* sid_12_let = (&(global_workspace_1_var[238112]));
  void* sid_8_let = (&(global_workspace_1_var[172320]));
  void* sid_1_let = (&(global_workspace_1_var[83992]));
  void* sid_5_let = (&(global_workspace_1_var[74016]));
  void* sid_4_let = (&(global_workspace_1_var[163968]));
  void* sid_2_let = (&(global_workspace_1_var[196736]));
  void* sid_6_let = (&(global_workspace_1_var[193056]));
  if (tvmgen_default_fused_cast_subtract(input_1_int8_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip(sid_1_let, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_cast_subtract_1(sid_2_let, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245_(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_cast_subtract_fixed_point_multiply_add_nn_conv2d_add_cast_multiply_add_rig_5494088d2bce6f3f_(sid_2_let, sid_4_let, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__1(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_9b1cea826623845_(sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_866ca172ecfe2cfb_(sid_5_let, sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_clip_cast_s_8a376065fd35c245__2(sid_8_let, sid_9_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_9b1cea826623845__1(sid_9_let, sid_10_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_conv2d_add_cast_multiply_add_right_shift_cast_add_clip_cast_cast_subtra_8b760d480c798df_(sid_8_let, sid_10_let, sid_11_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_avg_pool2d_cast(sid_11_let, sid_12_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_reshape_cast_subtract(sid_12_let, sid_13_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_add_fixed_point_multiply_add_clip_cast_cast_subtract_cast_multipl_10ab34501c4d620e_(sid_13_let, sid_14_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_softmax_divide_round_add_clip_cast(sid_14_let, Identity_int8_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

