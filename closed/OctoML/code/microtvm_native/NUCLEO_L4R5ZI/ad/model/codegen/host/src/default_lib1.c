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
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_MOXKPGNY(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_MOXKPGNY(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_MOXKPGNY(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_MOXKPGNY(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_MOXKPGNY(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_MOXKPGNY(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_MOXKPGNY(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_MOXKPGNY(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_MOXKPGNY(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_MOXKPGNY(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_MOXKPGNY(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_MOXKPGNY(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_MOXKPGNY(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_MOXKPGNY(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_MOXKPGNY(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_MOXKPGNY(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_MOXKPGNY(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_MOXKPGNY(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_MOXKPGNY(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_MOXKPGNY(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_MOXKPGNY(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_RHYFOUWW(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_RHYFOUWW(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_RHYFOUWW(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_RHYFOUWW(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_RHYFOUWW(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_RHYFOUWW(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_RHYFOUWW(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_RHYFOUWW(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_RHYFOUWW(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_RHYFOUWW(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_RHYFOUWW(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_RHYFOUWW(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_RHYFOUWW(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_RHYFOUWW(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_RHYFOUWW(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_RHYFOUWW(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_RHYFOUWW(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_RHYFOUWW(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_RHYFOUWW(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_RHYFOUWW(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_RHYFOUWW(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_ZPYPZKRH(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_ZPYPZKRH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_ZPYPZKRH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_ZPYPZKRH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_ZPYPZKRH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_ZPYPZKRH(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_ZPYPZKRH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_ZPYPZKRH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_ZPYPZKRH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_ZPYPZKRH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_ZPYPZKRH(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_ZPYPZKRH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_ZPYPZKRH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_ZPYPZKRH(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_ZPYPZKRH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_ZPYPZKRH(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_ZPYPZKRH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_ZPYPZKRH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_ZPYPZKRH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_ZPYPZKRH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_ZPYPZKRH(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_TIIJTXHF(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_TIIJTXHF(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_TIIJTXHF(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_TIIJTXHF(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_TIIJTXHF(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_TIIJTXHF(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_TIIJTXHF(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_TIIJTXHF(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_TIIJTXHF(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_TIIJTXHF(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_TIIJTXHF(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_TIIJTXHF(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_TIIJTXHF(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_TIIJTXHF(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_TIIJTXHF(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_TIIJTXHF(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_TIIJTXHF(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_TIIJTXHF(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_TIIJTXHF(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_TIIJTXHF(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_TIIJTXHF(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_OGDTLKHL(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_OGDTLKHL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_OGDTLKHL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_OGDTLKHL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_OGDTLKHL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_OGDTLKHL(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_OGDTLKHL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_OGDTLKHL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_OGDTLKHL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_OGDTLKHL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_OGDTLKHL(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_OGDTLKHL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_OGDTLKHL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_OGDTLKHL(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_OGDTLKHL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_OGDTLKHL(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_OGDTLKHL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_OGDTLKHL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_OGDTLKHL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_OGDTLKHL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_OGDTLKHL(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_DAUJAEUX(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_DAUJAEUX(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_DAUJAEUX(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_DAUJAEUX(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_DAUJAEUX(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_DAUJAEUX(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_DAUJAEUX(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_DAUJAEUX(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_DAUJAEUX(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_DAUJAEUX(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_DAUJAEUX(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_DAUJAEUX(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_DAUJAEUX(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_DAUJAEUX(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_DAUJAEUX(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_DAUJAEUX(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_DAUJAEUX(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_DAUJAEUX(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_DAUJAEUX(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_DAUJAEUX(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_DAUJAEUX(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_XGKYQZSH(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_XGKYQZSH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_XGKYQZSH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_XGKYQZSH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_XGKYQZSH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_XGKYQZSH(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_XGKYQZSH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_XGKYQZSH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_XGKYQZSH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_XGKYQZSH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_XGKYQZSH(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_XGKYQZSH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_XGKYQZSH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_XGKYQZSH(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_XGKYQZSH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_XGKYQZSH(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_XGKYQZSH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_XGKYQZSH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_XGKYQZSH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_XGKYQZSH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_XGKYQZSH(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_FVLJURDJ(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_FVLJURDJ(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_FVLJURDJ(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_FVLJURDJ(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_FVLJURDJ(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_FVLJURDJ(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_FVLJURDJ(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_FVLJURDJ(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_FVLJURDJ(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_FVLJURDJ(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_FVLJURDJ(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_FVLJURDJ(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_FVLJURDJ(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_FVLJURDJ(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_FVLJURDJ(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_FVLJURDJ(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_FVLJURDJ(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_FVLJURDJ(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_FVLJURDJ(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_FVLJURDJ(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_FVLJURDJ(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_EBGNJZAH(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_EBGNJZAH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_EBGNJZAH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_EBGNJZAH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_EBGNJZAH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_EBGNJZAH(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_EBGNJZAH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_EBGNJZAH(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_EBGNJZAH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_EBGNJZAH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_EBGNJZAH(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_EBGNJZAH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_EBGNJZAH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_EBGNJZAH(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_EBGNJZAH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_EBGNJZAH(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_EBGNJZAH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_EBGNJZAH(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_EBGNJZAH(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_EBGNJZAH(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_EBGNJZAH(int32_t *cc, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      cc[i*C_stride + j] = 0;
    }
  }
  return 0;
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <arm_nnsupportfunctions.h>

#include <tvm/runtime/crt/error_codes.h>




#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_body_rest_FHBKVICL(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_loop_FHBKVICL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_body_FHBKVICL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_body_loop_FHBKVICL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_body_rest_FHBKVICL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x2_update_rest_FHBKVICL(
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_loop_FHBKVICL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm_1x4x2_update_FHBKVICL(
    int8_t *aa, int8_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int16_t bb_pad[8];
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm_1x4x2_update_loop_FHBKVICL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 4 / 4; j++)
      read_and_pad(&bb[i*B_stride + j*4], (int32_t*) &bb_pad[i*4 + j*4], (int32_t*) &bb_pad[i*4 + j*4 + 2]);

  for (int i = 0; i < 1; i++) {
    int16_t aa_pad_line[4];
    for (int l = 0; l < 4 / 4; l++)
      read_and_pad(&aa[i*A_stride + l*4], (int32_t*) &aa_pad_line[l*4], (int32_t*) &aa_pad_line[l*4 + 2]);

    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) aa_pad_line;
      int32_t *bb_ptr = (int32_t *) &bb_pad[j*4];
      int32_t sum = 0;
      for (int l = 0; l < 2 * (4 / 4); l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 4 != 0 )
    gemm_1x2_update_rest_FHBKVICL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_body_rest_FHBKVICL(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_loop_FHBKVICL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_body_FHBKVICL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_body_loop_FHBKVICL(aa, bb, cc, A_stride, B_stride, C_stride);
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
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      // NOTE: this is the line where `*_body` differs from `*_update`. here
      // we're *setting* the result, instead of accumulating, because we know
      // the `i` and `j` itervars span their entire respective axes.
      cc[i*C_stride + j] = sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_body_rest_FHBKVICL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm16_1x2_update_rest_FHBKVICL(
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_loop_FHBKVICL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t sum = 0;
      for (int l = 0; l < 4; l++) {
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
__STATIC_FORCEINLINE int32_t gemm16_1x4x2_update_FHBKVICL(
    int16_t *aa, int16_t *bb, int32_t *cc,
    int A_stride, int B_stride, int C_stride) {
  int32_t retcode = 0;

  if ( 1 < 2 && 2 < 2 ) {
    retcode = gemm16_1x4x2_update_loop_FHBKVICL(aa, bb, cc, A_stride, B_stride, C_stride);
    goto out;
  }

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 2; j++) {
      int32_t *aa_ptr = (int32_t *) &aa[i*A_stride];
      int32_t *bb_ptr = (int32_t *) &bb[j*B_stride];

      int32_t sum = 0;
      for (int l = 0; l < 4 / 2; l++) {
        sum = __SMLAD(*aa_ptr, *bb_ptr, sum);
        ++ aa_ptr; ++ bb_ptr;
      }
      cc[i*C_stride + j] += sum;
    }
  }

  if ( 4 % 2 != 0 )
    gemm16_1x2_update_rest_FHBKVICL(4, aa, bb, cc, A_stride, B_stride, C_stride);

out:
  return retcode;
}

#ifdef __cplusplus
extern "C"
#endif
__STATIC_FORCEINLINE int32_t gemm_1x4x2_reset_FHBKVICL(int32_t *cc, int C_stride) {
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
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_cast_subtract_cas_df8f6575c9a82565_(int8_t* p0, float* T_multiply, uint8_t* global_const_workspace_22_var, uint8_t* global_workspace_23_var) {
  void* fused_nn_dense_constant_28_let = (&(global_const_workspace_22_var[264704]));
  void* fused_nn_dense_subtract_constant_29_let = (&(global_const_workspace_22_var[262144]));
  void* fused_constant_27_let = (&(global_const_workspace_22_var[0]));
  void* dense_let = (&(global_workspace_23_var[0]));
  for (int32_t y_outer = 0; y_outer < 320; ++y_outer) {
    gemm_1x4x2_reset_MOXKPGNY((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_MOXKPGNY((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_27_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t ax1 = 0; ax1 < 640; ++ax1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_29_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_28_let)[ax1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[ax1] + ((int32_t*)fused_nn_dense_subtract_constant_29_let)[ax1]) - ((int32_t*)fused_nn_dense_constant_28_let)[ax1]))) * (int64_t)1417662827) + ((int64_t)1 << ((int64_t)((9 + 31) - 1)))) >> ((int64_t)(9 + 31)))) + 89;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    T_multiply[ax1] = (((float)(((int32_t)((int8_t)((__2) > (-128) ? (__2) : (-128)))) - 89)) * 3.760228e-01f);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_4_var, uint8_t* global_workspace_5_var) {
  void* fused_nn_dense_constant_1_let = (&(global_const_workspace_4_var[276992]));
  void* fused_nn_dense_subtract_constant_2_let = (&(global_const_workspace_4_var[271872]));
  void* fused_constant_0_let = (&(global_const_workspace_4_var[81920]));
  void* dense_let = (&(global_workspace_5_var[640]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_RHYFOUWW((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 160; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_RHYFOUWW((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_0_let)[((y_outer * 1280) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 640, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_2_let)[i1]) - ((int32_t*)fused_nn_dense_constant_1_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_2_let)[i1]) - ((int32_t*)fused_nn_dense_constant_1_let)[i1]))) * (int64_t)1695943312) + ((int64_t)1 << ((int64_t)((8 + 31) - 1)))) >> ((int64_t)(8 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_1(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_6_var, uint8_t* global_workspace_7_var) {
  void* fused_nn_dense_constant_4_let = (&(global_const_workspace_6_var[273920]));
  void* fused_nn_dense_subtract_constant_5_let = (&(global_const_workspace_6_var[269824]));
  void* fused_constant_3_let = (&(global_const_workspace_6_var[196608]));
  void* dense_let = (&(global_workspace_7_var[0]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_ZPYPZKRH((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_ZPYPZKRH((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_3_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_5_let)[i1]) - ((int32_t*)fused_nn_dense_constant_4_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_5_let)[i1]) - ((int32_t*)fused_nn_dense_constant_4_let)[i1]))) * (int64_t)1442659874) + ((int64_t)1 << ((int64_t)((5 + 31) - 1)))) >> ((int64_t)(5 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_2(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_8_var, uint8_t* global_workspace_9_var) {
  void* fused_nn_dense_constant_7_let = (&(global_const_workspace_8_var[273408]));
  void* fused_nn_dense_subtract_constant_8_let = (&(global_const_workspace_8_var[269312]));
  void* fused_constant_6_let = (&(global_const_workspace_8_var[180224]));
  void* dense_let = (&(global_workspace_9_var[0]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_TIIJTXHF((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_TIIJTXHF((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_6_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_8_let)[i1]) - ((int32_t*)fused_nn_dense_constant_7_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_8_let)[i1]) - ((int32_t*)fused_nn_dense_constant_7_let)[i1]))) * (int64_t)1650946042) + ((int64_t)1 << ((int64_t)((3 + 31) - 1)))) >> ((int64_t)(3 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_3(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_10_var, uint8_t* global_workspace_11_var) {
  void* fused_nn_dense_constant_10_let = (&(global_const_workspace_10_var[276480]));
  void* fused_nn_dense_subtract_constant_11_let = (&(global_const_workspace_10_var[272896]));
  void* fused_constant_9_let = (&(global_const_workspace_10_var[163840]));
  void* dense_let = (&(global_workspace_11_var[0]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_OGDTLKHL((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_OGDTLKHL((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_9_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_11_let)[i1]) - ((int32_t*)fused_nn_dense_constant_10_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_11_let)[i1]) - ((int32_t*)fused_nn_dense_constant_10_let)[i1]))) * (int64_t)2066955235) + ((int64_t)1 << ((int64_t)((4 + 31) - 1)))) >> ((int64_t)(4 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_4(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_12_var, uint8_t* global_workspace_13_var) {
  void* fused_nn_dense_constant_13_let = (&(global_const_workspace_12_var[277536]));
  void* fused_nn_dense_subtract_constant_14_let = (&(global_const_workspace_12_var[277504]));
  void* fused_constant_12_let = (&(global_const_workspace_12_var[268288]));
  void* dense_let = (&(global_workspace_13_var[640]));
  for (int32_t y_outer = 0; y_outer < 4; ++y_outer) {
    gemm_1x4x2_reset_DAUJAEUX((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_DAUJAEUX((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_12_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 8; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_14_let)[i1]) - ((int32_t*)fused_nn_dense_constant_13_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_14_let)[i1]) - ((int32_t*)fused_nn_dense_constant_13_let)[i1]))) * (int64_t)1085889771) + ((int64_t)1 << ((int64_t)((6 + 31) - 1)))) >> ((int64_t)(6 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_5(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_14_var, uint8_t* global_workspace_15_var) {
  void* fused_nn_dense_constant_16_let = (&(global_const_workspace_14_var[275968]));
  void* fused_nn_dense_subtract_constant_17_let = (&(global_const_workspace_14_var[272384]));
  void* fused_constant_15_let = (&(global_const_workspace_14_var[267264]));
  void* dense_let = (&(global_workspace_15_var[0]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_XGKYQZSH((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 2; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_XGKYQZSH((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_15_let)[((y_outer * 16) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 8, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_17_let)[i1]) - ((int32_t*)fused_nn_dense_constant_16_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_17_let)[i1]) - ((int32_t*)fused_nn_dense_constant_16_let)[i1]))) * (int64_t)1442237646) + ((int64_t)1 << ((int64_t)((5 + 31) - 1)))) >> ((int64_t)(5 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_6(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_16_var, uint8_t* global_workspace_17_var) {
  void* fused_nn_dense_constant_19_let = (&(global_const_workspace_16_var[275456]));
  void* fused_nn_dense_subtract_constant_20_let = (&(global_const_workspace_16_var[271360]));
  void* fused_constant_18_let = (&(global_const_workspace_16_var[245760]));
  void* dense_let = (&(global_workspace_17_var[0]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_FVLJURDJ((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_FVLJURDJ((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_18_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_20_let)[i1]) - ((int32_t*)fused_nn_dense_constant_19_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_20_let)[i1]) - ((int32_t*)fused_nn_dense_constant_19_let)[i1]))) * (int64_t)1312526225) + ((int64_t)1 << ((int64_t)((5 + 31) - 1)))) >> ((int64_t)(5 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_7(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_18_var, uint8_t* global_workspace_19_var) {
  void* fused_nn_dense_constant_22_let = (&(global_const_workspace_18_var[274944]));
  void* fused_nn_dense_subtract_constant_23_let = (&(global_const_workspace_18_var[270848]));
  void* fused_constant_21_let = (&(global_const_workspace_18_var[229376]));
  void* dense_let = (&(global_workspace_19_var[0]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_EBGNJZAH((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_EBGNJZAH((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_21_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_23_let)[i1]) - ((int32_t*)fused_nn_dense_constant_22_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_23_let)[i1]) - ((int32_t*)fused_nn_dense_constant_22_let)[i1]))) * (int64_t)1999134766) + ((int64_t)1 << ((int64_t)((6 + 31) - 1)))) >> ((int64_t)(6 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_8(int8_t* p0, int8_t* compute, uint8_t* global_const_workspace_20_var, uint8_t* global_workspace_21_var) {
  void* fused_nn_dense_constant_25_let = (&(global_const_workspace_20_var[274432]));
  void* fused_nn_dense_subtract_constant_26_let = (&(global_const_workspace_20_var[270336]));
  void* fused_constant_24_let = (&(global_const_workspace_20_var[212992]));
  void* dense_let = (&(global_workspace_21_var[0]));
  for (int32_t y_outer = 0; y_outer < 64; ++y_outer) {
    gemm_1x4x2_reset_FHBKVICL((&(((int32_t*)dense_let)[(y_outer * 2)])), 2);
    for (int32_t k_outer = 0; k_outer < 32; ++k_outer) {
      int32_t cse_var_1 = (k_outer * 4);
      gemm_1x4x2_update_FHBKVICL((&(p0[cse_var_1])), (&(((int8_t*)fused_constant_24_let)[((y_outer * 256) + cse_var_1)])), (&(((int32_t*)dense_let)[(y_outer * 2)])), 4, 128, 2);
    }
  }
  for (int32_t i1 = 0; i1 < 128; ++i1) {
    int32_t __1 = ((int32_t)(((((0 != 0) ? (((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_26_let)[i1]) - ((int32_t*)fused_nn_dense_constant_25_let)[i1])) << ((int64_t)0)) : ((int64_t)((((int32_t*)dense_let)[i1] + ((int32_t*)fused_nn_dense_subtract_constant_26_let)[i1]) - ((int32_t*)fused_nn_dense_constant_25_let)[i1]))) * (int64_t)1105921547) + ((int64_t)1 << ((int64_t)((6 + 31) - 1)))) >> ((int64_t)(6 + 31)))) - 128;
    int32_t __2 = (__1) < (127) ? (__1) : (127);
    int8_t __3 = (int8_t)((__2) > (-128) ? (__2) : (-128));
    int8_t __4 = (int8_t)127;
    int8_t __5 = (__3) < (__4) ? (__3) : (__4);
    int8_t __6 = (int8_t)-128;
    compute[i1] = ((__5) > (__6) ? (__5) : (__6));
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default___tvm_main__(float* input_1_buffer_var, float* Identity_buffer_var, uint8_t* global_const_workspace_0_var, uint8_t* global_workspace_1_var) {
  void* sid_7_let = (&(global_workspace_1_var[512]));
  void* sid_8_let = (&(global_workspace_1_var[512]));
  void* sid_1_let = (&(global_workspace_1_var[0]));
  void* sid_2_let = (&(global_workspace_1_var[1152]));
  void* sid_6_let = (&(global_workspace_1_var[672]));
  void* sid_9_let = (&(global_workspace_1_var[512]));
  void* sid_3_let = (&(global_workspace_1_var[512]));
  void* sid_4_let = (&(global_workspace_1_var[512]));
  void* sid_5_let = (&(global_workspace_1_var[512]));
  void* sid_10_let = (&(global_workspace_1_var[2560]));
  if (tvmgen_default_fused_divide_round_add_clip_cast_reshape(input_1_buffer_var, sid_1_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip(sid_1_let, sid_2_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_1(sid_2_let, sid_3_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_2(sid_3_let, sid_4_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_3(sid_4_let, sid_5_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_4(sid_5_let, sid_6_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_5(sid_6_let, sid_7_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_6(sid_7_let, sid_8_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_7(sid_8_let, sid_9_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_clip_8(sid_9_let, sid_10_let, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  if (tvmgen_default_fused_nn_dense_subtract_add_fixed_point_multiply_add_clip_cast_cast_subtract_cas_df8f6575c9a82565_(sid_10_let, Identity_buffer_var, global_const_workspace_0_var, global_workspace_1_var) != 0 ) return -1;
  return 0;
}

