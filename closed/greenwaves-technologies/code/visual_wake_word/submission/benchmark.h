// benchmark specific includes
#include "vww_96Info.h"
#include "vww_96Kernels.h"

#include "bsp/fs/hostfs.h"


#ifndef FP16
#define INPUT_SCALE       __PREFIX(_Input_1_OUT_SCALE)
#define INPUT_ZERO_POINT  __PREFIX(_Input_1_OUT_ZERO_POINT)
#define OUTPUT_SCALE      __PREFIX(_Output_1_OUT_SCALE)
#define OUTPUT_QSCALE     __PREFIX(_Output_1_OUT_QSCALE)
#define OUTPUT_QNORM      __PREFIX(_Output_1_OUT_QNORM)
#define OUTPUT_ZERO_POINT __PREFIX(_Output_1_OUT_ZERO_POINT)
#endif

#define AT_INPUT_SIZE (AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)
#define NUM_CLASSES (2)

char *ImageName;
FILE *ptr;

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 


// data buffer
typedef signed char uart_input_type;
uart_input_type Input_UART[AT_INPUT_SIZE];

#ifdef IN_OUT_INT8
L2_MEM unsigned short int ResOut[NUM_CLASSES];
typedef signed char input_type;
#endif

#ifdef IN_OUT_NE16
L2_MEM unsigned short int ResOut[NUM_CLASSES];
typedef unsigned char input_type;
#endif

#ifdef IN_OUT_FP16
L2_MEM F16 ResOut[NUM_CLASSES];
typedef F16 input_type;
#endif
L2_MEM input_type Input_1[AT_INPUT_SIZE];

// Input_1_real Needed to calculate AUC (Anomaly detection only)
L2_MEM float Input_1_real[AT_INPUT_SIZE];


void preprocessing(signed char * ptr_src, input_type * ptr_dst)
{
    // ptr_src have elements saved as uint8: ptr_src range [0:255] --> float range [0:1]
    // with quantization scheme for tflite, i.e. r = S*(q-Z) --> S=0.003921568859368563 Z=0
    // Before feeding NNTool dequantize and requantize

    for(uint i = 0; i < AT_INPUT_SIZE;i++){
        float dq_in_v = 0.003921568859368563 * (float) (ptr_src[i]); // CHECKME: (float) ( 0.003921568859368563 * (int) (ptr_src[i] + 128));

        #ifdef IN_OUT_FP16
        ptr_dst[i] = (F16) dq_in_v;
        #else
        if (dq_in_v < 0) ptr_dst[i] = (input_type) ((int) ((dq_in_v / INPUT_SCALE) - 0.5) + INPUT_ZERO_POINT);
        else             ptr_dst[i] = (input_type) ((int) ((dq_in_v / INPUT_SCALE) + 0.5) + INPUT_ZERO_POINT);
        #endif
    }
}

float ConvertToFloat(short int value) {
#ifndef FP16
  if(value < 0) {
    value = value - 1;
    printf("%d you shouldn't be here\n", value);
}
    return ((float) value - (float) OUTPUT_ZERO_POINT) * (float) OUTPUT_SCALE ;
  #endif
}

size_t load_image_for_test(uart_input_type * img_raw, int size)
{
    ImageName = __XSTR(AT_IMAGE);
    printf("Image = %s \n",ImageName);

    struct pi_hostfs_conf conf;
    pi_hostfs_conf_init(&conf);
    struct pi_device fs;

    pi_open_from_conf(&fs, &conf);

    if (pi_fs_mount(&fs))
     pmsis_exit(-1);

    void *file = pi_fs_open(&fs, ImageName, PI_FS_FLAGS_READ);

    if (file == 0)
    {
        printf("Failed to open file, %s\n", ImageName);
        pmsis_exit(-1);
    }


    pi_fs_read(file, img_raw, size);
    pi_fs_close(file);
    pi_fs_unmount(&fs);

    return (size_t) size;
}