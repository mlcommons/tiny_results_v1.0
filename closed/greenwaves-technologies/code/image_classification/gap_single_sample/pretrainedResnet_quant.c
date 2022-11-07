
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "pretrainedResnet_quant.h"
#include "pretrainedResnet_quantKernels.h"

#include "BUILD_MODEL/pretrainedResnet_quantInfo.h"


#include "bsp/fs/hostfs.h"

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE pretrainedResnet_quant_L3_Flash = 0;

#ifdef IN_OUT_INT8
L2_MEM signed char Input_1[3072];
L2_MEM short int Output_1[10];
#endif

#ifdef IN_OUT_NE16
L2_MEM unsigned char Input_1[3072];
L2_MEM short int Output_1[10];
#endif

#ifdef IN_OUT_FP16
L2_MEM F16 Input_1[3072];
L2_MEM F16 Output_1[10];
#endif

L2_MEM uint8_t img_raw[3072];
L2_MEM float Input_1_real[3072];

//L2_MEM signed char chw_buff[3072];

char *ImageName;
FILE *ptr;


/* Copy inputs function */
void copy_inputs() {
}

static signed char * hwc_to_chw(signed char * InBuffer, int W, int H, int BytesPerPixel)
{
    signed char temp_buff[3072];
    unsigned int RowSize = W*BytesPerPixel, ChannelSize = W * H;
    //signed char * pInBuffer = InBuffer;
    //unsigned char *InputBuf = (unsigned char *) __ALLOC_L2(RowSize * sizeof(unsigned char));
    // if(InputBuf == NULL)
    // {
    //     printf("Malloc failed when loading image\n");
    //     return -1;
    // }

    for (int CurRow=0; CurRow < H; CurRow++) {

        for (int i=0; i < W; i++) {

        //printf("W = %d \n",i);

            for (int j=0; j < BytesPerPixel; j++) {
                temp_buff[ChannelSize * j + W * CurRow + i] = InBuffer[i * BytesPerPixel + j];
                printf("index out = %d \n",ChannelSize * j + W * CurRow + i);
                printf("index in = %d \n",i * BytesPerPixel + j);
            }
        }
    }

    // for(int i = 0; i < 3072;i++)
    //     printf("%d \n",temp_buff[i]);

    return temp_buff;
}


static void cluster()
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    pretrainedResnet_quantCNN(Input_1, Output_1);
    printf("Runner completed\n");


    #ifdef IN_OUT_FP16
    float MaxPred=Output_1[0];
    int PredClass=0;
    float somme = Output_1[0];
    
    printf("Class %d --> %f \n",0,(float)Output_1[0]);

    for (int i=1; i<10; i++){
        somme += (float) Output_1[i];
        printf("Class %d --> %f \n",i,(float)Output_1[i]);

        if (Output_1[i] > MaxPred) {
            MaxPred = Output_1[i];
            PredClass = i;
        }
    }
    printf("Predicted Class: %d with confidence: %f\n", PredClass, MaxPred);
    printf("somme = %f \n",somme);
    
    #else
    int MaxPred=Output_1[0], PredClass=0;
    int somme = Output_1[0];
    printf("Class %d --> %d \n",0,(int)Output_1[0]);

    for (int i=1; i<10; i++){
        somme += Output_1[i];

        printf("Class %d --> %d \n",i,(int)Output_1[i]);

        if (Output_1[i] > MaxPred) {
            MaxPred = Output_1[i];
            PredClass = i;
        }
    }
    
    printf("Predicted Class: %d with confidence: %d\n", PredClass, MaxPred);
    printf("somme = %d \n",somme);
    #endif

}

int test_pretrainedResnet_quant(void)
{
    printf("Entering main controller\n");
    /* ----------------> 
     * Put here Your input settings
     * <---------------
     */

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

    pi_fs_read(file, img_raw, 3072);
    pi_fs_close(file);
    pi_fs_unmount(&fs);

    // for (int i = 0; i < 3072;i++)
    //     printf("%x",img_raw[i]);

    #ifdef IN_OUT_NE16
    for (int i = 0; i < 3072;i++)
    {
        Input_1[i] = (unsigned char) img_raw[i];
        //printf("%d \n",Input_1[i]);
    }

    #endif


#ifdef IN_OUT_INT8
for (int i = 0; i < 3072;i++)
{
    Input_1_real[i] = (float) (img_raw[i]);// - 128);
    Input_1[i] = (signed char) ((int) ((Input_1_real[i] / pretrainedResnet_quant_Input_1_OUT_SCALE) + 0.5) + pretrainedResnet_quant_Input_1_OUT_ZERO_POINT);
}
#endif

#ifdef IN_OUT_FP16
for (int i = 0; i < 3072;i++)
    Input_1[i] = (F16) (img_raw[i]);
#endif
//signed char * chw_buff = (signed char *) hwc_to_chw(Input_1, 32, 32, 1);

// for (int i = 0; i < 3072;i++)
// {
//     //printf("avant = %d\n",Input_1[i]);
//     Input_1[i] = chw_buff[i];
//     //printf("après = %d\n",Input_1[i]);
// }


#ifndef __EMUL__
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

    /* Frequency Settings: defined in the Makefile */
    int cur_fc_freq = pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
    int cur_cl_freq = pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
    int cur_pe_freq = pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
    if (cur_fc_freq == -1 || cur_cl_freq == -1 || cur_pe_freq == -1)
    {
        printf("Error changing frequency !\nTest failed...\n");
        pmsis_exit(-4);
    }
	printf("FC Frequency as %d Hz, CL Frequency = %d Hz, PERIIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

#endif
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = pretrainedResnet_quantCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file pretrainedResnet_quantKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
        pmsis_exit(-6);
    }

    copy_inputs();

    printf("Call cluster\n");
#ifndef __EMUL__
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    pi_cluster_send_task_to_cl(&cluster_dev, &task);
#else
    cluster();
#endif

    pretrainedResnet_quantCNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
      }
      printf("\n");
      printf("%45s: Cycles: %12u, Cyc%%: 100.0%%, Operations: %12u, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
      printf("\n");
    }
#endif

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL pretrainedResnet_quant Example ***\n\n");
    #ifdef __EMUL__
    test_pretrainedResnet_quant();
    #else
    return pmsis_kickoff((void *) test_pretrainedResnet_quant);
    #endif
    return 0;
}
