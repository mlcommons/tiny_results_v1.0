
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */
// #include <stdio.h>
// #include <stdlib.h>
// #include <errno.h>

#include "bsp/fs/hostfs.h"
#include "BUILD_MODEL/kws_ref_modelInfo.h"

#include <stdio.h>
#include <math.h>


/* Autotiler includes. */
#include "kws_ref_model.h"
#include "kws_ref_modelKernels.h"
#include "gaplib/ImgIO.h"

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE kws_ref_model_L3_Flash = 0;

#ifdef IN_OUT_INT8
L2_MEM signed char Input_1[490];//={0};
L2_MEM short int Output_1[12];
#endif

#ifdef IN_OUT_FP16
L2_MEM F16 Input_1[490];
L2_MEM F16 Output_1[12];
#endif

char *ImageName;
FILE *ptr;
signed char img_raw[490];

L2_MEM float Input_1_real[490];


//int somme = 0;

/* Copy inputs function */
void copy_inputs() {
}


static void cluster()
{
    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    // for(int i = 0; i < 490;i++)
    //     printf("in %d = %d \n",i,Input_1[i]);
    kws_ref_modelCNN(Input_1, Output_1);

    printf("Runner completed\n");

    #ifdef IN_OUT_FP16
    float MaxPred=Output_1[0];
    int PredClass=0;
    float somme = Output_1[0];
    
    printf("Class %d --> %f \n",0,(float)Output_1[0]);

    for (int i=1; i<12; i++){
        somme += (float) Output_1[i];
        printf("Class %d --> %f \n",i,(float)Output_1[i]);

        if (Output_1[i] > MaxPred) {
            MaxPred = Output_1[i];
            PredClass = i;
        }
    }
    printf("Predicted Class: %d with confidence: %f\n", PredClass, MaxPred);
    printf("somme = %f \n",somme);
    #endif



    #ifdef IN_OUT_INT8
    int MaxPred=Output_1[0], PredClass=0;
    int somme = Output_1[0];
    printf("Class %d --> %d \n",0,(int)Output_1[0]);

    for (int i=1; i<12; i++){
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

int test_kws_ref_model(void)
{
    printf("Entering main controller\n");
    /* ----------------> 
     * Put here Your input settings
     * <---------------
     */

    ImageName = __XSTR(AT_IMAGE);
    printf("Image = %s \n",ImageName);

    int Traspose2CHW = 0;

    printf("Reading image in %s\n", Traspose2CHW?"CHW":"HWC");

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

    pi_fs_read(file, img_raw, 490);
    pi_fs_close(file);
    pi_fs_unmount(&fs);
    // for(int i = 0; i < 490;i++)
    //     printf("in raw %d = %d \n",i,img_raw[i]);

           
    #ifdef IN_OUT_FP16
    for(int i = 0; i < sizeof(img_raw);i++)
        Input_1[i] = (F16) (0.5847029089927673 * (int) (img_raw[i] - 83));
    #endif

    #ifdef IN_OUT_INT8
    printf("flag \n");
    //scale to int8 quantized nntool
    for(int i = 0; i < sizeof(img_raw);i++){
        Input_1_real[i] = 0.5847029089927673 * (int) (img_raw[i] - 83);
        if (Input_1_real[i] < 0)
            Input_1[i] = (signed char) ((int) ((Input_1_real[i] / kws_ref_model_Input_1_OUT_SCALE) - 0.5) + kws_ref_model_Input_1_OUT_ZERO_POINT);
        else
            Input_1[i] = (signed char) ((int) ((Input_1_real[i] / kws_ref_model_Input_1_OUT_SCALE) + 0.5) + kws_ref_model_Input_1_OUT_ZERO_POINT);
        //Input_1[i] = img_raw[i];
        //Input_1[i] = 0;//img_raw[i];

    }
    #endif

    // for(int i = 0; i < sizeof(img_raw);i++)
    //     printf("data %d =%d \n",i,Input_1[i]);

#ifndef __EMUL__
    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    //pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }
    printf("stack size = %d \n",STACK_SIZE);

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

    //pi_time_wait_us(1000000);
    // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
    printf("Constructor\n");
    int ConstructorErr = kws_ref_modelCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file kws_ref_modelKernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
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

    kws_ref_modelCNN_Destruct();

#ifdef PERF
    {
      unsigned int TotalCycles = 0, TotalOper = 0;
      printf("\n");
      for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
        printf("%45s: Cycles: %12u, Operations: %12u, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
        TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
      }
      printf("\n");
      printf("%45s: Cycles: %12u, Operations: %12u, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
      printf("\n");
    }
#endif

    printf("Ended\n");
    pmsis_exit(0);
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL kws_ref_model Example ***\n\n");
    #ifdef __EMUL__
    test_kws_ref_model();
    #else
    return pmsis_kickoff((void *) test_kws_ref_model);
    #endif
    return 0;
}
