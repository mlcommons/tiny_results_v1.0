
/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include "vww_96_int8.h"
#include "vww_96_int8Kernels.h"

#include "BUILD_MODEL/vww_96_int8Info.h"


#include "bsp/fs/hostfs.h"

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 

#ifdef __EMUL__
#define pmsis_exit(n) exit(n)
#endif

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE vww_96_int8_L3_Flash = 0;

#ifdef IN_OUT_INT8
L2_MEM signed char Input_1[27648];
L2_MEM short int Output_1[2];
#endif

#ifdef IN_OUT_NE16
L2_MEM unsigned char Input_1[27648];
L2_MEM short int Output_1[2];
#endif

#ifdef IN_OUT_FP16
L2_MEM F16 Input_1[27648];
L2_MEM F16 Output_1[2];
#endif

L2_MEM signed char img_raw[27648];
L2_MEM float Input_1_real[27648];

//L2_MEM signed char chw_buff[3072];

char *ImageName;
FILE *ptr;


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

    vww_96_int8CNN(Input_1, Output_1);
    printf("Runner completed\n");


    #ifdef IN_OUT_FP16
    float MaxPred=Output_1[0];
    int PredClass=0;
    float somme = Output_1[0];
    
    printf("Class %d --> %f \n",0,(float)Output_1[0]);

    for (int i=1; i<2; i++){
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

    for (int i=1; i<2; i++){
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

int test_vww_96_int8(void)
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

    pi_fs_read(file, img_raw, 27648);
    pi_fs_close(file);
    pi_fs_unmount(&fs);

    // for (int i = 0; i < 3072;i++)
    //     printf("%x",img_raw[i]);

    #ifdef IN_OUT_NE16
    for (int i = 0; i < 27648;i++)
    {
        Input_1[i] = (unsigned char) img_raw[i];
    }

    #endif


    #ifdef IN_OUT_INT8
    for (int i = 0; i < 27648;i++)

    {
        Input_1_real[i] = (float) ( 0.003921568859368563 * (int) (img_raw[i] + 128));
        Input_1[i] = (signed char) ((int) ((Input_1_real[i] / vww_96_int8_Input_1_OUT_SCALE) + 0.5 ) + vww_96_int8_Input_1_OUT_ZERO_POINT);
    }
    #endif

    #ifdef IN_OUT_FP16
    for (int i = 0; i < 27648;i++){
        Input_1[i] = (F16) ( 0.003921568859368563 * (int) (img_raw[i] + 128));
}
    #endif

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
    int ConstructorErr = vww_96_int8CNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: %d\n(check the generated file vww_96_int8Kernels.c to see which memory have failed to be allocated)\n", ConstructorErr);
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

    vww_96_int8CNN_Destruct();

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
    printf("\n\n\t *** NNTOOL vww_96_int8 Example ***\n\n");
    #ifdef __EMUL__
    test_vww_96_int8();
    #else
    return pmsis_kickoff((void *) test_vww_96_int8);
    #endif
    return 0;
}
