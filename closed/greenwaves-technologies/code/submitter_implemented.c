/*
Copyright (C) EEMBC(R). All Rights Reserved

All EEMBC Benchmark Software are products of EEMBC and are provided under the
terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
are proprietary intellectual properties of EEMBC and its Members and is
protected under all applicable laws, including all applicable copyright laws.

If you received this EEMBC Benchmark Software without having a currently
effective EEMBC Benchmark License Agreement, you must discontinue use.

Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C++ implementations of submitter_implemented.h

#include "pmsis.h"

#include "api/internally_implemented.h"
#include "submitter_implemented.h"

#include "util/quantization_helpers.h"

#include "benchmark.h"

#include "bsp/fs/hostfs.h"

#include <stdio.h>
#include <stdarg.h> 

// Global Defines
#define DELAY_US (100)
struct pi_device uart;
struct pi_device gpio_ts;
#ifdef __gap9__
#define GPIO_TS       PI_GPIO_A43 
//#define GPIO_TS       PI_GPIO_A89
#else
#define GPIO_TS       PI_GPIO_A17_PAD_31_B11
#define GPIO_TS_PAD   PI_PAD_31_B11_TIMER0_CH0
#define GPIO_TS_FUNC  PI_PAD_31_B11_GPIO_A17_FUNC1
#endif

/*
    GAP specific variables
*/
AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;

// cluster resources
struct pi_cluster_task task;
struct pi_device cluster_dev;




// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {

  #ifndef TEST_IMAGE
  size_t bytes = ee_get_buffer(Input_UART, AT_INPUT_SIZE*sizeof(uart_input_type));
  int type_size = sizeof(uart_input_type);
  #else
  size_t bytes = load_image_for_test(Input_UART, AT_INPUT_SIZE*sizeof(uart_input_type));
  int type_size = sizeof(uart_input_type);
  #endif
  if (bytes / type_size != AT_INPUT_SIZE) {
    th_printf("Input db has %d elemented, expected %d\n",
              bytes / sizeof(int8_t), AT_INPUT_SIZE);
    return;
  }

  // data preprocessing
  preprocessing( Input_UART , Input_1 );

}

// Add to this method to return real inference results.
void th_results() {



const int nresults = NUM_CLASSES;
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[");
  int kCategoryCount = NUM_CLASSES;
  #ifdef PRINT_OUT
  th_printf("[");
  for (size_t i = 0; i < kCategoryCount; i++) {
    float converted =
        ConvertToFloat( ResOut[i] );

    // Some platforms don't implement floating point formatting.
    th_printf("%0.3f", converted);
    if (i < (kCategoryCount - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\n");
  #endif


  if (TH_MODEL_VERSION == EE_MODEL_VERSION_AD01)
  {
    float diffsum = 0;
    for (size_t i = 0; i < kCategoryCount; i++) {
      float converted = ConvertToFloat( ResOut[i] );
      float diff = converted - Input_1_real[i];
      diffsum += diff*diff;
    }
    diffsum /= kCategoryCount;

    th_printf("%0.3f", diffsum);
  }
  else
  {
  for (size_t i = 0; i < kCategoryCount; i++) {
    float converted =
        ConvertToFloat( ResOut[i] );

    // Some platforms don't implement floating point formatting.
    th_printf("%0.3f", converted);
    if (i < (nresults - 1)) {
      th_printf(",");
    }
  }
}
  th_printf("]\r\n");

  // Performance counters
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

}

// Implement this method with the logic to perform one inference cycle.
static void RunNetwork()
{

#ifdef PERF
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif

  AT_CNN(Input_1, ResOut);

}

void th_infer() {
  pi_cluster_send_task_to_cl(&cluster_dev, &task); 

  // This code is to run in the cluster than turn off the FC (IDLE_MODE) for some time 0xFFFF --> then wait for the cluster task to finish
  // pi_task_t fc_task;
  // pi_cluster_send_task_to_cl_async(&cluster_dev, &task, pi_task_block(&fc_task));
  // // FC Idle mode
  // *((uint32_t *) (0x1A104000+0xD4)) = 0x8000FFFF;
  // pi_task_wait_on(&fc_task);
}

/// \brief optional API.
void th_final_initialize(void) {


      /* Configure And open cluster. */
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        th_printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

    struct pi_gpio_conf gpio_conf = {0};
    struct pi_device gpio_out;

    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(&gpio_out, &gpio_conf);
    gpio_conf.port = (PI_GPIO_A00 & PI_GPIO_NUM_MASK) / 32;
    int errors = pi_gpio_open(&gpio_out);
    if (errors)
    {
        printf("Error opening GPIO %d\n", errors);
        pmsis_exit(errors);
    }
    /* Configure gpio output. */
    pi_gpio_pin_configure(&gpio_out, PI_GPIO_A00, PI_GPIO_OUTPUT|PI_GPIO_PULL_DISABLE|PI_GPIO_DRIVE_STRENGTH_LOW);
    pi_gpio_pin_write(&gpio_out, PI_GPIO_A00, 0);

    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(&gpio_out, &gpio_conf);
    gpio_conf.port = (PI_GPIO_A88 & PI_GPIO_NUM_MASK) / 32;
    errors = pi_gpio_open(&gpio_out);
    if (errors)
    {
        printf("Error opening GPIO %d\n", errors);
        pmsis_exit(errors);
    }
    /* Configure gpio output. */
    pi_gpio_pin_configure(&gpio_out, PI_GPIO_A88, PI_GPIO_OUTPUT|PI_GPIO_PULL_DISABLE|PI_GPIO_DRIVE_STRENGTH_LOW);
    pi_gpio_pin_write(&gpio_out, PI_GPIO_A88, 0);


    // Turn off the switchable IOs
    *((uint32_t *) (0x1A104000+0x164)) = 0x0000FF1F;
    // Explicit ABB settings
    *((uint32_t *) (0x1A104000+0x188)) = 0x00000003;

    pi_freq_set(PI_FREQ_DOMAIN_CL,FREQ_CL*1000*1000);
    pi_freq_set(PI_FREQ_DOMAIN_FC,FREQ_FC*1000*1000);

    #ifdef VOLTAGE
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
    pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
    #endif


    int err_const=AT_CONSTRUCT();
    if (err_const)
    {
      th_printf("Construct of network failed %d !\n", err_const);
      pmsis_exit(-5);
    }
    #ifdef PROMOTED_L1
    pi_cluster_task(&task, (void (*)(void *))AT_CONSTRUCT_CL, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);
    pi_cluster_send_task_to_cl(&cluster_dev, &task);   
    #endif

    pi_cluster_task(&task, (void (*)(void *))RunNetwork, NULL);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

}

void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strcpy(dest, src);
}

size_t th_strnlen(const char *str, size_t maxlen) {

  size_t t;
  for (t=0; t<maxlen; t++){
    if (str[t] == 0)  return t;
  }
  return t;

//  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }



static char str_ok_buf[100];
char *th_strtok(char *str1, const char *sep) { 
   
    static char* input = NULL;    // Stores the state of string
    char d = *sep;

    if (str1 != NULL)   // Initialize the input string
        input = str1;
  
    // Case for final token
    if (input == NULL)
        return NULL;
  
    // Stores the extracted string

    char* result = str_ok_buf;
    int i = 0;
  
    // Start extracting string and
    // store it in array
    for (; input[i] != '\0'; i++) {
  
        // If delimeter is not reached
        // then add the current character
        // to result[i]
        if (input[i] != d)
        {
            result[i] = input[i];
            //printf("res %d = %c \n",i,result[i]);
        }
  
        // Else store the string formed
        else {
            result[i] = '\0';
            input = input + i + 1;
            return result;
        }
    }
  
    // Case when loop ends
      result[i] = '\0';
    input = NULL;
  
    // Return the resultant pointer
    // to the string
    return result;
//  return strtok(str1, sep); 
}


int th_atoi(const char *str) { 
    // A simple atoi() function
    int res = 0;    // Initialize result   
    int sign = 1;   // Initialize sign as positive
    int i = 0;      // Initialize index of first digit
 
    // If number is negative, then update sign
    if (str[0] == '-') {
        sign = -1;
        i++;  // Also update index of first digit
    }
 
    // Iterate through all digits and update the result
    for (; str[i] != '\0'; ++i)
        res = res * 10 + str[i] - '0';
 
    // Return result with sign
    return sign * res;
//  return atoi(str); 
}

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }

char buff_uart_tx[EE_CMD_SIZE];
void th_printf(const char *p_fmt, ...) {
  //printf("flag \n");
  va_list args;
  va_start(args, p_fmt);
  vsnprintf(buff_uart_tx,EE_CMD_SIZE,p_fmt,args);
#ifdef NORMAL_PRINTF
  printf("%s", buff_uart_tx);
#else
  pi_uart_write(&uart, buff_uart_tx, strlen(buff_uart_tx));
#endif
  va_end(args);
}

char th_getchar() {

  char a_get;
  pi_uart_read(&uart, &a_get,1);
  return a_get;   
}

void th_serialport_initialize(void) {

  #ifdef __gap9__
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
  #endif

  pi_pad_set_function(PI_PAD_060, PI_PAD_FUNC0);
  pi_pad_set_function(PI_PAD_061, PI_PAD_FUNC0);

  struct pi_uart_conf conf;
  /* Init & open uart. */
  pi_uart_conf_init(&conf);
  conf.enable_tx = 1;
  conf.enable_rx = 1;
  conf.uart_id = 0;
  conf.use_fast_clk = 1;
  conf.word_size = PI_UART_WORD_SIZE_8_BITS;

# if EE_CFG_ENERGY_MODE==1
  conf.baudrate_bps = 9600;
# else
  conf.baudrate_bps = 115200;
  //conf.baudrate_bps = 921600;
  //conf.baudrate_bps = 115200;
# endif
  pi_open_from_conf(&uart, &conf);
  if (pi_uart_open(&uart))
  {
      th_printf("Uart open failed !\n");
      pmsis_exit(-1);
  }
}

void th_timestamp(void) {
# if EE_CFG_ENERGY_MODE==1
  pi_gpio_pin_write(&gpio_ts, GPIO_TS, 0);
  pi_time_wait_us(DELAY_US);
  pi_gpio_pin_write(&gpio_ts, GPIO_TS, 1);
# else
  uint32_t microSeconds = 0;

  /* USER CODE 2 BEGIN */
  microSeconds = (uint32_t) pi_time_get_us();
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, (uint32_t) microSeconds);
#endif
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  struct pi_gpio_conf gpio_conf = {0};
  pi_gpio_conf_init(&gpio_conf);
  pi_open_from_conf(&gpio_ts, &gpio_conf);
  gpio_conf.port = (GPIO_TS & PI_GPIO_NUM_MASK) / 32;
  int errors = pi_gpio_open(&gpio_ts);
  if (errors)
  {
      th_printf("Error opening GPIO %d\n", errors);
      pmsis_exit(errors);
  }
  pi_gpio_pin_configure(&gpio_ts, GPIO_TS, PI_GPIO_OUTPUT);

  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
