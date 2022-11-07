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


#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <Wire.h>
#include <mbed.h>

#include "quantization_helpers.h"
#include "mlperf_submitter.h"
#include "mlperf_internal.h"

using namespace mbed;

// There is a (char *)msg_buff in DSP's status struct used to pass messages from the DSP to the MCU
// MSG_BUFF_OFFSET is the location of status.msg_buff relative to &status in the DSP
#define MSG_BUFF_OFFSET 8
#define MSG_BUFF_LEN 512

static uint32_t mbin_toggle_value = 0x80;

extern struct syntiant_ndp_device_s *ndp;

DigitalOut dbg_test_pin(digitalPinToPinName(D9));
DigitalOut timestampPin(digitalPinToPinName(D7)); // we need a pin to strobe to mark time for the energy test

#if defined(AD_BENCHMARK)
// declaring this file-global because th_load_tensor sets it and 
// th_results() needs to access it.  I'm sorry.
float input[kInputSize];
uint16_t softmax[kInputSize];
#endif

void get_and_print_activations()
{
    int8_t acts[kCategoryCount]; 
     // Before 9/27/22 this was writing to reg 0x300835c0 (DSP main data memory), but why?  
    // syntiant_ndp120_write(ndp,1,/* NDP Reg*/ NDP120_DNN_CONFIG_DNNDATACFG0, /* with value */ 0x0F0F); 
    syntiant_ndp120_write(ndp,1,/* NDP Reg*/ 0x300835c0, /* with value */ 0x0F0F); 
    syntiant_ndp120_read_block(ndp,1, DNN_OUTPUT_ADDR, acts, kCategoryCount);
    printf("Acts (infer): [");
    for(size_t loop0=0; loop0 < 20; loop0++){
      printf("%6d, ", acts[loop0]); //int8
    }
    printf("]\n");
}

void toggle_gpio()
{
    int cnt; 
    uint32_t data;
    // Send MBOX Command TOGGLE_GPIO
    syntiant_ndp120_write(ndp, 0, NDP120_SPI_MBIN, CMD_TOGGLE_GPIO | mbin_toggle_value);
    mbin_toggle_value = 0x80 ^ (mbin_toggle_value & 0x80);
    // Wait MBOX Interrupt
    for (cnt=0; cnt <50000; cnt++) {   // was 50
        syntiant_ndp120_read(ndp, 0, NDP120_SPI_INTSTS, &data);
        if (data & 0x4) { // got interrupt
            break;
        } 
        //delay(1);  // was 10
    }
    
    syntiant_ndp120_write(ndp, 0, NDP120_SPI_INTSTS, data); // clear interrupt status

    if (!(data & 0x4)) {
        th_printf("error(toggle-gpio): didn't get interrupt\n");
    }
}

void send_zeros(uint16_t num_bytes) {
  uint8_t *zeros;
  th_printf("send_zeros %d bytes\r\n", num_bytes);
  zeros=(uint8_t*)malloc(num_bytes);
  memset(zeros, 0, num_bytes);
  syntiant_ndp120_write(ndp,1, NDP120_DNN_CONFIG_DNNDATACFG0, 0x0000);
  syntiant_ndp120_write_block(ndp, 0, NDP120_SPI_SAMPLE, (uint8_t *)zeros, num_bytes);
  delay(500); 
}

// Implement this method to prepare for inference and preprocess inputs.
// for the image classification benchmark, the reference uses unsigned INT8 inputs,
// but our model uses signed INT16
void th_load_tensor() {
  // almost the same except KWS uses int8, others use unsigned int8
  #if defined(KWS_BENCHMARK)
  int8_t input[kInputSize];
  int16_t input_final[kInputSize]; 
  #elif defined(IC_BENCHMARK)
  uint8_t input[kInputSize];
  int16_t input_final[kInputSize]; 
  #elif defined(VWW_BENCHMARK)
  //uint8_t input[kInputSize];
  uint8_t *input;
  input = (uint8_t *)malloc(kInputSize*sizeof(uint8_t));
  uint8_t *input_final;
  input_final = (uint8_t *)malloc(kInputSize*sizeof(uint8_t));
  printf("jhdbg: allocated input, input_final with %d bytes\r\n", kInputSize*sizeof(uint8_t));
  #elif defined(AD_BENCHMARK)
  // float input[kInputSize]; // using file-global input, declared at top
  int16_t input_final[kInputSize]; 
  
  
  #elif defined(TEST_BENCHMARK)
  uint8_t input[kInputSize*kBytesPerValue];
  #endif

  
  
  size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input),
                               kInputSize * sizeof(BIN_DATA_TYPE));

  if (bytes / sizeof(BIN_DATA_TYPE) != kInputSize) {
    th_printf("Input db has %d elemented, expected %d\n", bytes / sizeof(int8_t),
              kInputSize);
    th_printf("Printing input buffer before reformatting:\r\n");
    for (uint32_t j=0;j<bytes;j++)
      th_printf("%d ",input[j]);
    th_printf("\r\n");
    return;
  }

  #if defined(KWS_BENCHMARK)
  // Correcting for 0 point and scale
  double buff[kInputSize];
  for(int i = 0; i < kInputSize; i++) {
      buff[i] = 0.5847029089927673 * ((double)input[i] - 83.0);
  };
  // This part is only needed for the KWS benchmark (I think??) JHH30JUN2022
  //Casting the int8_t into int16_t
  int16_t *input_16 = (int16_t *)input;
  for(int i = 0; i < kInputSize; i++) {
      double temp_value = floor(buff[i] * 512 + 0.5);
      // clip values
      if (temp_value > 32767.0){
          temp_value = 32767.0;
      } else if (temp_value < -32768.0) {
          temp_value = -32768.0;
      }
      input_16[i] = (int16_t)temp_value;
  };

  // Rearrange Input Tensor
  
  int elem_index = 0;
  for(int i = 0; i < kNumCols; i++) {
      for (int j = 0; j < kNumRows; j++) {
      input_final[elem_index++] = input_16[kNumCols*j + i];
      }
  } 
  #elif defined(IC_BENCHMARK)
  // This part is only needed for the image classification benchmark
  // Rearrange Input Tensor -- How are the bin files arranged and how does the SYN model need it?
  // int16_t readback_buff[kInputSize];
  int elem_index = 0;  
  // input[] is  uint8_t for img classification
  // map uint8 [0,255] into int16 [-32768,+32767]
  // SIGNED16 fixed-point format, [sgn][6 int bits][9 frac bits]
  // so "1.0" => 512 as a binary int16.
  // model conversion included 1/8 scaling factor (512/8 => 64)
  // image.bin files are flattened in channel, column, row order
  for (int i = 0; i < kNumCols; i++) {
    for (int j = 0; j < kNumRows; j++) {
      for(int k = 0; k < kNumChannels; k++) {
        input_final[elem_index++] = ((int16_t)input[kNumCols*kNumChannels*j + kNumChannels*i + k]) * 64;
      }
    }
  }
  #elif defined(VWW_BENCHMARK)
  int elem_index = 0;  
  for (int i = 0; i < kNumCols; i++) {
    for (int j = 0; j < kNumRows; j++) {
      for(int k = 0; k < kNumChannels; k++) {
        input_final[elem_index] = (uint8_t)(input[kNumCols*kNumChannels*j + kNumChannels*i + k]);
        elem_index++;
      }
    }
  }
  
  #elif defined(AD_BENCHMARK)
  for (int i = 0; i < kFeatureElementCount; i++) {
        input_final[i] = (int16_t)(input[i]*512.0); // SIGNED16 is s6.9, so 1.0 float = 512 int16
  }
  
  #error "Only KWS, VWW, IC, AD are defined here"
  #endif
  
  // cast input_final as uint8 ptr because we're just writing bytes
  // th_printf("About to upload data\n");
  // jhdb_tstmp0 = micros(); 
  syntiant_ndp120_write(ndp,1, NDP120_DNN_CONFIG_DNNDATACFG0, 0x0000);
  syntiant_ndp120_write_block(ndp, 0, NDP120_SPI_SAMPLE, (uint8_t *)input_final, kInputSize*kBytesPerValue);
  delay(500); 
  
  
  syntiant_ndp120_write(ndp,1, NDP120_DNN_CONFIG_DNNDATACFG0, 0x0000);

  #if defined(VWW_BENCHMARK)
  free(input);
  free(input_final);
  #endif
  // th_printf("jhdbg: th_load_tensor ending at %lu", micros());
}

#if defined(AD_BENCHMARK)
float calculate_result(){
  float diffsum = 0;
  float out_float;
  float diff;
  for (size_t i = 0; i < kFeatureElementCount; i++) {
    // not actually softmax here, but still using that to hold DNN Output
    // the outputs are shifted up by 128 to hitchhike on the unsigned softmax 
    // array and scaled up by 2x to utilize the [-128,+127] range of int8
    out_float = ((float)(softmax[i]-128))/ 2.0;
    diff = out_float - input[i];
    diffsum += diff * diff;
    //if(i<20) printf("In=%f vs Out=%f => (cum) %f\r\n", input[i], out_float, diffsum/(i+1));
  }
  diffsum /= kFeatureElementCount;

  return diffsum;
}
#endif

void th_results() {
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  
  #ifdef AD_BENCHMARK
  float result;
  #else
  uint16_t softmax[kCategoryCount]; // declared globally for AD
  #endif
  int8_t check_acts[kCategoryCount];
  float softmax_float;

  // The read address here 0x1ffff0a0 comes from STATUS_SOFTMAX_PTR in the dsp code: tinymlperf_top.h
  syntiant_ndp120_read_block(ndp,1, 0x1ffff0a0, &softmax[0], (kCategoryCount*2));
  
  // last arg is mask to enable memory banks, not sure if this is necessary
  // bit = 1 => bank is asleep (power-gate or retention) ; 0 => bank is awake
  syntiant_ndp120_write(ndp,1,/* NDP Reg*/ NDP120_DNN_CONFIG_DNNDATACFG0, /* with value */ 0x0000); // 0x0F0F

  syntiant_ndp120_read_block(ndp,1, DNN_OUTPUT_ADDR, &check_acts[0], kCategoryCount);  // 0x6002E7EC for IC; 0x6002FC20 for KWS
 
  #ifdef AD_BENCHMARK
  result = calculate_result();
  th_printf("m-results-[");
  th_printf("%5.3f", result);
  #else
  th_printf("m-results-[");
  for(size_t loop0=0; loop0 < kCategoryCount; loop0++){
    softmax_float = ((softmax[loop0] -1)/ pow(2, 15));
    softmax_float = (softmax_float > 0) ? softmax_float : 0;

    //Some platforms don't implement floating point formatting.
    th_printf("0.%d", static_cast<int>(softmax_float*10));
    th_printf("%d", static_cast<int>(softmax_float * 100) % 10);
    th_printf("%d", static_cast<int>(softmax_float * 1000) % 10);

    if (loop0 < (kCategoryCount - 1)) {
      th_printf(",");
    }
  }
  #endif
  th_printf("]\r\n");

}

void th_get_msg() {
  unsigned char dsp_stat_buff[100];
  unsigned char dsp_msg_buff[MSG_BUFF_LEN];
  uint32_t dsp_status_ptr = 0;
  uint32_t dsp_msg_ptr = 0;

  // read the address of the status struct from STATUS_PTR_ADDR into dsp_status_ptr
  syntiant_ndp120_read_block(ndp,1, (uint32_t)STATUS_PTR_ADDR, &dsp_status_ptr, 4);
  // then read (for now) the whole status struct into dsp_msg_buff
  // later change to just read the msg_buff element from the status struct
  //printf("jhdbg: status pointer is at %p\r\n", (void *)dsp_status_ptr);
  syntiant_ndp120_read_block(ndp,1, dsp_status_ptr, &dsp_stat_buff, 100);
  dsp_msg_ptr = *((uint32_t*)(&dsp_stat_buff[MSG_BUFF_OFFSET]));
  //printf("DSP msg_ptr = %p\r\n", (void *)dsp_msg_ptr);
  syntiant_ndp120_read_block(ndp,1, dsp_msg_ptr, &(dsp_msg_buff[0]), MSG_BUFF_LEN);
  
  // printf("jhdbg: 1st 100 bytes of status:\r\n");
  // for(size_t loop0=0; loop0 < 100; loop0++){
  //   printf("%02X ", dsp_stat_buff[loop0]);
  //   if((loop0+1) % 10 == 0){
  //     printf("]]\r\n");
  //   }
  // }
  th_printf("DSP MSG buffer: [%s]\r\n", dsp_msg_buff);
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() {
  // was previously: runner->Invoke(); 
    uint32_t data, cnt = 50;
  // Use Ilib to exchange Mailbox command with DSP Code
  // -- Sent Run DNN
  // -- Wait for ACK 
  // -- Wait for Feature/Match/Watterk Interrupt

    // Send MBOX Command Run DNN 
    
    syntiant_ndp120_write(ndp, 0, NDP120_SPI_MBIN, 0x09 | mbin_toggle_value);
    mbin_toggle_value = 0x80 ^ (mbin_toggle_value & 0x80);
    // Wait MBOX Interrupt
    for (cnt = 0; cnt <50000; cnt++) {   // was 50
        syntiant_ndp120_read(ndp, 0, NDP120_SPI_INTSTS, &data);
        if (data & 0x4) { // got interrupt
            break;
        } 
        //delay(1);  // was 10
    }

    syntiant_ndp120_write(ndp, 0, NDP120_SPI_INTSTS, data); // clear interrupt status

    if (!(data & 0x4)) {
        th_printf("error(th_infer): didn't get interrupt\n");
        goto error;
    }

    syntiant_ndp120_read(ndp, 0, NDP120_SPI_MBOUT, &data); // read mbout register for ack
    if (!(data & 0x7f)) {
        ; // th_printf("got ack from dsp\n");
    } else {
        th_printf("error: no ack from dsp\n");
        goto error;
    }
    // wait for feature interrupt
    for (cnt = 0; cnt < 50000; cnt++) {  // was 100
        syntiant_ndp120_read(ndp, 0, NDP120_SPI_INTSTS, &data);
        if (data & 0x10) {
            break;
        }
        //delay(1);  // was 10
    }
    // printf("jhdbg: Done waiting for feature interrupt. cnt=%d\n", cnt);
    syntiant_ndp120_write(ndp, 0, NDP120_SPI_INTSTS, data); // clear interrupt status

    if (!(data & 0x10)) {
        th_printf("error (th_infer 2): didn't get interrupt\n");
    }
    else {
        ; // th_printf("got feature interrupt\n");
    }
error:
    return;
}

/// \brief optional API.
void th_final_initialize(void) {
  // TODO Any initialization can go here
}

void th_pre() {}
void th_post() {
  uint32_t data, cnt = 50;
  // Send MBOX Command TOGGLE GPIO
  // toggle_gpio();

  syntiant_ndp120_read(ndp, 0, NDP120_SPI_MBOUT, &data); // read mbout register for ack
  if (!(data & 0x7f)) {
      // th_printf("got ack from dsp\n");
  } else {
      th_printf("error: error ack from dsp\n");
      goto error;
  }

error:
  return;
  
} // end of th_post()

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
void th_printf(const char *p_fmt, ...) {
  // avoiding re-allocation only saves about 30us, but I'll leave it 
  // unless it causes a problem 5MAY2021JHH
  static char buffer[200]; 
  va_list args;
  va_start(args, p_fmt);
  vsprintf(buffer, p_fmt, args); 
  Serial1.print(buffer);
  va_end(args);
}

char th_getchar() {
  // return getchar();
  return Serial1.read();
}

void th_serialport_initialize(void) {
  // serial port setup is handled at the beginning of setup()
  // # if EE_CFG_ENERGY_MODE==1
  //   pc.baud(9600);
  // # else
  //   pc.baud(115200);
  // # endif
}

void toggle_dbg_pin(int pw_count) {
  dbg_test_pin = 0;
   for (int i=0; i<pw_count; ++i) { // was 100'000 jhdbg
    asm("nop");
  }
  dbg_test_pin = 1;
}

void th_timestamp(void) {
# if EE_CFG_ENERGY_MODE==1
  timestampPin = 0;
  for (int i=0; i<1000; ++i) { // was 100'000 jhdbg
    asm("nop");
  }
  timestampPin = 1;
# else
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  microSeconds = micros(); 
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
# endif

}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}
