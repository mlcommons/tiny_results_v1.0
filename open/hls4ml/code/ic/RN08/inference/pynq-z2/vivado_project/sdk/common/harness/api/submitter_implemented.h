/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. All wrapped libc
methods from th_libc.h and all testharness methods from th_lib.h are here.
==============================================================================*/
/// \file
/// \brief Submitter-implemented methods required to perform inference.
/// \detail All methods with names starting with th_ are to be implemented by
/// the submitter. All basic I/O, inference and timer APIs must be implemented
/// in order for the benchmark to output useful results, but some auxiliary
/// methods default to an empty implementation. These methods are provided to
/// enable submitter optimizations, and are not required for submission.

#ifndef MLPERF_TINY_V0_1_API_SUBMITTER_IMPLEMENTED_H_
#define MLPERF_TINY_V0_1_API_SUBMITTER_IMPLEMENTED_H_

/// \brief These defines set logging prefixes for test harness integration.
/// \detail This API is designed for performance evaluation only. In order to
/// gather energy measurments we recommend using the EEMBC test suite.
#define EE_MSG_TIMESTAMP "m-lap-us-%lu\r\n"
#define TH_VENDOR_NAME_STRING "Xilinx_PynqZ2_Z7020"

#define TH_MODEL_VERSION EE_MODEL_VERSION_IC01

#define MAX_DB_INPUT_SIZE 3072*2

#ifndef TH_MODEL_VERSION
// See "internally_implemented.h" for a list
#error "PLease set TH_MODEL_VERSION to one of the EE_MODEL_VERSION_* defines"
// e.g.: to inform the user of model `ic01` use this:
// #define TH_MODEL_VERSION EE_MODEL_VERSION_IC01
#endif

// Use this to switch between DUT-direct (perf) & DUT-inderrect (energy) modes
#ifndef EE_CFG_ENERGY_MODE
#define EE_CFG_ENERGY_MODE 0
#endif

// This is a visual cue to the user when reviewing logs or plugging an
// unknown device into the system.
#if EE_CFG_ENERGY_MODE == 1
#define EE_MSG_TIMESTAMP_MODE "m-timestamp-mode-energy\r\n"
#else
#define EE_MSG_TIMESTAMP_MODE "m-timestamp-mode-performance\r\n"
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//extra "standard" libs
#include "unistd.h"
//#include <math.h>

// Accelerator C Driver
#include "xmyproject_axi.h"

// Timer
#include "xtime_l.h"     /* to measure performance of the system */

// Uart config
#include "xuartps.h"  //Higher level UART library for baud config


//Misc platform config
#include "../platform.h"
#include "xil_cache.h"   /* enable/disable caches etc */

#if EE_CFG_ENERGY_MODE == 1
//GPIO Config for Timestamp
#include "xgpiops.h" // GPIO PS Control Library
#define TIMER_PIN 12 //Arduino RST, Labeled "RST" on board between "IOREF" and "3.3V" on arduino shield connectors
#endif


//Accelerator and preprocessing stuff
/* base address for the accelerator */
#define MEM_BASE_ADDR XPAR_PS7_DDR_0_S_AXI_BASEADDR

/* data offsets and pointers */
#define SRC_BUFFER_BASE (MEM_BASE_ADDR + 0x00000000)
//unsigned char *src_mem = (unsigned char*)SRC_BUFFER_BASE;

#define DST_BUFFER_BASE (MEM_BASE_ADDR + 0x00020000)

#define src_FEATURE_COUNT 3072
#define dst_FEATURE_COUNT 10

/// \brief required core API
void th_load_tensor();
void th_results();
void th_infer();
void th_timestamp(void);
void th_printf(const char *fmt, ...);
char th_getchar();

/// \brief optional API
void th_serialport_initialize(void);
void th_timestamp_initialize(void);
void th_final_initialize(void);
void th_pre();
void th_post();
void th_command_ready(char volatile *msg);

/// \brief libc hooks
int th_strncmp(const char *str1, const char *str2, size_t n);
char *th_strncpy(char *dest, const char *src, size_t n);
size_t th_strnlen(const char *str, size_t maxlen);
char *th_strcat(char *dest, const char *src);
char *th_strtok(/*@null@*/ char *str1, const char *sep);
int th_atoi(const char *str);
void *th_memset(void *b, int c, size_t len);
void *th_memcpy(void *dst, const void *src, size_t n);
int th_vprintf(const char *format, va_list ap);

#endif  // MLPERF_TINY_V0_1_API_SUBMITTER_IMPLEMENTED_H_
