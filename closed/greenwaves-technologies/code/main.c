/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/// \file
/// \brief Main function to run benchmark on device.

#include "api/internally_implemented.h"
#include "submitter_implemented.h"

#include "pmsis.h"

int kickoff(void) {
  ee_benchmark_initialize();
  pi_time_wait_us(1000000);
#ifdef TEST_WITHOUT_EEMBC_SETUP
  static char command[6] = "infer\0";
  ee_profile_parse(command);
  pmsis_exit(0);
#else
  while (1) {
    int c;
    c = th_getchar();
    ee_serial_callback(c);
  }
#endif
  return 0;
}


int main(int argc, char *argv[])
{
    return pmsis_kickoff((void *) kickoff); 
    return 0;
}
