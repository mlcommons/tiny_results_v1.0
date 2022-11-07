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
/// \brief Model runner for Plumerai Inference Engine.

#include "api/submitter_implemented.h"
#include "plumerai/inference_engine.h"

extern "C" void DebugLog(const char *s) {
  th_printf(s);
}

template <typename inputT, typename outputT> class PlumeraiModelRunner {
public:
  PlumeraiModelRunner(uint8_t *tensor_arena, int tensor_arena_size)
      : interpreter_(tensor_arena, tensor_arena_size) {
    auto allocate_status = interpreter_.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      th_printf("ERROR: AllocateTensors() failed");
    }
  }

  void Invoke() {
    // Run the model on this input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter_.Invoke();
    if (invoke_status != kTfLiteOk) {
      th_printf("ERROR: Invoke failed");
    }
  }

  void SetInput(const inputT *custom_input) {
    TfLiteTensor *input = interpreter_.input(0);
    inputT *input_buffer = tflite::GetTensorData<inputT>(input);
    int input_length = input->bytes / sizeof(inputT);
    for (int i = 0; i < input_length; i++) {
      input_buffer[i] = custom_input[i];
    }
  }

  outputT *GetOutput() {
    return tflite::GetTensorData<outputT>(interpreter_.output(0));
  }

  int input_size() { return interpreter_.input(0)->bytes / sizeof(inputT); }

  int output_size() { return interpreter_.output(0)->bytes / sizeof(outputT); }

  float output_scale() { return interpreter_.output(0)->params.scale; }

  int output_zero_point() { return interpreter_.output(0)->params.zero_point; }

  float input_scale() { return interpreter_.input(0)->params.scale; }

  int input_zero_point() { return interpreter_.input(0)->params.zero_point; }

private:
  plumerai::InferenceEngine<false> interpreter_;
};
