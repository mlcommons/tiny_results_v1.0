// Copyright (C) 2022, Plumerai Ltd.
// All rights reserved.
#ifndef PLUMERAI_INFERENCE_ENGINE_H
#define PLUMERAI_INFERENCE_ENGINE_H

#include <cstdint>

#include "tensorflow_compatibility.h"

namespace plumerai {

template <bool report_mode = false>
class InferenceEngine {
 public:
  // The lifetime of the tensor arena and optional profiler must be at least
  // as long as that of the interpreter object, since the interpreter may need
  // to access them at any time. The interpreter doesn't do any deallocation of
  // any of the pointed-to objects, ownership remains with the caller.
  // If `report_mode` is true then the `profiler` argument is ignored and a
  // profiler is allocated by the engine itself in the arena and `print_report`
  // becomes available.
  InferenceEngine(std::uint8_t* tensor_arena_ptr, int tensor_arena_size,
                  ::tflite::MicroProfiler* profiler = nullptr);
  ~InferenceEngine();

  // Runs through the model and allocates all necessary input, output and
  // intermediate tensors in the tensor arena.
  TfLiteStatus AllocateTensors();

  // Run inference, assumes input data is already set
  TfLiteStatus Invoke();

  TfLiteTensor* input(int input_id);
  TfLiteTensor* output(int output_id);
  size_t inputs_size() const;
  size_t outputs_size() const;

  // For debugging only.
  // This method gives the optimal arena size. It's only available after
  // `AllocateTensors` has been called.
  size_t arena_used_bytes() const;

  // For analysis, only available if `report_mode = true`
  void print_report() const;

 private:
  struct impl;
  impl* impl_;
};

}  // namespace plumerai

#endif  // PLUMERAI_INFERENCE_ENGINE_H
