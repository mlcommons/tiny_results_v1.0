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
/// \brief Visual wakewords model settings.

#ifndef VWW_MODEL_SETTINGS_H_
#define VWW_MODEL_SETTINGS_H_


// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
#define DNN_INPUT_ADDR (0x60029400)
#define DNN_OUTPUT_ADDR (0x6006FFFE)
#define BIN_DATA_TYPE uint8_t

constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 3;

constexpr int kInputSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 2;

constexpr int kBytesPerValue = 1; 

extern const char* kCategoryLabels[kCategoryCount];

#endif  // V0_1_IC_MODEL_SETTINGS_H_
