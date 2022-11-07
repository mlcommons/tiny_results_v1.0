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

#ifndef V0_1_IC_MODEL_SETTINGS_H_
#define V0_1_IC_MODEL_SETTINGS_H_


#define DNN_INPUT_ADDR (0x6002E800)
#define DNN_OUTPUT_ADDR (0x6006FFF6)

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
constexpr int kNumCols = 32;
constexpr int kNumRows = 32;
constexpr int kNumChannels = 3;

constexpr int kInputSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 10;
constexpr int kBytesPerValue = 2; // img class uses 16b inputs

extern const char* kCategoryLabels[kCategoryCount];

#define BIN_DATA_TYPE uint8_t
#endif  // V0_1_IC_MODEL_SETTINGS_H_
