/*======================= COPYRIGHT NOTICE ==================================*]
[* Copyright (c) 2021-2022 Qualcomm Technologies, Inc.                       *]
[* All Rights Reserved.                                                      *]
[* Confidential and Proprietary - Qualcomm Technologies, Inc.                *]
[*===========================================================================*/

#ifndef _AD_MODEL_SETTINGS_H
#define _AD_MODEL_SETTINGS_H

const char *modelName = "ad_model.eai";
const char *inputPath = "anomaly_id_01_00000003_hist_librosa.bin";
const int kFeatureSliceSize = 128;
const int kFeatureSliceCount = 5;
const int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);

const int kSpectrogramSliceCount = 200;
const int kInputSize = (kFeatureSliceSize * kFeatureSliceCount);
//get these quantization info from model
const int qIntput_Zero = 89;
const float qIntput_Sacle = 0.3910152316093445;
const int qOntput_Zero = 96;
const float qOntput_Sacle = 0.36449846625328064;
const bool SAVE_OUTPUT = false;
/*=============================== For UART ==================================*/
char const *PATH_TO_SERIAL = "/dev/ttyMSM0"; 
#endif //__EAI_TRAINING_H__