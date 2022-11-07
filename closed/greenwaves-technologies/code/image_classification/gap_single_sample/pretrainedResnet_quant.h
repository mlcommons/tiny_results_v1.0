#ifndef __pretrainedResnet_quant_H__
#define __pretrainedResnet_quant_H__

#define __PREFIX(x) pretrainedResnet_quant ## x
// Include basic GAP builtins defined in the Autotiler
#include "Gap.h"

#ifdef __EMUL__
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/param.h>
#include <string.h>
#endif

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE pretrainedResnet_quant_L3_Flash;
#endif