#ifndef __kws_ref_model_H__
#define __kws_ref_model_H__

#define __PREFIX(x) kws_ref_model ## x
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

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE kws_ref_model_L3_Flash;
#endif