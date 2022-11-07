#ifndef __vww_96_int8_H__
#define __vww_96_int8_H__

#define __PREFIX(x) vww_96_int8 ## x
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

extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE vww_96_int8_L3_Flash;
#endif