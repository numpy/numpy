#ifndef _RANDOM_BITGEN_H
#define _RANDOM_BITGEN_H

#ifndef __VMS
#pragma once
#endif
#include <stddef.h>
#include <stdbool.h>
#if defined(__VMS) && __CRTL_VER <= 80400000
#include "numpy/stdint_vms.h"
#else
#include <stdint.h>
#endif


/* Must match the declaration in numpy/random/<any>.pxd */

typedef struct bitgen {
  void *state;
  uint64_t (*next_uint64)(void *st);
  uint32_t (*next_uint32)(void *st);
  double (*next_double)(void *st);
  uint64_t (*next_raw)(void *st);
} bitgen_t;


#endif
