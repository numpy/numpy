#ifndef _NPY_NPY_CONFIG_H_
#define _NPY_NPY_CONFIG_H_

#include "config.h"

/* Disable broken MS math functions */
#ifdef _MSC_VER
#undef HAVE_ATAN2
#undef HAVE_HYPOT
#endif

#endif
