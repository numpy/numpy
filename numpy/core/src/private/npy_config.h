#ifndef _NPY_NPY_CONFIG_H_
#define _NPY_NPY_CONFIG_H_

#include "config.h"

/* Disable broken MS math functions */
#if defined(_MSC_VER) || defined(__MINGW32_VERSION)
#undef HAVE_ATAN2
#undef HAVE_HYPOT
#endif

/* Disable broken Sun Workshop Pro math functions */
#ifdef __SUNPRO_C
#undef HAVE_ATAN2
#endif

#endif
