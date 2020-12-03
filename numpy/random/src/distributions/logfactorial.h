
#ifndef LOGFACTORIAL_H
#define LOGFACTORIAL_H

#if defined(__VMS) && __CRTL_VER <= 80400000
#include "numpy/stdint_vms.h"
#else
#include <stdint.h>
#endif


double logfactorial(int64_t k);

#endif
