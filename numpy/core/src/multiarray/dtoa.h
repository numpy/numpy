#ifndef _NPY_DTOA_H_
#define _NPY_DTOA_H_
#include "numpy/ndarraytypes.h"

NPY_NO_EXPORT char *
dtoa_double_to_string(double val, char format_code, int precision, int flags,
                      int *type);
#endif

