#include "Python.h"
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "randomkit.h"

extern void
init_by_array(rk_state *self, unsigned long init_key[],
              npy_intp key_length);
