/*
 * Common headers needed by every npy math compilation unit
 */

/* Any file that includes Python.h must include it before any other files */
/* https://docs.python.org/3/extending/extending.html#a-simple-example */
/* npy_common.h includes Python.h so it also counts in this list */
#include <Python.h>
#include <math.h>
#include <float.h>

#include "npy_config.h"
#include "numpy/npy_math.h"
