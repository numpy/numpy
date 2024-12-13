#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_

#include "numpy/npy_common.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/halffloat.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

#include "common.h"
#include "numpyos.h"
#include "umathmodule.h"
#include "gil_utils.h"
#include "static_string.h"
#include "dtypemeta.h"
#include "dtype.h"
#include "utf8_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **get_casts();

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_CASTS_H_ */
