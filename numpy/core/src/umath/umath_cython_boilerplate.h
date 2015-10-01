#define _UMATHMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_config.h"

#ifdef ENABLE_SEPARATE_COMPILATION
#define PY_ARRAY_UNIQUE_SYMBOL _npy_umathmodule_ARRAY_API
#define NO_IMPORT_ARRAY
#endif

#undef PyMODINIT_FUNC
#if defined(NPY_PY3K)
#define PyMODINIT_FUNC NPY_NO_EXPORT PyObject*
#else
#define PyMODINIT_FUNC NPY_NO_EXPORT void
#endif
