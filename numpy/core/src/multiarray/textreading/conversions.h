#ifndef NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_CONVERSIONS_H_
#define NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_CONVERSIONS_H_

#include <stdbool.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"

#include "textreading/parser_config.h"

NPY_NO_EXPORT int
npy_to_bool(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

NPY_NO_EXPORT int
npy_to_float(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

NPY_NO_EXPORT int
npy_to_double(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

NPY_NO_EXPORT int
npy_to_cfloat(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

NPY_NO_EXPORT int
npy_to_cdouble(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

NPY_NO_EXPORT int
npy_to_string(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused);

NPY_NO_EXPORT int
npy_to_unicode(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused);

NPY_NO_EXPORT int
npy_to_generic_with_converter(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *unused, PyObject *func);

NPY_NO_EXPORT int
npy_to_generic(PyArray_Descr *descr,
        const Py_UCS4 *str, const Py_UCS4 *end, char *dataptr,
        parser_config *pconfig);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_TEXTREADING_CONVERSIONS_H_ */
