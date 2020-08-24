#ifndef _NPY_PRIVATE__COMPILED_BASE_H_
#define _NPY_PRIVATE__COMPILED_BASE_H_
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <Python.h>
#include <structmember.h>
#include <string.h>

#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include <numpy/ndarraytypes.h>
#include "numpy/npy_math.h"
#include "npy_config.h"
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "lowlevel_strided_loops.h" /* for npy_bswap8 */
#include "alloc.h"
#include "ctors.h"
#include "common.h"

NPY_NO_EXPORT PyObject *
arr_insert(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr__monotonicity(PyObject *, PyObject *, PyObject *kwds);
NPY_NO_EXPORT PyObject *
arr_interp(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_interp_complex(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_ravel_multi_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_unravel_index(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
io_pack(PyObject *, PyObject *, PyObject *);
NPY_NO_EXPORT PyObject *
io_unpack(PyObject *, PyObject *, PyObject *);

#endif
