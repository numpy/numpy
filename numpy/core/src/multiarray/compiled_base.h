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
#include "simd/simd.h"
#include "ctors.h"
#include "common.h"

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "compiled_base.dispatch.h"
#endif

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void simd_compiled_base_pack_inner,
(const char *inptr, npy_intp element_size, npy_intp n_in, npy_intp in_stride, char *outptr, npy_intp n_out, npy_intp out_stride, char order))

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
