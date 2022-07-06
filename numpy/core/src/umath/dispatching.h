#ifndef _NPY_DISPATCHING_H
#define _NPY_DISPATCHING_H

#define _UMATHMODULE

#include <numpy/ufuncobject.h>
#include "array_method.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int promoter_function(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

NPY_NO_EXPORT int
PyUFunc_AddLoop(PyUFuncObject *ufunc, PyObject *info, int ignore_duplicate);

NPY_NO_EXPORT int
PyUFunc_AddLoopFromSpec(PyObject *ufunc, PyArrayMethod_Spec *spec);

NPY_NO_EXPORT PyArrayMethodObject *
promote_and_get_ufuncimpl(PyUFuncObject *ufunc,
        PyArrayObject *const ops[],
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *op_dtypes[],
        npy_bool force_legacy_promotion,
        npy_bool allow_legacy_promotion,
        npy_bool promote_pyscalars,
        npy_bool ensure_reduce_compatible);

NPY_NO_EXPORT PyObject *
add_and_return_legacy_wrapping_ufunc_loop(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *operation_dtypes[], int ignore_duplicate);

NPY_NO_EXPORT int
default_ufunc_promoter(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

NPY_NO_EXPORT int
object_only_ufunc_promoter(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *NPY_UNUSED(op_dtypes[]),
        PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[]);

NPY_NO_EXPORT int
install_logical_ufunc_promoter(PyObject *ufunc);

#ifdef __cplusplus
}
#endif

#endif  /*_NPY_DISPATCHING_H */
