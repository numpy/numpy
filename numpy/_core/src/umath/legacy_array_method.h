#ifndef _NPY_LEGACY_ARRAY_METHOD_H
#define _NPY_LEGACY_ARRAY_METHOD_H

#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "array_method.h"


NPY_NO_EXPORT PyArrayMethodObject *
PyArray_NewLegacyWrappingArrayMethod(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *signature[]);



/*
 * The following two symbols are in the header so that other places can use
 * them to probe for special cases (or whether an ArrayMethod is a "legacy"
 * one).
 */
NPY_NO_EXPORT int
get_wrapped_legacy_ufunc_loop(PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT NPY_CASTING
wrapped_legacy_resolve_descriptors(PyArrayMethodObject *,
        PyArray_DTypeMeta **, PyArray_Descr **, PyArray_Descr **, npy_intp *);


#endif  /*_NPY_LEGACY_ARRAY_METHOD_H */
