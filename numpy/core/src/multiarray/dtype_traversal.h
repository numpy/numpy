#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRAVERSAL_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRAVERSAL_H_

#include "array_method.h"

/* NumPy DType clear (object DECREF + NULLing) implementations */

NPY_NO_EXPORT int
npy_get_clear_object_strided_loop(
        void *traverse_context, PyArray_Descr *descr, int aligned,
        npy_intp fixed_stride,
        traverse_loop_function **out_loop, NpyAuxData **out_traversedata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT int
npy_get_clear_void_and_legacy_user_dtype_loop(
        void *traverse_context, PyArray_Descr *descr, int aligned,
        npy_intp fixed_stride,
        traverse_loop_function **out_loop, NpyAuxData **out_traversedata,
        NPY_ARRAYMETHOD_FLAGS *flags);

/* NumPy DType zero-filling implementations */

NPY_NO_EXPORT int
npy_object_get_fill_zero_loop(
        void *NPY_UNUSED(traverse_context), PyArray_Descr *NPY_UNUSED(descr),
        int NPY_UNUSED(aligned), npy_intp NPY_UNUSED(fixed_stride),
        traverse_loop_function **out_loop, NpyAuxData **NPY_UNUSED(out_auxdata),
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT int
npy_get_zerofill_void_and_legacy_user_dtype_loop(
        void *traverse_context, PyArray_Descr *dtype, int aligned,
        npy_intp stride, traverse_loop_function **out_func,
        NpyAuxData **out_auxdata, NPY_ARRAYMETHOD_FLAGS *flags);


/* Helper to deal with calling or nesting simple strided loops */

typedef struct {
    traverse_loop_function *func;
    NpyAuxData *auxdata;
    PyArray_Descr *descr;
} NPY_traverse_info;


static inline void
NPY_traverse_info_init(NPY_traverse_info *cast_info)
{
    cast_info->func = NULL;  /* mark as uninitialized. */
    cast_info->auxdata = NULL;  /* allow leaving auxdata untouched */
    cast_info->descr = NULL;  /* mark as uninitialized. */
}


static inline void
NPY_traverse_info_xfree(NPY_traverse_info *traverse_info)
{
    if (traverse_info->func == NULL) {
        return;
    }
    traverse_info->func = NULL;
    NPY_AUXDATA_FREE(traverse_info->auxdata);
    Py_XDECREF(traverse_info->descr);
}


static inline int
NPY_traverse_info_copy(
        NPY_traverse_info *traverse_info, NPY_traverse_info *original)
{
    traverse_info->func = NULL;
    if (original->func == NULL) {
        /* Allow copying also of unused clear info */
        return 0;
    }
    traverse_info->auxdata = NULL;
    if (original->auxdata != NULL) {
        traverse_info->auxdata = NPY_AUXDATA_CLONE(original->auxdata);
        if (traverse_info->auxdata == NULL) {
            return -1;
        }
    }
    Py_INCREF(original->descr);
    traverse_info->descr = original->descr;
    traverse_info->func = original->func;

    return 0;
}


NPY_NO_EXPORT int
PyArray_GetClearFunction(
        int aligned, npy_intp stride, PyArray_Descr *dtype,
        NPY_traverse_info *clear_info, NPY_ARRAYMETHOD_FLAGS *flags);


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRAVERSAL_H_ */
