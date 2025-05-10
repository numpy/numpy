#ifndef NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRANSFER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRANSFER_H_

#include "array_method.h"


/*
 * More than for most functions, cast information needs to be stored in
 * a few places.  Most importantly, in many cases we need to chain or wrap
 * casts (e.g. structured dtypes).
 *
 * This struct provides a place to store all necessary information as
 * compact as possible.  It must be used with the inline functions below
 * to ensure correct setup and teardown.
 *
 * In general, the casting machinery currently handles the correct set up
 * of the struct.
 */
typedef struct {
    PyArrayMethod_StridedLoop *func;
    NpyAuxData *auxdata;
    PyArrayMethod_Context context;
    /* Storage to be linked from "context" */
    PyArray_Descr *descriptors[2];
} NPY_cast_info;


/*
 * Create a new cast-info struct with cast_info->context.descriptors linked.
 * Compilers should inline this to ensure the whole struct is not actually
 * copied.
 * If set up otherwise, func must be NULL'ed to indicate no-cleanup necessary.
 */
static inline void
NPY_cast_info_init(NPY_cast_info *cast_info)
{
    cast_info->func = NULL;  /* mark as uninitialized. */
    /*
     * Support for auxdata being unchanged, in the future, we might add
     * a scratch space to `NPY_cast_info` and link to that instead.
     */
    cast_info->auxdata = NULL;
    cast_info->context.descriptors = cast_info->descriptors;

    // TODO: Delete this again probably maybe create a new minimal init macro
    cast_info->context.caller = NULL;
}


/*
 * Free's all references and data held inside the struct (not the struct).
 * First checks whether `cast_info.func == NULL`, and assume it is
 * uninitialized in that case.
 */
static inline void
NPY_cast_info_xfree(NPY_cast_info *cast_info)
{
    if (cast_info->func == NULL) {
        return;
    }
    assert(cast_info->context.descriptors == cast_info->descriptors);
    NPY_AUXDATA_FREE(cast_info->auxdata);
    Py_DECREF(cast_info->descriptors[0]);
    Py_XDECREF(cast_info->descriptors[1]);
    Py_XDECREF(cast_info->context.method);
    cast_info->func = NULL;
}


/*
 * Move the data from `original` to `cast_info`. Original is cleared
 * (its func set to NULL).
 */
static inline void
NPY_cast_info_move(NPY_cast_info *cast_info, NPY_cast_info *original)
{
    *cast_info = *original;
    /* Fix internal pointer: */
    cast_info->context.descriptors = cast_info->descriptors;
    /* Mark original to not be cleaned up: */
    original->func = NULL;
}

/*
 * Finalize a copy (INCREF+auxdata clone). This assumes a previous `memcpy`
 * of the struct.
 * NOTE: It is acceptable to call this with the same struct if the struct
 *       has been filled by a valid memcpy from an initialized one.
 */
static inline int
NPY_cast_info_copy(NPY_cast_info *cast_info, NPY_cast_info *original)
{
    cast_info->context.descriptors = cast_info->descriptors;

    assert(original->func != NULL);
    cast_info->func = original->func;
    cast_info->descriptors[0] = original->descriptors[0];
    Py_XINCREF(cast_info->descriptors[0]);
    cast_info->descriptors[1] = original->descriptors[1];
    Py_XINCREF(cast_info->descriptors[1]);
    cast_info->context.caller = original->context.caller;
    Py_XINCREF(cast_info->context.caller);
    cast_info->context.method = original->context.method;
    Py_XINCREF(cast_info->context.method);
    if (original->auxdata == NULL) {
        cast_info->auxdata = NULL;
        return 0;
    }
    cast_info->auxdata = NPY_AUXDATA_CLONE(original->auxdata);
    if (NPY_UNLIKELY(cast_info->auxdata == NULL)) {
        /* No need for cleanup, everything but auxdata is initialized fine. */
        return -1;
    }
    return 0;
}


NPY_NO_EXPORT int
_strided_to_strided_move_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata));

NPY_NO_EXPORT int
_strided_to_strided_copy_references(
        PyArrayMethod_Context *NPY_UNUSED(context), char *const *args,
        const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *NPY_UNUSED(auxdata));


NPY_NO_EXPORT int
any_to_object_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT int
object_to_any_get_loop(
        PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int move_references,
        const npy_intp *NPY_UNUSED(strides),
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT int
wrap_aligned_transferfunction(
        int aligned, int must_wrap,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArray_Descr *src_wrapped_dtype, PyArray_Descr *dst_wrapped_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata, int *out_needs_api);


NPY_NO_EXPORT int
get_nbo_cast_datetime_transfer_function(int aligned,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata);

NPY_NO_EXPORT int
get_nbo_datetime_to_string_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata);

NPY_NO_EXPORT int
get_nbo_string_to_datetime_transfer_function(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata);

NPY_NO_EXPORT int
get_datetime_to_unicode_transfer_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api);

NPY_NO_EXPORT int
get_unicode_to_datetime_transfer_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api);

/* Creates a wrapper around copyswapn or legacy cast functions */
NPY_NO_EXPORT int
get_wrapped_legacy_cast_function(int aligned,
        npy_intp src_stride, npy_intp dst_stride,
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        int move_references,
        PyArrayMethod_StridedLoop **out_stransfer,
        NpyAuxData **out_transferdata,
        int *out_needs_api, int allow_wrapped);


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRANSFER_H_  */
