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


/* Selection kinds have separate contracts.  Add a kind and its parameter
 * structure for a new traversal instead of broadening an existing kind.
 */
typedef enum {
    NPY_TRANSFER_INDEXED_CONTIGUOUS,
    NPY_TRANSFER_REPEAT_CONTIGUOUS,
    NPY_TRANSFER_PUTMASK_CONTIGUOUS
} NPY_transfer_kind;

typedef struct {
    /* Number of contiguous elements in each block. */
    npy_intp blocksize;
    /* The number of independent groups that get the same indexing.
     * Scatter requires one group. */
    npy_intp outer_count;
    /* Contiguous, aligned, native-endian indices, read in order. */
    const npy_intp *indices;
    npy_intp index_dim;
    /* scatter is a boolean: 0 gathers indexed source blocks from
     * data[0] into consecutive destination blocks in data[1];
     * 1 scatters source blocks from data[0], cycling at value_period,
     * to indexed destination blocks in data[1].
     * take uses gather; put uses scatter. */
    int scatter, axis;
    NPY_CLIPMODE clipmode;
    /* Value blocks for scatter (>0); unused for gather. */
    npy_intp value_period;
} NPY_indexed_contiguous_transfer;

typedef struct {
    /* Number of contiguous elements in each block. */
    npy_intp blocksize;
    /* The number of independent groups that get the same repetition. */
    npy_intp outer_count;
    /* Repeat counts, shared by all outer groups. */
    const npy_intp *counts;
    /* Step in counts, in npy_intp elements: 0 uses counts[0] for every
     * block; 1 uses counts[j] for block j. */
    npy_intp count_stride;
} NPY_repeat_contiguous_transfer;

typedef struct {
    const npy_bool *mask;
    /* Values cycle at every mask position, including false entries. */
    npy_intp value_period;
} NPY_putmask_contiguous_transfer;

/* Arguments for copying selected elements from data[0] to data[1].
 * kind chooses take/put, repeat, or putmask; fill the matching member of
 * selection with that operation's arguments. The copy function writes the
 * destination without changing this struct.
 */
typedef struct {
    NPY_transfer_kind kind;
    char *data[2];
    /* Number of selections per outer iteration. */
    npy_intp count;
    /* Bytes per element. */
    npy_intp itemsize;
    union {
        NPY_indexed_contiguous_transfer indexed;
        NPY_repeat_contiguous_transfer repeat;
        NPY_putmask_contiguous_transfer putmask;
    } selection;
} NPY_transfer_loop;

typedef int NPY_transfer_loop_function(NPY_cast_info *, const NPY_transfer_loop *);

#ifdef __cplusplus
extern "C" {
#endif
/* Return a copy function for the cast in info and the requested operation,
 * or NULL when the caller should use its existing copy path (e.g. memcpy
 * for data without references).
 * Call the returned function as copy(info, &loop), with loop.kind == kind.
 *
 * The caller owns the arrays and info. Keep them alive for the call, and
 * hold the GIL if the cast requires Python API access.
 */
NPY_NO_EXPORT NPY_transfer_loop_function *
NPY_get_transfer_loop(NPY_cast_info *info, NPY_transfer_kind kind);
NPY_NO_EXPORT NPY_transfer_loop_function *
NPY_get_string_transfer_loop(PyArrayMethod_StridedLoop *loop,
                             NPY_transfer_kind kind);
NPY_NO_EXPORT int
NPY_transfer_check_index(npy_intp index, npy_intp dim, int axis);
#ifdef __cplusplus
}
#endif


static inline void
NPY_context_init(PyArrayMethod_Context *context, PyArray_Descr *descr[2])
{
    context->descriptors = descr;
    context->caller = NULL;
    context->_reserved = NULL;
    context->flags = 0;
    context->parameters = NULL;
}

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
    NPY_context_init(&(cast_info->context), cast_info->descriptors);
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
    cast_info->context.flags = original->context.flags;
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


typedef int NPY_transfer_copy_function(NPY_cast_info *, char *, char *,
                                       npy_intp, const npy_intp *);

NPY_FINLINE int
npy_transfer_loop_impl(NPY_cast_info *info,
        const NPY_transfer_loop *loop, int scatter,
        NPY_transfer_copy_function *copy)
{
    char *src = loop->data[0], *dst = loop->data[1];
    const npy_intp size = loop->count;
    switch (loop->kind) {
        case NPY_TRANSFER_INDEXED_CONTIGUOUS: {
            const npy_intp strides[2] = {loop->itemsize, loop->itemsize};
            const npy_intp blocksize = loop->selection.indexed.blocksize;
            const npy_intp chunk = blocksize * loop->itemsize;
            const npy_intp period = loop->selection.indexed.value_period;
            const npy_intp *indices = loop->selection.indexed.indices;
            const npy_intp dim = loop->selection.indexed.index_dim;
            const NPY_CLIPMODE clipmode = loop->selection.indexed.clipmode;
            char *base = loop->data[scatter];
            char *values = loop->data[1 - scatter];
            for (npy_intp i = 0;
                    i < loop->selection.indexed.outer_count; i++) {
                npy_intp value_index = 0;
                for (npy_intp j = 0; j < size; j++) {
                    /* Read each index when visited: DECREF may call Python. */
                    npy_intp index = indices[j];
                    if (clipmode == NPY_WRAP) {
                        if (index < 0 || index >= dim) {
                            index %= dim;
                            if (index < 0) {
                                index += dim;
                            }
                        }
                    }
                    else if (clipmode == NPY_CLIP) {
                        index = index < 0 ? 0 : (index >= dim ? dim - 1 : index);
                    }
                    else {
                        if (NPY_UNLIKELY(index < -dim || index >= dim)) {
                            return NPY_transfer_check_index(
                                    index, dim, loop->selection.indexed.axis);
                        }
                        if (index < 0) {
                            index += dim;
                        }
                    }
                    char *indexed = base + index * chunk;
                    if (copy(info, scatter ? values : indexed,
                            scatter ? indexed : values, blocksize, strides) < 0) {
                        return -1;
                    }
                    values += chunk;
                    if (scatter && ++value_index == period) {
                        values = loop->data[0];
                        value_index = 0;
                    }
                }
                base += dim * chunk;
            }
            return 0;
        }
        case NPY_TRANSFER_REPEAT_CONTIGUOUS: {
            const npy_intp strides[2] = {loop->itemsize, loop->itemsize};
            const npy_intp blocksize = loop->selection.repeat.blocksize;
            const npy_intp chunk = blocksize * loop->itemsize;
            const npy_intp outer_count = loop->selection.repeat.outer_count;
            const npy_intp *counts = loop->selection.repeat.counts;
            const npy_intp count_stride = loop->selection.repeat.count_stride;
            for (npy_intp i = 0; i < outer_count; i++) {
                for (npy_intp j = 0; j < size; j++) {
                    npy_intp repeats = counts[j * count_stride];
                    /* Scan zero counts without entering the copy loop. */
                    while (repeats == 0 && j + 1 < size) {
                        repeats = counts[++j * count_stride];
                    }
                    for (npy_intp k = 0; k < repeats; k++) {
                        if (copy(info, src + j*chunk, dst,
                                blocksize, strides) < 0) {
                            return -1;
                        }
                        dst += chunk;
                    }
                }
                src += size * chunk;
            }
            return 0;
        }
        case NPY_TRANSFER_PUTMASK_CONTIGUOUS: {
            const npy_intp strides[2] = {loop->itemsize, loop->itemsize};
            const npy_intp period = loop->selection.putmask.value_period;
            const npy_bool *mask = loop->selection.putmask.mask;
            for (npy_intp i = 0, j = 0; i < size; i++, j++) {
                if (j == period) {
                    j = 0;
                }
                if (mask[i] && copy(info, src + j*strides[0],
                        dst + i*strides[1], 1, strides) < 0) {
                    return -1;
                }
            }
            return 0;
        }
    }
    return -1;  /* Internal programming error. */
}

NPY_FINLINE int
npy_transfer_loop(NPY_cast_info *info, const NPY_transfer_loop *loop,
                  NPY_transfer_copy_function *copy)
{
    if (loop->kind == NPY_TRANSFER_INDEXED_CONTIGUOUS &&
            loop->selection.indexed.scatter) {
        return npy_transfer_loop_impl(info, loop, 1, copy);
    }
    return npy_transfer_loop_impl(info, loop, 0, copy);
}

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_DTYPE_TRANSFER_H_  */
