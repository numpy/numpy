#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_MASKED_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_METHOD_MASKED_H_

#include "npy_cpu_dispatch.h"
#include "array_method.h"
#include "numpy/ndarraytypes.h"
#include "array_method_masked.dispatch.h"

/*
 * Support for masked inner-strided loops.  Masked inner-strided loops are
 * only used in the ufunc machinery.  So this special cases them.
 */
typedef struct {
    NpyAuxData base;
    PyArrayMethod_StridedLoop *unmasked_stridedloop;
    NpyAuxData *unmasked_auxdata;
    int nargs;
    char *buf;
    char *dataptrs[1];
} _masked_stridedloop_data;

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT void
_masked_stridedloop_data_free(NpyAuxData *auxdata);

NPY_NO_EXPORT int
generic_masked_strided_loop_helper(PyArrayMethod_Context* context,
    char **dataptrs, const npy_intp *strides, char *mask, npy_intp N, int nargs,
    PyArrayMethod_StridedLoop* strided_loop, NpyAuxData* strided_loop_auxdata);

NPY_NO_EXPORT int
generic_masked_strided_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions,
        const npy_intp *strides, NpyAuxData *_auxdata);

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT int npy_get_masked_strided_loop, (
        PyArrayMethod_Context *context,
        int aligned, npy_intp *fixed_strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags))

#ifdef __cplusplus
}
#endif
#endif
