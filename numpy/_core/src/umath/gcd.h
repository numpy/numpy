#ifndef _NPY_CORE_SRC_UMATH_GCD_H_
#define _NPY_CORE_SRC_UMATH_GCD_H_

#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT int
gcd_get_reduction_loop(
        PyArrayMethod_Context *context,
        int aligned,
        int move_references,
        const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

#ifdef __cplusplus
}
#endif

#endif
