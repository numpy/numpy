#ifndef _NPY_PRIVATE__COMPILED_BASE_PACK_INNER_H_
#define _NPY_PRIVATE__COMPILED_BASE_PACK_INNER_H_
#include <numpy/ndarraytypes.h>
#include "npy_cpu_dispatch.h"

#ifndef NPY_DISABLE_OPTIMIZATION
    #include "compiled_base.dispatch.h"
#endif

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void simd_compiled_base_pack_inner,
(const char *inptr, npy_intp element_size, npy_intp n_in, npy_intp in_stride, char *outptr, npy_intp n_out, npy_intp out_stride, char order))

#endif
