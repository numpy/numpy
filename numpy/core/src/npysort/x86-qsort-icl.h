#include "numpy/npy_common.h"

#include "npy_cpu_dispatch.h"

#ifndef NPY_NO_EXPORT
#define NPY_NO_EXPORT NPY_VISIBILITY_HIDDEN
#endif

#ifndef NPY_DISABLE_OPTIMIZATION
#include "x86-qsort-icl.dispatch.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void x86_quicksort_half,
                         (void *start, npy_intp num))

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void x86_quicksort_short,
                         (void *start, npy_intp num))

NPY_CPU_DISPATCH_DECLARE(NPY_NO_EXPORT void x86_quicksort_ushort,
                         (void *start, npy_intp num))

#ifdef __cplusplus
}
#endif
