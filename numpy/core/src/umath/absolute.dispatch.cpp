/*@targets
 * $maxopt baseline
 * sse2 sse41 xop avx2 avx512_skx
 * asimd
 */

#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"

#include "fast_loop_macros.h"
#include "loops_utils.h"

#include "hwy/aligned_allocator.h"
#include <hwy/highway.h>

namespace numpy {
namespace HWY_NAMESPACE {  // required: unique per target

// Can skip hn:: prefixes if already inside hwy::HWY_NAMESPACE.
namespace hn = hwy::HWY_NAMESPACE;

// Alternative to per-function HWY_ATTR: see HWY_BEFORE_NAMESPACE
template <typename T>
HWY_ATTR void
SuperAbsolute(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    const T *HWY_RESTRICT input_array = (const T *)args[0];
    T *HWY_RESTRICT output_array = (T *)args[1];
    const size_t size = dimensions[0];
    const hn::ScalableTag<T> d;

    if (is_mem_overlap(input_array, steps[0], output_array, steps[1], size)) {
        for (size_t i = 0; i < size; i++) {
            const auto in = hn::LoadN(d, input_array + i, 1);
            auto x = hn::Abs(in);
            hn::StoreN(x, d, output_array + i, 1);
        }
    }
    else if (IS_UNARY_CONT(input_array, output_array)) {
        size_t full = size & -hn::Lanes(d);
        size_t remainder = size - full;
        for (size_t i = 0; i < full; i += hn::Lanes(d)) {
            const auto in = hn::LoadU(d, input_array + i);
            auto x = hn::Abs(in);
            hn::StoreU(x, d, output_array + i);
        }
        if (remainder) {
            const auto in = hn::LoadN(d, input_array + full, remainder);
            auto x = hn::Abs(in);
            hn::StoreN(x, d, output_array + full, remainder);
        }
    }
    else {
        using TI = hwy::MakeSigned<T>;
        const hn::Rebind<TI, hn::ScalableTag<T>> di;

        const int lsize = sizeof(input_array[0]);
        const npy_intp ssrc = steps[0] / lsize;
        const npy_intp sdst = steps[1] / lsize;
        auto load_index = hwy::AllocateAligned<TI>(hn::Lanes(d));
        for (size_t i = 0; i < hn::Lanes(d); ++i) {
            load_index[i] = i * ssrc;
        }
        auto store_index = hwy::AllocateAligned<TI>(hn::Lanes(d));
        for (size_t i = 0; i < hn::Lanes(d); ++i) {
            store_index[i] = i * sdst;
        }

        size_t full = size & -hn::Lanes(d);
        size_t remainder = size - full;
        for (size_t i = 0; i < full; i += hn::Lanes(d)) {
            const auto in = hn::GatherIndex(d, input_array + i * ssrc,
                                            Load(di, load_index.get()));
            auto x = hn::Abs(in);
            hn::ScatterIndex(x, d, output_array + i * sdst,
                             Load(di, store_index.get()));
        }
        if (remainder) {
            const auto in =
                    hn::GatherIndexN(d, input_array + full * ssrc,
                                     Load(di, load_index.get()), remainder);
            auto x = hn::Abs(in);
            hn::ScatterIndexN(x, d, output_array + full * sdst,
                              Load(di, store_index.get()), remainder);
        }
    }
}

HWY_ATTR void
INT_SuperAbsolute(char **args, npy_intp const *dimensions,
                  npy_intp const *steps)
{
    SuperAbsolute<npy_int>(args, dimensions, steps);
}

HWY_ATTR void
DOUBLE_SuperAbsolute(char **args, npy_intp const *dimensions,
                     npy_intp const *steps)
{
    SuperAbsolute<npy_double>(args, dimensions, steps);
}

HWY_ATTR void
FLOAT_SuperAbsolute(char **args, npy_intp const *dimensions,
                    npy_intp const *steps)
{
    SuperAbsolute<npy_float>(args, dimensions, steps);
}

}  // namespace HWY_NAMESPACE
}  // namespace numpy

namespace numpy {

extern "C" {

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(INT_absolute)(char **args, npy_intp const *dimensions,
                                     npy_intp const *steps,
                                     void *NPY_UNUSED(func))
{
    HWY_STATIC_DISPATCH(INT_SuperAbsolute)(args, dimensions, steps);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(DOUBLE_absolute)(char **args,
                                        npy_intp const *dimensions,
                                        npy_intp const *steps,
                                        void *NPY_UNUSED(func))
{
    HWY_STATIC_DISPATCH(DOUBLE_SuperAbsolute)(args, dimensions, steps);
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(FLOAT_absolute)(char **args, npy_intp const *dimensions,
                                       npy_intp const *steps,
                                       void *NPY_UNUSED(func))
{
    HWY_STATIC_DISPATCH(FLOAT_SuperAbsolute)(args, dimensions, steps);
}

}  // extern "C"
}  // namespace numpy
