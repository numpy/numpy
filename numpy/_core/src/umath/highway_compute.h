#ifndef _NPY_CORE_SRC_UMATH_HIGHWAY_COMPUTE_H_
#define _NPY_CORE_SRC_UMATH_HIGHWAY_COMPUTE_H_

#include "simd/simd.h"
#include <hwy/highway.h>
#include <numpy/npy_common.h>

#if NPY_SIMD
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Sets `out[idx]` to `Func(d, in[idx])`
// Useful for math routines.
template <class D, class Func, typename T = TFromD<D>>
HWY_ATTR void
ComputeUnary(D d, const T *HWY_RESTRICT in, const npy_intp in_stride,
           T *HWY_RESTRICT out, const npy_intp out_stride, size_t count,
           const Func &func)
{
    using VI = Vec<RebindToSigned<D>>;
    const RebindToSigned<D> di;

    const size_t lanes = Lanes(d);
    const VI src_index = Mul(Iota(di, 0), Set(di, in_stride));
    const VI dst_index = Mul(Iota(di, 0), Set(di, out_stride));

    Vec<D> v;

    for (; count >= lanes; in += in_stride*lanes, out += out_stride*lanes, count -= lanes) {
        if (in_stride == 1) {
            v = LoadU(d, in);
        }
        else {
            v = GatherIndex(d, in, src_index);
        }
        if (out_stride == 1) {
            StoreU(func(v), d, out);
        }
        else {
            ScatterIndex(func(v), d, out, dst_index);
        }
        npyv_cleanup();
    }

    // `count` was a multiple of the vector length `N`: already done.
    if (HWY_UNLIKELY(count == 0))
        return;

    HWY_DASSERT(count > 0 && count < lanes);
    if (in_stride == 1) {
        v = LoadN(d, in, count);
    }
    else {
        v = GatherIndexN(d, in, src_index, count);
    }
    if (out_stride == 1) {
        StoreN(func(v), d, out, count);
    }
    else {
        ScatterIndexN(func(v), d, out, dst_index, count);
    }
    npyv_cleanup();
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif // NPY_SIMD
#endif
