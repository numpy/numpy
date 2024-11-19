#ifndef _NPY_CORE_SRC_UMATH_HIGHWAY_MASKED_FALLBACK_H_
#define _NPY_CORE_SRC_UMATH_HIGHWAY_MASKED_FALLBACK_H_

#include "simd/simd.h"
#include <hwy/highway.h>
#include <numpy/npy_common.h>

#if NPY_SIMD
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// Sets `out[idx]` to `Func(d, orig_x[idx])` if `mask[idx]` is true else `transformed_x[idx]`
// Useful for special casing math routines
template <class D, class Func, typename T = TFromD<D>>
HWY_ATTR Vec<D>
MaskedScalarFallbackUnary(D d, Vec<D> orig_x, Vec<D> transformed_x, Mask<D> mask, const Func &fallback)
{
    if (!AllTrue(d, mask)) {
        npy_uint64 maski;
        StoreMaskBits(d, mask, (uint8_t *)&maski);
        HWY_ALIGN TFromD<D> ip_fback[Lanes(d)];
        Store(orig_x, d, ip_fback);

        // process elements using libc for large elements
        for (unsigned i = 0; i < Lanes(d); ++i) {
            if ((maski >> i) & 1) {
                continue;
            }
            ip_fback[i] = fallback(ip_fback[i]);
        }

        // Merge updates back into transformed_x
        Vec<D> specialcase_x = Load(d, ip_fback);
        transformed_x = IfThenElse(mask, transformed_x, specialcase_x);
    }

    return transformed_x;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif // NPY_SIMD
#endif
