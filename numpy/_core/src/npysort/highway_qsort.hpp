#ifndef NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP

#include "common.hpp"

#include <cstring>
#include <type_traits>

namespace np::highway::qsort_simd {

// Highway's VQSort compares keys with float comparisons, so -0.0 and +0.0 are
// the same key. Several of its code paths (the Min/Max sorting networks and
// MaybePartitionTwoValue, which overwrites runs of equal keys with copies of
// one sample) therefore create or destroy signed zeros. Either order of equal
// keys is a valid sort, but np.sort must not change the multiset of values.
// SignedZeroGuard counts -0.0 before the sort and Restore() rewrites the sign
// of the (already contiguous) zeros afterwards.
template <typename T>
struct SignedZeroGuard {
    using U = std::conditional_t<sizeof(T) == 2, uint16_t,
              std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>;
    static constexpr U kSign = U(1) << (sizeof(T) * 8 - 1);

    npy_intp neg = 0, zeros = 0;

    static U Bits(const T &x)
    {
        U u;
        std::memcpy(&u, &x, sizeof(u));
        return u;
    }
    explicit SignedZeroGuard(const T *arr, npy_intp size)
    {
        for (npy_intp i = 0; i < size; ++i) {
            U u = Bits(arr[i]);
            zeros += (U)(u & ~kSign) == 0;
            neg += u == kSign;
        }
    }
    // Only needed if both signs were present; otherwise nothing can change.
    void Restore(T *arr, npy_intp size) const
    {
        if (neg == 0 || neg == zeros) {
            return;
        }
        npy_intp left = neg;
        for (npy_intp i = 0; i < size; ++i) {
            U u = Bits(arr[i]);
            if ((U)(u & ~kSign) == 0) {
                u = left > 0 ? kSign : U(0);
                left -= 1;
                std::memcpy(&arr[i], &u, sizeof(u));
            }
        }
    }
};

#include "highway_qsort.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size, bool reverse))

#include "highway_qsort_16bit.dispatch.h"
NPY_CPU_DISPATCH_DECLARE(template <typename T> void QSort, (T *arr, npy_intp size, bool reverse))

} // np::highway::qsort_simd

#endif // NUMPY_SRC_COMMON_NPYSORT_HWY_SIMD_QSORT_HPP
