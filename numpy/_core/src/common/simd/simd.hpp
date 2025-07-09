#ifndef NUMPY__CORE_SRC_COMMON_SIMD_SIMD_HPP_
#define NUMPY__CORE_SRC_COMMON_SIMD_SIMD_HPP_

/**
 * This header provides a thin wrapper over Google's Highway SIMD library.
 *
 * The wrapper aims to simplify the SIMD interface of Google's Highway by
 * get ride of its class tags and use lane types directly which can be deduced
 * from the args in most cases.
 */
/**
 * Since `NPY_SIMD` is only limited to NumPy C universal intrinsics,
 * `NPY_HWY` is defined to indicate the SIMD availability for Google's Highway
 * C++ code.
 *
 * Highway SIMD is only available when optimization is enabled.
 * When NPY_DISABLE_OPTIMIZATION is defined, SIMD operations are disabled
 * and the code falls back to scalar implementations.
 */
#ifndef NPY_DISABLE_OPTIMIZATION
#include <hwy/highway.h>

/**
 * We avoid using Highway scalar operations for the following reasons:
 *
 * 1. NumPy already provides optimized kernels for scalar operations. Using these
 *    existing implementations is more consistent with NumPy's architecture and
 *    allows for compiler optimizations specific to standard library calls.
 *
 * 2. Not all Highway intrinsics are fully supported in scalar mode, which could
 *    lead to compilation errors or unexpected behavior for certain operations.
 *
 * 3. For NumPy's strict IEEE 754 floating-point compliance requirements, direct scalar
 *    implementations offer more predictable behavior than EMU128.
 *
 * Therefore, we only enable Highway SIMD when targeting actual SIMD instruction sets.
 */
#define NPY_HWY ((HWY_TARGET != HWY_SCALAR) && (HWY_TARGET != HWY_EMU128))

// Indicates if the SIMD operations are available for float16.
#define NPY_HWY_F16 (NPY_HWY && HWY_HAVE_FLOAT16)
// Note: Highway requires SIMD extentions with native float32 support, so we don't need
// to check for it.

// Indicates if the SIMD operations are available for float64.
#define NPY_HWY_F64 (NPY_HWY && HWY_HAVE_FLOAT64)

// Indicates if the SIMD floating operations are natively supports fma.
#define NPY_HWY_FMA (NPY_HWY && HWY_NATIVE_FMA)

#else
#define NPY_HWY 0
#define NPY_HWY_F16 0
#define NPY_HWY_F64 0
#define NPY_HWY_FMA 0
#endif

namespace np {

/// Represents the max SIMD width supported by the platform.
namespace simd {
#if NPY_HWY
/// The highway namespace alias.
/// We can not import all the symbols from the HWY_NAMESPACE because it will
/// conflict with the existing symbols in the numpy namespace.
namespace hn = hwy::HWY_NAMESPACE;
// internaly used by the template header
template <typename TLane>
using _Tag = hn::ScalableTag<TLane>;
#endif
#include "simd.inc.hpp"
}  // namespace simd

/// Represents the 128-bit SIMD width.
namespace simd128 {
#if NPY_HWY
namespace hn = hwy::HWY_NAMESPACE;
template <typename TLane>
using _Tag = hn::Full128<TLane>;
#endif
#include "simd.inc.hpp"
}  // namespace simd128

}  // namespace np

#endif  // NUMPY__CORE_SRC_COMMON_SIMD_SIMD_HPP_
