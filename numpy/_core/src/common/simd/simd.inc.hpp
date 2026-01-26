#ifndef NPY_HWY
#error "This is not a standalone header. Include simd.hpp instead."
#define NPY_HWY 1  // Prevent editors from graying out the happy branch
#endif

// Using anonymous namespace instead of inline to ensure each translation unit
// gets its own copy of constants based on local compilation flags
namespace {

// NOTE: This file is included by simd.hpp multiple times with different namespaces
// so avoid including any headers here

/**
 * Determines whether the specified lane type is supported by the SIMD extension.
 * Always defined as false when SIMD is not enabled, so it can be used in SFINAE.
 *
 * @tparam TLane The lane type to check for support.
 */
template <typename TLane>
constexpr bool kSupportLane = NPY_HWY != 0;

#if NPY_HWY
// Define lane type support based on Highway capabilities
template <>
constexpr bool kSupportLane<hwy::float16_t> = HWY_HAVE_FLOAT16 != 0;
template <>
constexpr bool kSupportLane<double> = HWY_HAVE_FLOAT64 != 0;
template <>
constexpr bool kSupportLane<long double> =
        HWY_HAVE_FLOAT64 != 0 && sizeof(long double) == sizeof(double);

/// Maximum number of lanes supported by the SIMD extension for the specified lane type.
template <typename TLane>
constexpr size_t kMaxLanes = HWY_MAX_LANES_D(_Tag<TLane>);

/// Represents an N-lane vector based on the specified lane type.
/// @tparam TLane The scalar type for each vector lane
template <typename TLane>
using Vec = hn::Vec<_Tag<TLane>>;

/// Represents a mask vector with boolean values or as a bitmask.
/// @tparam TLane The scalar type the mask corresponds to
template <typename TLane>
using Mask = hn::Mask<_Tag<TLane>>;

/// Unaligned load of a vector from memory.
template <typename TLane>
HWY_API Vec<TLane>
LoadU(const TLane *ptr)
{
    return hn::LoadU(_Tag<TLane>(), ptr);
}

/// Unaligned store of a vector to memory.
template <typename TLane>
HWY_API void
StoreU(const Vec<TLane> &a, TLane *ptr)
{
    hn::StoreU(a, _Tag<TLane>(), ptr);
}

/// Returns the number of vector lanes based on the lane type.
template <typename TLane>
HWY_API HWY_LANES_CONSTEXPR size_t
Lanes(TLane tag = 0)
{
    return hn::Lanes(_Tag<TLane>());
}

/// Returns an uninitialized N-lane vector.
template <typename TLane>
HWY_API Vec<TLane>
Undefined(TLane tag = 0)
{
    return hn::Undefined(_Tag<TLane>());
}

/// Returns N-lane vector with all lanes equal to zero.
template <typename TLane>
HWY_API Vec<TLane>
Zero(TLane tag = 0)
{
    return hn::Zero(_Tag<TLane>());
}

/// Returns N-lane vector with all lanes equal to the given value of type `TLane`.
template <typename TLane>
HWY_API Vec<TLane>
Set(TLane val)
{
    return hn::Set(_Tag<TLane>(), val);
}

/// Converts a mask to a vector based on the specified lane type.
template <typename TLane, typename TMask>
HWY_API Vec<TLane>
VecFromMask(const TMask &m)
{
    return hn::VecFromMask(_Tag<TLane>(), m);
}

/// Convert (Reinterpret) an N-lane vector to a different type without modifying the
/// underlying data.
template <typename TLaneTo, typename TVec>
HWY_API Vec<TLaneTo>
BitCast(const TVec &v)
{
    return hn::BitCast(_Tag<TLaneTo>(), v);
}

// Import common Highway intrinsics
using hn::Abs;
using hn::Add;
using hn::And;
using hn::AndNot;
using hn::Div;
using hn::Eq;
using hn::Ge;
using hn::Gt;
using hn::Le;
using hn::Lt;
using hn::Max;
using hn::Min;
using hn::Mul;
using hn::Or;
using hn::Sqrt;
using hn::Sub;
using hn::Xor;

#endif  // NPY_HWY

}  // namespace
