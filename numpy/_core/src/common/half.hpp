#ifndef NUMPY_CORE_SRC_COMMON_HALF_HPP
#define NUMPY_CORE_SRC_COMMON_HALF_HPP

#include "npstd.hpp"

#include "npy_cpu_dispatch.h" // NPY_HAVE_CPU_FEATURES
#include "numpy/halffloat.h"

// TODO(@seiko2plus):
// - covers half-precision operations that being supported by numpy/halffloat.h
// - add support for arithmetic operations
// - enables __fp16 causes massive FP exceptions on aarch64,
//   needs a deep investigation

namespace np {

/// @addtogroup cpp_core_types
/// @{

/// Provides a type that implements 16-bit floating point (half-precision).
/// This type is ensured to be 16-bit size.
#if 1 // ndef __ARM_FP16_FORMAT_IEEE
class Half final {
  public:
    /// Whether `Half` has a full native HW support.
    static constexpr bool kNative = false;
    /// Whether `Half` has a native HW support for single/double conversion.
    template<typename T>
    static constexpr bool kNativeConversion = (
        (
            std::is_same_v<T, float> &&
        #if defined(NPY_HAVE_FP16) || defined(NPY_HAVE_VSX3)
            true
        #else
            false
        #endif
        ) || (
            std::is_same_v<T, double> &&
        #if defined(NPY_HAVE_AVX512FP16) || (defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX3_HALF_DOUBLE))
            true
        #else
            false
        #endif
        )
    );

    /// Default constructor. initialize nothing.
    Half() = default;

    /// Construct from float
    /// If there are no hardware optimization available, rounding will always
    /// be set to ties to even.
    explicit Half(float f)
    {
        bits_ = npy_float_to_half(f);
    }

    /// Construct from double.
    /// If there are no hardware optimization available, rounding will always
    /// be set to ties to even.
    explicit Half(double f)
    {
        bits_ = npy_double_to_half(f);
    }

    /// Cast to float
    explicit operator float() const
    {
        return npy_half_to_float(bits_);
    }

    /// Cast to double
    explicit operator double() const
    {
        return npy_half_to_double(bits_);
    }

    /// Returns a new Half constructed from the IEEE 754 binary16.
    static constexpr Half FromBits(uint16_t bits)
    {
        Half h{};
        h.bits_ = bits;
        return h;
    }
    /// Returns the IEEE 754 binary16 representation.
    constexpr uint16_t Bits() const
    {
        return bits_;
    }

    /// @name Comparison operators (ordered)
    /// @{
    constexpr bool operator==(Half r) const
    {
        return !(IsNaN() || r.IsNaN()) && Equal(r);
    }
    constexpr bool operator<(Half r) const
    {
        return !(IsNaN() || r.IsNaN()) && Less(r);
    }
    constexpr bool operator<=(Half r) const
    {
        return !(IsNaN() || r.IsNaN()) && LessEqual(r);
    }
    constexpr bool operator>(Half r) const
    {
        return r < *this;
    }
    constexpr bool operator>=(Half r) const
    {
        return r <= *this;
    }
    /// @}

    /// @name Comparison operators (unordered)
    /// @{
    constexpr bool operator!=(Half r) const
    {
        return !(*this == r);
    }
    /// @} Comparison operators

    /// @name Comparison with no guarantee of NaN behavior
    /// @{
    constexpr bool Less(Half r) const
    {
        uint_fast16_t a = static_cast<uint_fast16_t>(bits_),
                      b = static_cast<uint_fast16_t>(r.bits_);
        bool sign_a = (a & 0x8000u) == 0x8000u;
        bool sign_b = (b & 0x8000u) == 0x8000u;
        // if both `a` and `b` have same sign
        //   Test if `a` > `b` when `a` has the sign
        //        or `a` < `b` when is not.
        //   And make sure they are not equal to each other
        //       in case of both are equal to +-0
        // else
        //   Test if  `a` has the sign.
        //        and `a` != -0.0 and `b` != 0.0
        return (sign_a == sign_b) ? (sign_a ^ (a < b)) && (a != b)
                                  : sign_a && ((a | b) != 0x8000u);
    }
    constexpr bool LessEqual(Half r) const
    {
        uint_fast16_t a = static_cast<uint_fast16_t>(bits_),
                      b = static_cast<uint_fast16_t>(r.bits_);
        bool sign_a = (a & 0x8000u) == 0x8000u;
        bool sign_b = (b & 0x8000u) == 0x8000u;
        // if both `a` and `b` have same sign
        //   Test if `a` > `b` when `a` has the sign
        //        or `a` < `b` when is not.
        //        or a == b (needed even if we used <= above instead
        //                   since testing +-0 still required)
        // else
        //   Test if `a` has the sign
        //        or `a` and `b` equal to +-0.0
        return (sign_a == sign_b) ? (sign_a ^ (a < b)) || (a == b)
                                  : sign_a || ((a | b) == 0x8000u);
    }
    constexpr bool Equal(Half r) const
    {
        // fast16 cast is not worth it, since unpack op should involved.
        uint16_t a = bits_, b = r.bits_;
        return a == b || ((a | b) == 0x8000u);
    }
    /// @} Comparison

    /// @name Properties
    // @{
    constexpr bool IsNaN() const
    {
        return ((bits_ & 0x7c00u) == 0x7c00u) &&
               ((bits_ & 0x03ffu) != 0);
    }
    /// @} Properties

  private:
    uint16_t bits_;
};
#else // __ARM_FP16_FORMAT_IEEE
class Half final {
  public:
    static constexpr bool kNative = true;
    template<typename T>
    static constexpr bool kNativeConversion = (
        std::is_same_v<T, float> || std::is_same_v<T, double>
    );
    Half() = default;
    constexpr Half(__fp16 h) : half_(h)
    {}
    constexpr operator __fp16() const
    { return half_; }
    static Half FromBits(uint16_t bits)
    {
        Half h;
        h.half_ = BitCast<__fp16>(bits);
        return h;
    }
    uint16_t Bits() const
    { return BitCast<uint16_t>(half_); }
    constexpr bool Less(Half r) const
    { return half_ < r.half_; }
    constexpr bool LessEqual(Half r) const
    { return half_ <= r.half_; }
    constexpr bool Equal(Half r) const
    { return half_ == r.half_; }
    constexpr bool IsNaN() const
    { return half_ != half_; }

  private:
    __fp16 half_;
};
#endif // __ARM_FP16_FORMAT_IEEE

/// @} cpp_core_types

} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_HALF_HPP
