#ifndef NUMPY_CORE_SRC_COMMON_HALF_HPP
#define NUMPY_CORE_SRC_COMMON_HALF_HPP

#include "npstd.hpp"

#include "npy_cpu_dispatch.h" // NPY_HAVE_CPU_FEATURES
#include "half_private.hpp"

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
    #if defined(NPY_HAVE_FP16)
        __m128 mf = _mm_load_ss(&f);
        bits_ = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(mf, _MM_FROUND_TO_NEAREST_INT)));
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX_ASM)
        __vector float vf32 = vec_splats(f);
        __vector unsigned short vf16;
        __asm__ __volatile__ ("xvcvsphp %x0,%x1" : "=wa" (vf16) : "wa" (vf32));
        #ifdef __BIG_ENDIAN__
        bits_ = vec_extract(vf16, 1);
        #else
        bits_ = vec_extract(vf16, 0);
        #endif
    #else
        bits_ = half_private::FromFloatBits(BitCast<uint32_t>(f));
    #endif
    }

    /// Construct from double.
    /// If there are no hardware optimization available, rounding will always
    /// be set to ties to even.
    explicit Half(double f)
    {
    #if defined(NPY_HAVE_AVX512FP16)
        __m128d md = _mm_load_sd(&f);
        bits_ = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_castph_si128(_mm_cvtpd_ph(md))));
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX3_HALF_DOUBLE)
        __asm__ __volatile__ ("xscvdphp %x0,%x1" : "=wa" (bits_) : "wa" (f));
    #else
        bits_ = half_private::FromDoubleBits(BitCast<uint64_t>(f));
    #endif
    }

    /// Cast to float
    explicit operator float() const
    {
    #if defined(NPY_HAVE_FP16)
        float ret;
        _mm_store_ss(&ret, _mm_cvtph_ps(_mm_cvtsi32_si128(bits_)));
        return ret;
    #elif defined(NPY_HAVE_VSX3) && defined(vec_extract_fp_from_shorth)
        return vec_extract(vec_extract_fp_from_shorth(vec_splats(bits_)), 0);
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX_ASM)
        __vector float vf32;
        __asm__ __volatile__("xvcvhpsp %x0,%x1"
                             : "=wa"(vf32)
                             : "wa"(vec_splats(bits_)));
        return vec_extract(vf32, 0);
    #else
        return BitCast<float>(half_private::ToFloatBits(bits_));
    #endif
    }

    /// Cast to double
    explicit operator double() const
    {
    #if defined(NPY_HAVE_AVX512FP16)
        double ret;
        _mm_store_sd(&ret, _mm_cvtph_pd(_mm_castsi128_ph(_mm_cvtsi32_si128(bits_))));
        return ret;
    #elif defined(NPY_HAVE_VSX3) && defined(NPY_HAVE_VSX3_HALF_DOUBLE)
        double f64;
        __asm__ __volatile__("xscvhpdp %x0,%x1"
                             : "=wa"(f64)
                             : "wa"(bits_));
        return f64;
    #else
        return BitCast<double>(half_private::ToDoubleBits(bits_));
    #endif
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
