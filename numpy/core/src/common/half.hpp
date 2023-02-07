#ifndef NUMPY_CORE_SRC_COMMON_HALF_HPP
#define NUMPY_CORE_SRC_COMMON_HALF_HPP

#include "npstd.hpp"

// TODO(@seiko2plus):
// - covers half-precision operations that being supported by numpy/halffloat.h
// - support __fp16
// - optimize x86 half<->single via cpu_fp16
// - optimize ppc64 half<->single via cpu_vsx3

namespace np {

/// @addtogroup cpp_core_types
/// @{

/// Provides a type that implements 16-bit floating point (half-precision).
/// This type is ensured to be 16-bit size.
class Half final {
 public:
    /// @name Public Constructors
    /// @{

    /// Default constructor. initialize nothing.
    Half() = default;
    /// Copy.
    Half(const Half &r)
    {
        data_.u = r.data_.u;
    }

    /// @}

    /// Returns a new Half constracted from the IEEE 754 binary16.
    /// @param b the value of binary16.
    static Half FromBits(uint16_t b)
    {
        Half f;
        f.data_.u = b;
        return f;
    }
    /// Returns the IEEE 754 binary16 representation.
    uint16_t Bits() const
    {
        return data_.u;
    }

 private:
    union {
        uint16_t u;
/*
TODO(@seiko2plus): support __fp16
#ifdef NPY_HAVE_HW_FP16
        __fp16 f;
#endif
*/
    } data_;
};

/// @} cpp_core_types

} // namespace np
#endif // NUMPY_CORE_SRC_COMMON_HALF_HPP
