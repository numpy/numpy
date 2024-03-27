#ifndef NUMPY_CORE_SRC_COMMON_MATH_HPP
#define NUMPY_CORE_SRC_COMMON_MATH_HPP

#include "npstd.hpp"
#include "half.hpp"

namespace np {

template<typename T>
inline std::enable_if_t<kIsFloat<T>, T> Copysign(const T &a, const T &b)
{
    if constexpr (std::is_same_v<T, Half>) {
        return Half::FromBits((a.Bits()&0x7fffu) | (b.Bits()&0x8000u));
    }
    else {
        return std::copysign(a, b);
    }
}

} // namespace np


#endif // NUMPY_CORE_SRC_COMMON_MATH_HPP

