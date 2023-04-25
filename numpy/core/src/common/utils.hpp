#ifndef NUMPY_CORE_SRC_COMMON_UTILS_HPP
#define NUMPY_CORE_SRC_COMMON_UTILS_HPP

#include "npdef.hpp"

#if NP_HAS_CPP20
    #include <bit>
#endif

#include <type_traits>
#include <string.h>

namespace np {

/** Create a value of type `To` from the bits of `from`.
 *
 * similar to `std::bit_cast` but compatible with C++17,
 * should perform similar to `*reinterpret_cast<To*>(&from)`
 * or through punning without expecting any undefined behaviors.
 */
template<typename To, typename From>
#if NP_HAS_BUILTIN(__builtin_bit_cast) || NP_HAS_CPP20
[[nodiscard]] constexpr
#else
inline
#endif
To BitCast(const From &from) noexcept
{
    static_assert(
        sizeof(To) == sizeof(From),
        "both data types must have the same size");

    static_assert(
        std::is_trivially_copyable_v<To> &&
        std::is_trivially_copyable_v<From>,
        "both data types must be trivially copyable");

#if NP_HAS_CPP20
    return std::bit_cast<To>(from);
#elif NP_HAS_BUILTIN(__builtin_bit_cast)
    return __builtin_bit_cast(To, from);
#else
    To to;
    memcpy(&to, &from, sizeof(from));
    return to;
#endif
}

} // namespace np
#endif // NUMPY_CORE_SRC_COMMON_UTILS_HPP

