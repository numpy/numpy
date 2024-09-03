#ifndef NUMPY_CORE_SRC_COMMON_UTILS_HPP
#define NUMPY_CORE_SRC_COMMON_UTILS_HPP

#include "npdef.hpp"

#if NP_HAS_CPP20
    #include <bit>
#endif

#include <type_traits>
#include <string.h>
#include <cstdint>
#include <cassert>

namespace np {

using std::uint32_t;
using std::uint64_t;

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

/// Bit-scan reverse for non-zeros.
/// Returns the index of the highest set bit. Equivalent to floor(log2(a))
template <typename T>
inline int BitScanReverse(uint32_t a)
{
#if NP_HAS_CPP20
    return std::countl_one(a);
#else
    if (a == 0) {
        // Due to use __builtin_clz which is undefined behavior
        return 0;
    }
    int r;
    #ifdef _MSC_VER
    unsigned long rl;
    (void)_BitScanReverse(&rl, (unsigned long)a);
    r = static_cast<int>(rl);
    #elif (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)) \
        &&  (defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64))
    __asm__("bsr %1, %0" : "=r" (r) : "r"(a));
    #elif defined(__GNUC__) || defined(__clang__)
    r = 31 - __builtin_clz(a); // performs on arm -> clz, ppc -> cntlzw
    #else
    r = 0;
    while (a >>= 1) {
        r++;
    }
    #endif
    return r;
#endif
}
/// Bit-scan reverse for non-zeros.
/// Returns the index of the highest set bit. Equivalent to floor(log2(a))
inline int BitScanReverse(uint64_t a)
{
#if NP_HAS_CPP20
    return std::countl_one(a);
#else
    if (a == 0) {
        // Due to use __builtin_clzll which is undefined behavior
        return 0;
    }
    #if defined(_M_AMD64) && defined(_MSC_VER)
    unsigned long rl;
    (void)_BitScanReverse64(&rl, a);
    return static_cast<int>(rl);
    #elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
    uint64_t r;
    __asm__("bsrq %1, %0" : "=r"(r) : "r"(a));
    return static_cast<int>(r);
    #elif defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(a);
    #else
    uint64_t a_hi = a >> 32;
    if (a_hi == 0) {
        return BitScanReverse(static_cast<uint32_t>(a));
    }
    return 32 + BitScanReverse(static_cast<uint32_t>(a_hi));
    #endif
#endif
}

} // namespace np
#endif // NUMPY_CORE_SRC_COMMON_UTILS_HPP

