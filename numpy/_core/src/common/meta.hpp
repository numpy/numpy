#ifndef NUMPY_CORE_SRC_COMMON_META_HPP
#define NUMPY_CORE_SRC_COMMON_META_HPP

#include "npstd.hpp"

namespace np { namespace meta {
/// @addtogroup cpp_core_meta
/// @{

namespace details {
template<int size, bool unsig>
struct IntBySize;

template<bool unsig>
struct IntBySize<sizeof(uint8_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint8_t, int8_t>::type;
};
template<bool unsig>
struct IntBySize<sizeof(uint16_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint16_t, int16_t>::type;
};
template<bool unsig>
struct IntBySize<sizeof(uint32_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint32_t, int32_t>::type;
};
template<bool unsig>
struct IntBySize<sizeof(uint64_t), unsig> {
    using Type = typename std::conditional<
        unsig, uint64_t, int64_t>::type;
};
} // namespace details

/// Provides safe conversion of any integer type synonyms
/// to a fixed-width integer type.
template<typename T>
struct FixedWidth {
    using TF_ = typename details::IntBySize<
        sizeof(T), std::is_unsigned<T>::value
    >::Type;

    using Type = typename std::conditional<
        std::is_integral<T>::value, TF_, T
    >::type;
};

/// @} cpp_core_meta

}} // namespace np::meta

#endif // NUMPY_CORE_SRC_COMMON_META_HPP

