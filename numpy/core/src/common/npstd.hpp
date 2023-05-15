#ifndef NUMPY_CORE_SRC_COMMON_NPSTD_HPP
#define NUMPY_CORE_SRC_COMMON_NPSTD_HPP

#include <cstddef>
#include <cstring>
#include <cctype>
#include <cstdint>

#include <string>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <type_traits>

#include <numpy/npy_common.h>

#include "npy_config.h"

namespace np {
/// @addtogroup cpp_core_types
/// @{
using std::uint8_t;
using std::int8_t;
using std::uint16_t;
using std::int16_t;
using std::uint32_t;
using std::int32_t;
using std::uint64_t;
using std::int64_t;
using std::uintptr_t;
using std::intptr_t;
using std::complex;
using std::uint_fast16_t;
using std::uint_fast32_t;

/** Guard for long double.
 *
 * The C implementation defines long double as double
 * on MinGW to provide compatibility with MSVC to unify
 * one behavior under Windows OS, which makes npy_longdouble
 * not fit to be used with template specialization or overloading.
 *
 * This type will be set to `void` when `npy_longdouble` is not defined
 * as `long double`.
 */
using LongDouble = typename std::conditional<
    !std::is_same<npy_longdouble, long double>::value,
     void, npy_longdouble
>::type;
/// @} cpp_core_types

} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_NPSTD_HPP

