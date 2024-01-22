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
using SSize = Py_ssize_t;
/// @} cpp_core_types

/// @addtogroup cpp_core_constants
/// @{
/// Enumerated values represents the basic 24 data types.
enum class TypeID : char {
    /// Represents `Bool`.
    kBool = 0,
    /// Represents `Byte`.
    kByte,
    /// Represents `UByte`.
    kUByte,
    /// Represents `Short`.
    kShort,
    /// Represents `UShort`.
    kUShort,
    /// Represents `Int`.
    kInt,
    /// Represents `UInt`.
    kUInt,
    /// Represents `Long`.
    kLong,
    /// Represents `ULong`.
    kULong,
    /// Represents `LongLong`.
    kLongLong,
    /// Represents `ULongLong`.
    kULongLong,
    /// Represents `Float`.
    kFloat,
    /// Represents `Double`.
    kDouble,
    /// Represents `LongDouble`.
    kLongDouble,
    /// Represents `CFloat`.
    kCFloat,
    /// Represents `CDouble`.
    kCDouble,
    /// Represents `CLongDouble`.
    kCLongDouble,
    /// Represents `Object`.
    kObject = 17,
    /// Represents `String`.
    kString,
    /// Represents `Unicode`.
    kUnicode,
    /// Represents `Void`.
    kVoid,
    /// Represents `DateTime`.
    kDateTime,
    /// Represents `TimeDelta`.
    kTimeDelta,
    /// Represents `Half`.
    kHalf
};

/// @name Types IDs alias
/// For lazy access to enumeration values of `TypeID`.
/// @{
constexpr auto kBool = TypeID::kBool;
constexpr auto kByte = TypeID::kByte;
constexpr auto kUByte = TypeID::kUByte;
constexpr auto kShort = TypeID::kShort;
constexpr auto kUShort = TypeID::kUShort;
constexpr auto kInt = TypeID::kInt;
constexpr auto kUInt = TypeID::kUInt;
constexpr auto kLong = TypeID::kLong;
constexpr auto kULong = TypeID::kULong;
constexpr auto kLongLong = TypeID::kLongLong;
constexpr auto kULongLong = TypeID::kULongLong;
constexpr auto kHalf = TypeID::kHalf;
constexpr auto kFloat = TypeID::kFloat;
constexpr auto kDouble = TypeID::kDouble;
constexpr auto kLongDouble = TypeID::kLongDouble;
constexpr auto kCFloat = TypeID::kCFloat;
constexpr auto kCDouble = TypeID::kCDouble;
constexpr auto kCLongDouble = TypeID::kCLongDouble;
constexpr auto kObject = TypeID::kObject;
constexpr auto kString = TypeID::kString;
constexpr auto kUnicode = TypeID::kUnicode;
constexpr auto kVoid = TypeID::kVoid;
constexpr auto kDateTime = TypeID::kDateTime;
constexpr auto kTimeDelta = TypeID::kTimeDelta;
/// @}

/// Number of supported basic data types (24).
constexpr int kNTypes = static_cast<int>(kHalf) + 1;
/// Whether long double is forced or has the same precision as double.
#if defined(WIN32) || defined(_WIN32)
constexpr bool kLongDoubleIsDouble = true;
#else
constexpr bool kLongDoubleIsDouble = sizeof(long double) == sizeof(double);
#endif

template<typename T, TypeID>
class Holder;
class Half;
class DateTime;
class TimeDelta;
class Object;

namespace details {
template<typename T>
struct Traits;
template<>
struct Traits<bool> { static constexpr TypeID kID = kBool; };
template<>
struct Traits<Half> { static constexpr TypeID kID = kHalf; };
template<>
struct Traits<float> { static constexpr TypeID kID = kFloat; };
template<>
struct Traits<double> { static constexpr TypeID kID = kDouble; };
template<>
struct Traits<long double> { static constexpr TypeID kID = kLongDouble;};
template<>
struct Traits<complex<float>> { static constexpr TypeID kID = kCFloat; };
template<>
struct Traits<complex<double>> { static constexpr TypeID kID = kCDouble; };
template<>
struct Traits<complex<long double>> { static constexpr TypeID kID = kCLongDouble; };
template<>
struct Traits<Object> { static constexpr TypeID kID = kObject; };
template<>
struct Traits<DateTime> { static constexpr TypeID kID = kDateTime; };
template<>
struct Traits<TimeDelta> { static constexpr TypeID kID = kTimeDelta; };
template<typename T, TypeID ID>
struct Traits<Holder<T, ID>> { static constexpr TypeID kID = ID; };
} // namespace details

/// Returns TypeID based on @tparam T.
template<typename T, typename TBase=std::remove_cv_t<std::remove_reference_t<T>>>
constexpr TypeID kTypeID = details::Traits<TBase>::kID;

/// Checks whether @tparam `T` is an boolean type.
template<typename T>
constexpr bool kIsBool = kTypeID<T> == kBool;

/// Checks whether `T` is an unsigned integer type.
template<typename T, TypeID ID = kTypeID<T>>
constexpr bool kIsUnsigned = ID == kUByte || ID == kUShort || ID == kUInt ||
                             ID == kULong || ID == kULongLong;

/// Checks whether `T` is an signed integer type.
template<typename T, TypeID ID = kTypeID<T>>
constexpr bool kIsSigned = ID == kByte || ID == kShort || ID == kInt ||
                           ID == kLong || ID == kLongLong;

/// Checks whether `T` is an integer type.
template<typename T, TypeID ID = kTypeID<T>>
constexpr bool kIsIntegral = kIsSigned<T> || kIsUnsigned<T>;

/// Checks whether `T` is an floating-point type.
template<typename T, TypeID ID = kTypeID<T>>
constexpr bool kIsFloat = ID == kHalf || ID == kFloat ||
                          ID == kDouble || ID == kLongDouble;

/// Checks whether `T` is an complex type.
template<typename T, TypeID ID = kTypeID<T>>
constexpr bool kIsComplex = ID == kCFloat || ID == kCDouble ||
                            ID == kCLongDouble;

/// Checks whether `T` is an time type (`DateTime`, `Timedelta`).
template<typename T, TypeID ID = kTypeID<T>>
constexpr bool kIsTime = ID == kDateTime || ID == kTimeDelta;
/// @} cpp_core_constants

/// @addtogroup cpp_core_containers
/// @{
/// Class wraps builtin data types.
/// To distinct them, and gives the room for dealing
/// with platform compatiblty.
template <typename T, TypeID>
class Holder {
  public:
    /// initlize nothing.
    Holder() = default;
    /// Construct from `T`
    constexpr Holder(const T &val) : val_(val)
    {}
    /// cast
    constexpr operator T () const
    { return val_; }

  private:
    T val_;
};
template<template<class> class C, typename T, TypeID ID>
class Holder<C<T>, ID> : public C<T> {
  public:
    using C<T>::C;
};
/// @} cpp_core_containers

/// @addtogroup cpp_core_types
/// @{
/// Signed integer with at least 8-bit width.
using Byte = Holder<signed char, kByte>;
/// Unsigned integer with at least 8-bit width.
using UByte = Holder<unsigned char, kUByte>;
/// Signed integer with at least 16-bit width.
using Short = Holder<short, kShort>;
/// Unsigned integer with at least 16-bit width.
using UShort = Holder<unsigned short, kUShort>;
/// Signed integer with at least 16-bit width.
using Int = Holder<int, kInt>;
/// Unsigned integer with at least 16-bit width.
using UInt = Holder<unsigned int, kUInt>;
/// Signed integer with at least 32-bit width.
using Long = Holder<long, kLong>;
/// Unsigned integer with at least 32-bit width.
using ULong = Holder<unsigned long, kULong>;
/// Signed integer with at least 64-bit width.
using LongLong = Holder<long long, kLongLong>;
/// Unsigned integer with at least 64-bit width.
using ULongLong = Holder<unsigned long long, kULongLong>;
/// Boolean type, stored as `Byte`.
/// It may only be set to the values false and true.
using Bool = std::conditional_t<sizeof(bool) == sizeof(unsigned char),
    bool, Holder<unsigned char, kBool>>;
/// single-precision floating-point type, compatible with IEEE-754.
using Float = float;
/// double-precision floating-point type, compatible with IEEE-754.
using Double = double;
/**
 * Extended precision floating-point type, compatible with IEEE-754.
 * Provides at least the same precision of `Double`.
 * Note:
 *  The C implementation defines long double as double
 *  on MinGW to provide compatibility with MSVC to unify
 *  one behavior under Windows OS.
 */
using LongDouble = std::conditional_t<kLongDoubleIsDouble,
      Holder<double, kLongDouble>, long double>;
/// Complex type made up of two `Float` values.
using CFloat = complex<float>;
/// Complex type made up of two `Double` values.
using CDouble = complex<double>;
/// Complex type made up of two `LongDouble` values.
using CLongDouble = std::conditional_t<kLongDoubleIsDouble,
      Holder<complex<double>, kCLongDouble>, complex<long double>>;

using String = Holder<unsigned char, kString>;
using Unicode = Holder<uint32_t, kUnicode>;
/** An abstract type that implements general date/time
 * operations for types `DateTime` and `Timedelta`.
 *
 * This type is ensured to be 64-bits size.
 */
class AbstractTime {
 public:
    /// initlize nothing.
    AbstractTime() = default;
    /// Constract from int64_t
    constexpr AbstractTime(int64_t d) : bits_(d)
    {}
    /// cast to int64_t.
    constexpr operator int64_t() const
    {
        return bits_;
    }
    /// Initialize with not-a-time value.
    constexpr static AbstractTime NaT()
    {
        return AbstractTime(std::numeric_limits<int64_t>::min());
    }
    /// @name Comparison operators
    /// @{
    constexpr bool operator==(const AbstractTime &r) const
    {
        return bits_ == r.bits_ && IsFinite() && r.IsFinite();
    }
    constexpr bool operator!=(const AbstractTime &r) const
    {
        return bits_ != r.bits_ || IsNaT() || r.IsNaT();
    }
    constexpr bool operator<(const AbstractTime &r) const
    {
        return bits_ < r.bits_ && IsFinite() && r.IsFinite();
    }
    constexpr bool operator>(const AbstractTime &r) const
    {
        return r < *this;
    }
    constexpr bool operator<=(const AbstractTime &r) const
    {
        return bits_ <= r.bits_ && IsFinite() && r.IsFinite();
    }
    constexpr bool operator>=(const AbstractTime &r) const
    {
        return r <= *this;
    }
    /// @}

    /// @name Arithmetic operators
    /// @{
    constexpr AbstractTime operator+(const AbstractTime &r) const
    {
        if (IsNaT() || r.IsNaT()) {
            return NaT();
        }
        return AbstractTime(bits_ + r.bits_);
    }
    constexpr AbstractTime &operator+=(const AbstractTime &r)
    {
        *this = (*this) + r;
        return *this;
    }
    constexpr AbstractTime operator-(const AbstractTime &r) const
    {
        if (IsNaT() || r.IsNaT()) {
            return NaT();
        }
        return AbstractTime(bits_ - r.bits_);
    }
    constexpr AbstractTime &operator-=(const AbstractTime &r)
    {
        *this = (*this) - r;
        return *this;
    }
    constexpr AbstractTime operator*(const AbstractTime &r) const
    {
        if (IsNaT() || r.IsNaT()) {
            return NaT();
        }
        return AbstractTime(bits_ * r.bits_);
    }
    constexpr AbstractTime &operator*=(const AbstractTime &r)
    {
        *this = (*this) * r;
        return *this;
    }
    constexpr AbstractTime operator/(const AbstractTime &r) const
    {
        if (IsNaT() || r.IsNaT()) {
            return NaT();
        }
        return AbstractTime(bits_ / r.bits_);
    }
    constexpr AbstractTime &operator/=(const AbstractTime &r)
    {
        *this = (*this) / r;
        return *this;
    }
    /// @}

    /// @name Properties
    /// @{
    constexpr bool IsNaT() const
    {
        return bits_ == static_cast<int64_t>(NaT());
    }
    constexpr bool IsFinite() const
    {
        return bits_ != static_cast<int64_t>(NaT());
    }
    /// @}

  private:
    int64_t bits_;
};

/// Holds dates or datetimes with a precision based on selectable date or time units.
class DateTime : public AbstractTime
{
  public:
    using AbstractTime::AbstractTime;
};
/// Holds lengths of times in integers of selectable date or time units.
class TimeDelta : public AbstractTime
{
  public:
    using AbstractTime::AbstractTime;
};



namespace details {
template<int size, bool unsig>
struct IntBySize;

template<bool unsig>
struct IntBySize<sizeof(uint8_t), unsig> {
    using Type = typename std::conditional_t<
        unsig, uint8_t, int8_t>;
};
template<bool unsig>
struct IntBySize<sizeof(uint16_t), unsig> {
    using Type = typename std::conditional_t<
        unsig, uint16_t, int16_t>;
};
template<bool unsig>
struct IntBySize<sizeof(uint32_t), unsig> {
    using Type = typename std::conditional_t<
        unsig, uint32_t, int32_t>;
};
template<bool unsig>
struct IntBySize<sizeof(uint64_t), unsig> {
    using Type = typename std::conditional_t<
        unsig, uint64_t, int64_t>;
};
} // namespace details

template <typename T>
using MakeFixedIntegral = std::conditional_t<
    kIsIntegral<T>, typename details::IntBySize<sizeof(T), kIsUnsigned<T>>::Type, T
>;

/// @} cpp_core_types

} // namespace np

#endif // NUMPY_CORE_SRC_COMMON_NPSTD_HPP

