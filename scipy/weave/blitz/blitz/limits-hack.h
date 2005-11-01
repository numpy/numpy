/*
 * Copyright (c) 1997
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

/* NOTE: This is not portable code.  Parts of numeric_limits<> are
 * inherently machine-dependent, and this file is written for the MIPS
 * architecture and the SGI MIPSpro C++ compiler.  Parts of it (in
 * particular, some of the characteristics of floating-point types)
 * are almost certainly incorrect for any other platform.
 */

#include <blitz/wrap-climits.h> 
#include <float.h>

BZ_NAMESPACE(std)

enum float_round_style {
  round_indeterminate       = -1,
  round_toward_zero         =  0,
  round_to_nearest          =  1,
  round_toward_infinity     =  2,
  round_toward_neg_infinity =  3
};

enum float_denorm_style {
  denorm_indeterminate = -1,
  denorm_absent        =  0,
  denorm_present       =  1
};

// Base class for all specializations of numeric_limits.

template <typename __number>
class _Numeric_limits_base {
public:
  static const bool is_specialized = false;

  static __number min()  { return __number(); }
  static __number max()  { return __number(); }

  static const int digits   = 0;
  static const int digits10 = 0;

  static const bool is_signed  = false;
  static const bool is_integer = false;
  static const bool is_exact   = false;

  static const int radix = 0;

  static __number epsilon()      { return __number(); }
  static __number round_error()  { return __number(); }

  static const int min_exponent   = 0;
  static const int min_exponent10 = 0;
  static const int max_exponent   = 0;
  static const int max_exponent10 = 0;

  static const bool has_infinity      = false;
  static const bool has_quiet_NaN     = false;
  static const bool has_signaling_NaN = false;
  static const float_denorm_style has_denorm = denorm_absent;
  static const bool has_denorm_loss   = false;

  static __number infinity()       { return __number(); }
  static __number quiet_NaN()      { return __number(); }
  static __number signaling_NaN()  { return __number(); }
  static __number denorm_min()     { return __number(); }

  static const bool is_iec559  = false;
  static const bool is_bounded = false;
  static const bool is_modulo  = false;

  static const bool traps           = false;
  static const bool tinyness_before = false;
  static const float_round_style round_style = round_toward_zero;
};

#define __declare_numeric_base_member(__type, __mem) \
template <typename __number> \
  const __type _Numeric_limits_base<__number>:: __mem

__declare_numeric_base_member(bool, is_specialized);
__declare_numeric_base_member(int, digits);
__declare_numeric_base_member(int, digits10);
__declare_numeric_base_member(bool, is_signed);
__declare_numeric_base_member(bool, is_integer);
__declare_numeric_base_member(bool, is_exact);
__declare_numeric_base_member(int, radix);
__declare_numeric_base_member(int, min_exponent);
__declare_numeric_base_member(int, max_exponent);
__declare_numeric_base_member(int, min_exponent10);
__declare_numeric_base_member(int, max_exponent10);
__declare_numeric_base_member(bool, has_infinity);
__declare_numeric_base_member(bool, has_quiet_NaN);
__declare_numeric_base_member(bool, has_signaling_NaN);
__declare_numeric_base_member(float_denorm_style, has_denorm);
__declare_numeric_base_member(bool, has_denorm_loss);
__declare_numeric_base_member(bool, is_iec559);
__declare_numeric_base_member(bool, is_bounded);
__declare_numeric_base_member(bool, is_modulo);
__declare_numeric_base_member(bool, traps);
__declare_numeric_base_member(bool, tinyness_before);
__declare_numeric_base_member(float_round_style, round_style);

#undef __declare_numeric_base_member

// Base class for integers.

template <typename _Int,
          _Int __imin,
          _Int __imax,
          int __idigits = -1>
class _Integer_limits : public _Numeric_limits_base<_Int> {
public:
  static const bool is_specialized = true;

  static _Int min()  { return __imin; }
  static _Int max()  { return __imax; }

  static const int digits = 
    (__idigits < 0) ? sizeof(_Int) * CHAR_BIT - (__imin == 0 ? 0 : 1) 
                    : __idigits;
  static const int digits10 = (digits * 301) / 1000; 
                                // log 2 = 0.301029995664...

  static const bool is_signed = __imin != 0;
  static const bool is_integer = true;
  static const bool is_exact = true;
  static const int radix = 2;


  static const bool is_bounded = true;
  static const bool is_modulo = true;
};

#define __declare_integer_limits_member(__type, __mem) \
template <typename _Int, _Int __imin, _Int __imax, int __idigits> \
  const __type _Integer_limits<_Int, __imin, __imax, __idigits>:: __mem

__declare_integer_limits_member(bool, is_specialized);
__declare_integer_limits_member(int, digits);
__declare_integer_limits_member(int, digits10);
__declare_integer_limits_member(bool, is_signed);
__declare_integer_limits_member(bool, is_integer);
__declare_integer_limits_member(bool, is_exact);
__declare_integer_limits_member(int, radix);
__declare_integer_limits_member(bool, is_bounded);
__declare_integer_limits_member(bool, is_modulo);

#undef __declare_integer_limits_member

// Base class for floating-point numbers.
template <typename __number,
         int __Digits, int __Digits10,
         int __MinExp, int __MaxExp,
         int __MinExp10, int __MaxExp10,
         unsigned int __InfinityWord,
         unsigned int __QNaNWord, unsigned int __SNaNWord,
         bool __IsIEC559,
         float_round_style __RoundStyle>
class _Floating_limits : public _Numeric_limits_base<__number>
{
public:
  static const bool is_specialized = true;

  static const int digits   = __Digits;
  static const int digits10 = __Digits10;

  static const bool is_signed = true;

  static const int radix = 2;

  static const int min_exponent   = __MinExp;
  static const int max_exponent   = __MaxExp;
  static const int min_exponent10 = __MinExp10;
  static const int max_exponent10 = __MaxExp10;

  static const bool has_infinity      = true;
  static const bool has_quiet_NaN     = true;
  static const bool has_signaling_NaN = true;
  static const float_denorm_style has_denorm = denorm_indeterminate;
  static const bool has_denorm_loss   = false;

  static __number infinity()  {
    static unsigned int _S_inf[sizeof(__number) / sizeof(int)] = 
      { __InfinityWord };
    return *reinterpret_cast<__number*>(&_S_inf);
  }
  static __number quiet_NaN()  {
    static unsigned int _S_nan[sizeof(__number) / sizeof(int)] = 
      { __QNaNWord };
    return *reinterpret_cast<__number*>(&_S_nan);
  }
  static __number signaling_NaN()  {
    static unsigned int _S_nan[sizeof(__number) / sizeof(int)] = 
      { __SNaNWord };
    return *reinterpret_cast<__number*>(&_S_nan);
  }

  static const bool is_iec559       = __IsIEC559;
  static const bool is_bounded      = true;
  static const bool traps           = true;
  static const bool tinyness_before = false;

  static const float_round_style round_style = __RoundStyle;
};

#define __declare_float_limits_member(__type, __mem) \
template <typename __Num, int __Dig, int __Dig10, \
          int __MnX, int __MxX, int __MnX10, int __MxX10, \
          unsigned int __Inf, unsigned int __QNaN, unsigned int __SNaN, \
          bool __IsIEEE, float_round_style __Sty> \
const __type _Floating_limits<__Num, __Dig, __Dig10, \
                              __MnX, __MxX, __MnX10, __MxX10, \
                              __Inf, __QNaN, __SNaN,__IsIEEE, __Sty>:: __mem

__declare_float_limits_member(bool, is_specialized);  
__declare_float_limits_member(int, digits);  
__declare_float_limits_member(int, digits10);  
__declare_float_limits_member(bool, is_signed);  
__declare_float_limits_member(int, radix);  
__declare_float_limits_member(int, min_exponent);  
__declare_float_limits_member(int, max_exponent);  
__declare_float_limits_member(int, min_exponent10);  
__declare_float_limits_member(int, max_exponent10);  
__declare_float_limits_member(bool, has_infinity);
__declare_float_limits_member(bool, has_quiet_NaN);
__declare_float_limits_member(bool, has_signaling_NaN);
__declare_float_limits_member(float_denorm_style, has_denorm);
__declare_float_limits_member(bool, has_denorm_loss);
__declare_float_limits_member(bool, is_iec559);
__declare_float_limits_member(bool, is_bounded);
__declare_float_limits_member(bool, traps);
__declare_float_limits_member(bool, tinyness_before);
__declare_float_limits_member(float_round_style, round_style);

#undef __declare_float_limits_member

// Class numeric_limits

// The unspecialized class.

template<typename T> 
class numeric_limits : public _Numeric_limits_base<T> {};

// Specializations for all built-in integral types.

#ifndef __STL_NO_BOOL

template<>
class numeric_limits<bool>
  : public _Integer_limits<bool, false, true, 0> {};

#endif /* __STL_NO_BOOL */

template<>
class numeric_limits<char>
  : public _Integer_limits<char, CHAR_MIN, CHAR_MAX> {};

template<>
class numeric_limits<signed char>
  : public _Integer_limits<signed char, SCHAR_MIN, SCHAR_MAX> {};

template<>
class numeric_limits<unsigned char>
  : public _Integer_limits<unsigned char, 0, UCHAR_MAX> {};

#ifdef __STL_HAS_WCHAR_T

template<>
class numeric_limits<wchar_t>
  : public _Integer_limits<wchar_t, INT_MIN, INT_MAX> {};

#endif

template<>
class numeric_limits<short>
  : public _Integer_limits<short, SHRT_MIN, SHRT_MAX> {};

template<>
class numeric_limits<unsigned short>
  : public _Integer_limits<unsigned short, 0, USHRT_MAX> {};

template<>
class numeric_limits<int>
  : public _Integer_limits<int, INT_MIN, INT_MAX> {};

template<>
class numeric_limits<unsigned int>
  : public _Integer_limits<unsigned int, 0, UINT_MAX> {};

template<>
class numeric_limits<long>
  : public _Integer_limits<long, LONG_MIN, LONG_MAX> {};

template<>
class numeric_limits<unsigned long>
  : public _Integer_limits<unsigned long, 0, ULONG_MAX> {};

#ifdef __STL_LONG_LONG
#ifdef LONG_LONG_MIN

// CYGNUS LOCAL 9/4/1998
// fixed LONGLONG to be LONG_LONG
template<>
class numeric_limits<long long>
  : public _Integer_limits<long long, LONG_LONG_MIN, LONG_LONG_MAX> {};

// CYGNUS LOCAL 9/4/1998
// fixed LONGLONG to be LONG_LONG
template<>
class numeric_limits<unsigned long long>
  : public _Integer_limits<unsigned long long, 0, ULONG_LONG_MAX> {};

#endif
#endif /* __STL_LONG_LONG */

// Specializations for all built-in floating-point type.

template<> class numeric_limits<float>
  : public _Floating_limits<float, 
                            FLT_MANT_DIG,   // Binary digits of precision
                            FLT_DIG,        // Decimal digits of precision
                            FLT_MIN_EXP,    // Minimum exponent
                            FLT_MAX_EXP,    // Maximum exponent
                            FLT_MIN_10_EXP, // Minimum base 10 exponent
                            FLT_MAX_10_EXP, // Maximum base 10 exponent
                            0x7f800000u,    // First word of +infinity
                            0x7f810000u,    // First word of quiet NaN
                            0x7fc10000u,    // First word of signaling NaN
                            true,           // conforms to iec559
                            round_to_nearest>
{
public:
  static float min()  { return FLT_MIN; }
  static float denorm_min()  { return FLT_MIN; }
  static float max()  { return FLT_MAX; }
  static float epsilon()  { return FLT_EPSILON; }
  static float round_error()  { return 0.5f; } // Units: ulps.
};

template<> class numeric_limits<double>
  : public _Floating_limits<double, 
                            DBL_MANT_DIG,   // Binary digits of precision
                            DBL_DIG,        // Decimal digits of precision
                            DBL_MIN_EXP,    // Minimum exponent
                            DBL_MAX_EXP,    // Maximum exponent
                            DBL_MIN_10_EXP, // Minimum base 10 exponent
                            DBL_MAX_10_EXP, // Maximum base 10 exponent
                            0x7ff00000u,    // First word of +infinity
                            0x7ff10000u,    // First word of quiet NaN
                            0x7ff90000u,    // First word of signaling NaN
                            true,           // conforms to iec559
                            round_to_nearest>
{
public:
  static double min()  { return DBL_MIN; }
  static double denorm_min()  { return DBL_MIN; }
  static double max()  { return DBL_MAX; }
  static double epsilon()  { return DBL_EPSILON; }
  static double round_error()  { return 0.5; } // Units: ulps.
};

template<> class numeric_limits<long double>
  : public _Floating_limits<long double, 
                            LDBL_MANT_DIG,  // Binary digits of precision
                            LDBL_DIG,       // Decimal digits of precision
                            LDBL_MIN_EXP,   // Minimum exponent
                            LDBL_MAX_EXP,   // Maximum exponent
                            LDBL_MIN_10_EXP,// Minimum base 10 exponent
                            LDBL_MAX_10_EXP,// Maximum base 10 exponent
                            0x7ff00000u,    // First word of +infinity
                            0x7ff10000u,    // First word of quiet NaN
                            0x7ff90000u,    // First word of signaling NaN
                            false,          // Doesn't conform to iec559
                            round_to_nearest>
{
public:
  static long double min()  { return LDBL_MIN; }
  static long double denorm_min()  { return LDBL_MIN; }
  static long double max()  { return LDBL_MAX; }
  static long double epsilon()  { return LDBL_EPSILON; }
  static long double round_error()  { return 4; } // Units: ulps.
};

BZ_NAMESPACE_END

