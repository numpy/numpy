/***************************************************************************
 * blitz/applics.h      Applicative template classes
 *
 * $Id$
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org    
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************/

#ifndef BZ_APPLICS_H
#define BZ_APPLICS_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_PROMOTE_H
 #include <blitz/promote.h>
#endif

#ifndef BZ_NUMTRAIT_H
 #include <blitz/numtrait.h>
#endif

BZ_NAMESPACE(blitz)

// These base classes are included for no other reason than to keep
// the applicative templates clustered together in a graphical
// class browser.
class ApplicativeTemplatesBase { };
class TwoOperandApplicativeTemplatesBase : public ApplicativeTemplatesBase { };
class OneOperandApplicativeTemplatesBase : public ApplicativeTemplatesBase { };

template<typename P_numtype1, typename P_numtype2>
class _bz_Add : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote; 
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x + y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Subtract : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;
 
    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x - y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Multiply : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x * y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Divide : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x / y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Mod : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x % y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_BitwiseXOR : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x ^ y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_BitwiseAnd : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x & y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_BitwiseOr : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x | y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_ShiftRight : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x >> y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_ShiftLeft : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x << y; }
};


template<typename P_numtype1, typename P_numtype2>
class _bz_Min : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return (x < y ? x : y); }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Max : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return (x > y ? x : y); }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Greater : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x > y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Less : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x < y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_GreaterOrEqual : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x >= y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_LessOrEqual : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x <= y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_Equal : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x == y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_NotEqual : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x != y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_LogicalAnd : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x && y; }
};

template<typename P_numtype1, typename P_numtype2>
class _bz_LogicalOr : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef bool       T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x || y; }
};


template<typename P_numtype_in, typename P_numtype_out>
class _bz_Cast : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype_in T_numtype1;
    typedef P_numtype_out T_promote;
    typedef T_promote     T_numtype;

    static inline P_numtype_out apply(P_numtype_in x)
    { return P_numtype_out(x); }
};

template<typename P_numtype>
class _bz_LogicalNot : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype T_numtype1;
    typedef bool      T_promote;
    typedef T_promote T_numtype;

    static inline P_numtype apply(P_numtype x)
    { return !x; }
};

template<typename P_numtype>
class _bz_BitwiseNot : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype     T_numtype1;
    typedef T_numtype1    T_promote;
    typedef T_promote     T_numtype;

    static inline P_numtype apply(P_numtype x)
    { return ~x; }
};



/*****************************************************************************
 * Math Functions
 *****************************************************************************/

// Applicative templates for these functions are defined in
// <blitz/mathfunc.h>, which is included below:
//
// abs(i), labs(l)                     Absolute value
// acos(d), acols(ld)                  Inverse cosine
// acosh(d)                            Inverse hyperbolic cosine
// asin(d), asinl(ld)                  Inverse sine
// asinh(d)                            Inverse hyperbolic sine
// atan(d), atanl(ld)                  Inverse tangent
// atan2(d,d), atan2l(ld,ld)           Inverse tangent
// atanh(d)                            Inverse hyperbolic tangent
// cbrt(x)                             Cube root
// ceil(d), ceill(ld)                  Smallest f-int not less than x
// int class(d)                        Classification of x (FP_XXXXX)
// cos(d), cosl(ld)                    Cosine
// cosh(d), coshl(ld)                  Hyperbolic cosine
// copysign(d,d)                       Return 1st arg with same sign as 2nd
// drem(x,x)                           IEEE remainder
// exp(d), expl(ld)                    Exponential
// expm1(d)                            Exp(x)-1     
// erf(d), erfl(ld)                    Error function
// erfc(d), erfcl(ld)                  Complementary error function
// fabs(d), fabsl(ld)                  Floating point absolute value
// int finite(d)                       Nonzero if finite
// floor(d), floor(ld)                 Largest f-int not greater than x
// fmod(d,d), fmodl(ld,ld)             Floating point remainder
// frexp(d, int* e)                    Break into mantissa/exponent  (*)
// frexpl(ld, int* e)                  Break into mantissa/exponent  (*)
// gammaFunc(d)                        Gamma function (** needs special 
//                                     implementation using lgamma)
// hypot(d,d)                          Hypotenuse: sqrt(x*x+y*y)
// int ilogb(d)                        Integer unbiased exponent
// int isnan(d)                        Nonzero if NaNS or NaNQ
// int itrunc(d)                       Truncate and convert to integer
// j0(d)                               Bessel function first kind, order 0
// j1(d)                               Bessel function first kind, order 1
// jn(int, double)                     Bessel function first kind, order i
// ldexp(d,i), ldexpl(ld,i)            Compute d * 2^i
// lgamma(d), lgammald(ld)             Log absolute gamma
// log(d), logl(ld)                    Natural logarithm
// logb(d)                             Unbiased exponent (IEEE)
// log1p(d)                            Compute log(1 + x)
// log10(d), log10l(ld)                Logarithm base 10
// modf(d, int* i), modfl(ld, int* i)  Break into integral/fractional part
// double nearest(double)              Nearest floating point integer
// nextafter(d, d)                     Next representable neighbor of 1st
//                                     in direction of 2nd
// pow(d,d), pow(ld,ld)                Computes x ^ y
// d remainder(d,d)                    IEEE remainder
// d rint(d)                           Round to f-integer (depends on mode)
// d rsqrt(d)                          Reciprocal square root
// d scalb(d,d)                        Return x * (2^y)
// sin(d), sinl(ld)                    Sine 
// sinh(d), sinhl(ld)                  Hyperbolic sine
// sqr(x)                              Return x * x
// sqrt(d), sqrtl(ld)                  Square root
// tan(d), tanl(ld)                    Tangent
// tanh(d), tanhl(ld)                  Hyperbolic tangent
// trunc(d)                            Nearest f-int in the direction of 0
// unsigned uitrunc(d)                 Truncate and convert to unsigned
// int unordered(d,d)                  Nonzero if comparison is unordered
// y0(d)                               Bessel function 2nd kind, order 0
// y1(d)                               Bessel function 2nd kind, order 1
// yn(i,d)                             Bessel function 2nd kind, order d


BZ_NAMESPACE_END

#ifndef BZ_MATHFUNC_H
 #include <blitz/mathfunc.h>
#endif

#ifndef BZ_MATHF2_H
 #include <blitz/mathf2.h>
#endif

#endif // BZ_APPLICS_H
