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
 ***************************************************************************
 * $Log$
 * Revision 1.2  2002/09/12 07:04:04  eric
 * major rewrite of weave.
 *
 * 0.
 * The underlying library code is significantly re-factored and simpler. There used to be a xxx_spec.py and xxx_info.py file for every group of type conversion classes.  The spec file held the python code that handled the conversion and the info file had most of the C code templates that were generated.  This proved pretty confusing in practice, so the two files have mostly been merged into the spec file.
 *
 * Also, there was quite a bit of code duplication running around.  The re-factoring was able to trim the standard conversion code base (excluding blitz and accelerate stuff) by about 40%.  This should be a huge maintainability and extensibility win.
 *
 * 1.
 * With multiple months of using Numeric arrays, I've found some of weave's "magic variable" names unwieldy and want to change them.  The following are the old declarations for an array x of Float32 type:
 *
 *         PyArrayObject* x = convert_to_numpy(...);
 *         float* x_data = (float*) x->data;
 *         int*   _Nx = x->dimensions;
 *         int*   _Sx = x->strides;
 *         int    _Dx = x->nd;
 *
 * The new declaration looks like this:
 *
 *         PyArrayObject* x_array = convert_to_numpy(...);
 *         float* x = (float*) x->data;
 *         int*   Nx = x->dimensions;
 *         int*   Sx = x->strides;
 *         int    Dx = x->nd;
 *
 * This is obviously not backward compatible, and will break some code (including a lot of mine).  It also makes inline() code more readable and natural to write.
 *
 * 2.
 * I've switched from CXX to Gordon McMillan's SCXX for list, tuples, and dictionaries.  I like CXX pretty well, but its use of advanced C++ (templates, etc.) caused some portability problems.  The SCXX library is similar to CXX but doesn't use templates at all.  This, like (1) is not an
 * API compatible change and requires repairing existing code.
 *
 * I have also thought about boost python, but it also makes heavy use of templates.  Moving to SCXX gets rid of almost all template usage for the standard type converters which should help portability.  std::complex and std::string from the STL are the only templates left.  Of course blitz still uses templates in a major way so weave.blitz will continue to be hard on compilers.
 *
 * I've actually considered scrapping the C++ classes for list, tuples, and
 * dictionaries, and just fall back to the standard Python C API because the classes are waaay slower than the raw API in many cases.  They are also more convenient and less error prone in many cases, so I've decided to stick with them.  The PyObject variable will always be made available for variable "x" under the name "py_x" for more speedy operations.  You'll definitely want to use these for anything that needs to be speedy.
 *
 * 3.
 * strings are converted to std::string now.  I found this to be the most useful type in for strings in my code.  Py::String was used previously.
 *
 * 4.
 * There are a number of reference count "errors" in some of the less tested conversion codes such as instance, module, etc.  I've cleaned most of these up.  I put errors in quotes here because I'm actually not positive that objects passed into "inline" really need reference counting applied to them.  The dictionaries passed in by inline() hold references to these objects so it doesn't seem that they could ever be garbage collected inadvertently.  Variables used by ext_tools, though, definitely need the reference counting done.  I don't think this is a major cost in speed, so it probably isn't worth getting rid of the ref count code.
 *
 * 5.
 * Unicode objects are now supported.  This was necessary to support rendering Unicode strings in the freetype wrappers for Chaco.
 *
 * 6.
 * blitz++ was upgraded to the latest CVS.  It compiles about twice as fast as the old blitz and looks like it supports a large number of compilers (though only gcc 2.95.3 is tested).  Compile times now take about 9 seconds on my 850 MHz PIII laptop.
 *
 * Revision 1.2  2001/01/24 20:22:49  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.5  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.4  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.3  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 *
 */

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

template<class P_numtype1, class P_numtype2>
class _bz_Add : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote; 
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x + y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_Subtract : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;
 
    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x - y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_Multiply : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x * y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_Divide : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x / y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_Mod : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x % y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_BitwiseXOR : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x ^ y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_BitwiseAnd : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x & y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_BitwiseOr : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x | y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_ShiftRight : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x >> y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_ShiftLeft : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef BZ_PROMOTE(T_numtype1,T_numtype2) T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x << y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_Greater : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x > y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_Less : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x < y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_GreaterOrEqual : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x >= y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_LessOrEqual : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x <= y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_Equal : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x == y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_NotEqual : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x != y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_LogicalAnd : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x && y; }
};

template<class P_numtype1, class P_numtype2>
class _bz_LogicalOr : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef _bz_bool   T_promote;
    typedef T_promote  T_numtype;

    static inline T_promote apply(P_numtype1 x, P_numtype2 y)
    { return x || y; }
};

template<class P_numtype_in, class P_numtype_out>
class _bz_Cast : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype_in T_numtype1;
    typedef P_numtype_out T_promote;
    typedef T_promote     T_numtype;

    static inline P_numtype_out apply(P_numtype_in x)
    { return P_numtype_out(x); }
};

template<class P_numtype>
class _bz_LogicalNot : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype     T_numtype1;
    typedef _bz_bool      T_promote;
    typedef T_promote     T_numtype;

    static inline P_numtype apply(P_numtype x)
    { return !x; }
};

template<class P_numtype>
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
