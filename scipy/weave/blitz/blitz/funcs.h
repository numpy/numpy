// -*- C++ -*-
/***************************************************************************
 * blitz/funcs.h            Function objects for math functions
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
 *************************************************************************/

#ifndef BZ_FUNCS_H
#define BZ_FUNCS_H

#include <blitz/blitz.h>
#include <blitz/promote.h>
#include <blitz/prettyprint.h>

BZ_NAMESPACE(blitz)
    
/* Helper functions */
    
template <typename T>
inline T blitz_sqr(T x)
{ return x*x; }

template <typename T>
inline T blitz_cube(T x)
{ return x*x*x; }

template <typename T>
inline T blitz_pow4(T x)
{ return x*x*x*x; }

template <typename T>
inline T blitz_pow5(T x)
{ return x*x*x*x*x; }

template <typename T>
inline T blitz_pow6(T x)
{ return x*x*x*x*x*x; }

template <typename T>
inline T blitz_pow7(T x)
{ return x*x*x*x*x*x*x; }

template <typename T>
inline T blitz_pow8(T x)
{ return x*x*x*x*x*x*x*x; }


/* Unary functions that return same type as argument */
    
#define BZ_DEFINE_UNARY_FUNC(name,fun)                         \
template<typename T_numtype1>                                  \
struct name {                                                  \
    typedef T_numtype1 T_numtype;                              \
                                                               \
    static inline T_numtype                                    \
    apply(T_numtype1 a)                                        \
    { return fun(a); }                                         \
							       \
    template<typename T1>                                      \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,  \
        prettyPrintFormat& format, const T1& t1)               \
    {                                                          \
        str += #fun;                                           \
        str += "(";                                            \
        t1.prettyPrint(str, format);                           \
        str += ")";                                            \
    }                                                          \
};

BZ_DEFINE_UNARY_FUNC(Fn_acos,BZ_MATHFN_SCOPE(acos))
BZ_DEFINE_UNARY_FUNC(Fn_asin,BZ_MATHFN_SCOPE(asin))
BZ_DEFINE_UNARY_FUNC(Fn_atan,BZ_MATHFN_SCOPE(atan))
BZ_DEFINE_UNARY_FUNC(Fn_ceil,BZ_MATHFN_SCOPE(ceil))
BZ_DEFINE_UNARY_FUNC(Fn_cos,BZ_MATHFN_SCOPE(cos))
BZ_DEFINE_UNARY_FUNC(Fn_cosh,BZ_MATHFN_SCOPE(cosh))
BZ_DEFINE_UNARY_FUNC(Fn_exp,BZ_MATHFN_SCOPE(exp))
BZ_DEFINE_UNARY_FUNC(Fn_fabs,BZ_MATHFN_SCOPE(fabs))
BZ_DEFINE_UNARY_FUNC(Fn_floor,BZ_MATHFN_SCOPE(floor))
BZ_DEFINE_UNARY_FUNC(Fn_log,BZ_MATHFN_SCOPE(log))
BZ_DEFINE_UNARY_FUNC(Fn_log10,BZ_MATHFN_SCOPE(log10))
BZ_DEFINE_UNARY_FUNC(Fn_sin,BZ_MATHFN_SCOPE(sin))
BZ_DEFINE_UNARY_FUNC(Fn_sinh,BZ_MATHFN_SCOPE(sinh))
BZ_DEFINE_UNARY_FUNC(Fn_sqrt,BZ_MATHFN_SCOPE(sqrt))
BZ_DEFINE_UNARY_FUNC(Fn_tan,BZ_MATHFN_SCOPE(tan))
BZ_DEFINE_UNARY_FUNC(Fn_tanh,BZ_MATHFN_SCOPE(tanh))

#ifdef BZ_HAVE_IEEE_MATH
BZ_DEFINE_UNARY_FUNC(Fn_acosh,BZ_IEEEMATHFN_SCOPE(acosh))
BZ_DEFINE_UNARY_FUNC(Fn_asinh,BZ_IEEEMATHFN_SCOPE(asinh))
BZ_DEFINE_UNARY_FUNC(Fn_atanh,BZ_IEEEMATHFN_SCOPE(atanh))
BZ_DEFINE_UNARY_FUNC(Fn_cbrt,BZ_IEEEMATHFN_SCOPE(cbrt))
BZ_DEFINE_UNARY_FUNC(Fn_erf,BZ_IEEEMATHFN_SCOPE(erf))
BZ_DEFINE_UNARY_FUNC(Fn_erfc,BZ_IEEEMATHFN_SCOPE(erfc))
BZ_DEFINE_UNARY_FUNC(Fn_expm1,BZ_IEEEMATHFN_SCOPE(expm1))
BZ_DEFINE_UNARY_FUNC(Fn_j0,BZ_IEEEMATHFN_SCOPE(j0))
BZ_DEFINE_UNARY_FUNC(Fn_j1,BZ_IEEEMATHFN_SCOPE(j1))
BZ_DEFINE_UNARY_FUNC(Fn_lgamma,BZ_IEEEMATHFN_SCOPE(lgamma))
BZ_DEFINE_UNARY_FUNC(Fn_logb,BZ_IEEEMATHFN_SCOPE(logb))
BZ_DEFINE_UNARY_FUNC(Fn_log1p,BZ_IEEEMATHFN_SCOPE(log1p))
BZ_DEFINE_UNARY_FUNC(Fn_rint,BZ_IEEEMATHFN_SCOPE(rint))
BZ_DEFINE_UNARY_FUNC(Fn_y0,BZ_IEEEMATHFN_SCOPE(y0))
BZ_DEFINE_UNARY_FUNC(Fn_y1,BZ_IEEEMATHFN_SCOPE(y1))
#endif
    
#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DEFINE_UNARY_FUNC(Fn__class,BZ_IEEEMATHFN_SCOPE(_class))
BZ_DEFINE_UNARY_FUNC(Fn_nearest,BZ_IEEEMATHFN_SCOPE(nearest))
BZ_DEFINE_UNARY_FUNC(Fn_rsqrt,BZ_IEEEMATHFN_SCOPE(rsqrt))
#endif
    
BZ_DEFINE_UNARY_FUNC(Fn_sqr,BZ_BLITZ_SCOPE(blitz_sqr))
BZ_DEFINE_UNARY_FUNC(Fn_cube,BZ_BLITZ_SCOPE(blitz_cube))
BZ_DEFINE_UNARY_FUNC(Fn_pow4,BZ_BLITZ_SCOPE(blitz_pow4))
BZ_DEFINE_UNARY_FUNC(Fn_pow5,BZ_BLITZ_SCOPE(blitz_pow5))
BZ_DEFINE_UNARY_FUNC(Fn_pow6,BZ_BLITZ_SCOPE(blitz_pow6))
BZ_DEFINE_UNARY_FUNC(Fn_pow7,BZ_BLITZ_SCOPE(blitz_pow7))
BZ_DEFINE_UNARY_FUNC(Fn_pow8,BZ_BLITZ_SCOPE(blitz_pow8))

/* Unary functions that return a specified type */
    
#define BZ_DEFINE_UNARY_FUNC_RET(name,fun,ret)                 \
template<typename T_numtype1>                                  \
struct name {                                                  \
    typedef ret T_numtype;                                     \
                                                               \
    static inline T_numtype                                    \
    apply(T_numtype1 a)                                        \
    { return fun(a); }                                         \
							       \
    template<typename T1>                                      \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,  \
        prettyPrintFormat& format, const T1& t1)               \
    {                                                          \
        str += #fun;                                           \
        str += "(";                                            \
        t1.prettyPrint(str, format);                           \
        str += ")";                                            \
    }                                                          \
};

#ifdef BZ_HAVE_IEEE_MATH
BZ_DEFINE_UNARY_FUNC_RET(Fn_ilogb,BZ_IEEEMATHFN_SCOPE(ilogb),int)
#endif
    
#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DEFINE_UNARY_FUNC_RET(Fn_itrunc,BZ_IEEEMATHFN_SCOPE(itrunc),int)
BZ_DEFINE_UNARY_FUNC_RET(Fn_uitrunc,BZ_IEEEMATHFN_SCOPE(uitrunc),unsigned int)
#endif
    
    
#ifdef BZ_HAVE_COMPLEX
/* Specialization of unary functor for complex type */
    
#define BZ_DEFINE_UNARY_CFUNC(name,fun)                        \
template<typename T>                                           \
struct name< complex<T> > {                                    \
    typedef complex<T> T_numtype1;                             \
    typedef complex<T> T_numtype;                              \
                                                               \
    static inline T_numtype                                    \
    apply(T_numtype1 a)                                        \
    { return fun(a); }                                         \
							       \
    template<typename T1>                                      \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,  \
        prettyPrintFormat& format, const T1& t1)               \
    {                                                          \
        str += #fun;                                           \
        str += "(";                                            \
        t1.prettyPrint(str, format);                           \
        str += ")";                                            \
    }                                                          \
};

#ifdef BZ_HAVE_COMPLEX_FCNS
BZ_DEFINE_UNARY_FUNC(Fn_conj,BZ_CMATHFN_SCOPE(conj))
#endif

#ifdef BZ_HAVE_COMPLEX_MATH1
BZ_DEFINE_UNARY_CFUNC(Fn_cos,BZ_CMATHFN_SCOPE(cos))
BZ_DEFINE_UNARY_CFUNC(Fn_cosh,BZ_CMATHFN_SCOPE(cosh))
BZ_DEFINE_UNARY_CFUNC(Fn_exp,BZ_CMATHFN_SCOPE(exp))
BZ_DEFINE_UNARY_CFUNC(Fn_log,BZ_CMATHFN_SCOPE(log))
BZ_DEFINE_UNARY_CFUNC(Fn_log10,BZ_CMATHFN_SCOPE(log10))
BZ_DEFINE_UNARY_CFUNC(Fn_sin,BZ_CMATHFN_SCOPE(sin))
BZ_DEFINE_UNARY_CFUNC(Fn_sinh,BZ_CMATHFN_SCOPE(sinh))
BZ_DEFINE_UNARY_CFUNC(Fn_sqrt,BZ_CMATHFN_SCOPE(sqrt))
BZ_DEFINE_UNARY_CFUNC(Fn_tan,BZ_CMATHFN_SCOPE(tan))
BZ_DEFINE_UNARY_CFUNC(Fn_tanh,BZ_CMATHFN_SCOPE(tanh))
#endif // BZ_HAVE_COMPLEX_MATH1

BZ_DEFINE_UNARY_CFUNC(Fn_sqr,BZ_BLITZ_SCOPE(blitz_sqr))
BZ_DEFINE_UNARY_CFUNC(Fn_cube,BZ_BLITZ_SCOPE(blitz_cube))
BZ_DEFINE_UNARY_CFUNC(Fn_pow4,BZ_BLITZ_SCOPE(blitz_pow4))
BZ_DEFINE_UNARY_CFUNC(Fn_pow5,BZ_BLITZ_SCOPE(blitz_pow5))
BZ_DEFINE_UNARY_CFUNC(Fn_pow6,BZ_BLITZ_SCOPE(blitz_pow6))
BZ_DEFINE_UNARY_CFUNC(Fn_pow7,BZ_BLITZ_SCOPE(blitz_pow7))
BZ_DEFINE_UNARY_CFUNC(Fn_pow8,BZ_BLITZ_SCOPE(blitz_pow8))

/* Unary functions that apply only to complex<T> and return T */
    
#define BZ_DEFINE_UNARY_CFUNC2(name,fun)                       \
template<typename T_numtype1>                                  \
struct name;                                                   \
                                                               \
template<typename T>                                           \
struct name< complex<T> > {                                    \
    typedef complex<T> T_numtype1;                             \
    typedef T T_numtype;                                       \
                                                               \
    static inline T_numtype                                    \
    apply(T_numtype1 a)                                        \
    { return fun(a); }                                         \
							       \
    template<typename T1>                                      \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,  \
        prettyPrintFormat& format, const T1& t1)               \
    {                                                          \
        str += #fun;                                           \
        str += "(";                                            \
        t1.prettyPrint(str, format);                           \
        str += ")";                                            \
    }                                                          \
};

#ifdef BZ_HAVE_COMPLEX_FCNS
BZ_DEFINE_UNARY_CFUNC2(Fn_arg,BZ_CMATHFN_SCOPE(arg))
BZ_DEFINE_UNARY_CFUNC2(Fn_imag,BZ_CMATHFN_SCOPE(imag))
BZ_DEFINE_UNARY_CFUNC2(Fn_norm,BZ_CMATHFN_SCOPE(norm))
BZ_DEFINE_UNARY_CFUNC2(Fn_real,BZ_CMATHFN_SCOPE(real))
#endif // BZ_HAVE_COMPLEX_FCNS
    
#endif // BZ_HAVE_COMPLEX
    
/* Binary functions that return type based on type promotion */
    
#define BZ_DEFINE_BINARY_FUNC(name,fun)                           \
template<typename T_numtype1, typename T_numtype2>                \
struct name {                                                     \
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;         \
                                                                  \
    static inline T_numtype                                       \
    apply(T_numtype1 a, T_numtype2 b)                             \
    { return fun(a,b); }                                          \
							          \
    template<typename T1, typename T2>                            \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,     \
        prettyPrintFormat& format, const T1& t1,                  \
        const T2& t2)                                             \
    {                                                             \
        str += #fun;                                              \
        str += "(";                                               \
        t1.prettyPrint(str, format);                              \
        str += ",";                                               \
        t2.prettyPrint(str, format);                              \
        str += ")";                                               \
    }                                                             \
};

BZ_DEFINE_BINARY_FUNC(Fn_atan2,BZ_MATHFN_SCOPE(atan2))
BZ_DEFINE_BINARY_FUNC(Fn_fmod,BZ_MATHFN_SCOPE(fmod))
BZ_DEFINE_BINARY_FUNC(Fn_pow,BZ_MATHFN_SCOPE(pow))
    
#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DEFINE_BINARY_FUNC(Fn_copysign,BZ_IEEEMATHFN_SCOPE(copysign))
BZ_DEFINE_BINARY_FUNC(Fn_drem,BZ_IEEEMATHFN_SCOPE(drem))
BZ_DEFINE_BINARY_FUNC(Fn_hypot,BZ_IEEEMATHFN_SCOPE(hypot))
BZ_DEFINE_BINARY_FUNC(Fn_nextafter,BZ_IEEEMATHFN_SCOPE(nextafter))
BZ_DEFINE_BINARY_FUNC(Fn_remainder,BZ_IEEEMATHFN_SCOPE(remainder))
BZ_DEFINE_BINARY_FUNC(Fn_scalb,BZ_IEEEMATHFN_SCOPE(scalb))
#endif
    
/* Binary functions that return a specified type */
    
#define BZ_DEFINE_BINARY_FUNC_RET(name,fun,ret)                   \
template<typename T_numtype1, typename T_numtype2>                \
struct name {                                                     \
    typedef ret T_numtype;                                        \
                                                                  \
    static inline T_numtype                                       \
    apply(T_numtype1 a, T_numtype2 b)                             \
    { return fun(a,b); }                                          \
							          \
    template<typename T1, typename T2>                            \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,     \
        prettyPrintFormat& format, const T1& t1,                  \
        const T2& t2)                                             \
    {                                                             \
        str += #fun;                                              \
        str += "(";                                               \
        t1.prettyPrint(str, format);                              \
        str += ",";                                               \
        t2.prettyPrint(str, format);                              \
        str += ")";                                               \
    }                                                             \
};

#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DEFINE_BINARY_FUNC_RET(Fn_unordered,BZ_IEEEMATHFN_SCOPE(unordered),int)
#endif
    
#ifdef BZ_HAVE_COMPLEX
/* Specialization of binary functor for complex type */
    
#define BZ_DEFINE_BINARY_CFUNC(name,fun)                       \
template<typename T>                                           \
struct name< complex<T>, complex<T> > {                        \
    typedef complex<T> T_numtype1;                             \
    typedef complex<T> T_numtype2;                             \
    typedef complex<T> T_numtype;                              \
                                                               \
    static inline T_numtype                                    \
    apply(T_numtype1 a, T_numtype2 b)                          \
    { return fun(a,b); }                                       \
							       \
    template<typename T1, typename T2>                         \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,  \
        prettyPrintFormat& format, const T1& t1,               \
        const T2& t2)                                          \
    {                                                          \
        str += #fun;                                           \
        str += "(";                                            \
        t1.prettyPrint(str, format);                           \
        str += ",";                                            \
        t2.prettyPrint(str, format);                           \
        str += ")";                                            \
    }                                                          \
};                                                             \
                                                               \
template<typename T>                                           \
struct name< complex<T>, T > {                                 \
    typedef complex<T> T_numtype1;                             \
    typedef T T_numtype2;                                      \
    typedef complex<T> T_numtype;                              \
                                                               \
    static inline T_numtype                                    \
    apply(T_numtype1 a, T_numtype2 b)                          \
    { return fun(a,b); }                                       \
							       \
    template<typename T1, typename T2>                         \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,  \
        prettyPrintFormat& format, const T1& t1,               \
        const T2& t2)                                          \
    {                                                          \
        str += #fun;                                           \
        str += "(";                                            \
        t1.prettyPrint(str, format);                           \
        str += ",";                                            \
        t2.prettyPrint(str, format);                           \
        str += ")";                                            \
    }                                                          \
};                                                             \
                                                               \
template<typename T>                                           \
struct name< T, complex<T> > {                                 \
    typedef T T_numtype1;                                      \
    typedef complex<T> T_numtype2;                             \
    typedef complex<T> T_numtype;                              \
                                                               \
    static inline T_numtype                                    \
    apply(T_numtype1 a, T_numtype2 b)                          \
    { return fun(a,b); }                                       \
							       \
    template<typename T1, typename T2>                         \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,  \
        prettyPrintFormat& format, const T1& t1,               \
        const T2& t2)                                          \
    {                                                          \
        str += #fun;                                           \
        str += "(";                                            \
        t1.prettyPrint(str, format);                           \
        str += ",";                                            \
        t2.prettyPrint(str, format);                           \
        str += ")";                                            \
    }                                                          \
};

#ifdef BZ_HAVE_COMPLEX_MATH1
BZ_DEFINE_BINARY_CFUNC(Fn_pow,BZ_CMATHFN_SCOPE(pow))
#endif

/* Binary functions that apply only to T and return complex<T> */
    
#define BZ_DEFINE_BINARY_FUNC_CRET(name,fun)                      \
template<typename T_numtype1, typename T_numtype2>                \
struct name;                                                      \
                                                                  \
template<typename T>                                              \
struct name<T, T> {                                               \
    typedef T T_numtype1;                                         \
    typedef T T_numtype2;                                         \
    typedef complex<T> T_numtype;                                 \
                                                                  \
    static inline T_numtype                                       \
    apply(T_numtype1 a, T_numtype2 b)                             \
    { return fun(a,b); }                                          \
							          \
    template<typename T1, typename T2>                            \
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,     \
        prettyPrintFormat& format, const T1& t1,                  \
        const T2& t2)                                             \
    {                                                             \
        str += #fun;                                              \
        str += "(";                                               \
        t1.prettyPrint(str, format);                              \
        str += ",";                                               \
        t2.prettyPrint(str, format);                              \
        str += ")";                                               \
    }                                                             \
};

#ifdef BZ_HAVE_COMPLEX_FCNS
BZ_DEFINE_BINARY_FUNC_CRET(Fn_polar,BZ_CMATHFN_SCOPE(polar))
#endif
    
#endif // BZ_HAVE_COMPLEX
    
/* Ternary functions that return type based on type promotion */
    
#define BZ_DEFINE_TERNARY_FUNC(name,fun)                          \
template <typename P_numtype1, typename P_numtype2,               \
          typename P_numtype3>                                    \
struct name {                                                     \
    typedef BZ_PROMOTE(P_numtype1,                                \
            BZ_PROMOTE(P_numtype2,P_numtype3)) T_numtype;         \
                                                                  \
    static inline T_numtype                                       \
    apply(P_numtype1 x, P_numtype2 y, P_numtype3 z)               \
    { return fun(x,y,z); }                                        \
                                                                  \
    template <typename T1, typename T2, typename T3>              \
    static void prettyPrint(BZ_STD_SCOPE(string) &str,            \
        prettyPrintFormat& format,                                \
        const T1& a,const T2& b, const T3& c)                     \
    {                                                             \
        str += #fun;                                              \
        str += "(";                                               \
        a.prettyPrint(str,format);                                \
        str += ",";                                               \
        b.prettyPrint(str,format);                                \
        str += ",";                                               \
        c.prettyPrint(str,format);                                \
        str += ")";                                               \
    }                                                             \
};

/* Ternary functions that return a specified type */
    
#define BZ_DEFINE_TERNARY_FUNC_RET(name,fun,ret)                  \
template <typename P_numtype1, typename P_numtype2,               \
          typename P_numtype3>                                    \
struct name {                                                     \
    typedef ret T_numtype;                                        \
                                                                  \
    static inline T_numtype                                       \
    apply(P_numtype1 x, P_numtype2 y, P_numtype3 z)               \
    { return fun(x,y,z); }                                        \
                                                                  \
    template <typename T1, typename T2, typename T3>              \
    static void prettyPrint(BZ_STD_SCOPE(string) &str,            \
        prettyPrintFormat& format,                                \
        const T1& a,const T2& b, const T3& c)                     \
    {                                                             \
        str += #fun;                                              \
        str += "(";                                               \
        a.prettyPrint(str,format);                                \
        str += ",";                                               \
        b.prettyPrint(str,format);                                \
        str += ",";                                               \
        c.prettyPrint(str,format);                                \
        str += ")";                                               \
    }                                                             \
};

    
/* These functions don't quite fit the usual patterns */
    
// abs()    Absolute value
template<typename T_numtype1>
struct Fn_abs;

// abs(int)
template<>
struct Fn_abs< int > {
    typedef int T_numtype1;
    typedef int T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_MATHFN_SCOPE(abs)(a); }
    
    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};

// abs(long)
template<>
struct Fn_abs< long int > {
    typedef long int T_numtype1;
    typedef long int T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_MATHFN_SCOPE(labs)(a); }
    
    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};

// abs(float)
template<>
struct Fn_abs< float > {
    typedef float T_numtype1;
    typedef float T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_MATHFN_SCOPE(fabs)(a); }
    
    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};

// abs(double)
template<>
struct Fn_abs< double > {
    typedef double T_numtype1;
    typedef double T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_MATHFN_SCOPE(fabs)(a); }
    
    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};

// abs(long double)
template<>
struct Fn_abs< long double > {
    typedef long double T_numtype1;
    typedef long double T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_MATHFN_SCOPE(fabs)(a); }
    
    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};

#ifdef BZ_HAVE_COMPLEX_FCNS
// abs(complex<T>)
template<typename T>
struct Fn_abs< complex<T> > {
    typedef complex<T> T_numtype1;
    typedef T T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_CMATHFN_SCOPE(abs)(a); }
    
    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};
#endif // BZ_HAVE_COMPLEX_FCNS


#ifdef BZ_HAVE_IEEE_MATH
// isnan()    Nonzero if NaNS or NaNQ
template<typename T_numtype1>
struct Fn_isnan {
    typedef int T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    {
#ifdef BZ_ISNAN_IN_NAMESPACE_STD
        return BZ_STD_SCOPE(isnan)(a);
#else
        return BZ_IEEEMATHFN_SCOPE(isnan)(a);
#endif
    }
    
    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "isnan";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};
#endif // BZ_HAVE_IEEE_MATH


// Blitz cast() function
template<typename T_numtype1, typename T_cast>
struct Cast {
    typedef T_cast T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return T_numtype(a); }

    template<typename T1>
    static inline void prettyPrint(BZ_STD_SCOPE(string) &str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_cast);
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};


BZ_NAMESPACE_END

#endif // BZ_FUNCS_H
