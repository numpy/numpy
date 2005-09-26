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
 *************************************************************************
 * $Log$
 * Revision 1.1  2002/09/12 07:04:04  eric
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
 * Revision 1.1  2002/07/02 19:01:07  jcumming
 * New version of classes to provide ET support for unary and binary math functions.
 *
 *
 */

#ifndef BZ_FUNCS_H
#define BZ_FUNCS_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_PROMOTE_H
 #include <blitz/promote.h>
#endif

#ifndef BZ_PRETTYPRINT_H
 #include <blitz/prettyprint.h>
#endif

BZ_NAMESPACE(blitz)
    
/* Helper functions */
    
template <class T>
inline T blitz_sqr(T x)
{ return x*x; }

template <class T>
inline T blitz_cube(T x)
{ return x*x*x; }

template <class T>
inline T blitz_pow4(T x)
{ return x*x*x*x; }

template <class T>
inline T blitz_pow5(T x)
{ return x*x*x*x*x; }

template <class T>
inline T blitz_pow6(T x)
{ return x*x*x*x*x*x; }

template <class T>
inline T blitz_pow7(T x)
{ return x*x*x*x*x*x*x; }

template <class T>
inline T blitz_pow8(T x)
{ return x*x*x*x*x*x*x*x; }


/* Unary functions that return same type as argument */
    
#define BZ_DEFINE_UNARY_FUNC(name,fun)                      \
template<class T_numtype1>                                  \
struct name {                                               \
    typedef T_numtype1 T_numtype;                           \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a)                                     \
    { return fun(a); }                                      \
							    \
    template<class T1>                                      \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1)            \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
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

#ifdef BZ_HAVE_COMPLEX_MATH
BZ_DEFINE_UNARY_FUNC(Fn_conj,BZ_CMATHFN_SCOPE(conj))

/* Specialization of unary functor for complex type */
    
#define BZ_DEFINE_UNARY_CFUNC(name,fun)                     \
template<class T>                                           \
struct name< complex<T> > {                                 \
    typedef complex<T> T_numtype1;                          \
    typedef complex<T> T_numtype;                           \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a)                                     \
    { return fun(a); }                                      \
							    \
    template<class T1>                                      \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1)            \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

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

BZ_DEFINE_UNARY_CFUNC(Fn_sqr,BZ_BLITZ_SCOPE(blitz_sqr))
BZ_DEFINE_UNARY_CFUNC(Fn_cube,BZ_BLITZ_SCOPE(blitz_cube))
BZ_DEFINE_UNARY_CFUNC(Fn_pow4,BZ_BLITZ_SCOPE(blitz_pow4))
BZ_DEFINE_UNARY_CFUNC(Fn_pow5,BZ_BLITZ_SCOPE(blitz_pow5))
BZ_DEFINE_UNARY_CFUNC(Fn_pow6,BZ_BLITZ_SCOPE(blitz_pow6))
BZ_DEFINE_UNARY_CFUNC(Fn_pow7,BZ_BLITZ_SCOPE(blitz_pow7))
BZ_DEFINE_UNARY_CFUNC(Fn_pow8,BZ_BLITZ_SCOPE(blitz_pow8))

/* Unary functions that apply only to complex<T> and return T */
    
#define BZ_DEFINE_UNARY_CFUNC2(name,fun)                    \
template<class T_numtype1>                                  \
struct name;                                                \
                                                            \
template<class T>                                           \
struct name< complex<T> > {                                 \
    typedef complex<T> T_numtype1;                          \
    typedef T T_numtype;                                    \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a)                                     \
    { return fun(a); }                                      \
							    \
    template<class T1>                                      \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1)            \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

BZ_DEFINE_UNARY_CFUNC2(Fn_arg,BZ_CMATHFN_SCOPE(arg))
BZ_DEFINE_UNARY_CFUNC2(Fn_imag,BZ_CMATHFN_SCOPE(imag))
BZ_DEFINE_UNARY_CFUNC2(Fn_norm,BZ_CMATHFN_SCOPE(norm))
BZ_DEFINE_UNARY_CFUNC2(Fn_real,BZ_CMATHFN_SCOPE(real))
#endif
    
    
/* Unary functions that return a specified type */
    
#define BZ_DEFINE_UNARY_FUNC_RET(name,fun,ret)              \
template<class T_numtype1>                                  \
struct name {                                               \
    typedef ret T_numtype;                                  \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a)                                     \
    { return fun(a); }                                      \
							    \
    template<class T1>                                      \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1)            \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

#ifdef BZ_HAVE_IEEE_MATH
BZ_DEFINE_UNARY_FUNC_RET(Fn_ilogb,BZ_IEEEMATHFN_SCOPE(ilogb),int)
#endif
    
#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DEFINE_UNARY_FUNC_RET(Fn_itrunc,BZ_IEEEMATHFN_SCOPE(itrunc),int)
BZ_DEFINE_UNARY_FUNC_RET(Fn_uitrunc,BZ_IEEEMATHFN_SCOPE(uitrunc),unsigned int)
#endif
    
    
/* Binary functions that return type based on type promotion */
    
#define BZ_DEFINE_BINARY_FUNC(name,fun)                     \
template<class T_numtype1, class T_numtype2>                \
struct name {                                               \
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;   \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return fun(a,b); }                                    \
							    \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ",";                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
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
    
#ifdef BZ_HAVE_COMPLEX_MATH
/* Specialization of binary functor for complex type */
    
#define BZ_DEFINE_BINARY_CFUNC(name,fun)                    \
template<class T>                                           \
struct name< complex<T>, complex<T> > {                     \
    typedef complex<T> T_numtype1;                          \
    typedef complex<T> T_numtype2;                          \
    typedef complex<T> T_numtype;                           \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return fun(a,b); }                                    \
							    \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ",";                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};                                                          \
                                                            \
template<class T>                                           \
struct name< complex<T>, T > {                              \
    typedef complex<T> T_numtype1;                          \
    typedef T T_numtype2;                                   \
    typedef complex<T> T_numtype;                           \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return fun(a,b); }                                    \
							    \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ",";                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};                                                          \
                                                            \
template<class T>                                           \
struct name< T, complex<T> > {                              \
    typedef T T_numtype1;                                   \
    typedef complex<T> T_numtype2;                          \
    typedef complex<T> T_numtype;                           \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return fun(a,b); }                                    \
							    \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ",";                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

BZ_DEFINE_BINARY_CFUNC(Fn_pow,BZ_CMATHFN_SCOPE(pow))

/* Binary functions that apply only to T and return complex<T> */
    
#define BZ_DEFINE_BINARY_CFUNC2(name,fun)                   \
template<class T_numtype1, class T_numtype2>                \
struct name;                                                \
                                                            \
template<class T>                                           \
struct name<T, T> {                                         \
    typedef T T_numtype1;                                   \
    typedef T T_numtype2;                                   \
    typedef complex<T> T_numtype;                           \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return fun(a,b); }                                    \
							    \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ",";                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

BZ_DEFINE_BINARY_CFUNC2(Fn_polar,BZ_CMATHFN_SCOPE(polar))
#endif
    
    
/* Binary functions that return a specified type */
    
#define BZ_DEFINE_BINARY_FUNC_RET(name,fun,ret)             \
template<class T_numtype1, class T_numtype2>                \
struct name {                                               \
    typedef ret T_numtype;                                  \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return fun(a,b); }                                    \
							    \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += #fun;                                        \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += ",";                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DEFINE_BINARY_FUNC_RET(Fn_unordered,BZ_IEEEMATHFN_SCOPE(unordered),int)
#endif
    
    
/* These functions don't quite fit the usual patterns */
    
// abs()    Absolute value
template<class T_numtype1>
struct Fn_abs;

// abs(int)
template<>
struct Fn_abs< int > {
    typedef int T_numtype1;
    typedef int T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_MATHFN_SCOPE(abs)(a); }
    
    template<class T1>
    static inline void prettyPrint(string& str,
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
    
    template<class T1>
    static inline void prettyPrint(string& str,
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
    
    template<class T1>
    static inline void prettyPrint(string& str,
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
    
    template<class T1>
    static inline void prettyPrint(string& str,
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
    
    template<class T1>
    static inline void prettyPrint(string& str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};

#ifdef BZ_HAVE_COMPLEX_MATH
// abs(complex<T>)
template<class T>
struct Fn_abs< complex<T> > {
    typedef complex<T> T_numtype1;
    typedef T T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return BZ_CMATHFN_SCOPE(abs)(a); }
    
    template<class T1>
    static inline void prettyPrint(string& str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};
#endif


#ifdef BZ_HAVE_IEEE_MATH
// isnan()    Nonzero if NaNS or NaNQ
template<class T_numtype1>
struct Fn_isnan {
    typedef int T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    {
#ifdef isnan
        // Some platforms define isnan as a macro, which causes the
        // BZ_IEEEMATHFN_SCOPE macro to break.
        return isnan(a); 
#else
        return BZ_IEEEMATHFN_SCOPE(isnan)(a);
#endif
    }
    
    template<class T1>
    static inline void prettyPrint(string& str,
        prettyPrintFormat& format, const T1& t1)
    {
        str += "isnan";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
    }
};
#endif


// Blitz cast() function
template<class T_numtype1, class T_cast>
struct Cast {
    typedef T_cast T_numtype;
    
    static inline T_numtype
    apply(T_numtype1 a)
    { return T_numtype(a); }

    template<class T1>
    static inline void prettyPrint(string& str,
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
