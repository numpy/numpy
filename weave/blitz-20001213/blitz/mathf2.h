/***************************************************************************
 * blitz/mathf2.h  Declaration of additional math functions
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
 * Revision 1.2  2001/01/25 00:25:54  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_MATHF2_H
#define BZ_MATHF2_H

#ifndef BZ_APPLICS_H
 #error <blitz/mathf2.h> should be included via <blitz/applics.h>
#endif

#ifndef BZ_PRETTYPRINT_H
 #include <blitz/prettyprint.h>
#endif

BZ_NAMESPACE(blitz)

// cexp(z)     Complex exponential
template<class P_numtype1>
class _bz_cexp : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return _bz_exp<T_numtype1>::apply(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cexp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// csqrt(z)    Complex square root
template<class P_numtype1>
class _bz_csqrt : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return _bz_sqrt<T_numtype1>::apply(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "csqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow2        Square
template<class P_numtype1>
class _bz_pow2 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { 
        return BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x);
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow2(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow3        Cube
template<class P_numtype1>
class _bz_pow3 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { 
        return BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x) *
          BZ_NO_PROPAGATE(x);
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow3(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow4        Fourth power
template<class P_numtype1>
class _bz_pow4 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { 
        T_numtype t1 = BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x);
        return BZ_NO_PROPAGATE(t1) * BZ_NO_PROPAGATE(t1);
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow4(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow5        Fifth power
template<class P_numtype1>
class _bz_pow5 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    {
        T_numtype t1 = BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x);
        return BZ_NO_PROPAGATE(t1) * BZ_NO_PROPAGATE(t1)
            * BZ_NO_PROPAGATE(t1);
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow5(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow6        Sixth power
template<class P_numtype1>
class _bz_pow6 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    {
        T_numtype t1 = BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x) 
            * BZ_NO_PROPAGATE(x);
        return BZ_NO_PROPAGATE(t1) * BZ_NO_PROPAGATE(t1);
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow6(";
        a.prettyPrint(str,format);
        str += ")";
    }
};


// pow7        Seventh power
template<class P_numtype1>
class _bz_pow7 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    {
        T_numtype t1 = BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x) 
            * BZ_NO_PROPAGATE(x);
        return BZ_NO_PROPAGATE(t1) * BZ_NO_PROPAGATE(t1)
            * BZ_NO_PROPAGATE(x);
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow7(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow8        Eighth power
template<class P_numtype1>
class _bz_pow8 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    {
        T_numtype t1 = BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x);
        T_numtype t2 = BZ_NO_PROPAGATE(t1) * BZ_NO_PROPAGATE(t1);
        return BZ_NO_PROPAGATE(t2) * BZ_NO_PROPAGATE(t2);
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow8(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

/*
 * These scalar versions of pow2, pow3, ..., pow8 are provided for
 * convenience.
 *
 * NEEDS_WORK -- include BZ_NO_PROPAGATE for these scalar versions.
 */

// NEEDS_WORK -- make these templates.  Rely on specialization to
// handle expression template versions.

#define BZ_DECLARE_POW(T)  \
 inline T pow2(T x) { return x*x; }                  \
 inline T pow3(T x) { return x*x*x; }                \
 inline T pow4(T x) { T t1 = x*x; return t1*t1; }    \
 inline T pow5(T x) { T t1 = x*x; return t1*t1*x; }  \
 inline T pow6(T x) { T t1 = x*x*x; return t1*t1; }  \
 inline T pow7(T x) { T t1 = x*x; return t1*t1*t1*x; } \
 inline T pow8(T x) { T t1 = x*x, t2=t1*t1; return t2*t2; }  

BZ_DECLARE_POW(int)
BZ_DECLARE_POW(float)
BZ_DECLARE_POW(double)
BZ_DECLARE_POW(long double)

#ifdef BZ_HAVE_COMPLEX
BZ_DECLARE_POW(complex<float>)
BZ_DECLARE_POW(complex<double>)
BZ_DECLARE_POW(complex<long double>)
#endif

BZ_NAMESPACE_END

#endif
