// -*- C++ -*-
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
 ***************************************************************************/

#ifndef BZ_MATHF2_H
#define BZ_MATHF2_H

#ifndef BZ_APPLICS_H
 #error <blitz/mathf2.h> should be included via <blitz/applics.h>
#endif

#include <blitz/prettyprint.h>

BZ_NAMESPACE(blitz)

// cexp(z)     Complex exponential
template<typename P_numtype1>
class _bz_cexp : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return _bz_exp<T_numtype1>::apply(x); }

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cexp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// csqrt(z)    Complex square root
template<typename P_numtype1>
class _bz_csqrt : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return _bz_sqrt<T_numtype1>::apply(x); }

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "csqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow2        Square
template<typename P_numtype1>
class _bz_pow2 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { 
        return BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x);
    }

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow2(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow3        Cube
template<typename P_numtype1>
class _bz_pow3 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { 
        return BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x) *
          BZ_NO_PROPAGATE(x);
    }

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow3(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow4        Fourth power
template<typename P_numtype1>
class _bz_pow4 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { 
        T_numtype t1 = BZ_NO_PROPAGATE(x) * BZ_NO_PROPAGATE(x);
        return BZ_NO_PROPAGATE(t1) * BZ_NO_PROPAGATE(t1);
    }

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow4(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow5        Fifth power
template<typename P_numtype1>
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

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow5(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow6        Sixth power
template<typename P_numtype1>
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

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow6(";
        a.prettyPrint(str,format);
        str += ")";
    }
};


// pow7        Seventh power
template<typename P_numtype1>
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

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "pow7(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// pow8        Eighth power
template<typename P_numtype1>
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

    template<typename T1>
    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,
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
