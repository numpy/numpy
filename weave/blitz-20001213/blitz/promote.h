/***********************************************************************
 * promote.h   Arithmetic type promotion trait class
 * Author: Todd Veldhuizen         (tveldhui@oonumerics.org)
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
 * $Id$
 *
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
 * Revision 1.5  2002/07/02 19:34:50  jcumming
 * Added BZ_BLITZ_SCOPE to promote_trait in BZ_PROMOTE macro definition
 * so that this macro works correctly outside the blitz namespace.
 *
 * Revision 1.4  2002/03/06 16:58:19  patricg
 *
 * typename replaced by _bz_typename
 *
 * Revision 1.3  2001/01/25 00:25:55  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_PROMOTE_H
#define BZ_PROMOTE_H

#include <blitz/blitz.h>

BZ_NAMESPACE(blitz)

#ifdef BZ_TEMPLATE_QUALIFIED_RETURN_TYPE
    #define BZ_PROMOTE(A,B) _bz_typename BZ_BLITZ_SCOPE(promote_trait)<A,B>::T_promote
#else
    #define BZ_PROMOTE(A,B) A
#endif

#if defined(BZ_PARTIAL_SPECIALIZATION) && !defined(BZ_DISABLE_NEW_PROMOTE)

/*
 * This compiler supports partial specialization, so type promotion
 * can be done the elegant way.  This implementation is after ideas
 * by Jean-Louis Leroy.
 */

template<class T>
struct precision_trait {
    enum { precisionRank = 0,
           knowPrecisionRank = 0 };
};

#define BZ_DECLARE_PRECISION(T,rank)          \
    template<>                                \
    struct precision_trait< T > {             \
        enum { precisionRank = rank,          \
           knowPrecisionRank = 1 };           \
    };

BZ_DECLARE_PRECISION(int,100)
BZ_DECLARE_PRECISION(unsigned int,200)
BZ_DECLARE_PRECISION(long,300)
BZ_DECLARE_PRECISION(unsigned long,400)
BZ_DECLARE_PRECISION(float,500)
BZ_DECLARE_PRECISION(double,600)
BZ_DECLARE_PRECISION(long double,700)

#ifdef BZ_HAVE_COMPLEX
BZ_DECLARE_PRECISION(complex<float>,800)
BZ_DECLARE_PRECISION(complex<double>,900)
BZ_DECLARE_PRECISION(complex<long double>,1000)
#endif

template<class T>
struct autopromote_trait {
    typedef T T_numtype;
};

#define BZ_DECLARE_AUTOPROMOTE(T1,T2)     \
    template<>                            \
    struct autopromote_trait<T1> {        \
      typedef T2 T_numtype;               \
    };

// These are the odd cases where small integer types
// are automatically promoted to int or unsigned int for
// arithmetic.
BZ_DECLARE_AUTOPROMOTE(bool, int)
BZ_DECLARE_AUTOPROMOTE(char, int)
BZ_DECLARE_AUTOPROMOTE(unsigned char, int)
BZ_DECLARE_AUTOPROMOTE(short int, int)
BZ_DECLARE_AUTOPROMOTE(short unsigned int, unsigned int)

template<class T1, class T2, int promoteToT1>
struct _bz_promote2 {
    typedef T1 T_promote;
};

template<class T1, class T2>
struct _bz_promote2<T1,T2,0> {
    typedef T2 T_promote;
};

template<class T1_orig, class T2_orig>
struct promote_trait {
    // Handle promotion of small integers to int/unsigned int
    typedef _bz_typename autopromote_trait<T1_orig>::T_numtype T1;
    typedef _bz_typename autopromote_trait<T2_orig>::T_numtype T2;

    // True if T1 is higher ranked
    enum {
      T1IsBetter =
        BZ_ENUM_CAST(precision_trait<T1>::precisionRank) >
          BZ_ENUM_CAST(precision_trait<T2>::precisionRank),

    // True if we know ranks for both T1 and T2
      knowBothRanks =
        BZ_ENUM_CAST(precision_trait<T1>::knowPrecisionRank)
      && BZ_ENUM_CAST(precision_trait<T2>::knowPrecisionRank),

    // True if we know T1 but not T2
      knowT1butNotT2 =  BZ_ENUM_CAST(precision_trait<T1>::knowPrecisionRank)
        && !(BZ_ENUM_CAST(precision_trait<T2>::knowPrecisionRank)),

    // True if we know T2 but not T1
      knowT2butNotT1 =  BZ_ENUM_CAST(precision_trait<T2>::knowPrecisionRank)
        && !(BZ_ENUM_CAST(precision_trait<T1>::knowPrecisionRank)),

    // True if T1 is bigger than T2
      T1IsLarger = sizeof(T1) >= sizeof(T2),

    // We know T1 but not T2: true
    // We know T2 but not T1: false
    // Otherwise, if T1 is bigger than T2: true
      defaultPromotion = knowT1butNotT2 ? _bz_false : 
         (knowT2butNotT1 ? _bz_true : T1IsLarger)
    };

    // If we have both ranks, then use them.
    // If we have only one rank, then use the unknown type.
    // If we have neither rank, then promote to the larger type.

    enum {
      promoteToT1 = (BZ_ENUM_CAST(knowBothRanks) ? BZ_ENUM_CAST(T1IsBetter)
        : BZ_ENUM_CAST(defaultPromotion)) ? 1 : 0
    };

    typedef _bz_typename _bz_promote2<T1,T2,promoteToT1>::T_promote T_promote;
};

#else  // !BZ_PARTIAL_SPECIALIZATION

  // No partial specialization -- have to do it the ugly way.
  #include <blitz/promote-old.h>

#endif // !BZ_PARTIAL_SPECIALIZATION

BZ_NAMESPACE_END

#endif // BZ_PROMOTE_H
