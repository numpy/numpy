/***********************************************************************
 * promote.h   Arithmetic type promotion trait class
 * Author: Todd Veldhuizen         (tveldhui@oonumerics.org)
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 */

#ifndef BZ_PROMOTE_H
#define BZ_PROMOTE_H

#include <blitz/blitz.h>

BZ_NAMESPACE(blitz)

#ifdef BZ_TEMPLATE_QUALIFIED_RETURN_TYPE
    #define BZ_PROMOTE(A,B) _bz_typename promote_trait<A,B>::T_promote
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

    typedef typename _bz_promote2<T1,T2,promoteToT1>::T_promote T_promote;
};

#else  // !BZ_PARTIAL_SPECIALIZATION

  // No partial specialization -- have to do it the ugly way.
  #include <blitz/promote-old.h>

#endif // !BZ_PARTIAL_SPECIALIZATION

BZ_NAMESPACE_END

#endif // BZ_PROMOTE_H
