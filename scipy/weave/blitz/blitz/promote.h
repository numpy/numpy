// -*- C++ -*-
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
 ***************************************************************************/

#ifndef BZ_PROMOTE_H
#define BZ_PROMOTE_H

#include <blitz/blitz.h>

BZ_NAMESPACE(blitz)

#ifdef BZ_HAVE_TEMPLATE_QUALIFIED_RETURN_TYPE
    #define BZ_PROMOTE(A,B) _bz_typename BZ_BLITZ_SCOPE(promote_trait)<A,B>::T_promote
#else
    #define BZ_PROMOTE(A,B) A
#endif

#if defined(BZ_HAVE_PARTIAL_SPECIALIZATION) && !defined(BZ_DISABLE_NEW_PROMOTE)

/*
 * This compiler supports partial specialization, so type promotion
 * can be done the elegant way.  This implementation is after ideas
 * by Jean-Louis Leroy.
 */

template<typename T>
struct precision_trait {
    static const int precisionRank = 0;
    static const bool knowPrecisionRank = false;
};

#define BZ_DECLARE_PRECISION(T,rank)                  \
    template<>                                        \
    struct precision_trait< T > {                     \
        static const int precisionRank = rank;        \
        static const bool knowPrecisionRank = true;   \
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

template<typename T>
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

template<typename T1, typename T2, bool promoteToT1>
struct _bz_promote2 {
    typedef T1 T_promote;
};

template<typename T1, typename T2>
struct _bz_promote2<T1,T2,false> {
    typedef T2 T_promote;
};

template<typename T1_orig, typename T2_orig>
struct promote_trait {
    // Handle promotion of small integers to int/unsigned int
    typedef _bz_typename autopromote_trait<T1_orig>::T_numtype T1;
    typedef _bz_typename autopromote_trait<T2_orig>::T_numtype T2;

    // True if T1 is higher ranked
    static const bool
        T1IsBetter =
            precision_trait<T1>::precisionRank >
            precision_trait<T2>::precisionRank;

    // True if we know ranks for both T1 and T2
    static const bool
        knowBothRanks =
            precision_trait<T1>::knowPrecisionRank && 
            precision_trait<T2>::knowPrecisionRank;

    // True if we know T1 but not T2
    static const bool
        knowT1butNotT2 =  
            precision_trait<T1>::knowPrecisionRank && 
            !precision_trait<T2>::knowPrecisionRank;

    // True if we know T2 but not T1
    static const bool
        knowT2butNotT1 =  
            precision_trait<T2>::knowPrecisionRank && 
            !precision_trait<T1>::knowPrecisionRank;

    // True if T1 is bigger than T2
    static const bool
        T1IsLarger = sizeof(T1) >= sizeof(T2);

    // We know T1 but not T2: false
    // We know T2 but not T1: true
    // Otherwise, if T1 is bigger than T2: true
//     static const bool
//         defaultPromotion = knowT1butNotT2 ? false : 
//             (knowT2butNotT1 ? true : T1IsLarger);

    // If we have both ranks, then use them.
    // If we have only one rank, then use the unknown type.
    // If we have neither rank, then promote to the larger type.
    static const bool
        promoteToT1 = knowBothRanks ? T1IsBetter : (knowT1butNotT2 ? false : 
            (knowT2butNotT1 ? true : T1IsLarger));
//     static const bool
//         promoteToT1 = knowBothRanks ? T1IsBetter : defaultPromotion;

    typedef _bz_typename _bz_promote2<T1,T2,promoteToT1>::T_promote T_promote;
};

#else  // !BZ_HAVE_PARTIAL_SPECIALIZATION

  // No partial specialization -- have to do it the ugly way.
  #include <blitz/promote-old.h>

#endif // !BZ_HAVE_PARTIAL_SPECIALIZATION

BZ_NAMESPACE_END

#endif // BZ_PROMOTE_H
