/***************************************************************************
 * blitz/tvecglobs.h     TinyVector global functions
 *
 * $Id$
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

#ifndef BZ_TVECGLOBS_H
#define BZ_TVECGLOBS_H

#ifndef BZ_META_METAPROG_H
 #include <blitz/meta/metaprog.h>
#endif

#ifndef BZ_NUMTRAIT_H
 #include <blitz/numtrait.h>
#endif

#include <blitz/tvcross.h>       // Cross products
#include <blitz/meta/dot.h>
#include <blitz/meta/product.h>
#include <blitz/meta/sum.h>

BZ_NAMESPACE(blitz)

template<typename T_numtype1, typename T_numtype2, int N_length>
inline BZ_PROMOTE(T_numtype1, T_numtype2)
dot(const TinyVector<T_numtype1, N_length>& a, 
    const TinyVector<T_numtype2, N_length>& b)
{
    return _bz_meta_vectorDot<N_length, 0>::f(a,b);
}

template<typename T_expr1, typename T_numtype2, int N_length>
inline BZ_PROMOTE(_bz_typename T_expr1::T_numtype, T_numtype2)
dot(_bz_VecExpr<T_expr1> a, const TinyVector<T_numtype2, N_length>& b)
{
    return _bz_meta_vectorDot<N_length, 0>::f_value_ref(a,b);
}

template<typename T_numtype1, typename T_expr2, int N_length>
inline BZ_PROMOTE(T_numtype1, _bz_typename T_expr2::T_numtype)
dot(const TinyVector<T_numtype1, N_length>& a, _bz_VecExpr<T_expr2> b)
{
    return _bz_meta_vectorDot<N_length, 0>::f_ref_value(a,b);
}

template<typename T_numtype1, int N_length>
inline BZ_SUMTYPE(T_numtype1)
product(const TinyVector<T_numtype1, N_length>& a)
{
    return _bz_meta_vectorProduct<N_length, 0>::f(a);
}

template<typename T_numtype, int N_length>
inline T_numtype
sum(const TinyVector<T_numtype, N_length>& a)
{
    return _bz_meta_vectorSum<N_length, 0>::f(a);
}

BZ_NAMESPACE_END

#endif // BZ_TVECGLOBS_H

