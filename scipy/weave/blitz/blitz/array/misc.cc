/***************************************************************************
 * blitz/array/misc.cc  Miscellaneous operators for arrays
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
 ****************************************************************************/
#ifndef BZ_ARRAYMISC_CC
#define BZ_ARRAYMISC_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/misc.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

#define BZ_ARRAY_DECLARE_UOP(fn, fnobj)                                \
template<typename T_numtype, int N_rank>                                  \
inline                                                                 \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>, \
    fnobj<T_numtype> > >                                               \
fn(const Array<T_numtype,N_rank>& array)                               \
{                                                                      \
    return _bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,   \
        fnobj<T_numtype> >(array.beginFast());                         \
}                                                                      \
                                                                       \
template<typename T_expr>                                                 \
inline                                                                 \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,              \
    fnobj<_bz_typename T_expr::T_numtype> > >                          \
fn(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)                              \
{                                                                      \
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,                 \
        fnobj<_bz_typename T_expr::T_numtype> >(expr);                 \
}                                                                      
                                                                       
BZ_ARRAY_DECLARE_UOP(operator!, LogicalNot)
BZ_ARRAY_DECLARE_UOP(operator~, BitwiseNot)
BZ_ARRAY_DECLARE_UOP(operator-, Negate)

/*
 * cast() functions, for explicit type casting
 */

template<typename T_numtype, int N_rank, typename T_cast>
inline                                                                 
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,   
    Cast<T_numtype, T_cast> > >
cast(const Array<T_numtype,N_rank>& array, T_cast)
{                                                                      
    return _bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,      
        Cast<T_numtype,T_cast> >(array.beginFast());                            
}                                                                      
                                                                       
template<typename T_expr, typename T_cast>
inline                                                                 
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,              
    Cast<_bz_typename T_expr::T_numtype,T_cast> > >                          
cast(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr, T_cast)
{                                                                      
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,                 
        Cast<_bz_typename T_expr::T_numtype,T_cast> >(expr);                 
}                                                                      
                                                                       
BZ_NAMESPACE_END

#endif // BZ_ARRAYMISC_CC

