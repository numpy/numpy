/***************************************************************************
 * blitz/array/ops.cc  Basic operators for arrays.
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
#ifndef BZ_ARRAYOPS_CC
#define BZ_ARRAYOPS_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/ops.cc> must be included via <blitz/array.h>
#endif

#include <blitz/update.h>

BZ_NAMESPACE(blitz)

/*
 * Constant operands
 */

template<typename P_numtype, int N_rank>
Array<P_numtype, N_rank>& Array<P_numtype,N_rank>::initialize(T_numtype x)
{
    (*this) = _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

#ifdef BZ_NEW_EXPRESSION_TEMPLATES

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype,N_rank>&
Array<P_numtype,N_rank>::operator=(const ETBase<T_expr>& expr)
{
    evaluate(expr.unwrap(), 
        _bz_update<T_numtype, _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator=(const Array<T_numtype,N_rank>& x)
{
    (*this) = _bz_ArrayExpr<FastArrayIterator<T_numtype, N_rank> >
        (x.beginFast());
    return *this;
}

#define BZ_ARRAY_UPDATE(op,name) \
template<typename P_numtype, int N_rank> \
template<typename T> \
inline Array<P_numtype,N_rank>& \
Array<P_numtype,N_rank>::operator op(const T& expr) \
{ \
    evaluate(_bz_typename asExpr<T>::T_expr(expr), \
      name<T_numtype, _bz_typename asExpr<T>::T_expr::T_numtype>()); \
    return *this; \
}

BZ_ARRAY_UPDATE(+=, _bz_plus_update)
BZ_ARRAY_UPDATE(-=, _bz_minus_update)
BZ_ARRAY_UPDATE(*=, _bz_multiply_update)
BZ_ARRAY_UPDATE(/=, _bz_divide_update)
BZ_ARRAY_UPDATE(%=, _bz_mod_update)
BZ_ARRAY_UPDATE(^=, _bz_xor_update)
BZ_ARRAY_UPDATE(&=, _bz_bitand_update)
BZ_ARRAY_UPDATE(|=, _bz_bitor_update)
BZ_ARRAY_UPDATE(<<=, _bz_shiftl_update)
BZ_ARRAY_UPDATE(>>=, _bz_shiftr_update)

#else

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>& 
Array<P_numtype,N_rank>::operator+=(T_numtype x)
{
    (*this) += _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator-=(T_numtype x)
{
    (*this) -= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator*=(T_numtype x)
{
    (*this) *= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator/=(T_numtype x)
{
    (*this) /= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator%=(T_numtype x)
{
    (*this) %= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator^=(T_numtype x)
{
    (*this) ^= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator&=(T_numtype x)
{
    (*this) &= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator|=(T_numtype x)
{
    (*this) |= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator>>=(T_numtype x)
{
    (*this) <<= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype,N_rank>::operator<<=(T_numtype x)
{
    (*this) <<= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

/*
 * Array operands
 */

template<typename P_numtype, int N_rank>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator=(const Array<T_numtype,N_rank>& x)
{
    (*this) = _bz_ArrayExpr<FastArrayIterator<T_numtype, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator=(const Array<P_numtype2,N_rank>& x)
{
    (*this) = _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator+=(const Array<P_numtype2,N_rank>& x)
{
    (*this) += _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator-=(const Array<P_numtype2,N_rank>& x)
{
    (*this) -= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator*=(const Array<P_numtype2,N_rank>& x)
{
    (*this) *= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator/=(const Array<P_numtype2,N_rank>& x)
{
    (*this) /= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator%=(const Array<P_numtype2,N_rank>& x)
{
    (*this) %= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator^=(const Array<P_numtype2,N_rank>& x)
{
    (*this) ^= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator&=(const Array<P_numtype2,N_rank>& x)
{
    (*this) &= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator|=(const Array<P_numtype2,N_rank>& x)
{
    (*this) |= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator>>=(const Array<P_numtype2,N_rank>& x)
{
    (*this) >>= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator<<=(const Array<P_numtype2,N_rank>& x)
{
    (*this) <<= _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
    return *this;
}

/*
 * Array expression operands
 */

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_update<T_numtype, _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator+=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_plus_update<T_numtype, _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator-=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_minus_update<T_numtype, 
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator*=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_multiply_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator/=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_divide_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator%=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_mod_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator^=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_xor_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator&=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_bitand_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator|=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_bitor_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator>>=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_shiftr_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<typename P_numtype, int N_rank> template<typename T_expr>
inline Array<P_numtype, N_rank>&
Array<P_numtype, N_rank>::operator<<=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_shiftl_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

#endif // BZ_NEW_EXPRESSION_TEMPLATES

BZ_NAMESPACE_END

#endif // BZ_ARRAYOPS_CC
