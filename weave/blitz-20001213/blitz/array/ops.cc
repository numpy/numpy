#ifndef BZ_ARRAYOPS_CC
#define BZ_ARRAYOPS_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/ops.cc> must be included via <blitz/array.h>
#endif

#ifndef BZ_UPDATE_H
 #include <blitz/update.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * Constant operands
 */

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>& Array<T_numtype,N_rank>::initialize(
    T_numtype x)
{
    (*this) = _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

#ifdef BZ_NEW_EXPRESSION_TEMPLATES

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype,N_rank>&
Array<T_numtype,N_rank>::operator=(const ETBase<T_expr>& expr)
{
    evaluate(static_cast<const T_expr&>(expr), 
        _bz_update<T_numtype, _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator=(const Array<T_numtype,N_rank>& x)
{
    (*this) = _bz_ArrayExpr<FastArrayIterator<T_numtype, N_rank> >
        (x.beginFast());
    return *this;
}

#define BZ_ARRAY_UPDATE(op,name) \
template<class T_numtype, int N_rank> \
template<class T> \
inline Array<T_numtype,N_rank>& \
Array<T_numtype,N_rank>::operator op(const T& expr) \
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

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>& 
Array<T_numtype,N_rank>::operator+=(T_numtype x)
{
    (*this) += _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator-=(T_numtype x)
{
    (*this) -= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator*=(T_numtype x)
{
    (*this) *= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator/=(T_numtype x)
{
    (*this) /= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator%=(T_numtype x)
{
    (*this) %= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator^=(T_numtype x)
{
    (*this) ^= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator&=(T_numtype x)
{
    (*this) &= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator|=(T_numtype x)
{
    (*this) |= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator>>=(T_numtype x)
{
    (*this) <<= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype,N_rank>::operator<<=(T_numtype x)
{
    (*this) <<= _bz_ArrayExpr<_bz_ArrayExprConstant<T_numtype> >(x);
    return *this;
}

/*
 * Array operands
 */

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator=(const Array<T_numtype,N_rank>& x)
{
    (*this) = _bz_ArrayExpr<FastArrayIterator<T_numtype, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator=(const Array<T_numtype2,N_rank>& x)
{
    (*this) = _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator+=(const Array<T_numtype2,N_rank>& x)
{
    (*this) += _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator-=(const Array<T_numtype2,N_rank>& x)
{
    (*this) -= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator*=(const Array<T_numtype2,N_rank>& x)
{
    (*this) *= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator/=(const Array<T_numtype2,N_rank>& x)
{
    (*this) /= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator%=(const Array<T_numtype2,N_rank>& x)
{
    (*this) %= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator^=(const Array<T_numtype2,N_rank>& x)
{
    (*this) ^= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator&=(const Array<T_numtype2,N_rank>& x)
{
    (*this) &= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator|=(const Array<T_numtype2,N_rank>& x)
{
    (*this) |= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator>>=(const Array<T_numtype2,N_rank>& x)
{
    (*this) >>= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_numtype2>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator<<=(const Array<T_numtype2,N_rank>& x)
{
    (*this) <<= _bz_ArrayExpr<FastArrayIterator<T_numtype2, N_rank> >(x.beginFast());
    return *this;
}

/*
 * Array expression operands
 */

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_update<T_numtype, _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator+=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_plus_update<T_numtype, _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator-=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_minus_update<T_numtype, 
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator*=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_multiply_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator/=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_divide_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator%=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_mod_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator^=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_xor_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator&=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_bitand_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator|=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_bitor_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator>>=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_shiftr_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::operator<<=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)
{
    evaluate(expr, _bz_shiftl_update<T_numtype,
        _bz_typename T_expr::T_numtype>());
    return *this;
}

#endif // BZ_NEW_EXPRESSION_TEMPLATES

BZ_NAMESPACE_END

#endif // BZ_ARRAYOPS_CC
