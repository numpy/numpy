#ifndef BZ_TINYVEC_CC
#define BZ_TINYVEC_CC

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

#ifndef BZ_VECPICK_H
 #include <blitz/vecpick.h>
#endif

#ifndef BZ_RANGE_H
 #include <blitz/range.h>
#endif

#include <blitz/meta/vecassign.h>

BZ_NAMESPACE(blitz)

template<class T_numtype, int N_length>
inline TinyVector<T_numtype, N_length>::TinyVector(T_numtype initValue)
{
    for (int i=0; i < N_length; ++i)
        data_[i] = initValue;
}

template<class T_numtype, int N_length>
inline TinyVector<T_numtype, N_length>::TinyVector(const 
    TinyVector<T_numtype, N_length>& x)
{
    for (int i=0; i < N_length; ++i)
        data_[i] = x.data_[i];
}

template<class T_numtype, int N_length> template<class P_expr, class P_updater>
inline
void TinyVector<T_numtype, N_length>::_bz_assign(P_expr expr, P_updater up)
{
    BZPRECHECK(expr.length(N_length) == N_length,
        "An expression with length " << expr.length(N_length)
        << " was assigned to a TinyVector<"
        << BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype)
        << "," << N_length << ">");

    if (expr._bz_hasFastAccess())
    {
        _bz_meta_vecAssign<N_length, 0>::fastAssign(*this, expr, up);
    }
    else {
        _bz_meta_vecAssign<N_length, 0>::assign(*this, expr, up);
    }
}

// Constructor added by Peter Nordlund (peter.nordlund@ind.af.se)
template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>::TinyVector(_bz_VecExpr<P_expr> expr) 
{
  _bz_assign(expr, _bz_update<P_numtype, _bz_typename P_expr::T_numtype>());
}

/*****************************************************************************
 * Assignment operators with vector expression operand
 */

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>& 
TinyVector<P_numtype, N_length>::operator=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_update<P_numtype, _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_plus_update<P_numtype, 
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_minus_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_multiply_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_divide_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_mod_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_xor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitand_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftl_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype, int N_length> template<class P_expr>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftr_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

/*****************************************************************************
 * Assignment operators with scalar operand
 */

template<class P_numtype, int N_length> 
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::initialize(P_numtype x)
{
#ifndef BZ_KCC_COPY_PROPAGATION_KLUDGE
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) = _bz_VecExpr<T_expr>(T_expr(x));
#else
    // Avoid using the copy propagation kludge for this simple
    // operation.
    for (int i=0; i < N_length; ++i)
        data_[i] = x;
#endif
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) += _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(int x)
{
    typedef _bz_VecExprConstant<int> T_expr;
    (*this) <<= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(int x)
{
    typedef _bz_VecExprConstant<int> T_expr;
    (*this) >>= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

/*****************************************************************************
 * Assignment operators with TinyVector operand
 */

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(const 
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) = _bz_VecExpr<_bz_typename 
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) += _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) -= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) *= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) /= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) %= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) ^= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) &= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) |= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) <<= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const
    TinyVector<P_numtype2, N_length>& x)
{
    (*this) >>= _bz_VecExpr<_bz_typename
        TinyVector<P_numtype2, N_length>::T_constIterator>(x.begin());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Vector operand
 */

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(const Vector<P_numtype2>& x)
{
    (*this) = x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const Vector<P_numtype2>& x)
{
    (*this) += x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const Vector<P_numtype2>& x)
{
    (*this) -= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const Vector<P_numtype2>& x)
{
    (*this) *= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const Vector<P_numtype2>& x)
{
    (*this) /= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const Vector<P_numtype2>& x)
{
    (*this) %= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const Vector<P_numtype2>& x)
{
    (*this) ^= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const Vector<P_numtype2>& x)
{
    (*this) &= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const Vector<P_numtype2>& x)
{
    (*this) |= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(const Vector<P_numtype2>& x)
{
    (*this) <<= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const Vector<P_numtype2>& x)
{
    (*this) >>= x._bz_asVecExpr();
    return *this;
}

/*****************************************************************************
 * Assignment operators with Range operand
 */

template<class P_numtype, int N_length> 
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(Range r)
{
    (*this) = r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(Range r)
{
    (*this) += r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(Range r)
{
    (*this) -= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(Range r)
{
    (*this) *= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(Range r)
{
    (*this) /= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(Range r)
{
    (*this) %= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(Range r)
{
    (*this) ^= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(Range r)
{
    (*this) &= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(Range r)
{
    (*this) |= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator<<=(Range r)
{
    (*this) <<= r._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(Range r)
{
    (*this) >>= r._bz_asVecExpr();
    return *this;
}

/*****************************************************************************
 * Assignment operators with VectorPick operand
 */

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator=(const VectorPick<P_numtype2>& x)
{
    (*this) = x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator+=(const VectorPick<P_numtype2>& x)
{
    (*this) += x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator-=(const VectorPick<P_numtype2>& x)
{
    (*this) -= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator*=(const VectorPick<P_numtype2>& x)
{
    (*this) *= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator/=(const VectorPick<P_numtype2>& x)
{
    (*this) /= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator%=(const VectorPick<P_numtype2>& x)
{
    (*this) %= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator^=(const VectorPick<P_numtype2>& x)
{
    (*this) ^= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator&=(const VectorPick<P_numtype2>& x)
{
    (*this) &= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator|=(const VectorPick<P_numtype2>& x)
{
    (*this) |= x._bz_asVecExpr();
    return *this;
}

template<class P_numtype, int N_length> template<class P_numtype2>
inline TinyVector<P_numtype, N_length>&
TinyVector<P_numtype, N_length>::operator>>=(const VectorPick<P_numtype2>& x)
{
    (*this) <<= x._bz_asVecExpr();
    return *this;
}

BZ_NAMESPACE_END

#endif // BZ_TINYVEC_CC
