/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:25:28  ej
 * Looks like I need all the .cc files for blitz also
 *
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.5  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.4  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.3  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECPICK_CC
#define BZ_VECPICK_CC

#ifndef BZ_VECPICK_H
 #include <blitz/vecpick.h>
#endif

#ifndef BZ_UPDATE_H
 #include <blitz/update.h>
#endif

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif

#ifndef BZ_VECEXPR_H
 #include <blitz/vecexpr.h>
#endif

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif
BZ_NAMESPACE(blitz)

/*****************************************************************************
 * Assignment operators with vector expression operand
 */

template<class P_numtype> template<class P_expr, class P_updater>
inline
void VectorPick<P_numtype>::_bz_assign(P_expr expr, P_updater)
{
    BZPRECONDITION(expr.length(length()) == length());

    // If all vectors in the expression, plus the vector to which the
    // result is being assigned have unit stride, then avoid stride
    // calculations.
    if (_bz_hasFastAccess() && expr._bz_hasFastAccess())
    {
#ifndef BZ_PARTIAL_LOOP_UNROLL
        for (int i=0; i < length(); ++i)
            P_updater::update(vector_(index_(i)), expr._bz_fastAccess(i));
#else
        // Unwind the inner loop, five elements at a time.
        int leftover = length() % 5;

        int i=0;
        for (; i < leftover; ++i)
            P_updater::update(vector_(index_(i)), expr._bz_fastAccess(i));

        for (; i < length(); i += 5)
        {
            P_updater::update(vector_(index_(i)), expr._bz_fastAccess(i));
            P_updater::update(vector_(index_(i+1)), expr._bz_fastAccess(i+1));
            P_updater::update(vector_(index_(i+2)), expr._bz_fastAccess(i+2));
            P_updater::update(vector_(index_(i+3)), expr._bz_fastAccess(i+3));
            P_updater::update(vector_(index_(i+4)), expr._bz_fastAccess(i+4));
        }
#endif
    }
    else {
        // Not all unit strides -- have to access all the vector elements
        // as data_[i*stride_], which is slower.
        for (int i=0; i < length(); ++i)
            P_updater::update(vector_[index_[i]], expr[i]);
    }
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator=(_bz_VecExpr<P_expr> expr)
{
    BZPRECONDITION(expr.length(length()) == length());

    // If all vectors in the expression, plus the vector to which the
    // result is being assigned have unit stride, then avoid stride
    // calculations.
    if (_bz_hasFastAccess() && expr._bz_hasFastAccess())
    {
#ifndef BZ_PARTIAL_LOOP_UNROLL
        for (int i=0; i < length(); ++i)
            (*this)(i) = expr._bz_fastAccess(i);
#else
        // Unwind the inner loop, five elements at a time.
        int leftover = length() % 5;

        int i=0;
        for (; i < leftover; ++i)
            (*this)(i) = expr._bz_fastAccess(i);

        for (; i < length(); i += 5)
        {
            (*this)(i) = expr._bz_fastAccess(i);
            (*this)(i+1) = expr._bz_fastAccess(i+1);
            (*this)(i+2) = expr._bz_fastAccess(i+2);
            (*this)(i+3) = expr._bz_fastAccess(i+3);
            (*this)(i+4) = expr._bz_fastAccess(i+4);
        }
#endif
    }
    else {
        // Not all unit strides -- have to access all the vector elements
        // as data_[i*stride_], which is slower.
        for (int i=0; i < length(); ++i)
            (*this)[i] = expr[i];
    }
    return *this;
}


template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator+=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_plus_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator-=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_minus_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator*=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_multiply_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator/=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_divide_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator%=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_mod_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator^=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_xor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator&=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitand_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator|=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator>>=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftr_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator<<=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftl_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

/*****************************************************************************
 * Assignment operators with scalar operand
 */

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) = _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator+=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) += _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator-=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator*=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator/=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator%=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator^=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator&=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator|=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator>>=(int x)
{
    typedef _bz_VecExprConstant<int> T_expr;
    (*this) >>= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator<<=(int x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) <<= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

/*****************************************************************************
 * Assignment operators with vector operand
 */

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator=(const Vector<P_numtype2>& x)
{
    (*this) = _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator+=(const Vector<P_numtype2>& x)
{
    (*this) += _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator-=(const Vector<P_numtype2>& x)
{
    (*this) -= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator*=(const Vector<P_numtype2>& x)
{
    (*this) *= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator/=(const Vector<P_numtype2>& x)
{
    (*this) /= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator%=(const Vector<P_numtype2>& x)
{
    (*this) %= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator^=(const Vector<P_numtype2>& x)
{
    (*this) ^= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator&=(const Vector<P_numtype2>& x)
{
    (*this) &= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator|=(const Vector<P_numtype2>& x)
{
    (*this) |= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator<<=(const Vector<P_numtype2>& x)
{
    (*this) <<= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator>>=(const Vector<P_numtype2>& x)
{
    (*this) >>= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Range operand
 */

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator=(Range r)
{
    (*this) = _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator+=(Range r)
{
    (*this) += _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator-=(Range r)
{
    (*this) -= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator*=(Range r)
{
    (*this) *= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator/=(Range r)
{
    (*this) /= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator%=(Range r)
{
    (*this) %= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator^=(Range r)
{
    (*this) ^= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator&=(Range r)
{
    (*this) &= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator|=(Range r)
{
    (*this) |= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator>>=(Range r)
{
    (*this) >>= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator<<=(Range r)
{
    (*this) <<= _bz_VecExpr<Range>(r);
    return *this;
}

/*****************************************************************************
 * Assignment operators with VectorPick operand
 */

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator=(const 
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) = _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator+=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) += _bz_VecExpr<T_expr>(x.begin());
    return *this;
}


template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator-=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator*=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator/=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator%=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator^=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator&=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator|=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Random operand
 */

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator=(Random<P_distribution>& rand)
{
    (*this) = _bz_VecExpr<_bz_VecExprRandom<P_distribution> > 
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator+=(Random<P_distribution>& rand)
{
    (*this) += _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator-=(Random<P_distribution>& rand)
{
    (*this) -= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator*=(Random<P_distribution>& rand)
{
    (*this) *= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator/=(Random<P_distribution>& rand)
{
    (*this) /= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator%=(Random<P_distribution>& rand)
{
    (*this) %= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator^=(Random<P_distribution>& rand)
{
    (*this) ^= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator&=(Random<P_distribution>& rand)
{
    (*this) &= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator|=(Random<P_distribution>& rand)
{
    (*this) |= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

BZ_NAMESPACE_END

#endif // BZ_VECPICK_CC
