// -*- C++ -*-
/***************************************************************************
 * blitz/vecexpr.h      Vector<P_numtype> expression templates
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


#ifndef BZ_VECEXPR_H
#define BZ_VECEXPR_H

#include <blitz/vector.h>
#include <blitz/applics.h>
#include <blitz/meta/metaprog.h>
#include <blitz/vecexprwrap.h>           // _bz_VecExpr wrapper class

BZ_NAMESPACE(blitz)

template<typename P_expr1, typename P_expr2, typename P_op>
class _bz_VecExprOp {

public:
    typedef P_expr1 T_expr1;
    typedef P_expr2 T_expr2;
    typedef _bz_typename T_expr1::T_numtype T_numtype1;
    typedef _bz_typename T_expr2::T_numtype T_numtype2;
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;
    typedef P_op    T_op;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_VecExprOp(T_expr1 a, T_expr2 b)
        : iter1_(a), iter2_(b)
    { }
#else
    _bz_VecExprOp(const T_expr1& a, const T_expr2& b)
        : iter1_(a), iter2_(b)
    { }
#endif

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprOp(const _bz_VecExprOp<P_expr1, P_expr2, P_op>& x)
        : iter1_(x.iter1_), iter2_(x.iter2_)
    { }
#endif

    T_numtype operator[](int i) const
    { return T_op::apply(iter1_[i], iter2_[i]); }

    T_numtype operator()(int i) const
    { return T_op::apply(iter1_(i), iter2_(i)); }

    int length(int recommendedLength) const
    { 
        BZPRECONDITION(iter2_.length(recommendedLength) == 
            iter1_.length(recommendedLength));
        return iter1_.length(recommendedLength); 
    }

    static const int 
        _bz_staticLengthCount = P_expr1::_bz_staticLengthCount
                              + P_expr2::_bz_staticLengthCount,
        _bz_dynamicLengthCount = P_expr1::_bz_dynamicLengthCount
                               + P_expr2::_bz_dynamicLengthCount,
        _bz_staticLength =
            (P_expr1::_bz_staticLength > P_expr2::_bz_staticLength) ?
            P_expr1::_bz_staticLength : P_expr2::_bz_staticLength;

//      _bz_meta_max<P_expr1::_bz_staticLength, P_expr2::_bz_staticLength>::max 

    int _bz_suggestLength() const
    {
        int length1 = iter1_._bz_suggestLength();
        if (length1 != 0)
            return length1;
        return iter2_._bz_suggestLength();
    }

    bool _bz_hasFastAccess() const
    { return iter1_._bz_hasFastAccess() && iter2_._bz_hasFastAccess(); }

    T_numtype _bz_fastAccess(int i) const
    { 
        return T_op::apply(iter1_._bz_fastAccess(i),
            iter2_._bz_fastAccess(i)); 
    }
    
private:
    _bz_VecExprOp();

    T_expr1 iter1_;
    T_expr2 iter2_;
};

template<typename P_expr, typename P_unaryOp>
class _bz_VecExprUnaryOp {

public:
    typedef P_expr T_expr;
    typedef P_unaryOp T_unaryOp;
    typedef _bz_typename T_unaryOp::T_numtype T_numtype;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_VecExprUnaryOp(T_expr iter)
        : iter_(iter)
    { }
#else
    _bz_VecExprUnaryOp(const T_expr& iter)
        : iter_(iter)
    { }
#endif

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprUnaryOp(const _bz_VecExprUnaryOp<P_expr, P_unaryOp>& x)
        : iter_(x.iter_)
    { }
#endif

    T_numtype operator[](int i) const
    { return T_unaryOp::apply(iter_[i]); }

    T_numtype operator()(int i) const
    { return T_unaryOp::apply(iter_(i)); }

    int length(int recommendedLength) const
    { return iter_.length(recommendedLength); }

    static const int
        _bz_staticLengthCount = P_expr::_bz_staticLengthCount,
        _bz_dynamicLengthCount = P_expr::_bz_dynamicLengthCount,
        _bz_staticLength = P_expr::_bz_staticLength;

    int _bz_suggestLength() const
    { return iter_._bz_suggestLength(); }

    bool _bz_hasFastAccess() const
    { return iter_._bz_hasFastAccess(); }

    T_numtype _bz_fastAccess(int i) const
    { return T_unaryOp::apply(iter_._bz_fastAccess(i)); }

private:
    _bz_VecExprUnaryOp() { }

    T_expr iter_;    
};

template<typename P_numtype>
class _bz_VecExprConstant {
public:
    typedef P_numtype T_numtype;

    _bz_VecExprConstant(P_numtype value)
        : value_(BZ_NO_PROPAGATE(value))
    { 
    }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprConstant(const _bz_VecExprConstant<P_numtype>& x)
        : value_(x.value_)
    { }
#endif

    T_numtype operator[](int) const
    { return value_; }

    T_numtype operator()(int) const
    { return value_; }

    int length(int recommendedLength) const
    { return recommendedLength; }

    static const int
        _bz_staticLengthCount = 0,
        _bz_dynamicLengthCount = 0,
        _bz_staticLength = 0;

    int _bz_suggestLength() const
    { return 0; }

    bool _bz_hasFastAccess() const
    { return 1; }

    T_numtype _bz_fastAccess(int) const
    { return value_; }

private:

    _bz_VecExprConstant() { }

    T_numtype value_;
};

BZ_NAMESPACE_END


#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif


BZ_NAMESPACE(blitz)

// Some miscellaneous operators that don't seem to belong anywhere else.

template<typename P_expr>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr>, 
    _bz_negate<_bz_typename P_expr::T_numtype> > >
operator-(_bz_VecExpr<P_expr> a)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr>,
        _bz_negate<_bz_typename P_expr::T_numtype> > T_expr;
    return _bz_VecExpr<T_expr>(T_expr(a));
}

template<typename P_numtype>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype>,
    _bz_negate<P_numtype> > >
operator-(const Vector<P_numtype>& a)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype>,
        _bz_negate<P_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(a.beginFast()));
}

inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range, _bz_negate<Range::T_numtype> > >
operator-(Range r)
{
    typedef _bz_VecExprUnaryOp<Range, _bz_negate<Range::T_numtype> > T_expr;
    
    return _bz_VecExpr<T_expr>(T_expr(r));
}

template<typename P_numtype>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype>,
    _bz_negate<P_numtype> > >
operator-(const VectorPick<P_numtype>& a)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype>,
        _bz_negate<P_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(a.beginFast()));
}

template<typename P_numtype, int N_length>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype,N_length>,
    _bz_negate<P_numtype> > >
operator-(const TinyVector<P_numtype,N_length>& a)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype,N_length>,
        _bz_negate<P_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(a.beginFast()));
}

BZ_NAMESPACE_END

#endif // BZ_VECEXPR_H
