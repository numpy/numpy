/***************************************************************************
 * blitz/vecuops.cc	Expression templates for vectors, unary functions
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
 ***************************************************************************
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:25:28  ej
 * Looks like I need all the .cc files for blitz also
 *
 * Revision 1.1.1.1  2000/06/19 12:26:10  tveldhui
 * Imported sources
 *
 */ 

// Generated source file.  Do not edit. 
// genvecuops.cpp Dec  5 1998 17:06:25

#ifndef BZ_VECUOPS_CC
#define BZ_VECUOPS_CC

#ifndef BZ_VECEXPR_H
 #error <blitz/vecuops.cc> must be included via <blitz/vecexpr.h>
#endif // BZ_VECEXPR_H

BZ_NAMESPACE(blitz)

/****************************************************************************
 * abs
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_abs<P_numtype1> > >
abs(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_abs<_bz_typename P_expr1::T_numtype> > >
abs(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_abs<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_abs<P_numtype1> > >
abs(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_abs<int> > >
abs(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_abs<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_abs<P_numtype1> > >
abs(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * acos
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_acos<P_numtype1> > >
acos(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_acos<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_acos<_bz_typename P_expr1::T_numtype> > >
acos(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_acos<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_acos<P_numtype1> > >
acos(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_acos<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_acos<int> > >
acos(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_acos<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_acos<P_numtype1> > >
acos(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_acos<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * acosh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_acosh<P_numtype1> > >
acosh(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_acosh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_acosh<_bz_typename P_expr1::T_numtype> > >
acosh(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_acosh<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_acosh<P_numtype1> > >
acosh(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_acosh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_acosh<int> > >
acosh(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_acosh<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_acosh<P_numtype1> > >
acosh(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_acosh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * asin
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_asin<P_numtype1> > >
asin(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_asin<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_asin<_bz_typename P_expr1::T_numtype> > >
asin(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_asin<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_asin<P_numtype1> > >
asin(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_asin<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_asin<int> > >
asin(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_asin<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_asin<P_numtype1> > >
asin(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_asin<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * asinh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_asinh<P_numtype1> > >
asinh(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_asinh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_asinh<_bz_typename P_expr1::T_numtype> > >
asinh(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_asinh<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_asinh<P_numtype1> > >
asinh(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_asinh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_asinh<int> > >
asinh(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_asinh<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_asinh<P_numtype1> > >
asinh(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_asinh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * atan
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_atan<P_numtype1> > >
atan(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_atan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_atan<_bz_typename P_expr1::T_numtype> > >
atan(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_atan<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_atan<P_numtype1> > >
atan(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_atan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_atan<int> > >
atan(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_atan<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_atan<P_numtype1> > >
atan(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_atan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * atanh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_atanh<P_numtype1> > >
atanh(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_atanh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_atanh<_bz_typename P_expr1::T_numtype> > >
atanh(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_atanh<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_atanh<P_numtype1> > >
atanh(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_atanh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_atanh<int> > >
atanh(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_atanh<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_atanh<P_numtype1> > >
atanh(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_atanh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * _class
 ****************************************************************************/

#ifdef BZ_HAVE_SYSV_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz__class<P_numtype1> > >
_class(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz__class<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz__class<_bz_typename P_expr1::T_numtype> > >
_class(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz__class<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz__class<P_numtype1> > >
_class(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz__class<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz__class<int> > >
_class(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz__class<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz__class<P_numtype1> > >
_class(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz__class<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * cbrt
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_cbrt<P_numtype1> > >
cbrt(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_cbrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_cbrt<_bz_typename P_expr1::T_numtype> > >
cbrt(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_cbrt<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_cbrt<P_numtype1> > >
cbrt(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_cbrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_cbrt<int> > >
cbrt(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_cbrt<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_cbrt<P_numtype1> > >
cbrt(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_cbrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * ceil
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_ceil<P_numtype1> > >
ceil(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_ceil<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_ceil<_bz_typename P_expr1::T_numtype> > >
ceil(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_ceil<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_ceil<P_numtype1> > >
ceil(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_ceil<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_ceil<int> > >
ceil(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_ceil<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_ceil<P_numtype1> > >
ceil(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_ceil<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * cos
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_cos<P_numtype1> > >
cos(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_cos<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_cos<_bz_typename P_expr1::T_numtype> > >
cos(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_cos<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_cos<P_numtype1> > >
cos(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_cos<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_cos<int> > >
cos(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_cos<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_cos<P_numtype1> > >
cos(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_cos<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * cosh
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_cosh<P_numtype1> > >
cosh(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_cosh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_cosh<_bz_typename P_expr1::T_numtype> > >
cosh(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_cosh<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_cosh<P_numtype1> > >
cosh(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_cosh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_cosh<int> > >
cosh(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_cosh<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_cosh<P_numtype1> > >
cosh(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_cosh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * exp
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_exp<P_numtype1> > >
exp(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_exp<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_exp<_bz_typename P_expr1::T_numtype> > >
exp(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_exp<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_exp<P_numtype1> > >
exp(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_exp<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_exp<int> > >
exp(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_exp<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_exp<P_numtype1> > >
exp(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_exp<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * expm1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_expm1<P_numtype1> > >
expm1(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_expm1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_expm1<_bz_typename P_expr1::T_numtype> > >
expm1(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_expm1<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_expm1<P_numtype1> > >
expm1(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_expm1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_expm1<int> > >
expm1(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_expm1<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_expm1<P_numtype1> > >
expm1(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_expm1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * erf
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_erf<P_numtype1> > >
erf(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_erf<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_erf<_bz_typename P_expr1::T_numtype> > >
erf(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_erf<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_erf<P_numtype1> > >
erf(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_erf<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_erf<int> > >
erf(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_erf<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_erf<P_numtype1> > >
erf(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_erf<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * erfc
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_erfc<P_numtype1> > >
erfc(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_erfc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_erfc<_bz_typename P_expr1::T_numtype> > >
erfc(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_erfc<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_erfc<P_numtype1> > >
erfc(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_erfc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_erfc<int> > >
erfc(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_erfc<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_erfc<P_numtype1> > >
erfc(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_erfc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * fabs
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_abs<P_numtype1> > >
fabs(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_abs<_bz_typename P_expr1::T_numtype> > >
fabs(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_abs<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_abs<P_numtype1> > >
fabs(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_abs<int> > >
fabs(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_abs<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_abs<P_numtype1> > >
fabs(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * floor
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_floor<P_numtype1> > >
floor(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_floor<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_floor<_bz_typename P_expr1::T_numtype> > >
floor(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_floor<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_floor<P_numtype1> > >
floor(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_floor<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_floor<int> > >
floor(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_floor<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_floor<P_numtype1> > >
floor(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_floor<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * ilogb
 ****************************************************************************/

#ifdef BZ_HAVE_SYSV_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_ilogb<P_numtype1> > >
ilogb(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_ilogb<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_ilogb<_bz_typename P_expr1::T_numtype> > >
ilogb(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_ilogb<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_ilogb<P_numtype1> > >
ilogb(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_ilogb<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_ilogb<int> > >
ilogb(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_ilogb<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_ilogb<P_numtype1> > >
ilogb(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_ilogb<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * isnan
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_isnan<P_numtype1> > >
isnan(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_isnan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_isnan<_bz_typename P_expr1::T_numtype> > >
isnan(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_isnan<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_isnan<P_numtype1> > >
isnan(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_isnan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_isnan<int> > >
isnan(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_isnan<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_isnan<P_numtype1> > >
isnan(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_isnan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * itrunc
 ****************************************************************************/

#ifdef BZ_HAVE_SYSV_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_itrunc<P_numtype1> > >
itrunc(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_itrunc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_itrunc<_bz_typename P_expr1::T_numtype> > >
itrunc(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_itrunc<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_itrunc<P_numtype1> > >
itrunc(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_itrunc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_itrunc<int> > >
itrunc(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_itrunc<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_itrunc<P_numtype1> > >
itrunc(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_itrunc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * j0
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_j0<P_numtype1> > >
j0(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_j0<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_j0<_bz_typename P_expr1::T_numtype> > >
j0(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_j0<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_j0<P_numtype1> > >
j0(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_j0<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_j0<int> > >
j0(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_j0<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_j0<P_numtype1> > >
j0(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_j0<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * j1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_j1<P_numtype1> > >
j1(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_j1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_j1<_bz_typename P_expr1::T_numtype> > >
j1(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_j1<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_j1<P_numtype1> > >
j1(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_j1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_j1<int> > >
j1(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_j1<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_j1<P_numtype1> > >
j1(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_j1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * lgamma
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_lgamma<P_numtype1> > >
lgamma(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_lgamma<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_lgamma<_bz_typename P_expr1::T_numtype> > >
lgamma(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_lgamma<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_lgamma<P_numtype1> > >
lgamma(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_lgamma<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_lgamma<int> > >
lgamma(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_lgamma<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_lgamma<P_numtype1> > >
lgamma(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_lgamma<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * log
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_log<P_numtype1> > >
log(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_log<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_log<_bz_typename P_expr1::T_numtype> > >
log(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_log<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_log<P_numtype1> > >
log(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_log<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_log<int> > >
log(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_log<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_log<P_numtype1> > >
log(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_log<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * logb
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_logb<P_numtype1> > >
logb(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_logb<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_logb<_bz_typename P_expr1::T_numtype> > >
logb(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_logb<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_logb<P_numtype1> > >
logb(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_logb<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_logb<int> > >
logb(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_logb<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_logb<P_numtype1> > >
logb(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_logb<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * log1p
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_log1p<P_numtype1> > >
log1p(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_log1p<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_log1p<_bz_typename P_expr1::T_numtype> > >
log1p(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_log1p<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_log1p<P_numtype1> > >
log1p(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_log1p<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_log1p<int> > >
log1p(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_log1p<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_log1p<P_numtype1> > >
log1p(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_log1p<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * log10
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_log10<P_numtype1> > >
log10(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_log10<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_log10<_bz_typename P_expr1::T_numtype> > >
log10(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_log10<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_log10<P_numtype1> > >
log10(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_log10<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_log10<int> > >
log10(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_log10<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_log10<P_numtype1> > >
log10(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_log10<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * nearest
 ****************************************************************************/

#ifdef BZ_HAVE_SYSV_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_nearest<P_numtype1> > >
nearest(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_nearest<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_nearest<_bz_typename P_expr1::T_numtype> > >
nearest(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_nearest<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_nearest<P_numtype1> > >
nearest(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_nearest<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_nearest<int> > >
nearest(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_nearest<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_nearest<P_numtype1> > >
nearest(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_nearest<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * rint
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_rint<P_numtype1> > >
rint(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_rint<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_rint<_bz_typename P_expr1::T_numtype> > >
rint(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_rint<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_rint<P_numtype1> > >
rint(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_rint<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_rint<int> > >
rint(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_rint<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_rint<P_numtype1> > >
rint(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_rint<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * rsqrt
 ****************************************************************************/

#ifdef BZ_HAVE_SYSV_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_rsqrt<P_numtype1> > >
rsqrt(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_rsqrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_rsqrt<_bz_typename P_expr1::T_numtype> > >
rsqrt(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_rsqrt<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_rsqrt<P_numtype1> > >
rsqrt(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_rsqrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_rsqrt<int> > >
rsqrt(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_rsqrt<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_rsqrt<P_numtype1> > >
rsqrt(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_rsqrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * sin
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_sin<P_numtype1> > >
sin(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_sin<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_sin<_bz_typename P_expr1::T_numtype> > >
sin(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_sin<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_sin<P_numtype1> > >
sin(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_sin<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_sin<int> > >
sin(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_sin<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_sin<P_numtype1> > >
sin(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_sin<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * sinh
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_sinh<P_numtype1> > >
sinh(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_sinh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_sinh<_bz_typename P_expr1::T_numtype> > >
sinh(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_sinh<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_sinh<P_numtype1> > >
sinh(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_sinh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_sinh<int> > >
sinh(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_sinh<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_sinh<P_numtype1> > >
sinh(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_sinh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * sqr
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_sqr<P_numtype1> > >
sqr(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_sqr<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_sqr<_bz_typename P_expr1::T_numtype> > >
sqr(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_sqr<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_sqr<P_numtype1> > >
sqr(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_sqr<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_sqr<int> > >
sqr(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_sqr<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_sqr<P_numtype1> > >
sqr(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_sqr<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * sqrt
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_sqrt<P_numtype1> > >
sqrt(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_sqrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_sqrt<_bz_typename P_expr1::T_numtype> > >
sqrt(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_sqrt<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_sqrt<P_numtype1> > >
sqrt(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_sqrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_sqrt<int> > >
sqrt(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_sqrt<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_sqrt<P_numtype1> > >
sqrt(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_sqrt<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * tan
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_tan<P_numtype1> > >
tan(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_tan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_tan<_bz_typename P_expr1::T_numtype> > >
tan(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_tan<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_tan<P_numtype1> > >
tan(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_tan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_tan<int> > >
tan(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_tan<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_tan<P_numtype1> > >
tan(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_tan<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * tanh
 ****************************************************************************/

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_tanh<P_numtype1> > >
tanh(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_tanh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_tanh<_bz_typename P_expr1::T_numtype> > >
tanh(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_tanh<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_tanh<P_numtype1> > >
tanh(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_tanh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_tanh<int> > >
tanh(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_tanh<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_tanh<P_numtype1> > >
tanh(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_tanh<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


/****************************************************************************
 * uitrunc
 ****************************************************************************/

#ifdef BZ_HAVE_SYSV_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_uitrunc<P_numtype1> > >
uitrunc(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_uitrunc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_uitrunc<_bz_typename P_expr1::T_numtype> > >
uitrunc(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_uitrunc<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_uitrunc<P_numtype1> > >
uitrunc(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_uitrunc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_uitrunc<int> > >
uitrunc(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_uitrunc<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_uitrunc<P_numtype1> > >
uitrunc(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_uitrunc<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * y0
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_y0<P_numtype1> > >
y0(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_y0<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_y0<_bz_typename P_expr1::T_numtype> > >
y0(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_y0<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_y0<P_numtype1> > >
y0(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_y0<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_y0<int> > >
y0(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_y0<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_y0<P_numtype1> > >
y0(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_y0<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

/****************************************************************************
 * y1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
    _bz_y1<P_numtype1> > >
y1(const Vector<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype1>,
        _bz_y1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
    _bz_y1<_bz_typename P_expr1::T_numtype> > >
y1(_bz_VecExpr<P_expr1> d1)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr1>,
        _bz_y1<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
    _bz_y1<P_numtype1> > >
y1(const VectorPick<P_numtype1>& d1)
{
    typedef _bz_VecExprUnaryOp<VectorPickIterConst<P_numtype1>,
        _bz_y1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}


inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range,
    _bz_y1<int> > >
y1(Range d1)
{
    typedef _bz_VecExprUnaryOp<Range,
        _bz_y1<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1));
}

template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
    _bz_y1<P_numtype1> > >
y1(const TinyVector<P_numtype1, N_length1>& d1)
{
    typedef _bz_VecExprUnaryOp<TinyVectorIterConst<P_numtype1, N_length1>,
        _bz_y1<P_numtype1> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin()));
}

#endif

BZ_NAMESPACE_END

#endif
