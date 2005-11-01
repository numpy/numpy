/***************************************************************************
 * blitz/../vecbfn.cc	Vector expression binary functions (2 operands)
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
 * Suggestions:          blitz-suggest@cybervision.com
 * Bugs:                 blitz-bugs@cybervision.com
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://seurat.uwaterloo.ca/blitz/
 *
 ***************************************************************************
 *
 */ 

// Generated source file.  Do not edit. 
// genvecbfn.cpp Oct  6 2005 15:58:50

#ifndef BZ_VECBFN_CC
#define BZ_VECBFN_CC

#ifndef BZ_VECEXPR_H
 #error <blitz/vecbfn.cc> must be included via <blitz/vecexpr.h>
#endif

BZ_NAMESPACE(blitz)

/****************************************************************************
 * Minimum Operators
 ****************************************************************************/

// Vector<P_numtype1> min Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// Vector<P_numtype1> min _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<P_numtype1, typename P_expr2::T_numtype > > >
min(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// Vector<P_numtype1> min VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// Vector<P_numtype1> min Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Min<P_numtype1, int > > >
min(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Min<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// Vector<P_numtype1> min TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// Vector<P_numtype1> min int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Min<P_numtype1, int > > >
min(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Min<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> min float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Min<P_numtype1, float > > >
min(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Min<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> min double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Min<P_numtype1, double > > >
min(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Min<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> min long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Min<P_numtype1, long double > > >
min(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Min<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> min complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Min<P_numtype1, complex<T2>  > > >
min(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Min<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> min Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<typename P_expr1::T_numtype, P_numtype2 > > >
min(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// _bz_VecExpr<P_expr1> min _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
min(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> min VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<typename P_expr1::T_numtype, P_numtype2 > > >
min(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// _bz_VecExpr<P_expr1> min Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Min<typename P_expr1::T_numtype, int > > >
min(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Min<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> min TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<typename P_expr1::T_numtype, P_numtype2 > > >
min(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// _bz_VecExpr<P_expr1> min int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Min<typename P_expr1::T_numtype, int > > >
min(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Min<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> min float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Min<typename P_expr1::T_numtype, float > > >
min(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Min<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> min double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Min<typename P_expr1::T_numtype, double > > >
min(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Min<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> min long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Min<typename P_expr1::T_numtype, long double > > >
min(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Min<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> min complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Min<typename P_expr1::T_numtype, complex<T2>  > > >
min(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Min<typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> min Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// VectorPick<P_numtype1> min _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<P_numtype1, typename P_expr2::T_numtype > > >
min(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// VectorPick<P_numtype1> min VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// VectorPick<P_numtype1> min Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Min<P_numtype1, int > > >
min(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Min<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// VectorPick<P_numtype1> min TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// VectorPick<P_numtype1> min int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Min<P_numtype1, int > > >
min(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Min<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> min float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Min<P_numtype1, float > > >
min(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Min<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> min double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Min<P_numtype1, double > > >
min(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Min<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> min long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Min<P_numtype1, long double > > >
min(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Min<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> min complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Min<P_numtype1, complex<T2>  > > >
min(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Min<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range min Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Min<int, P_numtype2 > > >
min(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// Range min _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<int, typename P_expr2::T_numtype > > >
min(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<int, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range min VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<int, P_numtype2 > > >
min(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// Range min Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Min<int, int > > >
min(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Min<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range min TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<int, P_numtype2 > > >
min(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// Range min float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Min<int, float > > >
min(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Min<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range min double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Min<int, double > > >
min(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Min<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range min long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Min<int, long double > > >
min(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Min<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range min complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Min<int, complex<T2>  > > >
min(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Min<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> min Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// TinyVector<P_numtype1, N_length1> min _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<P_numtype1, typename P_expr2::T_numtype > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> min VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// TinyVector<P_numtype1, N_length1> min Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Min<P_numtype1, int > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Min<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> min TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// TinyVector<P_numtype1, N_length1> min int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Min<P_numtype1, int > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Min<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> min float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Min<P_numtype1, float > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Min<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> min double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Min<P_numtype1, double > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Min<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> min long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Min<P_numtype1, long double > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Min<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> min complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Min<P_numtype1, complex<T2>  > > >
min(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Min<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int min Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<int, P_numtype2 > > >
min(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.beginFast()));
}

// int min _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<int, typename P_expr2::T_numtype > > >
min(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<int, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int min VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<int, P_numtype2 > > >
min(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.beginFast()));
}

// int min TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<int, P_numtype2 > > >
min(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.beginFast()));
}

// float min Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<float, P_numtype2 > > >
min(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.beginFast()));
}

// float min _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<float, typename P_expr2::T_numtype > > >
min(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<float, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float min VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<float, P_numtype2 > > >
min(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.beginFast()));
}

// float min Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Min<float, int > > >
min(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Min<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float min TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<float, P_numtype2 > > >
min(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.beginFast()));
}

// double min Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<double, P_numtype2 > > >
min(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.beginFast()));
}

// double min _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<double, typename P_expr2::T_numtype > > >
min(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<double, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double min VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<double, P_numtype2 > > >
min(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.beginFast()));
}

// double min Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Min<double, int > > >
min(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Min<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double min TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<double, P_numtype2 > > >
min(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.beginFast()));
}

// long double min Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Min<long double, P_numtype2 > > >
min(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Min<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.beginFast()));
}

// long double min _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Min<long double, typename P_expr2::T_numtype > > >
min(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<long double, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double min VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<long double, P_numtype2 > > >
min(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.beginFast()));
}

// long double min Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Min<long double, int > > >
min(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Min<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double min TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<long double, P_numtype2 > > >
min(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.beginFast()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> min Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Min<complex<T1> , P_numtype2 > > >
min(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Min<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.beginFast()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> min _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Min<complex<T1> , typename P_expr2::T_numtype > > >
min(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Min<complex<T1> , typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> min VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Min<complex<T1> , P_numtype2 > > >
min(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Min<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.beginFast()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> min Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Min<complex<T1> , int > > >
min(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Min<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> min TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Min<complex<T1> , P_numtype2 > > >
min(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Min<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.beginFast()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Maximum Operators
 ****************************************************************************/

// Vector<P_numtype1> max Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// Vector<P_numtype1> max _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<P_numtype1, typename P_expr2::T_numtype > > >
max(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// Vector<P_numtype1> max VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// Vector<P_numtype1> max Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Max<P_numtype1, int > > >
max(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Max<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// Vector<P_numtype1> max TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// Vector<P_numtype1> max int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Max<P_numtype1, int > > >
max(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Max<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> max float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Max<P_numtype1, float > > >
max(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Max<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> max double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Max<P_numtype1, double > > >
max(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Max<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> max long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Max<P_numtype1, long double > > >
max(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Max<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> max complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Max<P_numtype1, complex<T2>  > > >
max(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Max<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> max Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<typename P_expr1::T_numtype, P_numtype2 > > >
max(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// _bz_VecExpr<P_expr1> max _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
max(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> max VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<typename P_expr1::T_numtype, P_numtype2 > > >
max(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// _bz_VecExpr<P_expr1> max Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Max<typename P_expr1::T_numtype, int > > >
max(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Max<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> max TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<typename P_expr1::T_numtype, P_numtype2 > > >
max(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// _bz_VecExpr<P_expr1> max int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Max<typename P_expr1::T_numtype, int > > >
max(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Max<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> max float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Max<typename P_expr1::T_numtype, float > > >
max(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Max<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> max double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Max<typename P_expr1::T_numtype, double > > >
max(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Max<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> max long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Max<typename P_expr1::T_numtype, long double > > >
max(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Max<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> max complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Max<typename P_expr1::T_numtype, complex<T2>  > > >
max(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Max<typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> max Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// VectorPick<P_numtype1> max _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<P_numtype1, typename P_expr2::T_numtype > > >
max(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// VectorPick<P_numtype1> max VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// VectorPick<P_numtype1> max Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Max<P_numtype1, int > > >
max(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Max<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// VectorPick<P_numtype1> max TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// VectorPick<P_numtype1> max int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Max<P_numtype1, int > > >
max(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Max<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> max float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Max<P_numtype1, float > > >
max(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Max<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> max double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Max<P_numtype1, double > > >
max(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Max<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> max long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Max<P_numtype1, long double > > >
max(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Max<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> max complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Max<P_numtype1, complex<T2>  > > >
max(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Max<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range max Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Max<int, P_numtype2 > > >
max(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// Range max _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<int, typename P_expr2::T_numtype > > >
max(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<int, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range max VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<int, P_numtype2 > > >
max(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// Range max Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Max<int, int > > >
max(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Max<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range max TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<int, P_numtype2 > > >
max(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.beginFast()));
}

// Range max float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Max<int, float > > >
max(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Max<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range max double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Max<int, double > > >
max(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Max<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range max long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Max<int, long double > > >
max(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Max<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range max complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Max<int, complex<T2>  > > >
max(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Max<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> max Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// TinyVector<P_numtype1, N_length1> max _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<P_numtype1, typename P_expr2::T_numtype > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> max VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// TinyVector<P_numtype1, N_length1> max Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Max<P_numtype1, int > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Max<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> max TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast()));
}

// TinyVector<P_numtype1, N_length1> max int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Max<P_numtype1, int > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Max<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> max float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Max<P_numtype1, float > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Max<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> max double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Max<P_numtype1, double > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Max<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> max long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Max<P_numtype1, long double > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Max<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> max complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Max<P_numtype1, complex<T2>  > > >
max(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Max<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int max Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<int, P_numtype2 > > >
max(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.beginFast()));
}

// int max _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<int, typename P_expr2::T_numtype > > >
max(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<int, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int max VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<int, P_numtype2 > > >
max(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.beginFast()));
}

// int max TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<int, P_numtype2 > > >
max(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.beginFast()));
}

// float max Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<float, P_numtype2 > > >
max(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.beginFast()));
}

// float max _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<float, typename P_expr2::T_numtype > > >
max(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<float, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float max VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<float, P_numtype2 > > >
max(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.beginFast()));
}

// float max Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Max<float, int > > >
max(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Max<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float max TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<float, P_numtype2 > > >
max(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.beginFast()));
}

// double max Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<double, P_numtype2 > > >
max(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.beginFast()));
}

// double max _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<double, typename P_expr2::T_numtype > > >
max(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<double, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double max VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<double, P_numtype2 > > >
max(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.beginFast()));
}

// double max Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Max<double, int > > >
max(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Max<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double max TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<double, P_numtype2 > > >
max(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.beginFast()));
}

// long double max Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Max<long double, P_numtype2 > > >
max(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Max<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.beginFast()));
}

// long double max _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Max<long double, typename P_expr2::T_numtype > > >
max(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<long double, typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double max VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<long double, P_numtype2 > > >
max(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.beginFast()));
}

// long double max Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Max<long double, int > > >
max(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Max<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double max TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<long double, P_numtype2 > > >
max(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.beginFast()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> max Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Max<complex<T1> , P_numtype2 > > >
max(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Max<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.beginFast()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> max _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Max<complex<T1> , typename P_expr2::T_numtype > > >
max(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Max<complex<T1> , typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> max VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Max<complex<T1> , P_numtype2 > > >
max(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Max<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.beginFast()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> max Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Max<complex<T1> , int > > >
max(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Max<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> max TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Max<complex<T1> , P_numtype2 > > >
max(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Max<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.beginFast()));
}
#endif // BZ_HAVE_COMPLEX

BZ_NAMESPACE_END

#endif
