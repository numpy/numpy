/***************************************************************************
 * blitz/vecbops.cc	Vector expression templates (2 operands)
 *
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.   Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 * Licensing inquiries:  blitz-licenses@oonumerics.org
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
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 */ 

// Generated source file.  Do not edit. 
// genvecbops.cpp Aug  7 1997 15:15:07

#ifndef BZ_VECBOPS_CC
#define BZ_VECBOPS_CC

#ifndef BZ_VECEXPR_H
 #error <blitz/vecbops.cc> must be included via <blitz/vecexpr.h>
#endif

BZ_NAMESPACE(blitz)

/****************************************************************************
 * Addition Operators
 ****************************************************************************/

// Vector<P_numtype1> + Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> + _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator+(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> + VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> + Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Add<P_numtype1, int > > >
operator+(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Add<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> + TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> + int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Add<P_numtype1, int > > >
operator+(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Add<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> + float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Add<P_numtype1, float > > >
operator+(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Add<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> + double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Add<P_numtype1, double > > >
operator+(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Add<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> + long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Add<P_numtype1, long double > > >
operator+(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Add<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> + complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Add<P_numtype1, complex<T2>  > > >
operator+(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Add<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> + Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> + _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> + VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> + Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Add<_bz_typename P_expr1::T_numtype, int > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Add<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> + TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> + int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Add<_bz_typename P_expr1::T_numtype, int > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> + float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Add<_bz_typename P_expr1::T_numtype, float > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> + double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Add<_bz_typename P_expr1::T_numtype, double > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> + long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Add<_bz_typename P_expr1::T_numtype, long double > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> + complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Add<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator+(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Add<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> + Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> + _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator+(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> + VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> + Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Add<P_numtype1, int > > >
operator+(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Add<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> + TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> + int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Add<P_numtype1, int > > >
operator+(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Add<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> + float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Add<P_numtype1, float > > >
operator+(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Add<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> + double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Add<P_numtype1, double > > >
operator+(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Add<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> + long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Add<P_numtype1, long double > > >
operator+(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Add<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> + complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Add<P_numtype1, complex<T2>  > > >
operator+(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Add<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range + Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Add<int, P_numtype2 > > >
operator+(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range + _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<int, _bz_typename P_expr2::T_numtype > > >
operator+(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range + VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<int, P_numtype2 > > >
operator+(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range + Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Add<int, int > > >
operator+(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Add<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range + TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<int, P_numtype2 > > >
operator+(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range + float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Add<int, float > > >
operator+(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Add<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range + double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Add<int, double > > >
operator+(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Add<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range + long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Add<int, long double > > >
operator+(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Add<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range + complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Add<int, complex<T2>  > > >
operator+(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Add<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> + Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> + _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> + VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> + Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Add<P_numtype1, int > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Add<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> + TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> + int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Add<P_numtype1, int > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Add<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> + float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Add<P_numtype1, float > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Add<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> + double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Add<P_numtype1, double > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Add<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> + long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Add<P_numtype1, long double > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Add<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> + complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Add<P_numtype1, complex<T2>  > > >
operator+(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Add<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int + Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<int, P_numtype2 > > >
operator+(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int + _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<int, _bz_typename P_expr2::T_numtype > > >
operator+(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int + VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<int, P_numtype2 > > >
operator+(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int + TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<int, P_numtype2 > > >
operator+(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float + Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<float, P_numtype2 > > >
operator+(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float + _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<float, _bz_typename P_expr2::T_numtype > > >
operator+(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float + VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<float, P_numtype2 > > >
operator+(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float + Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Add<float, int > > >
operator+(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Add<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float + TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<float, P_numtype2 > > >
operator+(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double + Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<double, P_numtype2 > > >
operator+(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double + _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<double, _bz_typename P_expr2::T_numtype > > >
operator+(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double + VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<double, P_numtype2 > > >
operator+(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double + Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Add<double, int > > >
operator+(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Add<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double + TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<double, P_numtype2 > > >
operator+(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double + Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Add<long double, P_numtype2 > > >
operator+(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Add<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double + _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Add<long double, _bz_typename P_expr2::T_numtype > > >
operator+(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double + VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<long double, P_numtype2 > > >
operator+(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double + Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Add<long double, int > > >
operator+(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Add<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double + TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<long double, P_numtype2 > > >
operator+(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> + Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Add<complex<T1> , P_numtype2 > > >
operator+(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Add<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> + _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Add<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator+(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Add<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> + VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Add<complex<T1> , P_numtype2 > > >
operator+(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Add<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> + Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Add<complex<T1> , int > > >
operator+(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Add<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> + TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Add<complex<T1> , P_numtype2 > > >
operator+(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Add<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Subtraction Operators
 ****************************************************************************/

// Vector<P_numtype1> - Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> - _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator-(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> - VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> - Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Subtract<P_numtype1, int > > >
operator-(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Subtract<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> - TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> - int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Subtract<P_numtype1, int > > >
operator-(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Subtract<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> - float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Subtract<P_numtype1, float > > >
operator-(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Subtract<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> - double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Subtract<P_numtype1, double > > >
operator-(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Subtract<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> - long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Subtract<P_numtype1, long double > > >
operator-(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Subtract<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> - complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Subtract<P_numtype1, complex<T2>  > > >
operator-(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Subtract<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> - Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> - _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> - VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> - Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, int > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> - TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> - int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, int > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> - float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, float > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> - double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, double > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> - long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, long double > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> - complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator-(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> - Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> - _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator-(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> - VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> - Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Subtract<P_numtype1, int > > >
operator-(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Subtract<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> - TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> - int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Subtract<P_numtype1, int > > >
operator-(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Subtract<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> - float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Subtract<P_numtype1, float > > >
operator-(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Subtract<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> - double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Subtract<P_numtype1, double > > >
operator-(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Subtract<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> - long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Subtract<P_numtype1, long double > > >
operator-(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Subtract<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> - complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Subtract<P_numtype1, complex<T2>  > > >
operator-(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Subtract<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range - Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<int, P_numtype2 > > >
operator-(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range - _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<int, _bz_typename P_expr2::T_numtype > > >
operator-(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range - VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<int, P_numtype2 > > >
operator-(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range - Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Subtract<int, int > > >
operator-(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Subtract<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range - TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<int, P_numtype2 > > >
operator-(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range - float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Subtract<int, float > > >
operator-(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Subtract<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range - double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Subtract<int, double > > >
operator-(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Subtract<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range - long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Subtract<int, long double > > >
operator-(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Subtract<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range - complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Subtract<int, complex<T2>  > > >
operator-(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Subtract<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> - Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> - _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> - VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> - Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Subtract<P_numtype1, int > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Subtract<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> - TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> - int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Subtract<P_numtype1, int > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Subtract<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> - float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Subtract<P_numtype1, float > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Subtract<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> - double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Subtract<P_numtype1, double > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Subtract<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> - long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Subtract<P_numtype1, long double > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Subtract<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> - complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Subtract<P_numtype1, complex<T2>  > > >
operator-(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Subtract<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int - Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<int, P_numtype2 > > >
operator-(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int - _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<int, _bz_typename P_expr2::T_numtype > > >
operator-(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int - VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<int, P_numtype2 > > >
operator-(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int - TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<int, P_numtype2 > > >
operator-(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float - Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<float, P_numtype2 > > >
operator-(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float - _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<float, _bz_typename P_expr2::T_numtype > > >
operator-(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float - VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<float, P_numtype2 > > >
operator-(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float - Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Subtract<float, int > > >
operator-(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Subtract<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float - TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<float, P_numtype2 > > >
operator-(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double - Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<double, P_numtype2 > > >
operator-(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double - _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<double, _bz_typename P_expr2::T_numtype > > >
operator-(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double - VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<double, P_numtype2 > > >
operator-(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double - Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Subtract<double, int > > >
operator-(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Subtract<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double - TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<double, P_numtype2 > > >
operator-(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double - Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<long double, P_numtype2 > > >
operator-(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double - _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<long double, _bz_typename P_expr2::T_numtype > > >
operator-(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double - VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<long double, P_numtype2 > > >
operator-(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double - Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Subtract<long double, int > > >
operator-(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Subtract<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double - TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<long double, P_numtype2 > > >
operator-(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> - Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Subtract<complex<T1> , P_numtype2 > > >
operator-(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Subtract<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> - _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Subtract<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator-(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Subtract<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> - VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Subtract<complex<T1> , P_numtype2 > > >
operator-(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Subtract<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> - Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Subtract<complex<T1> , int > > >
operator-(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Subtract<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> - TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Subtract<complex<T1> , P_numtype2 > > >
operator-(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Subtract<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Multiplication Operators
 ****************************************************************************/

// Vector<P_numtype1> * Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> * _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator*(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> * VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> * Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Multiply<P_numtype1, int > > >
operator*(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Multiply<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> * TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> * int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Multiply<P_numtype1, int > > >
operator*(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Multiply<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> * float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Multiply<P_numtype1, float > > >
operator*(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Multiply<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> * double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Multiply<P_numtype1, double > > >
operator*(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Multiply<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> * long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Multiply<P_numtype1, long double > > >
operator*(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Multiply<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> * complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Multiply<P_numtype1, complex<T2>  > > >
operator*(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Multiply<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> * Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> * _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> * VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> * Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, int > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> * TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> * int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, int > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> * float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, float > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> * double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, double > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> * long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, long double > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> * complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator*(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> * Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> * _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator*(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> * VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> * Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Multiply<P_numtype1, int > > >
operator*(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Multiply<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> * TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> * int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Multiply<P_numtype1, int > > >
operator*(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Multiply<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> * float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Multiply<P_numtype1, float > > >
operator*(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Multiply<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> * double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Multiply<P_numtype1, double > > >
operator*(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Multiply<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> * long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Multiply<P_numtype1, long double > > >
operator*(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Multiply<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> * complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Multiply<P_numtype1, complex<T2>  > > >
operator*(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Multiply<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range * Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<int, P_numtype2 > > >
operator*(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range * _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<int, _bz_typename P_expr2::T_numtype > > >
operator*(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range * VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<int, P_numtype2 > > >
operator*(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range * Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Multiply<int, int > > >
operator*(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Multiply<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range * TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<int, P_numtype2 > > >
operator*(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range * float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Multiply<int, float > > >
operator*(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Multiply<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range * double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Multiply<int, double > > >
operator*(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Multiply<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range * long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Multiply<int, long double > > >
operator*(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Multiply<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range * complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Multiply<int, complex<T2>  > > >
operator*(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Multiply<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> * Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> * _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> * VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> * Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Multiply<P_numtype1, int > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Multiply<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> * TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> * int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Multiply<P_numtype1, int > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Multiply<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> * float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Multiply<P_numtype1, float > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Multiply<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> * double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Multiply<P_numtype1, double > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Multiply<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> * long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Multiply<P_numtype1, long double > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Multiply<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> * complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Multiply<P_numtype1, complex<T2>  > > >
operator*(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Multiply<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int * Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<int, P_numtype2 > > >
operator*(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int * _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<int, _bz_typename P_expr2::T_numtype > > >
operator*(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int * VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<int, P_numtype2 > > >
operator*(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int * TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<int, P_numtype2 > > >
operator*(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float * Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<float, P_numtype2 > > >
operator*(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float * _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<float, _bz_typename P_expr2::T_numtype > > >
operator*(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float * VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<float, P_numtype2 > > >
operator*(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float * Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Multiply<float, int > > >
operator*(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Multiply<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float * TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<float, P_numtype2 > > >
operator*(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double * Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<double, P_numtype2 > > >
operator*(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double * _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<double, _bz_typename P_expr2::T_numtype > > >
operator*(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double * VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<double, P_numtype2 > > >
operator*(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double * Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Multiply<double, int > > >
operator*(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Multiply<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double * TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<double, P_numtype2 > > >
operator*(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double * Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<long double, P_numtype2 > > >
operator*(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double * _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<long double, _bz_typename P_expr2::T_numtype > > >
operator*(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double * VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<long double, P_numtype2 > > >
operator*(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double * Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Multiply<long double, int > > >
operator*(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Multiply<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double * TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<long double, P_numtype2 > > >
operator*(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> * Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Multiply<complex<T1> , P_numtype2 > > >
operator*(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Multiply<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> * _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Multiply<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator*(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Multiply<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> * VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Multiply<complex<T1> , P_numtype2 > > >
operator*(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Multiply<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> * Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Multiply<complex<T1> , int > > >
operator*(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Multiply<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> * TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Multiply<complex<T1> , P_numtype2 > > >
operator*(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Multiply<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Division Operators
 ****************************************************************************/

// Vector<P_numtype1> / Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> / _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator/(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> / VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> / Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Divide<P_numtype1, int > > >
operator/(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Divide<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> / TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> / int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Divide<P_numtype1, int > > >
operator/(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Divide<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> / float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Divide<P_numtype1, float > > >
operator/(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Divide<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> / double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Divide<P_numtype1, double > > >
operator/(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Divide<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> / long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Divide<P_numtype1, long double > > >
operator/(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Divide<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> / complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Divide<P_numtype1, complex<T2>  > > >
operator/(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Divide<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> / Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> / _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> / VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> / Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Divide<_bz_typename P_expr1::T_numtype, int > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> / TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> / int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, int > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> / float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, float > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> / double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, double > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> / long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, long double > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> / complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Divide<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator/(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Divide<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> / Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> / _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator/(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> / VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> / Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Divide<P_numtype1, int > > >
operator/(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Divide<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> / TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> / int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Divide<P_numtype1, int > > >
operator/(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Divide<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> / float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Divide<P_numtype1, float > > >
operator/(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Divide<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> / double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Divide<P_numtype1, double > > >
operator/(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Divide<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> / long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Divide<P_numtype1, long double > > >
operator/(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Divide<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> / complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Divide<P_numtype1, complex<T2>  > > >
operator/(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Divide<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range / Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<int, P_numtype2 > > >
operator/(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range / _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<int, _bz_typename P_expr2::T_numtype > > >
operator/(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range / VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<int, P_numtype2 > > >
operator/(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range / Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Divide<int, int > > >
operator/(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Divide<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range / TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<int, P_numtype2 > > >
operator/(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range / float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Divide<int, float > > >
operator/(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Divide<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range / double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Divide<int, double > > >
operator/(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Divide<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range / long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Divide<int, long double > > >
operator/(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Divide<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range / complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Divide<int, complex<T2>  > > >
operator/(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Divide<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> / Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> / _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> / VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> / Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Divide<P_numtype1, int > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Divide<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> / TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> / int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Divide<P_numtype1, int > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Divide<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> / float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Divide<P_numtype1, float > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Divide<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> / double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Divide<P_numtype1, double > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Divide<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> / long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Divide<P_numtype1, long double > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Divide<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> / complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Divide<P_numtype1, complex<T2>  > > >
operator/(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Divide<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int / Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<int, P_numtype2 > > >
operator/(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int / _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<int, _bz_typename P_expr2::T_numtype > > >
operator/(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int / VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<int, P_numtype2 > > >
operator/(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int / TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<int, P_numtype2 > > >
operator/(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float / Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<float, P_numtype2 > > >
operator/(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float / _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<float, _bz_typename P_expr2::T_numtype > > >
operator/(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float / VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<float, P_numtype2 > > >
operator/(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float / Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Divide<float, int > > >
operator/(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Divide<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float / TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<float, P_numtype2 > > >
operator/(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double / Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<double, P_numtype2 > > >
operator/(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double / _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<double, _bz_typename P_expr2::T_numtype > > >
operator/(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double / VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<double, P_numtype2 > > >
operator/(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double / Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Divide<double, int > > >
operator/(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Divide<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double / TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<double, P_numtype2 > > >
operator/(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double / Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Divide<long double, P_numtype2 > > >
operator/(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double / _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<long double, _bz_typename P_expr2::T_numtype > > >
operator/(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double / VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<long double, P_numtype2 > > >
operator/(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double / Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Divide<long double, int > > >
operator/(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Divide<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double / TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<long double, P_numtype2 > > >
operator/(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> / Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Divide<complex<T1> , P_numtype2 > > >
operator/(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Divide<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> / _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Divide<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator/(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Divide<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> / VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Divide<complex<T1> , P_numtype2 > > >
operator/(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Divide<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> / Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Divide<complex<T1> , int > > >
operator/(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Divide<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> / TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Divide<complex<T1> , P_numtype2 > > >
operator/(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Divide<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Modulus Operators
 ****************************************************************************/

// Vector<P_numtype1> % Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> % _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator%(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> % VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> % Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Mod<P_numtype1, int > > >
operator%(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Mod<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> % TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> % int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Mod<P_numtype1, int > > >
operator%(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Mod<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> % Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator%(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> % _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator%(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> % VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator%(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> % Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Mod<_bz_typename P_expr1::T_numtype, int > > >
operator%(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> % TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator%(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> % int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, int > > >
operator%(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> % Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> % _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator%(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> % VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> % Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Mod<P_numtype1, int > > >
operator%(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Mod<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> % TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> % int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Mod<P_numtype1, int > > >
operator%(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Mod<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range % Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Mod<int, P_numtype2 > > >
operator%(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Mod<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range % _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Mod<int, _bz_typename P_expr2::T_numtype > > >
operator%(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Mod<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range % VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Mod<int, P_numtype2 > > >
operator%(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Mod<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range % Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Mod<int, int > > >
operator%(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Mod<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range % TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Mod<int, P_numtype2 > > >
operator%(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Mod<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> % Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> % _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator%(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> % VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> % Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Mod<P_numtype1, int > > >
operator%(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Mod<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> % TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> % int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Mod<P_numtype1, int > > >
operator%(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Mod<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int % Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Mod<int, P_numtype2 > > >
operator%(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Mod<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int % _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Mod<int, _bz_typename P_expr2::T_numtype > > >
operator%(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Mod<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int % VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Mod<int, P_numtype2 > > >
operator%(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Mod<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int % TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Mod<int, P_numtype2 > > >
operator%(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Mod<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
/****************************************************************************
 * Bitwise XOR Operators
 ****************************************************************************/

// Vector<P_numtype1> ^ Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> ^ _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator^(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> ^ VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> ^ Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_BitwiseXOR<P_numtype1, int > > >
operator^(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_BitwiseXOR<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> ^ TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> ^ int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseXOR<P_numtype1, int > > >
operator^(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseXOR<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> ^ Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator^(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> ^ _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator^(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> ^ VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator^(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> ^ Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, int > > >
operator^(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> ^ TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator^(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> ^ int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, int > > >
operator^(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> ^ Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> ^ _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator^(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> ^ VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> ^ Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_BitwiseXOR<P_numtype1, int > > >
operator^(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_BitwiseXOR<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> ^ TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> ^ int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseXOR<P_numtype1, int > > >
operator^(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseXOR<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range ^ Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseXOR<int, P_numtype2 > > >
operator^(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseXOR<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range ^ _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseXOR<int, _bz_typename P_expr2::T_numtype > > >
operator^(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseXOR<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range ^ VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseXOR<int, P_numtype2 > > >
operator^(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseXOR<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range ^ Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_BitwiseXOR<int, int > > >
operator^(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_BitwiseXOR<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range ^ TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseXOR<int, P_numtype2 > > >
operator^(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseXOR<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> ^ Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> ^ _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator^(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> ^ VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> ^ Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_BitwiseXOR<P_numtype1, int > > >
operator^(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_BitwiseXOR<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> ^ TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> ^ int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseXOR<P_numtype1, int > > >
operator^(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseXOR<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int ^ Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseXOR<int, P_numtype2 > > >
operator^(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseXOR<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int ^ _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseXOR<int, _bz_typename P_expr2::T_numtype > > >
operator^(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseXOR<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int ^ VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseXOR<int, P_numtype2 > > >
operator^(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseXOR<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int ^ TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseXOR<int, P_numtype2 > > >
operator^(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseXOR<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
/****************************************************************************
 * Bitwise And Operators
 ****************************************************************************/

// Vector<P_numtype1> & Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> & _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> & VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> & Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_BitwiseAnd<P_numtype1, int > > >
operator&(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_BitwiseAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> & TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> & int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseAnd<P_numtype1, int > > >
operator&(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> & Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> & _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator&(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> & VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> & Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> & TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> & int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> & Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> & _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> & VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> & Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_BitwiseAnd<P_numtype1, int > > >
operator&(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_BitwiseAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> & TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> & int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseAnd<P_numtype1, int > > >
operator&(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range & Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseAnd<int, P_numtype2 > > >
operator&(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range & _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseAnd<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range & VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseAnd<int, P_numtype2 > > >
operator&(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range & Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_BitwiseAnd<int, int > > >
operator&(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_BitwiseAnd<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range & TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseAnd<int, P_numtype2 > > >
operator&(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> & Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> & _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> & VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> & Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_BitwiseAnd<P_numtype1, int > > >
operator&(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_BitwiseAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> & TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> & int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseAnd<P_numtype1, int > > >
operator&(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int & Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseAnd<int, P_numtype2 > > >
operator&(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int & _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseAnd<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int & VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseAnd<int, P_numtype2 > > >
operator&(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int & TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseAnd<int, P_numtype2 > > >
operator&(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
/****************************************************************************
 * Bitwise Or Operators
 ****************************************************************************/

// Vector<P_numtype1> | Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> | _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator|(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> | VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> | Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_BitwiseOr<P_numtype1, int > > >
operator|(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_BitwiseOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> | TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> | int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseOr<P_numtype1, int > > >
operator|(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> | Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator|(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> | _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator|(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> | VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator|(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> | Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, int > > >
operator|(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> | TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator|(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> | int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, int > > >
operator|(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> | Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> | _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator|(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> | VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> | Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_BitwiseOr<P_numtype1, int > > >
operator|(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_BitwiseOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> | TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> | int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseOr<P_numtype1, int > > >
operator|(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range | Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseOr<int, P_numtype2 > > >
operator|(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range | _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseOr<int, _bz_typename P_expr2::T_numtype > > >
operator|(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseOr<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range | VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseOr<int, P_numtype2 > > >
operator|(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range | Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_BitwiseOr<int, int > > >
operator|(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_BitwiseOr<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range | TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseOr<int, P_numtype2 > > >
operator|(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> | Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> | _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator|(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> | VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> | Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_BitwiseOr<P_numtype1, int > > >
operator|(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_BitwiseOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> | TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> | int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_BitwiseOr<P_numtype1, int > > >
operator|(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_BitwiseOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int | Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_BitwiseOr<int, P_numtype2 > > >
operator|(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_BitwiseOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int | _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_BitwiseOr<int, _bz_typename P_expr2::T_numtype > > >
operator|(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_BitwiseOr<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int | VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_BitwiseOr<int, P_numtype2 > > >
operator|(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_BitwiseOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int | TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_BitwiseOr<int, P_numtype2 > > >
operator|(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_BitwiseOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
/****************************************************************************
 * Shift right Operators
 ****************************************************************************/

// Vector<P_numtype1> >> Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> >> _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>>(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> >> VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> >> Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_ShiftRight<P_numtype1, int > > >
operator>>(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_ShiftRight<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> >> TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> >> int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftRight<P_numtype1, int > > >
operator>>(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftRight<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> >> Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>>(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> >> _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>>(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> >> VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>>(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> >> Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, int > > >
operator>>(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> >> TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>>(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> >> int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, int > > >
operator>>(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> >> Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> >> _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>>(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> >> VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> >> Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_ShiftRight<P_numtype1, int > > >
operator>>(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_ShiftRight<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> >> TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> >> int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftRight<P_numtype1, int > > >
operator>>(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftRight<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range >> Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftRight<int, P_numtype2 > > >
operator>>(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftRight<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range >> _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftRight<int, _bz_typename P_expr2::T_numtype > > >
operator>>(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftRight<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range >> VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftRight<int, P_numtype2 > > >
operator>>(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftRight<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range >> Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_ShiftRight<int, int > > >
operator>>(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_ShiftRight<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range >> TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftRight<int, P_numtype2 > > >
operator>>(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftRight<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> >> Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> >> _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>>(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> >> VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> >> Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_ShiftRight<P_numtype1, int > > >
operator>>(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_ShiftRight<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> >> TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> >> int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftRight<P_numtype1, int > > >
operator>>(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftRight<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int >> Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftRight<int, P_numtype2 > > >
operator>>(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftRight<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int >> _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftRight<int, _bz_typename P_expr2::T_numtype > > >
operator>>(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftRight<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int >> VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftRight<int, P_numtype2 > > >
operator>>(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftRight<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int >> TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftRight<int, P_numtype2 > > >
operator>>(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftRight<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
/****************************************************************************
 * Shift left Operators
 ****************************************************************************/

// Vector<P_numtype1> << Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> << _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<<(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> << VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> << Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_ShiftLeft<P_numtype1, int > > >
operator<<(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_ShiftLeft<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> << TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> << int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftLeft<P_numtype1, int > > >
operator<<(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftLeft<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> << Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<<(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> << _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<<(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> << VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<<(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> << Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, int > > >
operator<<(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> << TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<<(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> << int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, int > > >
operator<<(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> << Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> << _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<<(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> << VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> << Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_ShiftLeft<P_numtype1, int > > >
operator<<(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_ShiftLeft<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> << TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> << int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftLeft<P_numtype1, int > > >
operator<<(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftLeft<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range << Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftLeft<int, P_numtype2 > > >
operator<<(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftLeft<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range << _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftLeft<int, _bz_typename P_expr2::T_numtype > > >
operator<<(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftLeft<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range << VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftLeft<int, P_numtype2 > > >
operator<<(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftLeft<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range << Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_ShiftLeft<int, int > > >
operator<<(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_ShiftLeft<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range << TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftLeft<int, P_numtype2 > > >
operator<<(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftLeft<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> << Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> << _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<<(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> << VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> << Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_ShiftLeft<P_numtype1, int > > >
operator<<(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_ShiftLeft<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> << TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> << int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_ShiftLeft<P_numtype1, int > > >
operator<<(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_ShiftLeft<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int << Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_ShiftLeft<int, P_numtype2 > > >
operator<<(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_ShiftLeft<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int << _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_ShiftLeft<int, _bz_typename P_expr2::T_numtype > > >
operator<<(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_ShiftLeft<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int << VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_ShiftLeft<int, P_numtype2 > > >
operator<<(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_ShiftLeft<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int << TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_ShiftLeft<int, P_numtype2 > > >
operator<<(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_ShiftLeft<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
/****************************************************************************
 * Greater-than Operators
 ****************************************************************************/

// Vector<P_numtype1> > Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> > _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> > VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> > Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Greater<P_numtype1, int > > >
operator>(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Greater<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> > TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> > int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Greater<P_numtype1, int > > >
operator>(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Greater<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> > float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Greater<P_numtype1, float > > >
operator>(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Greater<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> > double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Greater<P_numtype1, double > > >
operator>(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Greater<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> > long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Greater<P_numtype1, long double > > >
operator>(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Greater<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> > complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Greater<P_numtype1, complex<T2>  > > >
operator>(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Greater<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> > Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> > _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> > VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> > Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Greater<_bz_typename P_expr1::T_numtype, int > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> > TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> > int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, int > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> > float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, float > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> > double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, double > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> > long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, long double > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> > complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Greater<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator>(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Greater<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> > Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> > _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> > VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> > Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Greater<P_numtype1, int > > >
operator>(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Greater<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> > TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> > int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Greater<P_numtype1, int > > >
operator>(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Greater<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> > float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Greater<P_numtype1, float > > >
operator>(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Greater<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> > double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Greater<P_numtype1, double > > >
operator>(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Greater<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> > long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Greater<P_numtype1, long double > > >
operator>(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Greater<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> > complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Greater<P_numtype1, complex<T2>  > > >
operator>(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Greater<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range > Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<int, P_numtype2 > > >
operator>(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range > _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<int, _bz_typename P_expr2::T_numtype > > >
operator>(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range > VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<int, P_numtype2 > > >
operator>(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range > Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Greater<int, int > > >
operator>(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Greater<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range > TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<int, P_numtype2 > > >
operator>(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range > float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Greater<int, float > > >
operator>(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Greater<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range > double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Greater<int, double > > >
operator>(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Greater<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range > long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Greater<int, long double > > >
operator>(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Greater<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range > complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Greater<int, complex<T2>  > > >
operator>(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Greater<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> > Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> > _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> > VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> > Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Greater<P_numtype1, int > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Greater<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> > TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> > int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Greater<P_numtype1, int > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Greater<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> > float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Greater<P_numtype1, float > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Greater<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> > double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Greater<P_numtype1, double > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Greater<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> > long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Greater<P_numtype1, long double > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Greater<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> > complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Greater<P_numtype1, complex<T2>  > > >
operator>(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Greater<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int > Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<int, P_numtype2 > > >
operator>(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int > _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<int, _bz_typename P_expr2::T_numtype > > >
operator>(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int > VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<int, P_numtype2 > > >
operator>(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int > TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<int, P_numtype2 > > >
operator>(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float > Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<float, P_numtype2 > > >
operator>(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float > _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<float, _bz_typename P_expr2::T_numtype > > >
operator>(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float > VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<float, P_numtype2 > > >
operator>(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float > Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Greater<float, int > > >
operator>(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Greater<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float > TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<float, P_numtype2 > > >
operator>(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double > Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<double, P_numtype2 > > >
operator>(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double > _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<double, _bz_typename P_expr2::T_numtype > > >
operator>(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double > VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<double, P_numtype2 > > >
operator>(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double > Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Greater<double, int > > >
operator>(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Greater<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double > TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<double, P_numtype2 > > >
operator>(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double > Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Greater<long double, P_numtype2 > > >
operator>(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double > _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<long double, _bz_typename P_expr2::T_numtype > > >
operator>(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double > VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<long double, P_numtype2 > > >
operator>(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double > Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Greater<long double, int > > >
operator>(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Greater<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double > TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<long double, P_numtype2 > > >
operator>(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> > Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Greater<complex<T1> , P_numtype2 > > >
operator>(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Greater<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> > _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Greater<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator>(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Greater<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> > VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Greater<complex<T1> , P_numtype2 > > >
operator>(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Greater<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> > Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Greater<complex<T1> , int > > >
operator>(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Greater<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> > TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Greater<complex<T1> , P_numtype2 > > >
operator>(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Greater<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Less-than Operators
 ****************************************************************************/

// Vector<P_numtype1> < Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> < _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> < VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> < Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Less<P_numtype1, int > > >
operator<(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Less<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> < TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> < int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Less<P_numtype1, int > > >
operator<(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Less<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> < float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Less<P_numtype1, float > > >
operator<(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Less<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> < double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Less<P_numtype1, double > > >
operator<(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Less<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> < long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Less<P_numtype1, long double > > >
operator<(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Less<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> < complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Less<P_numtype1, complex<T2>  > > >
operator<(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Less<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> < Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> < _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> < VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> < Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Less<_bz_typename P_expr1::T_numtype, int > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Less<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> < TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> < int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Less<_bz_typename P_expr1::T_numtype, int > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> < float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Less<_bz_typename P_expr1::T_numtype, float > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> < double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Less<_bz_typename P_expr1::T_numtype, double > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> < long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Less<_bz_typename P_expr1::T_numtype, long double > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> < complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Less<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator<(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Less<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> < Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> < _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> < VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> < Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Less<P_numtype1, int > > >
operator<(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Less<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> < TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> < int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Less<P_numtype1, int > > >
operator<(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Less<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> < float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Less<P_numtype1, float > > >
operator<(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Less<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> < double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Less<P_numtype1, double > > >
operator<(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Less<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> < long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Less<P_numtype1, long double > > >
operator<(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Less<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> < complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Less<P_numtype1, complex<T2>  > > >
operator<(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Less<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range < Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Less<int, P_numtype2 > > >
operator<(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range < _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<int, _bz_typename P_expr2::T_numtype > > >
operator<(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range < VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<int, P_numtype2 > > >
operator<(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range < Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Less<int, int > > >
operator<(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Less<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range < TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<int, P_numtype2 > > >
operator<(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range < float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Less<int, float > > >
operator<(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Less<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range < double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Less<int, double > > >
operator<(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Less<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range < long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Less<int, long double > > >
operator<(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Less<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range < complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Less<int, complex<T2>  > > >
operator<(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Less<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> < Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> < _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> < VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> < Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Less<P_numtype1, int > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Less<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> < TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> < int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Less<P_numtype1, int > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Less<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> < float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Less<P_numtype1, float > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Less<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> < double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Less<P_numtype1, double > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Less<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> < long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Less<P_numtype1, long double > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Less<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> < complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Less<P_numtype1, complex<T2>  > > >
operator<(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Less<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int < Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<int, P_numtype2 > > >
operator<(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int < _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<int, _bz_typename P_expr2::T_numtype > > >
operator<(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int < VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<int, P_numtype2 > > >
operator<(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int < TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<int, P_numtype2 > > >
operator<(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float < Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<float, P_numtype2 > > >
operator<(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float < _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<float, _bz_typename P_expr2::T_numtype > > >
operator<(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float < VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<float, P_numtype2 > > >
operator<(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float < Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Less<float, int > > >
operator<(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Less<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float < TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<float, P_numtype2 > > >
operator<(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double < Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<double, P_numtype2 > > >
operator<(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double < _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<double, _bz_typename P_expr2::T_numtype > > >
operator<(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double < VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<double, P_numtype2 > > >
operator<(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double < Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Less<double, int > > >
operator<(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Less<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double < TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<double, P_numtype2 > > >
operator<(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double < Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Less<long double, P_numtype2 > > >
operator<(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Less<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double < _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Less<long double, _bz_typename P_expr2::T_numtype > > >
operator<(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double < VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<long double, P_numtype2 > > >
operator<(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double < Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Less<long double, int > > >
operator<(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Less<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double < TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<long double, P_numtype2 > > >
operator<(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> < Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Less<complex<T1> , P_numtype2 > > >
operator<(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Less<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> < _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Less<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator<(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Less<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> < VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Less<complex<T1> , P_numtype2 > > >
operator<(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Less<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> < Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Less<complex<T1> , int > > >
operator<(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Less<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> < TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Less<complex<T1> , P_numtype2 > > >
operator<(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Less<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Greater or equal (>=) operators
 ****************************************************************************/

// Vector<P_numtype1> >= Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> >= _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>=(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> >= VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> >= Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_GreaterOrEqual<P_numtype1, int > > >
operator>=(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_GreaterOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> >= TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> >= int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_GreaterOrEqual<P_numtype1, int > > >
operator>=(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_GreaterOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> >= float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_GreaterOrEqual<P_numtype1, float > > >
operator>=(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_GreaterOrEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> >= double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_GreaterOrEqual<P_numtype1, double > > >
operator>=(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_GreaterOrEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> >= long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_GreaterOrEqual<P_numtype1, long double > > >
operator>=(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_GreaterOrEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> >= complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<P_numtype1, complex<T2>  > > >
operator>=(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> >= Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> >= _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> >= VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> >= Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> >= TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> >= int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> >= float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, float > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> >= double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, double > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> >= long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> >= complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator>=(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> >= Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> >= _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> >= VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> >= Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_GreaterOrEqual<P_numtype1, int > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_GreaterOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> >= TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> >= int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_GreaterOrEqual<P_numtype1, int > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_GreaterOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> >= float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_GreaterOrEqual<P_numtype1, float > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_GreaterOrEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> >= double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_GreaterOrEqual<P_numtype1, double > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_GreaterOrEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> >= long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_GreaterOrEqual<P_numtype1, long double > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_GreaterOrEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> >= complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<P_numtype1, complex<T2>  > > >
operator>=(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range >= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<int, P_numtype2 > > >
operator>=(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range >= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator>=(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range >= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<int, P_numtype2 > > >
operator>=(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range >= Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_GreaterOrEqual<int, int > > >
operator>=(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_GreaterOrEqual<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range >= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<int, P_numtype2 > > >
operator>=(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range >= float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_GreaterOrEqual<int, float > > >
operator>=(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_GreaterOrEqual<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range >= double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_GreaterOrEqual<int, double > > >
operator>=(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_GreaterOrEqual<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range >= long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_GreaterOrEqual<int, long double > > >
operator>=(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_GreaterOrEqual<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range >= complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<int, complex<T2>  > > >
operator>=(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> >= Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> >= _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> >= VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> >= Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_GreaterOrEqual<P_numtype1, int > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_GreaterOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> >= TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> >= int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_GreaterOrEqual<P_numtype1, int > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_GreaterOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> >= float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_GreaterOrEqual<P_numtype1, float > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_GreaterOrEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> >= double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_GreaterOrEqual<P_numtype1, double > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_GreaterOrEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> >= long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_GreaterOrEqual<P_numtype1, long double > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_GreaterOrEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> >= complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<P_numtype1, complex<T2>  > > >
operator>=(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int >= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<int, P_numtype2 > > >
operator>=(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int >= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator>=(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int >= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<int, P_numtype2 > > >
operator>=(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int >= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<int, P_numtype2 > > >
operator>=(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float >= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<float, P_numtype2 > > >
operator>=(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float >= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<float, _bz_typename P_expr2::T_numtype > > >
operator>=(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float >= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<float, P_numtype2 > > >
operator>=(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float >= Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_GreaterOrEqual<float, int > > >
operator>=(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_GreaterOrEqual<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float >= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<float, P_numtype2 > > >
operator>=(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double >= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<double, P_numtype2 > > >
operator>=(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double >= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<double, _bz_typename P_expr2::T_numtype > > >
operator>=(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double >= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<double, P_numtype2 > > >
operator>=(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double >= Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_GreaterOrEqual<double, int > > >
operator>=(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_GreaterOrEqual<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double >= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<double, P_numtype2 > > >
operator>=(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double >= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<long double, P_numtype2 > > >
operator>=(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double >= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator>=(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double >= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<long double, P_numtype2 > > >
operator>=(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double >= Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_GreaterOrEqual<long double, int > > >
operator>=(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_GreaterOrEqual<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double >= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<long double, P_numtype2 > > >
operator>=(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> >= Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_GreaterOrEqual<complex<T1> , P_numtype2 > > >
operator>=(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> >= _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_GreaterOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator>=(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_GreaterOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> >= VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_GreaterOrEqual<complex<T1> , P_numtype2 > > >
operator>=(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_GreaterOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> >= Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_GreaterOrEqual<complex<T1> , int > > >
operator>=(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_GreaterOrEqual<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> >= TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_GreaterOrEqual<complex<T1> , P_numtype2 > > >
operator>=(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_GreaterOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Less or equal (<=) operators
 ****************************************************************************/

// Vector<P_numtype1> <= Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> <= _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<=(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> <= VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> <= Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_LessOrEqual<P_numtype1, int > > >
operator<=(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_LessOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> <= TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> <= int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_LessOrEqual<P_numtype1, int > > >
operator<=(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_LessOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> <= float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_LessOrEqual<P_numtype1, float > > >
operator<=(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_LessOrEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> <= double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_LessOrEqual<P_numtype1, double > > >
operator<=(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_LessOrEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> <= long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_LessOrEqual<P_numtype1, long double > > >
operator<=(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_LessOrEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> <= complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_LessOrEqual<P_numtype1, complex<T2>  > > >
operator<=(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_LessOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> <= Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> <= _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> <= VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> <= Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> <= TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> <= int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> <= float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, float > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> <= double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, double > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> <= long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> <= complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator<=(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> <= Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> <= _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> <= VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> <= Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_LessOrEqual<P_numtype1, int > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_LessOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> <= TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> <= int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_LessOrEqual<P_numtype1, int > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_LessOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> <= float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_LessOrEqual<P_numtype1, float > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_LessOrEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> <= double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_LessOrEqual<P_numtype1, double > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_LessOrEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> <= long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_LessOrEqual<P_numtype1, long double > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_LessOrEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> <= complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_LessOrEqual<P_numtype1, complex<T2>  > > >
operator<=(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_LessOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range <= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<int, P_numtype2 > > >
operator<=(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range <= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator<=(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range <= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<int, P_numtype2 > > >
operator<=(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range <= Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_LessOrEqual<int, int > > >
operator<=(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_LessOrEqual<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range <= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<int, P_numtype2 > > >
operator<=(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range <= float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_LessOrEqual<int, float > > >
operator<=(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_LessOrEqual<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range <= double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_LessOrEqual<int, double > > >
operator<=(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_LessOrEqual<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range <= long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_LessOrEqual<int, long double > > >
operator<=(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_LessOrEqual<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range <= complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_LessOrEqual<int, complex<T2>  > > >
operator<=(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_LessOrEqual<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> <= Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> <= _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> <= VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> <= Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_LessOrEqual<P_numtype1, int > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_LessOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> <= TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> <= int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_LessOrEqual<P_numtype1, int > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_LessOrEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> <= float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_LessOrEqual<P_numtype1, float > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_LessOrEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> <= double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_LessOrEqual<P_numtype1, double > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_LessOrEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> <= long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_LessOrEqual<P_numtype1, long double > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_LessOrEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> <= complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_LessOrEqual<P_numtype1, complex<T2>  > > >
operator<=(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_LessOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int <= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<int, P_numtype2 > > >
operator<=(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int <= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator<=(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int <= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<int, P_numtype2 > > >
operator<=(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int <= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<int, P_numtype2 > > >
operator<=(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float <= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<float, P_numtype2 > > >
operator<=(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float <= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<float, _bz_typename P_expr2::T_numtype > > >
operator<=(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float <= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<float, P_numtype2 > > >
operator<=(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float <= Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_LessOrEqual<float, int > > >
operator<=(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_LessOrEqual<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float <= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<float, P_numtype2 > > >
operator<=(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double <= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<double, P_numtype2 > > >
operator<=(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double <= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<double, _bz_typename P_expr2::T_numtype > > >
operator<=(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double <= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<double, P_numtype2 > > >
operator<=(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double <= Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_LessOrEqual<double, int > > >
operator<=(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_LessOrEqual<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double <= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<double, P_numtype2 > > >
operator<=(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double <= Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<long double, P_numtype2 > > >
operator<=(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double <= _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator<=(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double <= VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<long double, P_numtype2 > > >
operator<=(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double <= Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_LessOrEqual<long double, int > > >
operator<=(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_LessOrEqual<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double <= TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<long double, P_numtype2 > > >
operator<=(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> <= Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_LessOrEqual<complex<T1> , P_numtype2 > > >
operator<=(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_LessOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> <= _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_LessOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator<=(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_LessOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> <= VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_LessOrEqual<complex<T1> , P_numtype2 > > >
operator<=(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_LessOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> <= Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_LessOrEqual<complex<T1> , int > > >
operator<=(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_LessOrEqual<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> <= TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LessOrEqual<complex<T1> , P_numtype2 > > >
operator<=(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LessOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Equality operators
 ****************************************************************************/

// Vector<P_numtype1> == Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> == _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator==(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> == VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> == Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_Equal<P_numtype1, int > > >
operator==(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_Equal<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> == TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> == int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Equal<P_numtype1, int > > >
operator==(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Equal<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> == float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Equal<P_numtype1, float > > >
operator==(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Equal<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> == double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Equal<P_numtype1, double > > >
operator==(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Equal<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> == long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Equal<P_numtype1, long double > > >
operator==(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Equal<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> == complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Equal<P_numtype1, complex<T2>  > > >
operator==(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Equal<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> == Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> == _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> == VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> == Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_Equal<_bz_typename P_expr1::T_numtype, int > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> == TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> == int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, int > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> == float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, float > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> == double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, double > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> == long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, long double > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> == complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Equal<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator==(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Equal<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> == Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> == _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator==(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> == VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> == Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_Equal<P_numtype1, int > > >
operator==(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_Equal<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> == TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> == int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_Equal<P_numtype1, int > > >
operator==(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_Equal<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> == float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_Equal<P_numtype1, float > > >
operator==(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_Equal<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> == double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_Equal<P_numtype1, double > > >
operator==(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_Equal<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> == long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_Equal<P_numtype1, long double > > >
operator==(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Equal<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> == complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Equal<P_numtype1, complex<T2>  > > >
operator==(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Equal<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range == Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<int, P_numtype2 > > >
operator==(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range == _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<int, _bz_typename P_expr2::T_numtype > > >
operator==(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range == VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<int, P_numtype2 > > >
operator==(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range == Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_Equal<int, int > > >
operator==(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_Equal<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range == TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<int, P_numtype2 > > >
operator==(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range == float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_Equal<int, float > > >
operator==(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_Equal<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range == double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_Equal<int, double > > >
operator==(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_Equal<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range == long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_Equal<int, long double > > >
operator==(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_Equal<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range == complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Equal<int, complex<T2>  > > >
operator==(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Equal<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> == Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> == _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> == VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> == Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_Equal<P_numtype1, int > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_Equal<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> == TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> == int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_Equal<P_numtype1, int > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_Equal<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> == float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_Equal<P_numtype1, float > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_Equal<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> == double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_Equal<P_numtype1, double > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_Equal<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> == long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_Equal<P_numtype1, long double > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_Equal<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> == complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_Equal<P_numtype1, complex<T2>  > > >
operator==(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_Equal<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int == Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<int, P_numtype2 > > >
operator==(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int == _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<int, _bz_typename P_expr2::T_numtype > > >
operator==(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int == VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<int, P_numtype2 > > >
operator==(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int == TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<int, P_numtype2 > > >
operator==(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float == Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<float, P_numtype2 > > >
operator==(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float == _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<float, _bz_typename P_expr2::T_numtype > > >
operator==(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float == VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<float, P_numtype2 > > >
operator==(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float == Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_Equal<float, int > > >
operator==(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_Equal<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float == TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<float, P_numtype2 > > >
operator==(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double == Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<double, P_numtype2 > > >
operator==(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double == _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<double, _bz_typename P_expr2::T_numtype > > >
operator==(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double == VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<double, P_numtype2 > > >
operator==(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double == Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_Equal<double, int > > >
operator==(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_Equal<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double == TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<double, P_numtype2 > > >
operator==(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double == Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_Equal<long double, P_numtype2 > > >
operator==(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double == _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<long double, _bz_typename P_expr2::T_numtype > > >
operator==(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double == VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<long double, P_numtype2 > > >
operator==(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double == Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_Equal<long double, int > > >
operator==(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_Equal<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double == TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<long double, P_numtype2 > > >
operator==(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> == Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_Equal<complex<T1> , P_numtype2 > > >
operator==(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_Equal<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> == _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_Equal<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator==(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_Equal<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> == VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_Equal<complex<T1> , P_numtype2 > > >
operator==(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_Equal<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> == Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_Equal<complex<T1> , int > > >
operator==(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_Equal<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> == TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_Equal<complex<T1> , P_numtype2 > > >
operator==(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_Equal<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Not-equal operators
 ****************************************************************************/

// Vector<P_numtype1> != Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> != _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator!=(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> != VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> != Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_NotEqual<P_numtype1, int > > >
operator!=(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_NotEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> != TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> != int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_NotEqual<P_numtype1, int > > >
operator!=(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_NotEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Vector<P_numtype1> != float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_NotEqual<P_numtype1, float > > >
operator!=(const Vector<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_NotEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// Vector<P_numtype1> != double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_NotEqual<P_numtype1, double > > >
operator!=(const Vector<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_NotEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// Vector<P_numtype1> != long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_NotEqual<P_numtype1, long double > > >
operator!=(const Vector<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_NotEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Vector<P_numtype1> != complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_NotEqual<P_numtype1, complex<T2>  > > >
operator!=(const Vector<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_NotEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_VecExpr<P_expr1> != Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> != _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> != VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> != Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, int > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> != TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> != int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, int > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> != float
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, float > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// _bz_VecExpr<P_expr1> != double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, double > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// _bz_VecExpr<P_expr1> != long double
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_VecExpr<P_expr1> != complex<T2>
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator!=(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// VectorPick<P_numtype1> != Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> != _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> != VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> != Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_NotEqual<P_numtype1, int > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_NotEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> != TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> != int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_NotEqual<P_numtype1, int > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_NotEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> != float
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>,
      _bz_NotEqual<P_numtype1, float > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_NotEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// VectorPick<P_numtype1> != double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>,
      _bz_NotEqual<P_numtype1, double > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_NotEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// VectorPick<P_numtype1> != long double
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>,
      _bz_NotEqual<P_numtype1, long double > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_NotEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// VectorPick<P_numtype1> != complex<T2>
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_NotEqual<P_numtype1, complex<T2>  > > >
operator!=(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_NotEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// Range != Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<int, P_numtype2 > > >
operator!=(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range != _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<int, _bz_typename P_expr2::T_numtype > > >
operator!=(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range != VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<int, P_numtype2 > > >
operator!=(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range != Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_NotEqual<int, int > > >
operator!=(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_NotEqual<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range != TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<int, P_numtype2 > > >
operator!=(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range != float

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>,
      _bz_NotEqual<int, float > > >
operator!=(Range d1, 
      float d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<float>, 
      _bz_NotEqual<int, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2)));
}

// Range != double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>,
      _bz_NotEqual<int, double > > >
operator!=(Range d1, 
      double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<double>, 
      _bz_NotEqual<int, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2)));
}

// Range != long double

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>,
      _bz_NotEqual<int, long double > > >
operator!=(Range d1, 
      long double d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_NotEqual<int, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Range != complex<T2>
template<class T2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_NotEqual<int, complex<T2>  > > >
operator!=(Range d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_NotEqual<int, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// TinyVector<P_numtype1, N_length1> != Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> != _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> != VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> != Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_NotEqual<P_numtype1, int > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_NotEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> != TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> != int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_NotEqual<P_numtype1, int > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_NotEqual<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// TinyVector<P_numtype1, N_length1> != float
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>,
      _bz_NotEqual<P_numtype1, float > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_NotEqual<P_numtype1, float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2)));
}

// TinyVector<P_numtype1, N_length1> != double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>,
      _bz_NotEqual<P_numtype1, double > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_NotEqual<P_numtype1, double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2)));
}

// TinyVector<P_numtype1, N_length1> != long double
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>,
      _bz_NotEqual<P_numtype1, long double > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_NotEqual<P_numtype1, long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// TinyVector<P_numtype1, N_length1> != complex<T2>
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > ,
      _bz_NotEqual<P_numtype1, complex<T2>  > > >
operator!=(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_NotEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int != Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<int, P_numtype2 > > >
operator!=(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int != _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<int, _bz_typename P_expr2::T_numtype > > >
operator!=(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int != VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<int, P_numtype2 > > >
operator!=(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int != TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<int, P_numtype2 > > >
operator!=(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// float != Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<float, P_numtype2 > > >
operator!=(float d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float != _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<float, _bz_typename P_expr2::T_numtype > > >
operator!=(float d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float != VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<float, P_numtype2 > > >
operator!=(float d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// float != Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range,
      _bz_NotEqual<float, int > > >
operator!=(float d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      Range, 
      _bz_NotEqual<float, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2));
}

// float != TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<float, P_numtype2 > > >
operator!=(float d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<float, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<float>(d1), 
      d2.begin()));
}

// double != Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<double, P_numtype2 > > >
operator!=(double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double != _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<double, _bz_typename P_expr2::T_numtype > > >
operator!=(double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double != VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<double, P_numtype2 > > >
operator!=(double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// double != Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range,
      _bz_NotEqual<double, int > > >
operator!=(double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      Range, 
      _bz_NotEqual<double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2));
}

// double != TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<double, P_numtype2 > > >
operator!=(double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<double>(d1), 
      d2.begin()));
}

// long double != Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<long double, P_numtype2 > > >
operator!=(long double d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double != _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator!=(long double d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double != VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<long double, P_numtype2 > > >
operator!=(long double d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}

// long double != Range

inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range,
      _bz_NotEqual<long double, int > > >
operator!=(long double d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      Range, 
      _bz_NotEqual<long double, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2));
}

// long double != TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<long double, P_numtype2 > > >
operator!=(long double d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<long double, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<long double>(d1), 
      d2.begin()));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> != Vector<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>,
      _bz_NotEqual<complex<T1> , P_numtype2 > > >
operator!=(complex<T1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorIterConst<P_numtype2>, 
      _bz_NotEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> != _bz_VecExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>,
      _bz_NotEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator!=(complex<T1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      _bz_VecExpr<P_expr2>, 
      _bz_NotEqual<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> != VectorPick<P_numtype2>
template<class T1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>,
      _bz_NotEqual<complex<T1> , P_numtype2 > > >
operator!=(complex<T1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      VectorPickIterConst<P_numtype2>, 
      _bz_NotEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> != Range
template<class T1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range,
      _bz_NotEqual<complex<T1> , int > > >
operator!=(complex<T1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      Range, 
      _bz_NotEqual<complex<T1> , int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> != TinyVector<P_numtype2, N_length2>
template<class T1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_NotEqual<complex<T1> , P_numtype2 > > >
operator!=(complex<T1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<complex<T1> > , 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_NotEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<complex<T1> > (d1), 
      d2.begin()));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Logical AND operators
 ****************************************************************************/

// Vector<P_numtype1> && Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> && _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&&(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> && VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> && Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_LogicalAnd<P_numtype1, int > > >
operator&&(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_LogicalAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> && TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> && int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalAnd<P_numtype1, int > > >
operator&&(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> && Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&&(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> && _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator&&(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> && VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&&(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> && Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&&(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> && TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&&(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> && int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&&(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> && Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> && _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&&(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> && VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> && Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_LogicalAnd<P_numtype1, int > > >
operator&&(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_LogicalAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> && TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> && int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalAnd<P_numtype1, int > > >
operator&&(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range && Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalAnd<int, P_numtype2 > > >
operator&&(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range && _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&&(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalAnd<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range && VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalAnd<int, P_numtype2 > > >
operator&&(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range && Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_LogicalAnd<int, int > > >
operator&&(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_LogicalAnd<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range && TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalAnd<int, P_numtype2 > > >
operator&&(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> && Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> && _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&&(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> && VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> && Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_LogicalAnd<P_numtype1, int > > >
operator&&(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_LogicalAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> && TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> && int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalAnd<P_numtype1, int > > >
operator&&(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalAnd<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int && Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalAnd<int, P_numtype2 > > >
operator&&(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int && _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&&(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalAnd<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int && VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalAnd<int, P_numtype2 > > >
operator&&(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int && TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalAnd<int, P_numtype2 > > >
operator&&(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalAnd<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
/****************************************************************************
 * Logical OR operators
 ****************************************************************************/

// Vector<P_numtype1> || Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> || _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator||(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> || VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> || Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range,
      _bz_LogicalOr<P_numtype1, int > > >
operator||(const Vector<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_LogicalOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// Vector<P_numtype1> || TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// Vector<P_numtype1> || int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalOr<P_numtype1, int > > >
operator||(const Vector<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// _bz_VecExpr<P_expr1> || Vector<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator||(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> || _bz_VecExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator||(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> || VectorPick<P_numtype2>
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator||(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> || Range
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, int > > >
operator||(_bz_VecExpr<P_expr1> d1, 
      Range d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_VecExpr<P_expr1> || TinyVector<P_numtype2, N_length2>
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator||(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// _bz_VecExpr<P_expr1> || int
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, int > > >
operator||(_bz_VecExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_VecExprOp<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2)));
}

// VectorPick<P_numtype1> || Vector<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> || _bz_VecExpr<P_expr2>
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator||(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> || VectorPick<P_numtype2>
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> || Range
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range,
      _bz_LogicalOr<P_numtype1, int > > >
operator||(const VectorPick<P_numtype1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_LogicalOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// VectorPick<P_numtype1> || TinyVector<P_numtype2, N_length2>
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// VectorPick<P_numtype1> || int
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalOr<P_numtype1, int > > >
operator||(const VectorPick<P_numtype1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// Range || Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalOr<int, P_numtype2 > > >
operator||(Range d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range || _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalOr<int, _bz_typename P_expr2::T_numtype > > >
operator||(Range d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalOr<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range || VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalOr<int, P_numtype2 > > >
operator||(Range d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// Range || Range

inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      Range,
      _bz_LogicalOr<int, int > > >
operator||(Range d1, 
      Range d2)
{
    typedef _bz_VecExprOp<Range, 
      Range, 
      _bz_LogicalOr<int, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2));
}

// Range || TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalOr<int, P_numtype2 > > >
operator||(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> || Vector<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> || _bz_VecExpr<P_expr2>
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator||(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> || VectorPick<P_numtype2>
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> || Range
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range,
      _bz_LogicalOr<P_numtype1, int > > >
operator||(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_LogicalOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2));
}

// TinyVector<P_numtype1, N_length1> || TinyVector<P_numtype2, N_length2>
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin()));
}

// TinyVector<P_numtype1, N_length1> || int
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>,
      _bz_LogicalOr<P_numtype1, int > > >
operator||(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2)
{
    typedef _bz_VecExprOp<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_LogicalOr<P_numtype1, int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2)));
}

// int || Vector<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>,
      _bz_LogicalOr<int, P_numtype2 > > >
operator||(int d1, 
      const Vector<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype2>, 
      _bz_LogicalOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int || _bz_VecExpr<P_expr2>
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>,
      _bz_LogicalOr<int, _bz_typename P_expr2::T_numtype > > >
operator||(int d1, 
      _bz_VecExpr<P_expr2> d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr2>, 
      _bz_LogicalOr<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2));
}

// int || VectorPick<P_numtype2>
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>,
      _bz_LogicalOr<int, P_numtype2 > > >
operator||(int d1, 
      const VectorPick<P_numtype2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_LogicalOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}

// int || TinyVector<P_numtype2, N_length2>
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>,
      _bz_LogicalOr<int, P_numtype2 > > >
operator||(int d1, 
      const TinyVector<P_numtype2, N_length2>& d2)
{
    typedef _bz_VecExprOp<_bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_LogicalOr<int, P_numtype2> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(_bz_VecExprConstant<int>(d1), 
      d2.begin()));
}
BZ_NAMESPACE_END

#endif
