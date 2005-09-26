// Generated source file.  Do not edit.
// Created by: genmatbops.cpp Jun 28 2002 15:25:04

#ifndef BZ_MATBOPS_H
#define BZ_MATBOPS_H

BZ_NAMESPACE(blitz)

#ifndef BZ_MATEXPR_H
 #error <blitz/matbops.h> must be included via <blitz/matexpr.h>
#endif

/****************************************************************************
 * Addition Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> + Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Add<P_numtype1, P_numtype2 > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> + _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> + int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Add<P_numtype1, int > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Add<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> + float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_Add<P_numtype1, float > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_Add<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> + double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_Add<P_numtype1, double > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_Add<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> + long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_Add<P_numtype1, long double > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Add<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> + complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Add<P_numtype1, complex<T2>  > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Add<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> + Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> + _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Add<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> + int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Add<_bz_typename P_expr1::T_numtype, int > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> + float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Add<_bz_typename P_expr1::T_numtype, float > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> + double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Add<_bz_typename P_expr1::T_numtype, double > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> + long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Add<_bz_typename P_expr1::T_numtype, long double > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Add<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> + complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Add<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Add<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int + Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Add<int, P_numtype2 > > >
operator+(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int + _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Add<int, _bz_typename P_expr2::T_numtype > > >
operator+(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float + Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Add<float, P_numtype2 > > >
operator+(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float + _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_Add<float, _bz_typename P_expr2::T_numtype > > >
operator+(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double + Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Add<double, P_numtype2 > > >
operator+(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double + _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Add<double, _bz_typename P_expr2::T_numtype > > >
operator+(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double + Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Add<long double, P_numtype2 > > >
operator+(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double + _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Add<long double, _bz_typename P_expr2::T_numtype > > >
operator+(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> + Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Add<complex<T1> , P_numtype2 > > >
operator+(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> + _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_Add<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator+(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Subtraction Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> - Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Subtract<P_numtype1, P_numtype2 > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> - _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> - int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Subtract<P_numtype1, int > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Subtract<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> - float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_Subtract<P_numtype1, float > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_Subtract<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> - double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_Subtract<P_numtype1, double > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_Subtract<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> - long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_Subtract<P_numtype1, long double > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Subtract<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> - complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Subtract<P_numtype1, complex<T2>  > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Subtract<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> - Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> - _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> - int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, int > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> - float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, float > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> - double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, double > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> - long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, long double > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> - complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Subtract<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Subtract<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int - Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Subtract<int, P_numtype2 > > >
operator-(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int - _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<int, _bz_typename P_expr2::T_numtype > > >
operator-(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float - Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Subtract<float, P_numtype2 > > >
operator-(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float - _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<float, _bz_typename P_expr2::T_numtype > > >
operator-(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double - Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Subtract<double, P_numtype2 > > >
operator-(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double - _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<double, _bz_typename P_expr2::T_numtype > > >
operator-(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double - Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Subtract<long double, P_numtype2 > > >
operator-(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double - _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<long double, _bz_typename P_expr2::T_numtype > > >
operator-(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> - Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Subtract<complex<T1> , P_numtype2 > > >
operator-(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> - _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator-(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Multiplication Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> * Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Multiply<P_numtype1, P_numtype2 > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> * _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> * int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Multiply<P_numtype1, int > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Multiply<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> * float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_Multiply<P_numtype1, float > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_Multiply<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> * double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_Multiply<P_numtype1, double > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_Multiply<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> * long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_Multiply<P_numtype1, long double > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Multiply<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> * complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Multiply<P_numtype1, complex<T2>  > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Multiply<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> * Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> * _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> * int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, int > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> * float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, float > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> * double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, double > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> * long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, long double > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> * complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Multiply<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Multiply<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int * Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Multiply<int, P_numtype2 > > >
operator*(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int * _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<int, _bz_typename P_expr2::T_numtype > > >
operator*(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float * Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Multiply<float, P_numtype2 > > >
operator*(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float * _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<float, _bz_typename P_expr2::T_numtype > > >
operator*(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double * Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Multiply<double, P_numtype2 > > >
operator*(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double * _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<double, _bz_typename P_expr2::T_numtype > > >
operator*(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double * Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Multiply<long double, P_numtype2 > > >
operator*(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double * _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<long double, _bz_typename P_expr2::T_numtype > > >
operator*(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> * Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Multiply<complex<T1> , P_numtype2 > > >
operator*(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> * _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator*(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Division Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> / Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Divide<P_numtype1, P_numtype2 > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> / _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> / int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Divide<P_numtype1, int > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Divide<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> / float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_Divide<P_numtype1, float > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_Divide<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> / double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_Divide<P_numtype1, double > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_Divide<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> / long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_Divide<P_numtype1, long double > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Divide<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> / complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Divide<P_numtype1, complex<T2>  > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Divide<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> / Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> / _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> / int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, int > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> / float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, float > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> / double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, double > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> / long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Divide<_bz_typename P_expr1::T_numtype, long double > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Divide<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> / complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Divide<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Divide<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int / Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Divide<int, P_numtype2 > > >
operator/(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int / _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<int, _bz_typename P_expr2::T_numtype > > >
operator/(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float / Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Divide<float, P_numtype2 > > >
operator/(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float / _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<float, _bz_typename P_expr2::T_numtype > > >
operator/(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double / Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Divide<double, P_numtype2 > > >
operator/(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double / _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<double, _bz_typename P_expr2::T_numtype > > >
operator/(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double / Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Divide<long double, P_numtype2 > > >
operator/(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double / _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<long double, _bz_typename P_expr2::T_numtype > > >
operator/(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> / Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Divide<complex<T1> , P_numtype2 > > >
operator/(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> / _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator/(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Modulus Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> % Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Mod<P_numtype1, P_numtype2 > > >
operator%(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Mod<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> % _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator%(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Mod<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> % int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Mod<P_numtype1, int > > >
operator%(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Mod<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> % Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator%(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> % _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator%(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> % int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Mod<_bz_typename P_expr1::T_numtype, int > > >
operator%(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Mod<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int % Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Mod<int, P_numtype2 > > >
operator%(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Mod<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int % _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Mod<int, _bz_typename P_expr2::T_numtype > > >
operator%(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Mod<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Bitwise XOR Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> ^ Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseXOR<P_numtype1, P_numtype2 > > >
operator^(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseXOR<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> ^ _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator^(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseXOR<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> ^ int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseXOR<P_numtype1, int > > >
operator^(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseXOR<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> ^ Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator^(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> ^ _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator^(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> ^ int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, int > > >
operator^(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseXOR<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int ^ Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseXOR<int, P_numtype2 > > >
operator^(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseXOR<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int ^ _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseXOR<int, _bz_typename P_expr2::T_numtype > > >
operator^(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseXOR<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Bitwise And Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> & Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseAnd<P_numtype1, P_numtype2 > > >
operator&(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> & _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> & int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseAnd<P_numtype1, int > > >
operator&(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseAnd<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> & Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> & _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator&(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> & int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseAnd<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int & Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseAnd<int, P_numtype2 > > >
operator&(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseAnd<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int & _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseAnd<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Bitwise Or Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> | Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseOr<P_numtype1, P_numtype2 > > >
operator|(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> | _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator|(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> | int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseOr<P_numtype1, int > > >
operator|(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseOr<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> | Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator|(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> | _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator|(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> | int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, int > > >
operator|(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseOr<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int | Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_BitwiseOr<int, P_numtype2 > > >
operator|(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseOr<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int | _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseOr<int, _bz_typename P_expr2::T_numtype > > >
operator|(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseOr<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Shift right Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> >> Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_ShiftRight<P_numtype1, P_numtype2 > > >
operator>>(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftRight<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> >> _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>>(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftRight<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> >> int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_ShiftRight<P_numtype1, int > > >
operator>>(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_ShiftRight<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> >> Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>>(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> >> _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>>(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> >> int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, int > > >
operator>>(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_ShiftRight<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int >> Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_ShiftRight<int, P_numtype2 > > >
operator>>(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftRight<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int >> _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftRight<int, _bz_typename P_expr2::T_numtype > > >
operator>>(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftRight<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Shift left Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> << Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_ShiftLeft<P_numtype1, P_numtype2 > > >
operator<<(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftLeft<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> << _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<<(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftLeft<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> << int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_ShiftLeft<P_numtype1, int > > >
operator<<(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_ShiftLeft<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> << Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<<(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> << _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<<(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> << int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, int > > >
operator<<(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_ShiftLeft<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int << Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_ShiftLeft<int, P_numtype2 > > >
operator<<(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftLeft<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int << _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftLeft<int, _bz_typename P_expr2::T_numtype > > >
operator<<(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftLeft<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Greater-than Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> > Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Greater<P_numtype1, P_numtype2 > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> > _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> > int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Greater<P_numtype1, int > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Greater<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> > float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_Greater<P_numtype1, float > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_Greater<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> > double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_Greater<P_numtype1, double > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_Greater<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> > long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_Greater<P_numtype1, long double > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Greater<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> > complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Greater<P_numtype1, complex<T2>  > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Greater<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> > Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> > _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> > int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, int > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> > float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, float > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> > double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, double > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> > long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Greater<_bz_typename P_expr1::T_numtype, long double > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Greater<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> > complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Greater<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Greater<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int > Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Greater<int, P_numtype2 > > >
operator>(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int > _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<int, _bz_typename P_expr2::T_numtype > > >
operator>(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float > Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Greater<float, P_numtype2 > > >
operator>(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float > _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<float, _bz_typename P_expr2::T_numtype > > >
operator>(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double > Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Greater<double, P_numtype2 > > >
operator>(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double > _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<double, _bz_typename P_expr2::T_numtype > > >
operator>(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double > Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Greater<long double, P_numtype2 > > >
operator>(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double > _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<long double, _bz_typename P_expr2::T_numtype > > >
operator>(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> > Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Greater<complex<T1> , P_numtype2 > > >
operator>(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> > _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator>(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Less-than Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> < Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Less<P_numtype1, P_numtype2 > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> < _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> < int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Less<P_numtype1, int > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Less<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> < float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_Less<P_numtype1, float > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_Less<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> < double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_Less<P_numtype1, double > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_Less<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> < long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_Less<P_numtype1, long double > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Less<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> < complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Less<P_numtype1, complex<T2>  > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Less<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> < Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> < _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Less<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> < int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Less<_bz_typename P_expr1::T_numtype, int > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> < float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Less<_bz_typename P_expr1::T_numtype, float > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> < double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Less<_bz_typename P_expr1::T_numtype, double > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> < long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Less<_bz_typename P_expr1::T_numtype, long double > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Less<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> < complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Less<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Less<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int < Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Less<int, P_numtype2 > > >
operator<(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int < _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Less<int, _bz_typename P_expr2::T_numtype > > >
operator<(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float < Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Less<float, P_numtype2 > > >
operator<(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float < _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_Less<float, _bz_typename P_expr2::T_numtype > > >
operator<(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double < Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Less<double, P_numtype2 > > >
operator<(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double < _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Less<double, _bz_typename P_expr2::T_numtype > > >
operator<(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double < Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Less<long double, P_numtype2 > > >
operator<(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double < _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Less<long double, _bz_typename P_expr2::T_numtype > > >
operator<(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> < Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Less<complex<T1> , P_numtype2 > > >
operator<(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> < _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_Less<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator<(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Greater or equal (>=) operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> >= Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_GreaterOrEqual<P_numtype1, P_numtype2 > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> >= _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> >= int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_GreaterOrEqual<P_numtype1, int > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_GreaterOrEqual<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> >= float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_GreaterOrEqual<P_numtype1, float > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_GreaterOrEqual<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> >= double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_GreaterOrEqual<P_numtype1, double > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_GreaterOrEqual<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> >= long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_GreaterOrEqual<P_numtype1, long double > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_GreaterOrEqual<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> >= complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<P_numtype1, complex<T2>  > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> >= Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> >= _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> >= int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> >= float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, float > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> >= double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, double > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> >= long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> >= complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int >= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_GreaterOrEqual<int, P_numtype2 > > >
operator>=(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int >= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator>=(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float >= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_GreaterOrEqual<float, P_numtype2 > > >
operator>=(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float >= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<float, _bz_typename P_expr2::T_numtype > > >
operator>=(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double >= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_GreaterOrEqual<double, P_numtype2 > > >
operator>=(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double >= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<double, _bz_typename P_expr2::T_numtype > > >
operator>=(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double >= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_GreaterOrEqual<long double, P_numtype2 > > >
operator>=(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double >= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator>=(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> >= Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_GreaterOrEqual<complex<T1> , P_numtype2 > > >
operator>=(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> >= _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator>=(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Less or equal (<=) operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> <= Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LessOrEqual<P_numtype1, P_numtype2 > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> <= _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> <= int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_LessOrEqual<P_numtype1, int > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_LessOrEqual<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> <= float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_LessOrEqual<P_numtype1, float > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_LessOrEqual<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> <= double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_LessOrEqual<P_numtype1, double > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_LessOrEqual<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> <= long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_LessOrEqual<P_numtype1, long double > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_LessOrEqual<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> <= complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_LessOrEqual<P_numtype1, complex<T2>  > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_LessOrEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> <= Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> <= _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> <= int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> <= float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, float > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> <= double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, double > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> <= long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> <= complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_LessOrEqual<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int <= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LessOrEqual<int, P_numtype2 > > >
operator<=(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int <= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator<=(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float <= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LessOrEqual<float, P_numtype2 > > >
operator<=(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float <= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<float, _bz_typename P_expr2::T_numtype > > >
operator<=(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double <= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LessOrEqual<double, P_numtype2 > > >
operator<=(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double <= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<double, _bz_typename P_expr2::T_numtype > > >
operator<=(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double <= Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LessOrEqual<long double, P_numtype2 > > >
operator<=(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double <= _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator<=(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> <= Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LessOrEqual<complex<T1> , P_numtype2 > > >
operator<=(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> <= _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator<=(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Equality operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> == Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Equal<P_numtype1, P_numtype2 > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> == _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> == int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Equal<P_numtype1, int > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Equal<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> == float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_Equal<P_numtype1, float > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_Equal<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> == double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_Equal<P_numtype1, double > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_Equal<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> == long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_Equal<P_numtype1, long double > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Equal<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> == complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Equal<P_numtype1, complex<T2>  > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Equal<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> == Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> == _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> == int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, int > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> == float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, float > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> == double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, double > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> == long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Equal<_bz_typename P_expr1::T_numtype, long double > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Equal<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> == complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Equal<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Equal<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int == Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Equal<int, P_numtype2 > > >
operator==(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int == _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<int, _bz_typename P_expr2::T_numtype > > >
operator==(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float == Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Equal<float, P_numtype2 > > >
operator==(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float == _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<float, _bz_typename P_expr2::T_numtype > > >
operator==(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double == Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Equal<double, P_numtype2 > > >
operator==(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double == _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<double, _bz_typename P_expr2::T_numtype > > >
operator==(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double == Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Equal<long double, P_numtype2 > > >
operator==(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double == _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<long double, _bz_typename P_expr2::T_numtype > > >
operator==(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> == Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Equal<complex<T1> , P_numtype2 > > >
operator==(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> == _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator==(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Not-equal operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> != Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_NotEqual<P_numtype1, P_numtype2 > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> != _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> != int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_NotEqual<P_numtype1, int > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_NotEqual<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// Matrix<P_numtype1, P_struct1> != float
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>,
      _bz_NotEqual<P_numtype1, float > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<float>, 
      _bz_NotEqual<P_numtype1, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<float>(d2)));
}

// Matrix<P_numtype1, P_struct1> != double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>,
      _bz_NotEqual<P_numtype1, double > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<double>, 
      _bz_NotEqual<P_numtype1, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<double>(d2)));
}

// Matrix<P_numtype1, P_struct1> != long double
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>,
      _bz_NotEqual<P_numtype1, long double > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<long double>, 
      _bz_NotEqual<P_numtype1, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// Matrix<P_numtype1, P_struct1> != complex<T2>
template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_NotEqual<P_numtype1, complex<T2>  > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_NotEqual<P_numtype1, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// _bz_MatExpr<P_expr1> != Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> != _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> != int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, int > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> != float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, float > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> != double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, double > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> != long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> != complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_NotEqual<_bz_typename P_expr1::T_numtype, complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<complex<T2> > (d2)));
}
#endif // BZ_HAVE_COMPLEX


// int != Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_NotEqual<int, P_numtype2 > > >
operator!=(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int != _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<int, _bz_typename P_expr2::T_numtype > > >
operator!=(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

// float != Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_NotEqual<float, P_numtype2 > > >
operator!=(float d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<float, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2._bz_getRef()));
}

// float != _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<float, _bz_typename P_expr2::T_numtype > > >
operator!=(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<float, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), 
      d2));
}

// double != Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_NotEqual<double, P_numtype2 > > >
operator!=(double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2._bz_getRef()));
}

// double != _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<double, _bz_typename P_expr2::T_numtype > > >
operator!=(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), 
      d2));
}

// long double != Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_NotEqual<long double, P_numtype2 > > >
operator!=(long double d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<long double, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2._bz_getRef()));
}

// long double != _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator!=(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<long double, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), 
      d2));
}
#ifdef BZ_HAVE_COMPLEX

// complex<T1> != Matrix<P_numtype2, P_struct2>
template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_NotEqual<complex<T1> , P_numtype2 > > >
operator!=(complex<T1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<complex<T1> , P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2._bz_getRef()));
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX

// complex<T1> != _bz_MatExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator!=(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<complex<T1> , _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), 
      d2));
}
#endif // BZ_HAVE_COMPLEX

/****************************************************************************
 * Logical AND operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> && Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LogicalAnd<P_numtype1, P_numtype2 > > >
operator&&(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalAnd<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> && _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&&(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalAnd<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> && int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_LogicalAnd<P_numtype1, int > > >
operator&&(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_LogicalAnd<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> && Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator&&(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> && _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator&&(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> && int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&&(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_LogicalAnd<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int && Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LogicalAnd<int, P_numtype2 > > >
operator&&(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalAnd<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int && _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&&(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalAnd<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Logical OR operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> || Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LogicalOr<P_numtype1, P_numtype2 > > >
operator||(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalOr<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> || _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype > > >
operator||(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalOr<P_numtype1, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> || int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_LogicalOr<P_numtype1, int > > >
operator||(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_LogicalOr<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> || Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2 > > >
operator||(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> || _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator||(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> || int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, int > > >
operator||(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_LogicalOr<_bz_typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int || Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_LogicalOr<int, P_numtype2 > > >
operator||(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalOr<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int || _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalOr<int, _bz_typename P_expr2::T_numtype > > >
operator||(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalOr<int, _bz_typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

BZ_NAMESPACE_END

#endif // BZ_MATBOPS_H
