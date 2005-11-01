// Generated source file.  Do not edit.
// Created by: genmatbops.cpp Dec 10 2003 17:58:20

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
      _bz_Add<P_numtype1, typename P_expr2::T_numtype > > >
operator+(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Add<typename P_expr1::T_numtype, P_numtype2 > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Add<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> + _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Add<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> + int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Add<typename P_expr1::T_numtype, int > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Add<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> + float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Add<typename P_expr1::T_numtype, float > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Add<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> + double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Add<typename P_expr1::T_numtype, double > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Add<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> + long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Add<typename P_expr1::T_numtype, long double > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Add<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> + complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Add<typename P_expr1::T_numtype, complex<T2>  > > >
operator+(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Add<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_Add<int, typename P_expr2::T_numtype > > >
operator+(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Add<float, typename P_expr2::T_numtype > > >
operator+(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Add<double, typename P_expr2::T_numtype > > >
operator+(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Add<long double, typename P_expr2::T_numtype > > >
operator+(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Add<complex<T1> , typename P_expr2::T_numtype > > >
operator+(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Add<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_Subtract<P_numtype1, typename P_expr2::T_numtype > > >
operator-(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Subtract<typename P_expr1::T_numtype, P_numtype2 > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Subtract<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> - _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Subtract<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> - int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Subtract<typename P_expr1::T_numtype, int > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Subtract<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> - float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Subtract<typename P_expr1::T_numtype, float > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Subtract<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> - double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Subtract<typename P_expr1::T_numtype, double > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Subtract<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> - long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Subtract<typename P_expr1::T_numtype, long double > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Subtract<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> - complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Subtract<typename P_expr1::T_numtype, complex<T2>  > > >
operator-(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Subtract<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_Subtract<int, typename P_expr2::T_numtype > > >
operator-(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Subtract<float, typename P_expr2::T_numtype > > >
operator-(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Subtract<double, typename P_expr2::T_numtype > > >
operator-(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Subtract<long double, typename P_expr2::T_numtype > > >
operator-(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Subtract<complex<T1> , typename P_expr2::T_numtype > > >
operator-(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Subtract<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_Multiply<P_numtype1, typename P_expr2::T_numtype > > >
operator*(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Multiply<typename P_expr1::T_numtype, P_numtype2 > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Multiply<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> * _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Multiply<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> * int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Multiply<typename P_expr1::T_numtype, int > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Multiply<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> * float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Multiply<typename P_expr1::T_numtype, float > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Multiply<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> * double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Multiply<typename P_expr1::T_numtype, double > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Multiply<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> * long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Multiply<typename P_expr1::T_numtype, long double > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Multiply<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> * complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Multiply<typename P_expr1::T_numtype, complex<T2>  > > >
operator*(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Multiply<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_Multiply<int, typename P_expr2::T_numtype > > >
operator*(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Multiply<float, typename P_expr2::T_numtype > > >
operator*(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Multiply<double, typename P_expr2::T_numtype > > >
operator*(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Multiply<long double, typename P_expr2::T_numtype > > >
operator*(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Multiply<complex<T1> , typename P_expr2::T_numtype > > >
operator*(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Multiply<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_Divide<P_numtype1, typename P_expr2::T_numtype > > >
operator/(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Divide<typename P_expr1::T_numtype, P_numtype2 > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Divide<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> / _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Divide<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> / int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Divide<typename P_expr1::T_numtype, int > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Divide<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> / float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Divide<typename P_expr1::T_numtype, float > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Divide<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> / double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Divide<typename P_expr1::T_numtype, double > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Divide<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> / long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Divide<typename P_expr1::T_numtype, long double > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Divide<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> / complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Divide<typename P_expr1::T_numtype, complex<T2>  > > >
operator/(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Divide<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_Divide<int, typename P_expr2::T_numtype > > >
operator/(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Divide<float, typename P_expr2::T_numtype > > >
operator/(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Divide<double, typename P_expr2::T_numtype > > >
operator/(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Divide<long double, typename P_expr2::T_numtype > > >
operator/(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Divide<complex<T1> , typename P_expr2::T_numtype > > >
operator/(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Divide<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_Mod<P_numtype1, typename P_expr2::T_numtype > > >
operator%(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Mod<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Mod<typename P_expr1::T_numtype, P_numtype2 > > >
operator%(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Mod<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> % _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Mod<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator%(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Mod<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> % int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Mod<typename P_expr1::T_numtype, int > > >
operator%(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Mod<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_Mod<int, typename P_expr2::T_numtype > > >
operator%(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Mod<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_BitwiseXOR<P_numtype1, typename P_expr2::T_numtype > > >
operator^(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseXOR<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_BitwiseXOR<typename P_expr1::T_numtype, P_numtype2 > > >
operator^(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseXOR<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> ^ _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseXOR<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator^(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseXOR<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> ^ int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseXOR<typename P_expr1::T_numtype, int > > >
operator^(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseXOR<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_BitwiseXOR<int, typename P_expr2::T_numtype > > >
operator^(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseXOR<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_BitwiseAnd<P_numtype1, typename P_expr2::T_numtype > > >
operator&(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseAnd<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_BitwiseAnd<typename P_expr1::T_numtype, P_numtype2 > > >
operator&(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseAnd<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> & _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator&(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> & int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseAnd<typename P_expr1::T_numtype, int > > >
operator&(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseAnd<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_BitwiseAnd<int, typename P_expr2::T_numtype > > >
operator&(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseAnd<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_BitwiseOr<P_numtype1, typename P_expr2::T_numtype > > >
operator|(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseOr<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_BitwiseOr<typename P_expr1::T_numtype, P_numtype2 > > >
operator|(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_BitwiseOr<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> | _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_BitwiseOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator|(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> | int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_BitwiseOr<typename P_expr1::T_numtype, int > > >
operator|(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_BitwiseOr<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_BitwiseOr<int, typename P_expr2::T_numtype > > >
operator|(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_BitwiseOr<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_ShiftRight<P_numtype1, typename P_expr2::T_numtype > > >
operator>>(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftRight<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_ShiftRight<typename P_expr1::T_numtype, P_numtype2 > > >
operator>>(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftRight<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> >> _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftRight<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator>>(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftRight<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> >> int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_ShiftRight<typename P_expr1::T_numtype, int > > >
operator>>(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_ShiftRight<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_ShiftRight<int, typename P_expr2::T_numtype > > >
operator>>(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftRight<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_ShiftLeft<P_numtype1, typename P_expr2::T_numtype > > >
operator<<(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftLeft<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_ShiftLeft<typename P_expr1::T_numtype, P_numtype2 > > >
operator<<(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_ShiftLeft<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> << _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_ShiftLeft<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator<<(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftLeft<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> << int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_ShiftLeft<typename P_expr1::T_numtype, int > > >
operator<<(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_ShiftLeft<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_ShiftLeft<int, typename P_expr2::T_numtype > > >
operator<<(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_ShiftLeft<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Greater<P_numtype1, typename P_expr2::T_numtype > > >
operator>(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Greater<typename P_expr1::T_numtype, P_numtype2 > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Greater<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> > _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Greater<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> > int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Greater<typename P_expr1::T_numtype, int > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Greater<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> > float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Greater<typename P_expr1::T_numtype, float > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Greater<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> > double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Greater<typename P_expr1::T_numtype, double > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Greater<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> > long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Greater<typename P_expr1::T_numtype, long double > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Greater<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> > complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Greater<typename P_expr1::T_numtype, complex<T2>  > > >
operator>(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Greater<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_Greater<int, typename P_expr2::T_numtype > > >
operator>(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Greater<float, typename P_expr2::T_numtype > > >
operator>(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Greater<double, typename P_expr2::T_numtype > > >
operator>(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Greater<long double, typename P_expr2::T_numtype > > >
operator>(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Greater<complex<T1> , typename P_expr2::T_numtype > > >
operator>(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Greater<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_Less<P_numtype1, typename P_expr2::T_numtype > > >
operator<(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Less<typename P_expr1::T_numtype, P_numtype2 > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Less<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> < _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Less<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> < int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Less<typename P_expr1::T_numtype, int > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Less<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> < float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Less<typename P_expr1::T_numtype, float > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Less<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> < double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Less<typename P_expr1::T_numtype, double > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Less<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> < long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Less<typename P_expr1::T_numtype, long double > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Less<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> < complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Less<typename P_expr1::T_numtype, complex<T2>  > > >
operator<(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Less<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_Less<int, typename P_expr2::T_numtype > > >
operator<(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Less<float, typename P_expr2::T_numtype > > >
operator<(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Less<double, typename P_expr2::T_numtype > > >
operator<(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Less<long double, typename P_expr2::T_numtype > > >
operator<(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Less<complex<T1> , typename P_expr2::T_numtype > > >
operator<(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Less<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_GreaterOrEqual<P_numtype1, typename P_expr2::T_numtype > > >
operator>=(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, P_numtype2 > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> >= _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> >= int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, int > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> >= float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, float > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> >= double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, double > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> >= long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, long double > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> >= complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, complex<T2>  > > >
operator>=(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_GreaterOrEqual<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_GreaterOrEqual<int, typename P_expr2::T_numtype > > >
operator>=(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_GreaterOrEqual<float, typename P_expr2::T_numtype > > >
operator>=(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_GreaterOrEqual<double, typename P_expr2::T_numtype > > >
operator>=(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_GreaterOrEqual<long double, typename P_expr2::T_numtype > > >
operator>=(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_GreaterOrEqual<complex<T1> , typename P_expr2::T_numtype > > >
operator>=(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_GreaterOrEqual<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_LessOrEqual<P_numtype1, typename P_expr2::T_numtype > > >
operator<=(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LessOrEqual<typename P_expr1::T_numtype, P_numtype2 > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LessOrEqual<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> <= _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LessOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> <= int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_LessOrEqual<typename P_expr1::T_numtype, int > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_LessOrEqual<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> <= float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_LessOrEqual<typename P_expr1::T_numtype, float > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_LessOrEqual<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> <= double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_LessOrEqual<typename P_expr1::T_numtype, double > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_LessOrEqual<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> <= long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_LessOrEqual<typename P_expr1::T_numtype, long double > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_LessOrEqual<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> <= complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_LessOrEqual<typename P_expr1::T_numtype, complex<T2>  > > >
operator<=(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_LessOrEqual<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_LessOrEqual<int, typename P_expr2::T_numtype > > >
operator<=(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LessOrEqual<float, typename P_expr2::T_numtype > > >
operator<=(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LessOrEqual<double, typename P_expr2::T_numtype > > >
operator<=(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LessOrEqual<long double, typename P_expr2::T_numtype > > >
operator<=(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LessOrEqual<complex<T1> , typename P_expr2::T_numtype > > >
operator<=(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_LessOrEqual<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_Equal<P_numtype1, typename P_expr2::T_numtype > > >
operator==(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Equal<typename P_expr1::T_numtype, P_numtype2 > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Equal<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> == _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Equal<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> == int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Equal<typename P_expr1::T_numtype, int > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Equal<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> == float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_Equal<typename P_expr1::T_numtype, float > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_Equal<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> == double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_Equal<typename P_expr1::T_numtype, double > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_Equal<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> == long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_Equal<typename P_expr1::T_numtype, long double > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_Equal<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> == complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_Equal<typename P_expr1::T_numtype, complex<T2>  > > >
operator==(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_Equal<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_Equal<int, typename P_expr2::T_numtype > > >
operator==(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Equal<float, typename P_expr2::T_numtype > > >
operator==(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Equal<double, typename P_expr2::T_numtype > > >
operator==(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Equal<long double, typename P_expr2::T_numtype > > >
operator==(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_Equal<complex<T1> , typename P_expr2::T_numtype > > >
operator==(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_Equal<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_NotEqual<P_numtype1, typename P_expr2::T_numtype > > >
operator!=(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_NotEqual<typename P_expr1::T_numtype, P_numtype2 > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_NotEqual<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> != _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_NotEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> != int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_NotEqual<typename P_expr1::T_numtype, int > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_NotEqual<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> != float
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>,
      _bz_NotEqual<typename P_expr1::T_numtype, float > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<float>, 
      _bz_NotEqual<typename P_expr1::T_numtype, float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<float>(d2)));
}

// _bz_MatExpr<P_expr1> != double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>,
      _bz_NotEqual<typename P_expr1::T_numtype, double > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<double>, 
      _bz_NotEqual<typename P_expr1::T_numtype, double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<double>(d2)));
}

// _bz_MatExpr<P_expr1> != long double
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>,
      _bz_NotEqual<typename P_expr1::T_numtype, long double > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<long double>, 
      _bz_NotEqual<typename P_expr1::T_numtype, long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<long double>(d2)));
}
#ifdef BZ_HAVE_COMPLEX

// _bz_MatExpr<P_expr1> != complex<T2>
template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > ,
      _bz_NotEqual<typename P_expr1::T_numtype, complex<T2>  > > >
operator!=(_bz_MatExpr<P_expr1> d1, 
      complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<complex<T2> > , 
      _bz_NotEqual<typename P_expr1::T_numtype, complex<T2> > > T_expr;

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
      _bz_NotEqual<int, typename P_expr2::T_numtype > > >
operator!=(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_NotEqual<float, typename P_expr2::T_numtype > > >
operator!=(float d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<float, typename P_expr2::T_numtype> > T_expr;

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
      _bz_NotEqual<double, typename P_expr2::T_numtype > > >
operator!=(double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_NotEqual<long double, typename P_expr2::T_numtype > > >
operator!=(long double d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<long double, typename P_expr2::T_numtype> > T_expr;

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
      _bz_NotEqual<complex<T1> , typename P_expr2::T_numtype > > >
operator!=(complex<T1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , 
      _bz_MatExpr<P_expr2>, 
      _bz_NotEqual<complex<T1> , typename P_expr2::T_numtype> > T_expr;

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
      _bz_LogicalAnd<P_numtype1, typename P_expr2::T_numtype > > >
operator&&(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalAnd<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LogicalAnd<typename P_expr1::T_numtype, P_numtype2 > > >
operator&&(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalAnd<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> && _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator&&(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> && int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_LogicalAnd<typename P_expr1::T_numtype, int > > >
operator&&(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_LogicalAnd<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_LogicalAnd<int, typename P_expr2::T_numtype > > >
operator&&(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalAnd<int, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LogicalOr<P_numtype1, typename P_expr2::T_numtype > > >
operator||(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalOr<P_numtype1, typename P_expr2::T_numtype> > T_expr;

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
      _bz_LogicalOr<typename P_expr1::T_numtype, P_numtype2 > > >
operator||(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_LogicalOr<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> || _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_LogicalOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator||(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> || int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_LogicalOr<typename P_expr1::T_numtype, int > > >
operator||(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_LogicalOr<typename P_expr1::T_numtype, int> > T_expr;

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
      _bz_LogicalOr<int, typename P_expr2::T_numtype > > >
operator||(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_LogicalOr<int, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Minimum Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> min Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Min<P_numtype1, P_numtype2 > > >
min(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Min<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> min _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Min<P_numtype1, typename P_expr2::T_numtype > > >
min(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Min<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> min int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Min<P_numtype1, int > > >
min(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Min<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> min Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Min<typename P_expr1::T_numtype, P_numtype2 > > >
min(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Min<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> min _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Min<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
min(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Min<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> min int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Min<typename P_expr1::T_numtype, int > > >
min(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Min<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int min Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Min<int, P_numtype2 > > >
min(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Min<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int min _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Min<int, typename P_expr2::T_numtype > > >
min(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Min<int, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}
/****************************************************************************
 * Maximum Operators
 ****************************************************************************/

// Matrix<P_numtype1, P_struct1> max Matrix<P_numtype2, P_struct2>
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Max<P_numtype1, P_numtype2 > > >
max(const Matrix<P_numtype1, P_struct1>& d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Max<P_numtype1, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2._bz_getRef()));
}

// Matrix<P_numtype1, P_struct1> max _bz_MatExpr<P_expr2>
template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Max<P_numtype1, typename P_expr2::T_numtype > > >
max(const Matrix<P_numtype1, P_struct1>& d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Max<P_numtype1, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      d2));
}

// Matrix<P_numtype1, P_struct1> max int
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>,
      _bz_Max<P_numtype1, int > > >
max(const Matrix<P_numtype1, P_struct1>& d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, 
      _bz_MatExprConstant<int>, 
      _bz_Max<P_numtype1, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), 
      _bz_MatExprConstant<int>(d2)));
}

// _bz_MatExpr<P_expr1> max Matrix<P_numtype2, P_struct2>
template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Max<typename P_expr1::T_numtype, P_numtype2 > > >
max(_bz_MatExpr<P_expr1> d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Max<typename P_expr1::T_numtype, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2._bz_getRef()));
}

// _bz_MatExpr<P_expr1> max _bz_MatExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>,
      _bz_Max<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
max(_bz_MatExpr<P_expr1> d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Max<typename P_expr1::T_numtype, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      d2));
}

// _bz_MatExpr<P_expr1> max int
template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>,
      _bz_Max<typename P_expr1::T_numtype, int > > >
max(_bz_MatExpr<P_expr1> d1, 
      int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, 
      _bz_MatExprConstant<int>, 
      _bz_Max<typename P_expr1::T_numtype, int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, 
      _bz_MatExprConstant<int>(d2)));
}

// int max Matrix<P_numtype2, P_struct2>
template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>,
      _bz_Max<int, P_numtype2 > > >
max(int d1, 
      const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatrixRef<P_numtype2, P_struct2>, 
      _bz_Max<int, P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2._bz_getRef()));
}

// int max _bz_MatExpr<P_expr2>
template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>,
      _bz_Max<int, typename P_expr2::T_numtype > > >
max(int d1, 
      _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, 
      _bz_MatExpr<P_expr2>, 
      _bz_Max<int, typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), 
      d2));
}

BZ_NAMESPACE_END

#endif // BZ_MATBOPS_H
