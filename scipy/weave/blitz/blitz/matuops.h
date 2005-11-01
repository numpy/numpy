// Generated source file.  Do not edit.
// Created by: genmatuops.cpp Dec 10 2003 17:58:05

#ifndef BZ_MATUOPS_H
#define BZ_MATUOPS_H

BZ_NAMESPACE(blitz)

#ifndef BZ_MATEXPR_H
 #error <blitz/matuops.h> must be included via <blitz/matexpr.h>
#endif

/****************************************************************************
 * abs
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_abs<P_numtype1> > >
abs(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_abs<typename P_expr1::T_numtype> > >
abs(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_abs<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * acos
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_acos<P_numtype1> > >
acos(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_acos<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_acos<typename P_expr1::T_numtype> > >
acos(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_acos<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * acosh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_acosh<P_numtype1> > >
acosh(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_acosh<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_acosh<typename P_expr1::T_numtype> > >
acosh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_acosh<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * asin
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_asin<P_numtype1> > >
asin(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_asin<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_asin<typename P_expr1::T_numtype> > >
asin(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_asin<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * asinh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_asinh<P_numtype1> > >
asinh(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_asinh<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_asinh<typename P_expr1::T_numtype> > >
asinh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_asinh<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * atan
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_atan<P_numtype1> > >
atan(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_atan<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_atan<typename P_expr1::T_numtype> > >
atan(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_atan<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * atan2
 ****************************************************************************/

template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_atan2<P_numtype1,P_numtype2> > >
atan2(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_atan2<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_atan2<P_numtype1,typename P_expr2::T_numtype> > >
atan2(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_atan2<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_atan2<P_numtype1,int> > >
atan2(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_atan2<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_atan2<P_numtype1,float> > >
atan2(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_atan2<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_atan2<P_numtype1,double> > >
atan2(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_atan2<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_atan2<P_numtype1,long double> > >
atan2(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_atan2<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_atan2<P_numtype1,complex<T2> > > >
atan2(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_atan2<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_atan2<typename P_expr1::T_numtype,P_numtype2> > >
atan2(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_atan2<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_atan2<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
atan2(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_atan2<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_atan2<typename P_expr1::T_numtype,int> > >
atan2(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_atan2<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_atan2<typename P_expr1::T_numtype,float> > >
atan2(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_atan2<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_atan2<typename P_expr1::T_numtype,double> > >
atan2(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_atan2<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_atan2<typename P_expr1::T_numtype,long double> > >
atan2(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_atan2<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_atan2<typename P_expr1::T_numtype,complex<T2> > > >
atan2(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_atan2<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_atan2<int,P_numtype2> > >
atan2(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_atan2<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_atan2<int,typename P_expr2::T_numtype> > >
atan2(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_atan2<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_atan2<float,P_numtype2> > >
atan2(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_atan2<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_atan2<float,typename P_expr2::T_numtype> > >
atan2(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_atan2<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_atan2<double,P_numtype2> > >
atan2(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_atan2<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_atan2<double,typename P_expr2::T_numtype> > >
atan2(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_atan2<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_atan2<long double,P_numtype2> > >
atan2(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_atan2<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_atan2<long double,typename P_expr2::T_numtype> > >
atan2(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_atan2<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_atan2<complex<T1> ,P_numtype2> > >
atan2(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_atan2<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_atan2<complex<T1> ,typename P_expr2::T_numtype> > >
atan2(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_atan2<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}


/****************************************************************************
 * atanh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_atanh<P_numtype1> > >
atanh(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_atanh<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_atanh<typename P_expr1::T_numtype> > >
atanh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_atanh<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * _class
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz__class<P_numtype1> > >
_class(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz__class<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz__class<typename P_expr1::T_numtype> > >
_class(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz__class<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * cbrt
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_cbrt<P_numtype1> > >
cbrt(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_cbrt<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_cbrt<typename P_expr1::T_numtype> > >
cbrt(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_cbrt<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * ceil
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_ceil<P_numtype1> > >
ceil(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_ceil<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_ceil<typename P_expr1::T_numtype> > >
ceil(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_ceil<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * copysign
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_copysign<P_numtype1,P_numtype2> > >
copysign(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_copysign<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_copysign<P_numtype1,typename P_expr2::T_numtype> > >
copysign(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_copysign<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_copysign<P_numtype1,int> > >
copysign(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_copysign<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_copysign<P_numtype1,float> > >
copysign(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_copysign<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_copysign<P_numtype1,double> > >
copysign(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_copysign<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_copysign<P_numtype1,long double> > >
copysign(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_copysign<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_copysign<P_numtype1,complex<T2> > > >
copysign(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_copysign<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_copysign<typename P_expr1::T_numtype,P_numtype2> > >
copysign(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_copysign<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_copysign<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
copysign(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_copysign<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_copysign<typename P_expr1::T_numtype,int> > >
copysign(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_copysign<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_copysign<typename P_expr1::T_numtype,float> > >
copysign(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_copysign<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_copysign<typename P_expr1::T_numtype,double> > >
copysign(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_copysign<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_copysign<typename P_expr1::T_numtype,long double> > >
copysign(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_copysign<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_copysign<typename P_expr1::T_numtype,complex<T2> > > >
copysign(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_copysign<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_copysign<int,P_numtype2> > >
copysign(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_copysign<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_copysign<int,typename P_expr2::T_numtype> > >
copysign(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_copysign<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_copysign<float,P_numtype2> > >
copysign(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_copysign<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_copysign<float,typename P_expr2::T_numtype> > >
copysign(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_copysign<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_copysign<double,P_numtype2> > >
copysign(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_copysign<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_copysign<double,typename P_expr2::T_numtype> > >
copysign(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_copysign<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_copysign<long double,P_numtype2> > >
copysign(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_copysign<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_copysign<long double,typename P_expr2::T_numtype> > >
copysign(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_copysign<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_copysign<complex<T1> ,P_numtype2> > >
copysign(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_copysign<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_copysign<complex<T1> ,typename P_expr2::T_numtype> > >
copysign(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_copysign<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}

#endif

/****************************************************************************
 * cos
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_cos<P_numtype1> > >
cos(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_cos<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_cos<typename P_expr1::T_numtype> > >
cos(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_cos<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * cosh
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_cosh<P_numtype1> > >
cosh(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_cosh<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_cosh<typename P_expr1::T_numtype> > >
cosh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_cosh<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * drem
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_drem<P_numtype1,P_numtype2> > >
drem(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_drem<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_drem<P_numtype1,typename P_expr2::T_numtype> > >
drem(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_drem<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_drem<P_numtype1,int> > >
drem(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_drem<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_drem<P_numtype1,float> > >
drem(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_drem<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_drem<P_numtype1,double> > >
drem(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_drem<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_drem<P_numtype1,long double> > >
drem(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_drem<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_drem<P_numtype1,complex<T2> > > >
drem(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_drem<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_drem<typename P_expr1::T_numtype,P_numtype2> > >
drem(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_drem<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_drem<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
drem(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_drem<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_drem<typename P_expr1::T_numtype,int> > >
drem(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_drem<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_drem<typename P_expr1::T_numtype,float> > >
drem(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_drem<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_drem<typename P_expr1::T_numtype,double> > >
drem(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_drem<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_drem<typename P_expr1::T_numtype,long double> > >
drem(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_drem<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_drem<typename P_expr1::T_numtype,complex<T2> > > >
drem(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_drem<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_drem<int,P_numtype2> > >
drem(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_drem<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_drem<int,typename P_expr2::T_numtype> > >
drem(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_drem<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_drem<float,P_numtype2> > >
drem(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_drem<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_drem<float,typename P_expr2::T_numtype> > >
drem(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_drem<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_drem<double,P_numtype2> > >
drem(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_drem<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_drem<double,typename P_expr2::T_numtype> > >
drem(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_drem<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_drem<long double,P_numtype2> > >
drem(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_drem<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_drem<long double,typename P_expr2::T_numtype> > >
drem(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_drem<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_drem<complex<T1> ,P_numtype2> > >
drem(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_drem<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_drem<complex<T1> ,typename P_expr2::T_numtype> > >
drem(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_drem<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}

#endif

/****************************************************************************
 * exp
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_exp<P_numtype1> > >
exp(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_exp<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_exp<typename P_expr1::T_numtype> > >
exp(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_exp<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * expm1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_expm1<P_numtype1> > >
expm1(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_expm1<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_expm1<typename P_expr1::T_numtype> > >
expm1(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_expm1<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * erf
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_erf<P_numtype1> > >
erf(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_erf<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_erf<typename P_expr1::T_numtype> > >
erf(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_erf<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * erfc
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_erfc<P_numtype1> > >
erfc(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_erfc<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_erfc<typename P_expr1::T_numtype> > >
erfc(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_erfc<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * fabs
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_abs<P_numtype1> > >
fabs(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_abs<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_abs<typename P_expr1::T_numtype> > >
fabs(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_abs<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * floor
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_floor<P_numtype1> > >
floor(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_floor<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_floor<typename P_expr1::T_numtype> > >
floor(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_floor<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * fmod
 ****************************************************************************/

template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_fmod<P_numtype1,P_numtype2> > >
fmod(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_fmod<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_fmod<P_numtype1,typename P_expr2::T_numtype> > >
fmod(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_fmod<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_fmod<P_numtype1,int> > >
fmod(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_fmod<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_fmod<P_numtype1,float> > >
fmod(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_fmod<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_fmod<P_numtype1,double> > >
fmod(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_fmod<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_fmod<P_numtype1,long double> > >
fmod(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_fmod<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_fmod<P_numtype1,complex<T2> > > >
fmod(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_fmod<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_fmod<typename P_expr1::T_numtype,P_numtype2> > >
fmod(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_fmod<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_fmod<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
fmod(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_fmod<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_fmod<typename P_expr1::T_numtype,int> > >
fmod(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_fmod<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_fmod<typename P_expr1::T_numtype,float> > >
fmod(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_fmod<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_fmod<typename P_expr1::T_numtype,double> > >
fmod(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_fmod<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_fmod<typename P_expr1::T_numtype,long double> > >
fmod(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_fmod<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_fmod<typename P_expr1::T_numtype,complex<T2> > > >
fmod(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_fmod<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_fmod<int,P_numtype2> > >
fmod(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_fmod<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_fmod<int,typename P_expr2::T_numtype> > >
fmod(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_fmod<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_fmod<float,P_numtype2> > >
fmod(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_fmod<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_fmod<float,typename P_expr2::T_numtype> > >
fmod(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_fmod<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_fmod<double,P_numtype2> > >
fmod(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_fmod<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_fmod<double,typename P_expr2::T_numtype> > >
fmod(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_fmod<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_fmod<long double,P_numtype2> > >
fmod(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_fmod<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_fmod<long double,typename P_expr2::T_numtype> > >
fmod(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_fmod<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_fmod<complex<T1> ,P_numtype2> > >
fmod(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_fmod<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_fmod<complex<T1> ,typename P_expr2::T_numtype> > >
fmod(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_fmod<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}


/****************************************************************************
 * hypot
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_hypot<P_numtype1,P_numtype2> > >
hypot(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_hypot<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_hypot<P_numtype1,typename P_expr2::T_numtype> > >
hypot(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_hypot<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_hypot<P_numtype1,int> > >
hypot(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_hypot<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_hypot<P_numtype1,float> > >
hypot(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_hypot<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_hypot<P_numtype1,double> > >
hypot(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_hypot<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_hypot<P_numtype1,long double> > >
hypot(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_hypot<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_hypot<P_numtype1,complex<T2> > > >
hypot(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_hypot<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_hypot<typename P_expr1::T_numtype,P_numtype2> > >
hypot(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_hypot<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_hypot<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
hypot(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_hypot<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_hypot<typename P_expr1::T_numtype,int> > >
hypot(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_hypot<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_hypot<typename P_expr1::T_numtype,float> > >
hypot(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_hypot<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_hypot<typename P_expr1::T_numtype,double> > >
hypot(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_hypot<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_hypot<typename P_expr1::T_numtype,long double> > >
hypot(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_hypot<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_hypot<typename P_expr1::T_numtype,complex<T2> > > >
hypot(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_hypot<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_hypot<int,P_numtype2> > >
hypot(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_hypot<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_hypot<int,typename P_expr2::T_numtype> > >
hypot(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_hypot<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_hypot<float,P_numtype2> > >
hypot(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_hypot<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_hypot<float,typename P_expr2::T_numtype> > >
hypot(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_hypot<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_hypot<double,P_numtype2> > >
hypot(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_hypot<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_hypot<double,typename P_expr2::T_numtype> > >
hypot(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_hypot<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_hypot<long double,P_numtype2> > >
hypot(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_hypot<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_hypot<long double,typename P_expr2::T_numtype> > >
hypot(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_hypot<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_hypot<complex<T1> ,P_numtype2> > >
hypot(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_hypot<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_hypot<complex<T1> ,typename P_expr2::T_numtype> > >
hypot(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_hypot<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}

#endif

/****************************************************************************
 * ilogb
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_ilogb<P_numtype1> > >
ilogb(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_ilogb<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_ilogb<typename P_expr1::T_numtype> > >
ilogb(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_ilogb<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * blitz_isnan
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_blitz_isnan<P_numtype1> > >
blitz_isnan(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_blitz_isnan<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_blitz_isnan<typename P_expr1::T_numtype> > >
blitz_isnan(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_blitz_isnan<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * itrunc
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_itrunc<P_numtype1> > >
itrunc(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_itrunc<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_itrunc<typename P_expr1::T_numtype> > >
itrunc(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_itrunc<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * j0
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_j0<P_numtype1> > >
j0(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_j0<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_j0<typename P_expr1::T_numtype> > >
j0(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_j0<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * j1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_j1<P_numtype1> > >
j1(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_j1<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_j1<typename P_expr1::T_numtype> > >
j1(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_j1<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * lgamma
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_lgamma<P_numtype1> > >
lgamma(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_lgamma<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_lgamma<typename P_expr1::T_numtype> > >
lgamma(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_lgamma<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * log
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_log<P_numtype1> > >
log(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_log<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_log<typename P_expr1::T_numtype> > >
log(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_log<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * logb
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_logb<P_numtype1> > >
logb(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_logb<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_logb<typename P_expr1::T_numtype> > >
logb(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_logb<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * log1p
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_log1p<P_numtype1> > >
log1p(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_log1p<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_log1p<typename P_expr1::T_numtype> > >
log1p(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_log1p<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * log10
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_log10<P_numtype1> > >
log10(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_log10<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_log10<typename P_expr1::T_numtype> > >
log10(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_log10<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * nearest
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_nearest<P_numtype1> > >
nearest(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_nearest<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_nearest<typename P_expr1::T_numtype> > >
nearest(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_nearest<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * nextafter
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_nextafter<P_numtype1,P_numtype2> > >
nextafter(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_nextafter<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_nextafter<P_numtype1,typename P_expr2::T_numtype> > >
nextafter(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_nextafter<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_nextafter<P_numtype1,int> > >
nextafter(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_nextafter<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_nextafter<P_numtype1,float> > >
nextafter(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_nextafter<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_nextafter<P_numtype1,double> > >
nextafter(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_nextafter<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_nextafter<P_numtype1,long double> > >
nextafter(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_nextafter<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_nextafter<P_numtype1,complex<T2> > > >
nextafter(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_nextafter<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_nextafter<typename P_expr1::T_numtype,P_numtype2> > >
nextafter(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_nextafter<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_nextafter<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
nextafter(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_nextafter<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_nextafter<typename P_expr1::T_numtype,int> > >
nextafter(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_nextafter<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_nextafter<typename P_expr1::T_numtype,float> > >
nextafter(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_nextafter<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_nextafter<typename P_expr1::T_numtype,double> > >
nextafter(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_nextafter<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_nextafter<typename P_expr1::T_numtype,long double> > >
nextafter(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_nextafter<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_nextafter<typename P_expr1::T_numtype,complex<T2> > > >
nextafter(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_nextafter<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_nextafter<int,P_numtype2> > >
nextafter(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_nextafter<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_nextafter<int,typename P_expr2::T_numtype> > >
nextafter(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_nextafter<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_nextafter<float,P_numtype2> > >
nextafter(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_nextafter<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_nextafter<float,typename P_expr2::T_numtype> > >
nextafter(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_nextafter<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_nextafter<double,P_numtype2> > >
nextafter(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_nextafter<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_nextafter<double,typename P_expr2::T_numtype> > >
nextafter(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_nextafter<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_nextafter<long double,P_numtype2> > >
nextafter(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_nextafter<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_nextafter<long double,typename P_expr2::T_numtype> > >
nextafter(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_nextafter<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_nextafter<complex<T1> ,P_numtype2> > >
nextafter(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_nextafter<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_nextafter<complex<T1> ,typename P_expr2::T_numtype> > >
nextafter(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_nextafter<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}

#endif

/****************************************************************************
 * pow
 ****************************************************************************/

template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_pow<P_numtype1,P_numtype2> > >
pow(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_pow<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_pow<P_numtype1,typename P_expr2::T_numtype> > >
pow(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_pow<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_pow<P_numtype1,int> > >
pow(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_pow<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_pow<P_numtype1,float> > >
pow(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_pow<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_pow<P_numtype1,double> > >
pow(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_pow<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_pow<P_numtype1,long double> > >
pow(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_pow<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_pow<P_numtype1,complex<T2> > > >
pow(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_pow<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_pow<typename P_expr1::T_numtype,P_numtype2> > >
pow(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_pow<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_pow<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
pow(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_pow<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_pow<typename P_expr1::T_numtype,int> > >
pow(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_pow<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_pow<typename P_expr1::T_numtype,float> > >
pow(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_pow<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_pow<typename P_expr1::T_numtype,double> > >
pow(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_pow<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_pow<typename P_expr1::T_numtype,long double> > >
pow(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_pow<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_pow<typename P_expr1::T_numtype,complex<T2> > > >
pow(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_pow<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_pow<int,P_numtype2> > >
pow(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_pow<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_pow<int,typename P_expr2::T_numtype> > >
pow(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_pow<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_pow<float,P_numtype2> > >
pow(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_pow<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_pow<float,typename P_expr2::T_numtype> > >
pow(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_pow<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_pow<double,P_numtype2> > >
pow(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_pow<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_pow<double,typename P_expr2::T_numtype> > >
pow(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_pow<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_pow<long double,P_numtype2> > >
pow(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_pow<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_pow<long double,typename P_expr2::T_numtype> > >
pow(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_pow<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_pow<complex<T1> ,P_numtype2> > >
pow(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_pow<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_pow<complex<T1> ,typename P_expr2::T_numtype> > >
pow(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_pow<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}


/****************************************************************************
 * remainder
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_remainder<P_numtype1,P_numtype2> > >
remainder(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_remainder<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_remainder<P_numtype1,typename P_expr2::T_numtype> > >
remainder(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_remainder<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_remainder<P_numtype1,int> > >
remainder(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_remainder<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_remainder<P_numtype1,float> > >
remainder(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_remainder<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_remainder<P_numtype1,double> > >
remainder(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_remainder<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_remainder<P_numtype1,long double> > >
remainder(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_remainder<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_remainder<P_numtype1,complex<T2> > > >
remainder(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_remainder<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_remainder<typename P_expr1::T_numtype,P_numtype2> > >
remainder(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_remainder<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_remainder<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
remainder(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_remainder<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_remainder<typename P_expr1::T_numtype,int> > >
remainder(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_remainder<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_remainder<typename P_expr1::T_numtype,float> > >
remainder(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_remainder<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_remainder<typename P_expr1::T_numtype,double> > >
remainder(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_remainder<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_remainder<typename P_expr1::T_numtype,long double> > >
remainder(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_remainder<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_remainder<typename P_expr1::T_numtype,complex<T2> > > >
remainder(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_remainder<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_remainder<int,P_numtype2> > >
remainder(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_remainder<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_remainder<int,typename P_expr2::T_numtype> > >
remainder(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_remainder<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_remainder<float,P_numtype2> > >
remainder(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_remainder<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_remainder<float,typename P_expr2::T_numtype> > >
remainder(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_remainder<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_remainder<double,P_numtype2> > >
remainder(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_remainder<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_remainder<double,typename P_expr2::T_numtype> > >
remainder(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_remainder<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_remainder<long double,P_numtype2> > >
remainder(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_remainder<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_remainder<long double,typename P_expr2::T_numtype> > >
remainder(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_remainder<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_remainder<complex<T1> ,P_numtype2> > >
remainder(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_remainder<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_remainder<complex<T1> ,typename P_expr2::T_numtype> > >
remainder(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_remainder<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}

#endif

/****************************************************************************
 * rint
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_rint<P_numtype1> > >
rint(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_rint<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_rint<typename P_expr1::T_numtype> > >
rint(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_rint<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * rsqrt
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_rsqrt<P_numtype1> > >
rsqrt(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_rsqrt<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_rsqrt<typename P_expr1::T_numtype> > >
rsqrt(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_rsqrt<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * scalb
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_scalb<P_numtype1,P_numtype2> > >
scalb(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_scalb<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_scalb<P_numtype1,typename P_expr2::T_numtype> > >
scalb(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_scalb<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_scalb<P_numtype1,int> > >
scalb(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_scalb<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_scalb<P_numtype1,float> > >
scalb(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_scalb<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_scalb<P_numtype1,double> > >
scalb(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_scalb<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_scalb<P_numtype1,long double> > >
scalb(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_scalb<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_scalb<P_numtype1,complex<T2> > > >
scalb(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_scalb<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_scalb<typename P_expr1::T_numtype,P_numtype2> > >
scalb(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_scalb<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_scalb<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
scalb(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_scalb<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_scalb<typename P_expr1::T_numtype,int> > >
scalb(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_scalb<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_scalb<typename P_expr1::T_numtype,float> > >
scalb(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_scalb<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_scalb<typename P_expr1::T_numtype,double> > >
scalb(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_scalb<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_scalb<typename P_expr1::T_numtype,long double> > >
scalb(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_scalb<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_scalb<typename P_expr1::T_numtype,complex<T2> > > >
scalb(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_scalb<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_scalb<int,P_numtype2> > >
scalb(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_scalb<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_scalb<int,typename P_expr2::T_numtype> > >
scalb(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_scalb<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_scalb<float,P_numtype2> > >
scalb(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_scalb<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_scalb<float,typename P_expr2::T_numtype> > >
scalb(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_scalb<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_scalb<double,P_numtype2> > >
scalb(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_scalb<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_scalb<double,typename P_expr2::T_numtype> > >
scalb(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_scalb<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_scalb<long double,P_numtype2> > >
scalb(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_scalb<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_scalb<long double,typename P_expr2::T_numtype> > >
scalb(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_scalb<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_scalb<complex<T1> ,P_numtype2> > >
scalb(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_scalb<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_scalb<complex<T1> ,typename P_expr2::T_numtype> > >
scalb(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_scalb<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}

#endif

/****************************************************************************
 * sin
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_sin<P_numtype1> > >
sin(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_sin<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_sin<typename P_expr1::T_numtype> > >
sin(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sin<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * sinh
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_sinh<P_numtype1> > >
sinh(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_sinh<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_sinh<typename P_expr1::T_numtype> > >
sinh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sinh<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * sqr
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_sqr<P_numtype1> > >
sqr(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_sqr<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_sqr<typename P_expr1::T_numtype> > >
sqr(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sqr<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * sqrt
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_sqrt<P_numtype1> > >
sqrt(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_sqrt<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_sqrt<typename P_expr1::T_numtype> > >
sqrt(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sqrt<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * tan
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_tan<P_numtype1> > >
tan(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_tan<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_tan<typename P_expr1::T_numtype> > >
tan(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_tan<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * tanh
 ****************************************************************************/

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_tanh<P_numtype1> > >
tanh(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_tanh<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_tanh<typename P_expr1::T_numtype> > >
tanh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_tanh<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


/****************************************************************************
 * uitrunc
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_uitrunc<P_numtype1> > >
uitrunc(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_uitrunc<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_uitrunc<typename P_expr1::T_numtype> > >
uitrunc(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_uitrunc<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * unordered
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_struct1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_unordered<P_numtype1,P_numtype2> > >
unordered(const Matrix<P_numtype1, P_struct1>& d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_unordered<P_numtype1,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2._bz_getRef()));
}

template<class P_numtype1, class P_struct1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
    _bz_unordered<P_numtype1,typename P_expr2::T_numtype> > >
unordered(const Matrix<P_numtype1, P_struct1>& d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExpr<P_expr2>,
        _bz_unordered<P_numtype1,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), d2));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
    _bz_unordered<P_numtype1,int> > >
unordered(const Matrix<P_numtype1, P_struct1>& d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<int>,
        _bz_unordered<P_numtype1,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<int>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
    _bz_unordered<P_numtype1,float> > >
unordered(const Matrix<P_numtype1, P_struct1>& d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<float>,
        _bz_unordered<P_numtype1,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<float>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
    _bz_unordered<P_numtype1,double> > >
unordered(const Matrix<P_numtype1, P_struct1>& d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<double>,
        _bz_unordered<P_numtype1,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<double>(d2)));
}

template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
    _bz_unordered<P_numtype1,long double> > >
unordered(const Matrix<P_numtype1, P_struct1>& d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<long double>,
        _bz_unordered<P_numtype1,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<long double>(d2)));
}

template<class P_numtype1, class P_struct1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_unordered<P_numtype1,complex<T2> > > >
unordered(const Matrix<P_numtype1, P_struct1>& d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatrixRef<P_numtype1, P_struct1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_unordered<P_numtype1,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef(), _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_expr1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_unordered<typename P_expr1::T_numtype,P_numtype2> > >
unordered(_bz_MatExpr<P_expr1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_unordered<typename P_expr1::T_numtype,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2._bz_getRef()));
}

template<class P_expr1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
    _bz_unordered<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
unordered(_bz_MatExpr<P_expr1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExpr<P_expr2>,
        _bz_unordered<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, d2));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
    _bz_unordered<typename P_expr1::T_numtype,int> > >
unordered(_bz_MatExpr<P_expr1> d1, int d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<int>,
        _bz_unordered<typename P_expr1::T_numtype,int> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<int>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
    _bz_unordered<typename P_expr1::T_numtype,float> > >
unordered(_bz_MatExpr<P_expr1> d1, float d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<float>,
        _bz_unordered<typename P_expr1::T_numtype,float> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<float>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
    _bz_unordered<typename P_expr1::T_numtype,double> > >
unordered(_bz_MatExpr<P_expr1> d1, double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<double>,
        _bz_unordered<typename P_expr1::T_numtype,double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<double>(d2)));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
    _bz_unordered<typename P_expr1::T_numtype,long double> > >
unordered(_bz_MatExpr<P_expr1> d1, long double d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<long double>,
        _bz_unordered<typename P_expr1::T_numtype,long double> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<long double>(d2)));
}

template<class P_expr1, class T2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
    _bz_unordered<typename P_expr1::T_numtype,complex<T2> > > >
unordered(_bz_MatExpr<P_expr1> d1, complex<T2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExpr<P_expr1>, _bz_MatExprConstant<complex<T2> > ,
        _bz_unordered<typename P_expr1::T_numtype,complex<T2> > > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1, _bz_MatExprConstant<complex<T2> > (d2)));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_unordered<int,P_numtype2> > >
unordered(int d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_unordered<int,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
    _bz_unordered<int,typename P_expr2::T_numtype> > >
unordered(int d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<int>, _bz_MatExpr<P_expr2>,
        _bz_unordered<int,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<int>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_unordered<float,P_numtype2> > >
unordered(float d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_unordered<float,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
    _bz_unordered<float,typename P_expr2::T_numtype> > >
unordered(float d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<float>, _bz_MatExpr<P_expr2>,
        _bz_unordered<float,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<float>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_unordered<double,P_numtype2> > >
unordered(double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_unordered<double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
    _bz_unordered<double,typename P_expr2::T_numtype> > >
unordered(double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<double>, _bz_MatExpr<P_expr2>,
        _bz_unordered<double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<double>(d1), d2));
}

template<class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_unordered<long double,P_numtype2> > >
unordered(long double d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_unordered<long double,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2._bz_getRef()));
}

template<class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
    _bz_unordered<long double,typename P_expr2::T_numtype> > >
unordered(long double d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<long double>, _bz_MatExpr<P_expr2>,
        _bz_unordered<long double,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<long double>(d1), d2));
}

template<class T1, class P_numtype2, class P_struct2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
    _bz_unordered<complex<T1> ,P_numtype2> > >
unordered(complex<T1> d1, const Matrix<P_numtype2, P_struct2>& d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatrixRef<P_numtype2, P_struct2>,
        _bz_unordered<complex<T1> ,P_numtype2> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2._bz_getRef()));
}

template<class T1, class P_expr2>
inline
_bz_MatExpr<_bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
    _bz_unordered<complex<T1> ,typename P_expr2::T_numtype> > >
unordered(complex<T1> d1, _bz_MatExpr<P_expr2> d2)
{
    typedef _bz_MatExprOp<_bz_MatExprConstant<complex<T1> > , _bz_MatExpr<P_expr2>,
        _bz_unordered<complex<T1> ,typename P_expr2::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(_bz_MatExprConstant<complex<T1> > (d1), d2));
}

#endif

/****************************************************************************
 * y0
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_y0<P_numtype1> > >
y0(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_y0<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_y0<typename P_expr1::T_numtype> > >
y0(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_y0<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif

/****************************************************************************
 * y1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1, class P_struct1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
    _bz_y1<P_numtype1> > >
y1(const Matrix<P_numtype1, P_struct1>& d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatrixRef<P_numtype1, P_struct1>,
        _bz_y1<P_numtype1> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1._bz_getRef()));
}

template<class P_expr1>
inline
_bz_MatExpr<_bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
    _bz_y1<typename P_expr1::T_numtype> > >
y1(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_y1<typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif


BZ_NAMESPACE_END

#endif // BZ_MATUOPS_H
