// Generated source file.  Do not edit.
// Created by: genmatuops.cpp Jun 28 2002 16:20:51

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
    _bz_abs<_bz_typename P_expr1::T_numtype> > >
abs(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_abs<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_acos<_bz_typename P_expr1::T_numtype> > >
acos(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_acos<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_acosh<_bz_typename P_expr1::T_numtype> > >
acosh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_acosh<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_asin<_bz_typename P_expr1::T_numtype> > >
asin(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_asin<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_asinh<_bz_typename P_expr1::T_numtype> > >
asinh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_asinh<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_atan<_bz_typename P_expr1::T_numtype> > >
atan(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_atan<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
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
    _bz_atanh<_bz_typename P_expr1::T_numtype> > >
atanh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_atanh<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz__class<_bz_typename P_expr1::T_numtype> > >
_class(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz__class<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_cbrt<_bz_typename P_expr1::T_numtype> > >
cbrt(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_cbrt<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_ceil<_bz_typename P_expr1::T_numtype> > >
ceil(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_ceil<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


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
    _bz_cos<_bz_typename P_expr1::T_numtype> > >
cos(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_cos<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_cosh<_bz_typename P_expr1::T_numtype> > >
cosh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_cosh<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


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
    _bz_exp<_bz_typename P_expr1::T_numtype> > >
exp(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_exp<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_expm1<_bz_typename P_expr1::T_numtype> > >
expm1(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_expm1<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_erf<_bz_typename P_expr1::T_numtype> > >
erf(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_erf<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_erfc<_bz_typename P_expr1::T_numtype> > >
erfc(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_erfc<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_abs<_bz_typename P_expr1::T_numtype> > >
fabs(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_abs<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_floor<_bz_typename P_expr1::T_numtype> > >
floor(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_floor<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}


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
    _bz_ilogb<_bz_typename P_expr1::T_numtype> > >
ilogb(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_ilogb<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_blitz_isnan<_bz_typename P_expr1::T_numtype> > >
blitz_isnan(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_blitz_isnan<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_itrunc<_bz_typename P_expr1::T_numtype> > >
itrunc(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_itrunc<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_j0<_bz_typename P_expr1::T_numtype> > >
j0(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_j0<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_j1<_bz_typename P_expr1::T_numtype> > >
j1(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_j1<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_lgamma<_bz_typename P_expr1::T_numtype> > >
lgamma(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_lgamma<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_log<_bz_typename P_expr1::T_numtype> > >
log(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_log<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_logb<_bz_typename P_expr1::T_numtype> > >
logb(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_logb<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_log1p<_bz_typename P_expr1::T_numtype> > >
log1p(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_log1p<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_log10<_bz_typename P_expr1::T_numtype> > >
log10(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_log10<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_nearest<_bz_typename P_expr1::T_numtype> > >
nearest(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_nearest<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
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
    _bz_rint<_bz_typename P_expr1::T_numtype> > >
rint(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_rint<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_rsqrt<_bz_typename P_expr1::T_numtype> > >
rsqrt(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_rsqrt<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
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
    _bz_sin<_bz_typename P_expr1::T_numtype> > >
sin(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sin<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_sinh<_bz_typename P_expr1::T_numtype> > >
sinh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sinh<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_sqr<_bz_typename P_expr1::T_numtype> > >
sqr(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sqr<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_sqrt<_bz_typename P_expr1::T_numtype> > >
sqrt(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_sqrt<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_tan<_bz_typename P_expr1::T_numtype> > >
tan(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_tan<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_tanh<_bz_typename P_expr1::T_numtype> > >
tanh(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_tanh<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_uitrunc<_bz_typename P_expr1::T_numtype> > >
uitrunc(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_uitrunc<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
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
    _bz_y0<_bz_typename P_expr1::T_numtype> > >
y0(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_y0<_bz_typename P_expr1::T_numtype> > T_expr;

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
    _bz_y1<_bz_typename P_expr1::T_numtype> > >
y1(_bz_MatExpr<P_expr1> d1)
{
    typedef _bz_MatExprUnaryOp<_bz_MatExpr<P_expr1>,
        _bz_y1<_bz_typename P_expr1::T_numtype> > T_expr;

    return _bz_MatExpr<T_expr>(T_expr(d1));
}

#endif


BZ_NAMESPACE_END

#endif // BZ_MATUOPS_H
