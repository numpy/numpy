#ifndef BZ_ARRAY_ET_H
#define BZ_ARRAY_ET_H

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/array/et.h> must be included after <blitz/arrayexpr.h>
#endif

#include <blitz/array/asexpr.h>

#ifndef BZ_OPS_H
 #include <blitz/ops.h>
#endif

#ifndef BZ_MATHFUNC_H
 #include <blitz/mathfunc.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * Array expression templates: the macro BZ_DECLARE_ARRAY_ET(X,Y)
 * declares a function or operator which takes two operands.
 * X is the function name (or operator), and Y is the functor object
 * which implements the operation.
 */

#define BZ_DECLARE_ARRAY_ET(name, applic)                                 \
                                                                          \
template<class T_numtype1, int N_rank1, class T_other>                    \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprOp<FastArrayIterator<T_numtype1, N_rank1>,     \
    _bz_typename asExpr<T_other>::T_expr,                                 \
    applic<T_numtype1,                                                    \
    _bz_typename asExpr<T_other>::T_expr::T_numtype> > >                  \
name (const Array<T_numtype1,N_rank1>& d1,                                \
    const T_other& d2)                                                    \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprOp<FastArrayIterator<T_numtype1,    \
        N_rank1>,                                                         \
        _bz_typename asExpr<T_other>::T_expr,                             \
        applic<T_numtype1,                                                \
        _bz_typename asExpr<T_other>::T_expr::T_numtype> > >              \
      (d1.beginFast(),d2);                                                \
}                                                                         \
                                                                          \
template<class T_expr1, class T_other>                                    \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<T_expr1>,                     \
    _bz_typename asExpr<T_other>::T_expr,                                 \
    applic<_bz_typename T_expr1::T_numtype,                               \
        _bz_typename asExpr<T_other>::T_expr::T_numtype> > >              \
name(const _bz_ArrayExpr<T_expr1>& d1,                                    \
    const T_other& d2)                                                    \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<T_expr1>,          \
        _bz_typename asExpr<T_other>::T_expr,                             \
        applic<_bz_typename T_expr1::T_numtype,                           \
            _bz_typename asExpr<T_other>::T_expr::T_numtype> > >(d1,d2);  \
}                                                                         \
                                                                          \
template<class T1, class T2>                                              \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_typename asExpr<T1>::T_expr,            \
    _bz_typename asExpr<T2>::T_expr,                                      \
    applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                    \
        _bz_typename asExpr<T2>::T_expr::T_numtype> > >                   \
name(const ETBase<T1>& d1, const T2& d2)                                  \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprOp<_bz_typename asExpr<T1>::T_expr, \
        _bz_typename asExpr<T2>::T_expr,                                  \
        applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                \
            _bz_typename asExpr<T2>::T_expr::T_numtype> > >               \
        (static_cast<const T1&>(d1), d2);                                 \
}                                                                         \
                                                                          \
template<class T1, class T2>                                              \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_typename asExpr<T1>::T_expr,            \
_bz_typename asExpr<T2>::T_expr,                                          \
    applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                    \
        _bz_typename asExpr<T2>::T_expr::T_numtype> > >                   \
name(const T1& d1,                                                        \
    const ETBase<T2>& d2)                                                 \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprOp<_bz_typename                     \
        asExpr<T1>::T_expr,                                               \
        _bz_typename asExpr<T2>::T_expr,                                  \
        applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                \
            _bz_typename asExpr<T2>::T_expr::T_numtype> > >               \
        (d1, static_cast<const T2&>(d2));                                 \
}                                                                         \
                                                                          \
template<int N1, class T_other>                                           \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N1>,                       \
    _bz_typename asExpr<T_other>::T_expr,                                 \
    applic<int,                                                           \
        _bz_typename asExpr<T_other>::T_expr::T_numtype> > >              \
name(IndexPlaceholder<N1> d1,                                             \
    const T_other& d2)                                                    \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N1>,            \
        _bz_typename asExpr<T_other>::T_expr,                             \
        applic<int,                                                       \
            _bz_typename asExpr<T_other>::T_expr::T_numtype> > >(d1,d2);  \
}                                                                         \


// operator<< has been commented out because it causes ambiguity
// with statements like "cout << A".  NEEDS_WORK
// ditto operator<<

BZ_DECLARE_ARRAY_ET(operator+,  Add)
BZ_DECLARE_ARRAY_ET(operator-,  Subtract)
BZ_DECLARE_ARRAY_ET(operator*,  Multiply)
BZ_DECLARE_ARRAY_ET(operator/,  Divide)
BZ_DECLARE_ARRAY_ET(operator%,  Modulo)
BZ_DECLARE_ARRAY_ET(operator^,  BitwiseXor)
BZ_DECLARE_ARRAY_ET(operator&,  BitwiseAnd)
BZ_DECLARE_ARRAY_ET(operator|,  BitwiseOr)
// BZ_DECLARE_ARRAY_ET(operator>>, ShiftRight)
// BZ_DECLARE_ARRAY_ET(operator<<, ShiftLeft)
BZ_DECLARE_ARRAY_ET(operator>,  Greater)
BZ_DECLARE_ARRAY_ET(operator<,  Less)
BZ_DECLARE_ARRAY_ET(operator>=, GreaterOrEqual)
BZ_DECLARE_ARRAY_ET(operator<=, LessOrEqual)
BZ_DECLARE_ARRAY_ET(operator==, Equal)
BZ_DECLARE_ARRAY_ET(operator!=, NotEqual)
BZ_DECLARE_ARRAY_ET(operator&&, LogicalAnd)
BZ_DECLARE_ARRAY_ET(operator||, LogicalOr)

BZ_DECLARE_ARRAY_ET(atan2,      _bz_atan2)
BZ_DECLARE_ARRAY_ET(pow,        _bz_pow)

#ifdef BZ_HAVE_COMPLEX_MATH
BZ_DECLARE_ARRAY_ET(polar,     _bz_polar)
#endif

#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DECLARE_ARRAY_ET(copysign,  _bz_copysign)
BZ_DECLARE_ARRAY_ET(drem,      _bz_drem)
BZ_DECLARE_ARRAY_ET(fmod,      _bz_fmod)
BZ_DECLARE_ARRAY_ET(hypot,     _bz_hypot)
BZ_DECLARE_ARRAY_ET(nextafter, _bz_nextafter)
BZ_DECLARE_ARRAY_ET(remainder, _bz_remainder)
BZ_DECLARE_ARRAY_ET(scalb,     _bz_scalb)
BZ_DECLARE_ARRAY_ET(unordered, _bz_unordered)
#endif

/*
 * Unary functions and operators
 */

#define BZ_DECLARE_ARRAY_ET_UOP(name, functor)                        \
template<class T1>                                                    \
_bz_inline_et                                                         \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<                                   \
    _bz_typename asExpr<T1>::T_expr,                                  \
    functor<_bz_typename asExpr<T1>::T_expr::T_numtype> > >           \
name(const ETBase<T1>& d1)                                            \
{                                                                     \
    return _bz_ArrayExpr<_bz_ArrayExprUnaryOp<                        \
        _bz_typename asExpr<T1>::T_expr,                              \
        functor<_bz_typename asExpr<T1>::T_expr::T_numtype> > >       \
      (static_cast<const T1&>(d1));                                   \
}

BZ_DECLARE_ARRAY_ET_UOP(operator-, _bz_negate)

// NEEDS_WORK: operator!, operator~

BZ_DECLARE_ARRAY_ET_UOP(abs,   _bz_abs)
BZ_DECLARE_ARRAY_ET_UOP(acos,  _bz_acos)
BZ_DECLARE_ARRAY_ET_UOP(asin,  _bz_asin)
BZ_DECLARE_ARRAY_ET_UOP(atan,  _bz_atan)
BZ_DECLARE_ARRAY_ET_UOP(ceil,  _bz_ceil)
BZ_DECLARE_ARRAY_ET_UOP(cexp,  _bz_cexp)
BZ_DECLARE_ARRAY_ET_UOP(cos,   _bz_cos)
BZ_DECLARE_ARRAY_ET_UOP(cosh,  _bz_cosh)
BZ_DECLARE_ARRAY_ET_UOP(csqrt, _bz_csqrt)
BZ_DECLARE_ARRAY_ET_UOP(exp,   _bz_exp)
BZ_DECLARE_ARRAY_ET_UOP(fabs,  _bz_abs)
BZ_DECLARE_ARRAY_ET_UOP(floor, _bz_floor)
BZ_DECLARE_ARRAY_ET_UOP(log,   _bz_log)
BZ_DECLARE_ARRAY_ET_UOP(log10, _bz_log10)
BZ_DECLARE_ARRAY_ET_UOP(pow2,  _bz_pow2)
BZ_DECLARE_ARRAY_ET_UOP(pow3,  _bz_pow3)
BZ_DECLARE_ARRAY_ET_UOP(pow4,  _bz_pow4)
BZ_DECLARE_ARRAY_ET_UOP(pow5,  _bz_pow5)
BZ_DECLARE_ARRAY_ET_UOP(pow6,  _bz_pow6)
BZ_DECLARE_ARRAY_ET_UOP(pow7,  _bz_pow7)
BZ_DECLARE_ARRAY_ET_UOP(pow8,  _bz_pow8)
BZ_DECLARE_ARRAY_ET_UOP(sin,   _bz_sin)
BZ_DECLARE_ARRAY_ET_UOP(sinh,  _bz_sinh)
BZ_DECLARE_ARRAY_ET_UOP(sqr,   _bz_sqr)
BZ_DECLARE_ARRAY_ET_UOP(sqrt,  _bz_sqrt)
BZ_DECLARE_ARRAY_ET_UOP(tan,   _bz_tan)
BZ_DECLARE_ARRAY_ET_UOP(tanh,  _bz_tanh)

#ifdef BZ_HAVE_COMPLEX_MATH
BZ_DECLARE_ARRAY_ET_UOP(arg,   _bz_arg)
BZ_DECLARE_ARRAY_ET_UOP(conj,  _bz_conj)
#endif

#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DECLARE_ARRAY_ET_UOP(_class,  _bz__class)
BZ_DECLARE_ARRAY_ET_UOP(ilogb,   _bz_ilogb)
BZ_DECLARE_ARRAY_ET_UOP(itrunc,  _bz_itrunc)
BZ_DECLARE_ARRAY_ET_UOP(nearest, _bz_nearest)
BZ_DECLARE_ARRAY_ET_UOP(rsqrt,   _bz_rsqrt)
BZ_DECLARE_ARRAY_ET_UOP(uitrunc, _bz_uitrunc)
#endif

#ifdef BZ_HAVE_IEEE_MATH

// finite and trunc omitted: blitz-bugs/archive/0189.html
BZ_DECLARE_ARRAY_ET_UOP(acosh,  _bz_acosh)
BZ_DECLARE_ARRAY_ET_UOP(asinh,  _bz_asinh)
BZ_DECLARE_ARRAY_ET_UOP(atanh,  _bz_atanh)
BZ_DECLARE_ARRAY_ET_UOP(cbrt,   _bz_cbrt)
BZ_DECLARE_ARRAY_ET_UOP(expm1,  _bz_expm1)
BZ_DECLARE_ARRAY_ET_UOP(erf,    _bz_erf)
BZ_DECLARE_ARRAY_ET_UOP(erfc,   _bz_erfc)
// BZ_DECLARE_ARRAY_ET_UOP(finite, _bz_finite)
BZ_DECLARE_ARRAY_ET_UOP(isnan,  _bz_isnan)
BZ_DECLARE_ARRAY_ET_UOP(j0,     _bz_j0)
BZ_DECLARE_ARRAY_ET_UOP(j1,     _bz_j1)
BZ_DECLARE_ARRAY_ET_UOP(lgamma, _bz_lgamma)
BZ_DECLARE_ARRAY_ET_UOP(logb,   _bz_logb)
BZ_DECLARE_ARRAY_ET_UOP(log1p,  _bz_log1p)
BZ_DECLARE_ARRAY_ET_UOP(rint,   _bz_rint)
// BZ_DECLARE_ARRAY_ET_UOP(trunc,  _bz_trunc)
BZ_DECLARE_ARRAY_ET_UOP(y0,     _bz_y0)
BZ_DECLARE_ARRAY_ET_UOP(y1,     _bz_y1)
#endif


/*
 * User-defined expression template routines
 */

#define BZ_DECLARE_FUNCTION(name)                                     \
  template<class P_numtype>                                           \
  struct name ## _impl {                                              \
    typedef P_numtype T_numtype;                                      \
    template<class T>                                                 \
    static inline T apply(T x)                                        \
    { return name(x); }                                               \
                                                                      \
    template<class T1>                                                \
    static void prettyPrint(string& str,                              \
        prettyPrintFormat& format, const T1& a)                       \
    {                                                                 \
        str += #name;                                                 \
        str += "(";                                                   \
        a.prettyPrint(str,format);                                    \
        str += ")";                                                   \
    }                                                                 \
  };                                                                  \
                                                                      \
  BZ_DECLARE_ARRAY_ET_UOP(name, name ## _impl)

#define BZ_DECLARE_FUNCTION_RET(name, return_type)                    \
  template<class P_numtype>                                           \
  struct name ## _impl {                                              \
    typedef return_type T_numtype;                                    \
    template<class T>                                                 \
    static inline return_type apply(T x)                              \
    { return name(x); }                                               \
                                                                      \
    template<class T1>                                                \
    static void prettyPrint(string& str,                              \
        prettyPrintFormat& format, const T1& a)                       \
    {                                                                 \
        str += #name;                                                 \
        str += "(";                                                   \
        a.prettyPrint(str,format);                                    \
        str += ")";                                                   \
    }                                                                 \
  };                                                                  \
                                                                      \
  BZ_DECLARE_ARRAY_ET_UOP(name, name ## _impl)


#define BZ_DECLARE_FUNCTION2(name)                                    \
  template<class P_numtype1, class P_numtype2>                        \
  struct name ## _impl {                                              \
    typedef _bz_typename promote_trait<P_numtype1,                    \
        P_numtype2>::T_promote T_numtype;                             \
    template<class T1, class T2>                                      \
    static inline T_numtype apply(T1 x, T2 y)                         \
    { return name(x,y); }                                             \
                                                                      \
    template<class T1, class T2>                                      \
    static void prettyPrint(string& str,                              \
        prettyPrintFormat& format, const T1& a, const T2& b)          \
    {                                                                 \
        str += #name;                                                 \
        str += "(";                                                   \
        a.prettyPrint(str,format);                                    \
        str += ",";                                                   \
        b.prettyPrint(str,format);                                    \
        str += ")";                                                   \
    }                                                                 \
  };                                                                  \
                                                                      \
  BZ_DECLARE_ARRAY_ET(name, name ## _impl)

#define BZ_DECLARE_FUNCTION2_RET(name, return_type)                   \
  template<class P_numtype1, class P_numtype2>                        \
  struct name ## _impl {                                              \
    typedef return_type T_numtype;                                    \
    template<class T1, class T2>                                      \
    static inline T_numtype apply(T1 x, T2 y)                         \
    { return name(x,y); }                                             \
                                                                      \
    template<class T1, class T2>                                      \
    static void prettyPrint(string& str,                              \
        prettyPrintFormat& format, const T1& a, const T2& b)          \
    {                                                                 \
        str += #name;                                                 \
        str += "(";                                                   \
        a.prettyPrint(str,format);                                    \
        str += ",";                                                   \
        b.prettyPrint(str,format);                                    \
        str += ")";                                                   \
    }                                                                 \
  };                                                                  \
                                                                      \
  BZ_DECLARE_ARRAY_ET(name, name ## _impl)

BZ_NAMESPACE_END

#endif
