#ifndef BZ_ARRAYBOPS_CC
#define BZ_ARRAYBOPS_CC

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/arraybops.cc> must be included after <blitz/arrayexpr.h>
#endif

#include <blitz/array/asexpr.h>

BZ_NAMESPACE(blitz)

#define BZ_DECLARE_ARRAY_ET(name, applic)                                 \
                                                                          \
template<class T_numtype1, int N_rank1, class T_other>                    \
inline                                                                    \
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>,         \
    _bz_typename asExpr<T_other>::T_expr,                                 \
    applic<T_numtype1,                                                    \
    _bz_typename asExpr<T_other>::T_expr::T_numtype> > >                  \
name (const Array<T_numtype1,N_rank1>& d1,                                \
    const T_other& d2)                                                    \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1,        \
        N_rank1>,                                                         \
        _bz_typename asExpr<T_other>::T_expr,                             \
        applic<T_numtype1,                                                \
        _bz_typename asExpr<T_other>::T_expr::T_numtype> > >              \
      (d1.begin(),d2);                                                    \
}                                                                         \
                                                                          \
template<class T_expr1, class T_other>                                    \
inline                                                                    \
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
inline                                                                    \
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_typename asExpr<T1>::T_expr,            \
    _bz_typename asExpr<T2>::T_expr,                                      \
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
}

BZ_DECLARE_ARRAY_ET(operator+, Add)
BZ_DECLARE_ARRAY_ET(operator-, Subtract)
BZ_DECLARE_ARRAY_ET(operator*, Multiply)
BZ_DECLARE_ARRAY_ET(operator/, Divide)
BZ_DECLARE_ARRAY_ET(operator%, Modulo)
BZ_DECLARE_ARRAY_ET(operator^, BitwiseXor)
BZ_DECLARE_ARRAY_ET(operator&, BitwiseAnd)
BZ_DECLARE_ARRAY_ET(operator|, BitwiseOr)
BZ_DECLARE_ARRAY_ET(operator>>,ShiftRight)
BZ_DECLARE_ARRAY_ET(operator<<,ShiftLeft)
BZ_DECLARE_ARRAY_ET(operator>, Greater)
BZ_DECLARE_ARRAY_ET(operator<, Less)
BZ_DECLARE_ARRAY_ET(operator>=, GreaterOrEqual)
BZ_DECLARE_ARRAY_ET(operator<=, LessOrEqual)
BZ_DECLARE_ARRAY_ET(operator==, Equal)
BZ_DECLARE_ARRAY_ET(operator!=, NotEqual)
BZ_DECLARE_ARRAY_ET(operator&&, LogicalAnd)
BZ_DECLARE_ARRAY_ET(operator||, LogicalOr)

BZ_DECLARE_ARRAY_ET(atan2, _bz_atan2)

BZ_NAMESPACE_END

#endif // BZ_ARRAYBOPS_CC
