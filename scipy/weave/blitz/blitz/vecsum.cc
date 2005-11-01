/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_VECSUM_CC
#define BZ_VECSUM_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecsum.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr>
inline
BZ_SUMTYPE(_bz_typename P_expr::T_numtype)
_bz_vec_sum(P_expr vector)
{
    typedef _bz_typename P_expr::T_numtype T_numtype;
    typedef BZ_SUMTYPE(T_numtype)          T_sumtype;

    T_sumtype sum = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            sum += vector._bz_fastAccess(i);
    }
    else {
        for (int i=0; i < length; ++i)
            sum += vector(i);
    }

    return sum;
}

template<typename P_numtype>
inline
BZ_SUMTYPE(P_numtype) sum(const Vector<P_numtype>& x)
{
    return _bz_vec_sum(x._bz_asVecExpr());
}

// sum(expr)
template<typename P_expr>
inline
BZ_SUMTYPE(_bz_typename P_expr::T_numtype)
sum(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_sum(expr);
}

// sum(vecpick)
template<typename P_numtype>
inline
BZ_SUMTYPE(P_numtype)
sum(const VectorPick<P_numtype>& x)
{
    return _bz_vec_sum(x._bz_asVecExpr());
}

// mean(vector)
template<typename P_numtype>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype)) mean(const Vector<P_numtype>& x)
{
    BZPRECONDITION(x.length() > 0);

    typedef BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype)) T_floattype;
    return _bz_vec_sum(x._bz_asVecExpr()) / (T_floattype) x.length();
}

// mean(expr)
template<typename P_expr>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(_bz_typename P_expr::T_numtype))
mean(_bz_VecExpr<P_expr> expr)
{
    BZPRECONDITION(expr._bz_suggestLength() > 0);

    typedef BZ_FLOATTYPE(BZ_SUMTYPE(_bz_typename P_expr::T_numtype)) 
        T_floattype;
    return _bz_vec_sum(expr) / (T_floattype) expr._bz_suggestLength();
}

// mean(vecpick)
template<typename P_numtype>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype))
mean(const VectorPick<P_numtype>& x)
{
    BZPRECONDITION(x.length() > 0);

    typedef BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype)) T_floattype;
    return _bz_vec_sum(x._bz_asVecExpr()) / (T_floattype) x.length();
}

BZ_NAMESPACE_END

#endif // BZ_VECSUM_CC

