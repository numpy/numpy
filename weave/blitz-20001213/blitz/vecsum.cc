/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:25:28  ej
 * Looks like I need all the .cc files for blitz also
 *
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.4  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.3  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECSUM_CC
#define BZ_VECSUM_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecsum.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_expr>
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

template<class P_numtype>
inline
BZ_SUMTYPE(P_numtype) sum(const Vector<P_numtype>& x)
{
    return _bz_vec_sum(x._bz_asVecExpr());
}

// sum(expr)
template<class P_expr>
inline
BZ_SUMTYPE(_bz_typename P_expr::T_numtype)
sum(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_sum(expr);
}

// sum(vecpick)
template<class P_numtype>
inline
BZ_SUMTYPE(P_numtype)
sum(const VectorPick<P_numtype>& x)
{
    return _bz_vec_sum(x._bz_asVecExpr());
}

// mean(vector)
template<class P_numtype>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype)) mean(const Vector<P_numtype>& x)
{
    BZPRECONDITION(x.length() > 0);

    typedef BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype)) T_floattype;
    return _bz_vec_sum(x._bz_asVecExpr()) / (T_floattype) x.length();
}

// mean(expr)
template<class P_expr>
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
template<class P_numtype>
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

