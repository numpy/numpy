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
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
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

#ifndef BZ_VECNORM1_CC
#define BZ_VECNORM1_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecnorm1.cc> must be included via <blitz/vecglobs.h>
#endif

#include <blitz/applics.h>

BZ_NAMESPACE(blitz)

template<class P_expr>
inline
BZ_SUMTYPE(_bz_typename P_expr::T_numtype)
_bz_vec_norm1(P_expr vector)
{
    typedef _bz_typename P_expr::T_numtype T_numtype;
    typedef BZ_SUMTYPE(T_numtype)          T_sumtype;

    T_sumtype sum = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            sum += _bz_abs<T_numtype>::apply(vector._bz_fastAccess(i));
    }
    else {
        for (int i=0; i < length; ++i)
            sum += _bz_abs<T_numtype>::apply(vector(i));
    }

    return sum;
}

// norm1(vector)
template<class P_numtype>
BZ_SUMTYPE(P_numtype) norm1(const Vector<P_numtype>& x)
{
    return _bz_vec_norm1(x._bz_asVecExpr());
}

// norm1(expr)
template<class P_expr>
BZ_SUMTYPE(_bz_typename P_expr::T_numtype) norm1(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_norm1(expr);
}

// norm1(vecpick)
template<class P_numtype>
BZ_SUMTYPE(P_numtype) norm1(const VectorPick<P_numtype>& x)
{
    return _bz_vec_norm1(x._bz_asVecExpr());
}

// norm1(TinyVector)
template<class P_numtype, int N_dimensions>
BZ_SUMTYPE(P_numtype) norm1(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_norm1(x._bz_asVecExpr());
}


BZ_NAMESPACE_END

#endif // BZ_VECNORM1_CC

