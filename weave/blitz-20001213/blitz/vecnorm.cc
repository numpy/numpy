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
 * Revision 1.5  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.4  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.3  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECNORM_CC
#define BZ_VECNORM_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecnorm.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_expr>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(_bz_typename P_expr::T_numtype))
_bz_vec_norm(P_expr vector)
{
    // An extreme use of traits here.  It's necessary to
    // handle odd cases such as the norm of a Vector<char>.
    // To avoid overflow, BZ_SUMTYPE(char) is int.
    // To take the sqrt of the sum, BZ_FLOATTYPE(char) is float.
    // So float is returned for Vector<char>.
    typedef _bz_typename P_expr::T_numtype T_numtype;
    typedef BZ_SUMTYPE(T_numtype)          T_sumtype;
    typedef BZ_FLOATTYPE(T_sumtype)        T_floattype;

    T_sumtype sum = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
        {
            T_numtype value = vector._bz_fastAccess(i);
            sum += value * T_sumtype(value);
        }
    }
    else {
        for (int i=0; i < length; ++i)
        {
            T_numtype value = vector(i);
            sum += value * T_sumtype(value);
        }
    }

    return _bz_sqrt<T_floattype>::apply(sum);
}

template<class P_numtype>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype)) norm(const Vector<P_numtype>& x)
{
    return _bz_vec_norm(x._bz_asVecExpr());
}

// norm(expr)
template<class P_expr>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(_bz_typename P_expr::T_numtype))
norm(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_norm(expr);
}

// norm(vecpick)
template<class P_numtype>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype))
norm(const VectorPick<P_numtype>& x)
{
    return _bz_vec_norm(x._bz_asVecExpr());
}

// norm(TinyVector)
template<class P_numtype, int N_dimensions>
inline
BZ_FLOATTYPE(BZ_SUMTYPE(P_numtype))
norm(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_norm(x._bz_asVecExpr());
}

BZ_NAMESPACE_END

#endif // BZ_VECNORM_CC

