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
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECANY_CC
#define BZ_VECANY_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecany.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_expr>
inline _bz_bool _bz_vec_any(P_expr vector)
{
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            if (vector._bz_fastAccess(i))
                return _bz_true;
    }
    else {
        for (int i=0; i < length; ++i)
            if (vector[i])
                return _bz_true;
    }

    return _bz_false;
}

template<class P_numtype>
inline _bz_bool any(const Vector<P_numtype>& x)
{
    return _bz_vec_any(x._bz_asVecExpr());
}

template<class P_expr>
inline _bz_bool any(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_any(expr);
}

template<class P_numtype>
inline _bz_bool any(const VectorPick<P_numtype>& x)
{
    return _bz_vec_any(x._bz_asVecExpr());
}

template<class P_numtype, int N_dimensions>
inline _bz_bool any(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_any(x._bz_asVecExpr());
}

BZ_NAMESPACE_END

#endif // BZ_VECANY_CC

