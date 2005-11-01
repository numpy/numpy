/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_VECANY_CC
#define BZ_VECANY_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecany.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr>
inline bool _bz_vec_any(P_expr vector)
{
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            if (vector._bz_fastAccess(i))
                return true;
    }
    else {
        for (int i=0; i < length; ++i)
            if (vector[i])
                return true;
    }

    return false;
}

template<typename P_numtype>
inline bool any(const Vector<P_numtype>& x)
{
    return _bz_vec_any(x._bz_asVecExpr());
}

template<typename P_expr>
inline bool any(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_any(expr);
}

template<typename P_numtype>
inline bool any(const VectorPick<P_numtype>& x)
{
    return _bz_vec_any(x._bz_asVecExpr());
}

template<typename P_numtype, int N_dimensions>
inline bool any(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_any(x._bz_asVecExpr());
}

BZ_NAMESPACE_END

#endif // BZ_VECANY_CC

