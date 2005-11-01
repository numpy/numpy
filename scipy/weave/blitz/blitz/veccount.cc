/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_VECCOUNT_CC
#define BZ_VECCOUNT_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/veccount.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr>
inline int _bz_vec_count(P_expr vector)
{
    int length = vector._bz_suggestLength();
    int count = 0;

    if (vector._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            if (vector._bz_fastAccess(i))
                ++count;
    }
    else {
        for (int i=0; i < length; ++i)
            if (vector[i])
                ++count;
    }

    return count;
}

template<typename P_numtype>
inline int count(const Vector<P_numtype>& x)
{
    return _bz_vec_count(x._bz_asVecExpr());
}

template<typename P_expr>
inline int count(_bz_VecExpr<P_expr> expr)
{
    return _bz_vec_count(expr);
}

template<typename P_numtype>
inline int count(const VectorPick<P_numtype>& x)
{
    return _bz_vec_count(x._bz_asVecExpr());
}

template<typename P_numtype, int N_dimensions>
inline int count(const TinyVector<P_numtype, N_dimensions>& x)
{
    return _bz_vec_count(x._bz_asVecExpr());
}

BZ_NAMESPACE_END

#endif // BZ_VECCOUNT_CC

