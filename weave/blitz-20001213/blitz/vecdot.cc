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

#ifndef BZ_VECDOT_CC
#define BZ_VECDOT_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecdot.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<class P1, class P2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P1::T_numtype, _bz_typename P2::T_numtype))
_bz_dot(P1 vector1, P2 vector2)
{
    BZPRECONDITION(vector1._bz_suggestLength() == vector2._bz_suggestLength());

    typedef BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P1::T_numtype,
        _bz_typename P2::T_numtype))  T_sumtype;

    T_sumtype sum = 0;
    int length = vector1._bz_suggestLength();

    if (vector1._bz_hasFastAccess() && vector2._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            sum += vector1._bz_fastAccess(i) 
                * vector2._bz_fastAccess(i);
    }
    else {
        for (int i=0; i < length; ++i)
            sum += vector1[i] * vector2[i];
    }

    return sum;
}


// dot()
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1,P_numtype2))
dot(const Vector<P_numtype1>& a, const Vector<P_numtype2>& b)
{
    return _bz_dot(a, b);
}

// dot(expr,expr)
template<class P_expr1, class P_expr2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P_expr1::T_numtype,
    _bz_typename P_expr2::T_numtype))
dot(_bz_VecExpr<P_expr1> expr1, _bz_VecExpr<P_expr2> expr2)
{
    return _bz_dot(expr1, expr2);
}

// dot(expr,vec)
template<class P_expr1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P_expr1::T_numtype, P_numtype2))
dot(_bz_VecExpr<P_expr1> expr1, const Vector<P_numtype2>& vector2)
{
    return _bz_dot(vector2, expr1);
}

// dot(vec,expr)
template<class P_numtype1, class P_expr2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, _bz_typename P_expr2::T_numtype))
dot(const Vector<P_numtype1>& vector1, _bz_VecExpr<P_expr2> expr2)
{
    return _bz_dot(vector1, expr2);
}

// dot(vec,vecpick)
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, P_numtype2))
dot(const Vector<P_numtype1>& vector1, const VectorPick<P_numtype2>& vector2)
{
    return _bz_dot(vector1, vector2);
}

// dot(vecpick,vec)
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, P_numtype2))
dot(const VectorPick<P_numtype1>& vector1, const Vector<P_numtype2>& vector2)
{
    return _bz_dot(vector1, vector2);
}

// dot(vecpick,vecpick)
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, P_numtype2))
dot(const VectorPick<P_numtype1>& vector1, const VectorPick<P_numtype2>& vector2)
{
    return _bz_dot(vector1, vector2);
}

// dot(expr, vecpick)
template<class P_expr1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P_expr1::T_numtype, P_numtype2))
dot(_bz_VecExpr<P_expr1> expr1, const VectorPick<P_numtype2>& vector2)
{
    return _bz_dot(expr1, vector2);
}

// dot(vecpick, expr)
template<class P_numtype1, class P_expr2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, _bz_typename P_expr2::T_numtype))
dot(const VectorPick<P_numtype1>& vector1, _bz_VecExpr<P_expr2> expr2)
{
    return _bz_dot(vector1, expr2);
}

BZ_NAMESPACE_END

#endif // BZ_VECDOT_CC

