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

#ifndef BZ_VECMAX_CC
#define BZ_VECMAX_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecmax.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_expr>
inline
Extremum<_bz_typename P_expr::T_numtype, int> _bz_vec_max(P_expr vector)
{
    typedef _bz_typename P_expr::T_numtype T_numtype;

    T_numtype maxValue = vector(0);
    int maxIndex = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=1; i < length; ++i)
        {
            T_numtype value = vector._bz_fastAccess(i);
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
        }
    }
    else {
        for (int i=1; i < length; ++i)
        {
            T_numtype value = vector(i);
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
        }
    }

    return Extremum<T_numtype, int>(maxValue, maxIndex);
}

// max(vector)
template<class P_numtype>
inline
Extremum<P_numtype, int> max(const Vector<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr());
}

// max(expr)
template<class P_expr>
inline
Extremum<_bz_typename P_expr::T_numtype,int> max(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_max(x);
}

// max(vecpick)
template<class P_numtype>
inline
Extremum<P_numtype, int> max(const VectorPick<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr());
}

// max(TinyVector)
template<class P_numtype, int N_length>
inline
Extremum<P_numtype, int>
max(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_max(x._bz_asVecExpr());
}


// maxIndex(vector)
template<class P_numtype>
inline
int  maxIndex(const Vector<P_numtype>& x)
{
    return _bz_vec_max(x).index();
}

// maxIndex(expr)
template<class P_expr>
inline
int maxIndex(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_max(x._bz_asVecExpr()).index();
}

// maxIndex(vecpick)
template<class P_numtype>
int  maxIndex(const VectorPick<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).index();
}

// maxIndex(TinyVector)
template<class P_numtype, int N_length>
int  maxIndex(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).index();
}

// maxValue(vector)
template<class P_numtype>
inline
int  maxValue(const Vector<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).value();
}

// maxValue(expr)
template<class P_expr>
inline
int  maxValue(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_max(x).value();
}

// maxValue(vecpick)
template<class P_numtype>
int  maxValue(const VectorPick<P_numtype>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).value();
}

// maxValue(TinyVector)
template<class P_numtype, int N_length>
int  maxValue(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_max(x._bz_asVecExpr()).value();
}

BZ_NAMESPACE_END

#endif // BZ_VECMAX_CC

