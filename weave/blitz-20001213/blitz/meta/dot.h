/***************************************************************************
 * blitz/meta/dot.h      Tiny vector dot product metaprogram
 *
 * $Id$
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************
 * $Log$
 * Revision 1.1  2002/01/03 19:50:38  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:13  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:08:44  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_META_DOT_H
#define BZ_META_DOT_H

#ifndef BZ_PROMOTE_H
 #include <blitz/promote.h>
#endif

#ifndef BZ_METAPROG_H
 #include <blitz/meta/metaprog.h>
#endif

BZ_NAMESPACE(blitz)

template<int N, int I>
class _bz_meta_vectorDot {
public:
    enum { loopFlag = (I < N-1) ? 1 : 0 };

    template<class T_expr1, class T_expr2>
    static inline BZ_PROMOTE(_bz_typename T_expr1::T_numtype,
        _bz_typename T_expr2::T_numtype)
    f(const T_expr1& a, const T_expr2& b)
    {
        return a[I] * b[I]
            + _bz_meta_vectorDot<loopFlag * N, loopFlag * (I+1)>::f(a,b);
    }

    template<class T_expr1, class T_expr2>
    static inline BZ_PROMOTE(_bz_typename T_expr1::T_numtype,
        _bz_typename T_expr2::T_numtype)
    f_value_ref(T_expr1 a, const T_expr2& b)
    {
        return a[I] * b[I]
            + _bz_meta_vectorDot<loopFlag * N, loopFlag * (I+1)>::f(a,b);
    }

    template<class T_expr1, class T_expr2>
    static inline BZ_PROMOTE(_bz_typename T_expr1::T_numtype,
        _bz_typename T_expr2::T_numtype)
    f_ref_value(const T_expr1& a, T_expr2 b)
    {
        return a[I] * b[I]
            + _bz_meta_vectorDot<loopFlag * N, loopFlag * (I+1)>::f(a,b);
    }

    template<class T_expr1, class P_numtype2>
    static inline BZ_PROMOTE(_bz_typename T_expr1::T_numtype,
        P_numtype2)
    dotWithArgs(const T_expr1& a, P_numtype2 i1, P_numtype2 i2=0,
        P_numtype2 i3=0, P_numtype2 i4=0, P_numtype2 i5=0, P_numtype2 i6=0,
        P_numtype2 i7=0, P_numtype2 i8=0, P_numtype2 i9=0, P_numtype2 i10=0)
    {
        return a[I] * i1  
            + _bz_meta_vectorDot<loopFlag * N, loopFlag * (I+1)>::dotWithArgs
                 (a, i2, i3, i4, i5, i6, i7, i8, i9);
    }
};

template<>
class _bz_meta_vectorDot<0,0> {
public:
    template<class T_expr1, class T_expr2>
    static inline _bz_meta_nullOperand f(const T_expr1&, const T_expr2&)
    { return _bz_meta_nullOperand(); }

    template<class T_expr1, class P_numtype2>
    static inline _bz_meta_nullOperand 
    dotWithArgs(const T_expr1& a, P_numtype2 i1, P_numtype2 i2=0,
        P_numtype2 i3=0, P_numtype2 i4=0, P_numtype2 i5=0, P_numtype2 i6=0,
        P_numtype2 i7=0, P_numtype2 i8=0, P_numtype2 i9=0, P_numtype2 i10=0)
    {
        return _bz_meta_nullOperand(); 
    }

};

BZ_NAMESPACE_END

#endif // BZ_META_DOT_H
