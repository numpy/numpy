/***************************************************************************
 * blitz/meta/matassign.h   TinyMatrix assignment metaprogram
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


#ifndef BZ_META_MATASSIGN_H
#define BZ_META_MATASSIGN_H

BZ_NAMESPACE(blitz)

template<int N_rows, int N_columns, int I, int J>
class _bz_meta_matAssign2 {
public:
    enum { go = (J < N_columns - 1) ? 1 : 0 };

    template<class T_matrix, class T_expr, class T_updater>
    static inline void f(T_matrix& mat, T_expr expr, T_updater u)
    {
        u.update(mat(I,J), expr(I,J));
        _bz_meta_matAssign2<N_rows * go, N_columns * go, I * go, (J+1) * go>
            ::f(mat, expr, u);
    }
};

template<>
class _bz_meta_matAssign2<0,0,0,0> {
public:
    template<class T_matrix, class T_expr, class T_updater>
    static inline void f(T_matrix& mat, T_expr expr, T_updater u)
    { }
};

template<int N_rows, int N_columns, int I> 
class _bz_meta_matAssign {
public:
    enum { go = (I < N_rows-1) ? 1 : 0 };

    template<class T_matrix, class T_expr, class T_updater>
    static inline void f(T_matrix& mat, T_expr expr, T_updater u)
    {
        _bz_meta_matAssign2<N_rows, N_columns, I, 0>::f(mat, expr, u);
        _bz_meta_matAssign<N_rows * go, N_columns * go, (I+1) * go>
            ::f(mat, expr, u);
    }
};

template<>
class _bz_meta_matAssign<0,0,0> {
public:
    template<class T_matrix, class T_expr, class T_updater>
    static inline void f(T_matrix& mat, T_expr expr, T_updater u)
    { }
};


BZ_NAMESPACE_END

#endif // BZ_META_ASSIGN_H
