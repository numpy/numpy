/***************************************************************************
 * blitz/meta/matmat.h   TinyMatrix matrix-matrix product metaprogram
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
 * Revision 1.3  1998/03/14 00:08:44  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_META_MATMAT_H
#define BZ_META_MATMAT_H

#ifndef BZ_TINYMAT_H
 #error <blitz/meta/matmat.h> must be included via <blitz/tinymat.h>
#endif

#include <blitz/meta/metaprog.h>
#include <blitz/tinymatexpr.h>

BZ_NAMESPACE(blitz)

// Template metaprogram for matrix-matrix multiplication
template<int N_rows1, int N_columns, int N_columns2, int N_rowStride1,
    int N_colStride1, int N_rowStride2, int N_colStride2, int K>
class _bz_meta_matrixMatrixProduct {
public:
    enum { go = (K != N_columns - 1) };

    template<class T_numtype1, class T_numtype2>
    static inline BZ_PROMOTE(T_numtype1, T_numtype2)
    f(const T_numtype1* matrix1, const T_numtype2* matrix2, int i, int j)
    {
        return matrix1[i * N_rowStride1 + K * N_colStride1]
            * matrix2[K * N_rowStride2 + j * N_colStride2]
            + _bz_meta_matrixMatrixProduct<N_rows1 * go, N_columns * go,
                N_columns2 * go, N_rowStride1 * go, N_colStride1 * go,
                N_rowStride2 * go, N_colStride2 * go, (K+1) * go>
              ::f(matrix1, matrix2, i, j);
    }
};

template<>
class _bz_meta_matrixMatrixProduct<0,0,0,0,0,0,0,0> {
public:
    static inline _bz_meta_nullOperand f(const void*, const void*, int, int)
    { return _bz_meta_nullOperand(); }
};




template<class T_numtype1, class T_numtype2, int N_rows1, int N_columns,
    int N_columns2, int N_rowStride1, int N_colStride1,
    int N_rowStride2, int N_colStride2>
class _bz_tinyMatrixMatrixProduct {
public:
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;

    enum { rows = N_rows1, columns = N_columns2 };

    _bz_tinyMatrixMatrixProduct(const T_numtype1* matrix1,
        const T_numtype2* matrix2)
        : matrix1_(matrix1), matrix2_(matrix2)
    { }

    _bz_tinyMatrixMatrixProduct(const _bz_tinyMatrixMatrixProduct<T_numtype1,
        T_numtype2, N_rows1, N_columns, N_columns2, N_rowStride1, N_colStride1,
        N_rowStride2, N_colStride2>& x)
        : matrix1_(x.matrix1_), matrix2_(x.matrix2_)
    { }

    const T_numtype1* matrix1() const
    { return matrix1_; }

    const T_numtype2* matrix2() const
    { return matrix2_; }

    T_numtype operator()(int i, int j) const
    {
        return _bz_meta_matrixMatrixProduct<N_rows1, N_columns,
            N_columns2, N_rowStride1, N_colStride1, N_rowStride2,
            N_colStride2, 0>::f(matrix1_, matrix2_, i, j);
    }

protected:
    const T_numtype1* matrix1_;
    const T_numtype2* matrix2_;    
};

template<class T_numtype1, class T_numtype2, int N_rows1, int N_columns1,
    int N_columns2>
inline
_bz_tinyMatExpr<_bz_tinyMatrixMatrixProduct<T_numtype1, T_numtype2, N_rows1, 
    N_columns1, N_columns2, N_columns1, 1, N_columns2, 1> >
product(const TinyMatrix<T_numtype1, N_rows1, N_columns1>& a,
    const TinyMatrix<T_numtype2, N_columns1, N_columns2>& b)
{
    typedef _bz_tinyMatrixMatrixProduct<T_numtype1, T_numtype2,
        N_rows1, N_columns1, N_columns2, N_columns1, 1, N_columns2, 1> T_expr;
    return _bz_tinyMatExpr<T_expr>(T_expr(a.data(), b.data()));
}

BZ_NAMESPACE_END

#endif // BZ_META_MATMAT_H

