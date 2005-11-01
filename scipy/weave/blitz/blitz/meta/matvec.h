// -*- C++ -*-
/***************************************************************************
 * blitz/tiny/matvec.h   TinyMatrix/TinyVector product metaprogram
 *
 * $Id$
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 ***************************************************************************/

#ifndef BZ_META_MATVEC_H
#define BZ_META_MATVEC_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_VECEXPRWRAP_H
 #include <blitz/vecexprwrap.h>
#endif

#ifndef BZ_METAPROG_H
 #include <blitz/meta/metaprog.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declarations
template<int N_rows, int N_columns, int N_rowStride, int N_colStride,
    int N_vecStride, int J>
class _bz_meta_matrixVectorProduct2;


template<typename T_numtype1, typename T_numtype2, int N_rows, int N_columns, 
    int N_rowStride, int N_colStride, int N_vecStride>
class _bz_tinyMatrixVectorProduct {
public:
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;
    
    _bz_tinyMatrixVectorProduct(const _bz_tinyMatrixVectorProduct<T_numtype1,
        T_numtype2, N_rows, N_columns, N_rowStride, N_colStride, 
        N_vecStride>& z)
            : matrix_(z.matrix_), vector_(z.vector_)
    { }

    _bz_tinyMatrixVectorProduct(const T_numtype1* matrix, 
        const T_numtype2* vector)
        : matrix_(matrix), vector_(vector)
    { }

    T_numtype operator[](unsigned i) const
    {
        return _bz_meta_matrixVectorProduct2<N_rows, N_columns, N_rowStride,
            N_colStride, N_vecStride, 0>::f(matrix_, vector_, i);
    }

    T_numtype operator()(unsigned i) const
    {
        return _bz_meta_matrixVectorProduct2<N_rows, N_columns, N_rowStride,
            N_colStride, N_vecStride, 0>::f(matrix_, vector_, i);
    }

    static const int
        _bz_staticLengthCount = 1,
        _bz_dynamicLengthCount = 0,
        _bz_staticLength = N_rows;

#ifdef BZ_HAVE_COSTS
    static const int 
        _bz_costPerEval = 2 * N_columns * costs::memoryAccess
                        + (N_columns-1) * costs::add;
#endif

    unsigned _bz_suggestLength() const
    {
        return N_rows;
    }

    bool _bz_hasFastAccess() const
    { return true; }

    T_numtype _bz_fastAccess(unsigned i) const
    {
        return _bz_meta_matrixVectorProduct2<N_rows, N_columns, N_rowStride,
            N_colStride, N_vecStride, 0>::f(matrix_, vector_, i);
    }

    unsigned length(unsigned recommendedLength) const
    { return N_rows; }

    const T_numtype1* matrix() const
    { return matrix_; }

    const T_numtype2* vector() const
    { return vector_; }

protected:
    const T_numtype1* matrix_;
    const T_numtype2* vector_;
};

template<typename T_numtype1, typename T_numtype2, int N_rows, int N_columns>
inline _bz_VecExpr<_bz_tinyMatrixVectorProduct<T_numtype1, T_numtype2, 
    N_rows, N_columns, N_columns, 1, 1> >
product(const TinyMatrix<T_numtype1, N_rows, N_columns>& matrix,
    const TinyVector<T_numtype2, N_columns>& vector)
{
    typedef _bz_tinyMatrixVectorProduct<T_numtype1, T_numtype2, N_rows, 
        N_columns, N_columns, 1, 1> T_expr;
    return _bz_VecExpr<T_expr>(T_expr(matrix.data(), vector.data()));
}

// Template metaprogram for matrix-vector multiplication

template<int N_rows, int N_columns, int N_rowStride, int N_colStride,
    int N_vecStride, int J>
class _bz_meta_matrixVectorProduct2 {

public:
    static const int go = J < (N_columns-1) ? 1 : 0;
   
    template<typename T_numtype1, typename T_numtype2> 
    static inline BZ_PROMOTE(T_numtype1, T_numtype2)
    f(const T_numtype1* matrix, const T_numtype2* vector, int i)
    {
        return matrix[i * N_rowStride + J * N_colStride]
            * vector[J * N_vecStride]

            + _bz_meta_matrixVectorProduct2<N_rows * go, N_columns * go,
                N_rowStride * go, N_colStride * go, N_vecStride * go, (J+1)*go>
                ::f(matrix, vector, i);
    }
    
};

template<>
class _bz_meta_matrixVectorProduct2<0,0,0,0,0,0> {
public:
    static inline _bz_meta_nullOperand f(const void*, const void*, int)
    { return _bz_meta_nullOperand(); }
};

template<int N_rows, int N_columns, int N_rowStride, int N_colStride,
    int N_vecStride, int I>
class _bz_meta_matrixVectorProduct {
public:
    static const int go = I < (N_rows - 1) ? 1 : 0;

    template<typename T_numtype1, typename T_numtype2, typename T_numtype3>
    static inline void f(TinyVector<T_numtype3, N_rows>& result,
        const T_numtype1* matrix, const T_numtype2* vector)
    {
        result[I] = _bz_meta_matrixVectorProduct2<N_rows, N_columns,
            N_rowStride, N_colStride, N_vecStride, 0>::f(matrix,vector, I);

        _bz_meta_matrixVectorProduct<N_rows * go, N_columns * go,
            N_rowStride * go, N_colStride * go, N_vecStride * go, (I+1)*go>
              ::f(result, matrix, vector);
    }
};

template<>
class _bz_meta_matrixVectorProduct<0,0,0,0,0,0> {
public:
    static inline void f(const _bz_tinyBase&, const void*, const void*)
    { }
};

BZ_NAMESPACE_END

#endif // BZ_META_MATVEC_H

