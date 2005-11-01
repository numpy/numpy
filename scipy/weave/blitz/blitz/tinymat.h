/***************************************************************************
 * blitz/tinymat.h       Declaration of TinyMatrix<T, N, M>
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

#ifndef BZ_TINYMAT_H
#define BZ_TINYMAT_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

#ifndef BZ_LISTINIT_H
 #include <blitz/listinit.h>
#endif

#include <blitz/tinymatexpr.h>
#include <blitz/meta/matassign.h>

BZ_NAMESPACE(blitz)

// Forward declarations
template<typename T_expr>
class _bz_tinyMatExpr;

template<typename T_numtype, int N_rows, int N_columns, int N_rowStride,
    int N_colStride>
class _bz_tinyMatrixRef {

public:
    _bz_tinyMatrixRef(T_numtype* restrict const data)
        : data_(data)
    { }

    T_numtype * restrict data()
    { return (T_numtype * restrict)data_; }

    T_numtype& restrict operator()(int i, int j)
    { return data_[i * N_rowStride + j * N_colStride]; }

    T_numtype operator()(int i, int j) const
    { return data_[i * N_rowStride + j * N_colStride]; }

protected:
    T_numtype * restrict const data_;
};

template<typename P_numtype, int N_rows, int N_columns>
class TinyMatrix {

public:
    typedef P_numtype T_numtype;
    typedef _bz_tinyMatrixRef<T_numtype, N_rows, N_columns, N_columns, 1> 
        T_reference;
    typedef TinyMatrix<T_numtype, N_rows, N_columns> T_matrix;

    TinyMatrix() { }

    T_numtype* restrict data()
    { return data_; }

    const T_numtype* restrict data() const
    { return data_; }

    T_numtype* restrict dataFirst()
    { return data_; }

    const T_numtype* restrict dataFirst() const
    { return data_; }

    // NEEDS_WORK -- precondition checks
    T_numtype& restrict operator()(int i, int j)
    { return data_[i*N_columns + j]; }

    T_numtype operator()(int i, int j) const
    { return data_[i*N_columns + j]; }

    T_reference getRef()
    { return T_reference((T_numtype*)data_); }

    const T_reference getRef() const
    { return T_reference((T_numtype*)data_); }

    // Scalar operand
    ListInitializationSwitch<T_matrix,T_numtype*>
    operator=(T_numtype x)
    {
        return ListInitializationSwitch<T_matrix,T_numtype*>(*this, x);
    }

    template<typename T_expr>
    TinyMatrix<T_numtype, N_rows, N_columns>&
    operator=(_bz_tinyMatExpr<T_expr> expr)
    {
        _bz_meta_matAssign<N_rows, N_columns, 0>::f(*this, expr,
            _bz_update<T_numtype, _bz_typename T_expr::T_numtype>());
        return *this;
    }

    void initialize(T_numtype x)
    { 
        for (int i=0; i < N_rows; ++i)
          for (int j=0; j < N_columns; ++j)
            (*this)(i,j) = x;
    }

    T_numtype* restrict getInitializationIterator()
    { return dataFirst(); }

protected:
    T_numtype data_[N_rows * N_columns];
};

BZ_NAMESPACE_END

#include <blitz/meta/matvec.h>     // Matrix-vector product metaprogram
#include <blitz/meta/matmat.h>     // Matrix-matrix products
#include <blitz/tinymatio.cc>      // I/O operations

#endif // BZ_TINYMAT_H

