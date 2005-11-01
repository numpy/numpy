/***************************************************************************
 * blitz/matrix.h      Declaration of the Matrix<T_type, T_structure> class
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

#ifndef BZ_MATRIX_H
#define BZ_MATRIX_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_MEMBLOCK_H
 #include <blitz/memblock.h>
#endif

#ifndef BZ_MSTRUCT_H
 #include <blitz/mstruct.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declarations
template<typename P_numtype, typename P_structure>
class _bz_MatrixRef;

template<typename P_expr>
class _bz_MatExpr;

// Declaration of class Matrix
template<typename P_numtype, typename P_structure BZ_TEMPLATE_DEFAULT(RowMajor)>
class Matrix : protected MemoryBlockReference<P_numtype> {

private:
    typedef MemoryBlockReference<P_numtype> T_base;
    using T_base::data_;

public:

    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    typedef P_numtype        T_numtype;
    typedef P_structure      T_structure;
    typedef Matrix<P_numtype, P_structure>   T_matrix;

    //////////////////////////////////////////////
    // Constructors                             //
    //////////////////////////////////////////////

    Matrix()
    { }

    Matrix(int rows, int cols, T_structure structure = T_structure())
        : structure_(structure) 
    {
        structure_.resize(rows, cols);
        MemoryBlockReference<T_numtype>::newBlock(structure_.numElements());
    }

    // Matrix(int rows, int cols, T_numtype initValue,
    //    T_structure structure = T_structure(rows, cols));
    // Matrix(int rows, int cols, Random);
    // Matrix(int rows, int cols, matrix-expression);
    // Matrix(int rows, int cols, T_numtype* data, int rowStride, int colStride);
    // explicit Matrix(Vector<T_numtype>& matrix);
    // explicit Matrix(unsigned length);

    // Create a vector view of an already allocated block of memory.
    // Note that the memory will not be freed when this vector is
    // destroyed.
    // Matrix(unsigned length, T_numtype* data, int stride = 1);

    //////////////////////////////////////////////
    // Member functions
    //////////////////////////////////////////////

    //T_iterator      begin()  const;
    //T_constIterator begin()  const;
    //T_vector        copy()   const;
    //T_iterator      end()    const;
    //T_constIterator end()    const;

    unsigned        cols()        const
    { return structure_.columns(); }

    unsigned        columns()     const
    { return structure_.columns(); }

    void            makeUnique()  const;

    unsigned        numElements() const
    { return structure_.numElements(); }

    void            reference(T_matrix&);

    void            resize(unsigned rows, unsigned cols)
    {
        structure_.resize(rows, cols);
        MemoryBlockReference<T_numtype>::newBlock(structure_.numElements());
    }

//    void            resizeAndPreserve(unsigned newLength);

    unsigned        rows()   const
    { return structure_.rows(); }

    _bz_MatrixRef<T_numtype, T_structure> _bz_getRef() const
    { return _bz_MatrixRef<T_numtype, T_structure>(*this); }

    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    T_numtype           operator()(unsigned i, unsigned j) const
    {
        return structure_.get(data_, i, j);
    }

    T_numtype& restrict operator()(unsigned i, unsigned j)
    {
        return structure_.get(data_, i, j);
    }

    // T_matrix      operator()(Range,Range);

    // T_matrixIndirect operator()(Vector<int>,Vector<int>);
    // T_matrixIndirect operator()(integer-placeholder-expression, Range);
    // T_matrix         operator()(difference-equation-expression)

    //////////////////////////////////////////////
    // Assignment operators
    //////////////////////////////////////////////

    // Scalar operand
    T_matrix& operator=(T_numtype);
    T_matrix& operator+=(T_numtype);
    T_matrix& operator-=(T_numtype);
    T_matrix& operator*=(T_numtype);
    T_matrix& operator/=(T_numtype);

    // Matrix operand

    template<typename P_numtype2, typename P_structure2> 
    T_matrix& operator=(const Matrix<P_numtype2, P_structure2> &);
    template<typename P_numtype2, typename P_structure2> 
    T_matrix& operator+=(const Matrix<P_numtype2, P_structure2>&);
    template<typename P_numtype2, typename P_structure2> 
    T_matrix& operator-=(const Matrix<P_numtype2, P_structure2> &);
    template<typename P_numtype2, typename P_structure2> 
    T_matrix& operator*=(const Matrix<P_numtype2, P_structure2> &);
    template<typename P_numtype2, typename P_structure2> 
    T_matrix& operator/=(const Matrix<P_numtype2, P_structure2> &);

    // Matrix expression operand
    template<typename P_expr>
    T_matrix& operator=(_bz_MatExpr<P_expr>);

    // Integer placeholder expression operand
    // MatrixPick operand

    //////////////////////////////////////////////
    // Unary operators
    //////////////////////////////////////////////

    T_matrix& operator++();
    void operator++(int);
    T_matrix& operator--();
    void operator--(int);
    
private:
    T_structure structure_;
};

template<typename P_numtype, typename P_structure>
ostream& operator<<(ostream& os, const Matrix<P_numtype, P_structure>& matrix);

// Global operators
// +,-,*,/ with all possible combinations of:
//    - scalar
//    - matrix
//    - matrix pick
//    - matrix expression
// Pointwise Math functions: sin, cos, etc.
// Global functions

BZ_NAMESPACE_END

#include <blitz/matrix.cc>
#include <blitz/matexpr.h>

#endif // BZ_MATRIX_H
