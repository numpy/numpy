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
 ***************************************************************************
 * $Log$
 * Revision 1.2  2002/09/12 07:04:04  eric
 * major rewrite of weave.
 *
 * 0.
 * The underlying library code is significantly re-factored and simpler. There used to be a xxx_spec.py and xxx_info.py file for every group of type conversion classes.  The spec file held the python code that handled the conversion and the info file had most of the C code templates that were generated.  This proved pretty confusing in practice, so the two files have mostly been merged into the spec file.
 *
 * Also, there was quite a bit of code duplication running around.  The re-factoring was able to trim the standard conversion code base (excluding blitz and accelerate stuff) by about 40%.  This should be a huge maintainability and extensibility win.
 *
 * 1.
 * With multiple months of using Numeric arrays, I've found some of weave's "magic variable" names unwieldy and want to change them.  The following are the old declarations for an array x of Float32 type:
 *
 *         PyArrayObject* x = convert_to_numpy(...);
 *         float* x_data = (float*) x->data;
 *         int*   _Nx = x->dimensions;
 *         int*   _Sx = x->strides;
 *         int    _Dx = x->nd;
 *
 * The new declaration looks like this:
 *
 *         PyArrayObject* x_array = convert_to_numpy(...);
 *         float* x = (float*) x->data;
 *         int*   Nx = x->dimensions;
 *         int*   Sx = x->strides;
 *         int    Dx = x->nd;
 *
 * This is obviously not backward compatible, and will break some code (including a lot of mine).  It also makes inline() code more readable and natural to write.
 *
 * 2.
 * I've switched from CXX to Gordon McMillan's SCXX for list, tuples, and dictionaries.  I like CXX pretty well, but its use of advanced C++ (templates, etc.) caused some portability problems.  The SCXX library is similar to CXX but doesn't use templates at all.  This, like (1) is not an
 * API compatible change and requires repairing existing code.
 *
 * I have also thought about boost python, but it also makes heavy use of templates.  Moving to SCXX gets rid of almost all template usage for the standard type converters which should help portability.  std::complex and std::string from the STL are the only templates left.  Of course blitz still uses templates in a major way so weave.blitz will continue to be hard on compilers.
 *
 * I've actually considered scrapping the C++ classes for list, tuples, and
 * dictionaries, and just fall back to the standard Python C API because the classes are waaay slower than the raw API in many cases.  They are also more convenient and less error prone in many cases, so I've decided to stick with them.  The PyObject variable will always be made available for variable "x" under the name "py_x" for more speedy operations.  You'll definitely want to use these for anything that needs to be speedy.
 *
 * 3.
 * strings are converted to std::string now.  I found this to be the most useful type in for strings in my code.  Py::String was used previously.
 *
 * 4.
 * There are a number of reference count "errors" in some of the less tested conversion codes such as instance, module, etc.  I've cleaned most of these up.  I put errors in quotes here because I'm actually not positive that objects passed into "inline" really need reference counting applied to them.  The dictionaries passed in by inline() hold references to these objects so it doesn't seem that they could ever be garbage collected inadvertently.  Variables used by ext_tools, though, definitely need the reference counting done.  I don't think this is a major cost in speed, so it probably isn't worth getting rid of the ref count code.
 *
 * 5.
 * Unicode objects are now supported.  This was necessary to support rendering Unicode strings in the freetype wrappers for Chaco.
 *
 * 6.
 * blitz++ was upgraded to the latest CVS.  It compiles about twice as fast as the old blitz and looks like it supports a large number of compilers (though only gcc 2.95.3 is tested).  Compile times now take about 9 seconds on my 850 MHz PIII laptop.
 *
 * Revision 1.4  2002/05/27 19:31:43  jcumming
 * Removed use of this-> as means of accessing members of templated base class.
 * Instead provided using declarations for these members within the derived
 * class definitions to bring them into the scope of the derived class.
 *
 * Revision 1.3  2002/03/06 16:30:24  patricg
 *
 * data_ replaced by this->data_ everywhere
 *
 * Revision 1.2  2001/01/24 20:22:49  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.6  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.5  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.4  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1996/11/11 17:29:13  tveldhui
 * Initial revision
 *
 * Revision 1.2  1996/10/31 21:06:54  tveldhui
 * Did away with multiple template parameters.  Only numeric type
 * and structure parameters now.
 *
 *
 */

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
template<class P_numtype, class P_structure>
class _bz_MatrixRef;

template<class P_expr>
class _bz_MatExpr;

// Declaration of class Matrix
template<class P_numtype, class P_structure BZ_TEMPLATE_DEFAULT(RowMajor)>
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
    // _bz_explicit Matrix(Vector<T_numtype>& matrix);
    // _bz_explicit Matrix(unsigned length);

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

    T_numtype& _bz_restrict operator()(unsigned i, unsigned j)
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

    template<class P_numtype2, class P_structure2> 
    T_matrix& operator=(const Matrix<P_numtype2, P_structure2> &);
    template<class P_numtype2, class P_structure2> 
    T_matrix& operator+=(const Matrix<P_numtype2, P_structure2>&);
    template<class P_numtype2, class P_structure2> 
    T_matrix& operator-=(const Matrix<P_numtype2, P_structure2> &);
    template<class P_numtype2, class P_structure2> 
    T_matrix& operator*=(const Matrix<P_numtype2, P_structure2> &);
    template<class P_numtype2, class P_structure2> 
    T_matrix& operator/=(const Matrix<P_numtype2, P_structure2> &);

    // Matrix expression operand
    template<class P_expr>
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

template<class P_numtype, class P_structure>
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
