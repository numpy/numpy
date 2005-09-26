/***************************************************************************
 * blitz/mstruct.h      Matrix structure classes
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
 * Revision 1.2  2001/01/24 20:22:49  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.5  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.4  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.3  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 *
 */

#ifndef BZ_MSTRUCT_H
#define BZ_MSTRUCT_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_ZERO_H
 #include <blitz/zero.h>
#endif

/*
 * Each matrix structure class encapsulates a storage format for matrix
 * data.  It is responsible for:
 * - Storing the size of the matrix
 * - Calculating how many unique elements the matrix will have
 * - Mapping indices (i,j) onto memory locations
 * - Performing any sign reversals or conjugations when matrix
 *   elements are retrieved (e.g. in a Hermitian or Skew symmetric
 *   matrix)
 *
 * Every matrix structure class must provide these methods:
 *
 * ctor()
 * ctor(unsigned rows, unsigned cols)
 * unsigned columns() const;
 * unsigned cols()    const;
 * unsigned firstInRow() const;
 * unsigned firstInCol() const;
 * template<class T> T& get(T* data, unsigned i, unsigned j);
 * template<class T> T  get(const T* data, unsigned i, unsigned j) const;
 * bool inRange(unsigned i, unsigned j) const
 * unsigned lastInRow() const;
 * unsigned lastInCol() const;
 * unsigned numElements() const;
 * void resize(unsigned rows, unsigned cols);
 * unsigned rows()    const;
 *
 * Each matrix structure class must declare a public type
 * T_iterator which is an iterator to scan through the unique
 * entries of the matrix.  The iterator class must provide
 * these methods:
 *
 * ctor(unsigned rows, unsigned cols)
 * unsigned offset() const
 * operator bool() const
 * unsigned row() const
 * unsigned col() const
 */

BZ_NAMESPACE(blitz)

class MatrixStructure { };

class AsymmetricMatrix : public MatrixStructure {
public:
    AsymmetricMatrix()
        : rows_(0), cols_(0)
    { }

    AsymmetricMatrix(unsigned rows, unsigned cols)
        : rows_(rows), cols_(cols)
    { }

    unsigned columns() const
    { return cols_; }

    unsigned cols() const
    { return cols_; }

    _bz_bool inRange(unsigned i, unsigned j) const
    {
        return (i < rows_) && (j < cols_);
    }

    void resize(unsigned rows, unsigned cols)
    {
        rows_ = rows;
        cols_ = cols;
    }

    unsigned rows() const
    { return rows_; }

protected:
    unsigned rows_, cols_;
};

// Still to be implemented:
// SkewSymmetric
// Hermitian
// Tridiagonal
// Banded<L,H>
// Upper bidiagonal
// Lower bidiagonal
// Upper Hessenberg
// Lower Hessenberg

BZ_NAMESPACE_END

#include <blitz/matgen.h>         // RowMajor and ColumnMajor general matrices
#include <blitz/matsymm.h>        // Symmetric
#include <blitz/matdiag.h>        // Diagonal
#include <blitz/mattoep.h>        // Toeplitz
#include <blitz/matltri.h>        // Lower triangular
#include <blitz/matutri.h>        // Upper triangular

#endif // BZ_MSTRUCT_H
