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
 * Revision 1.4  2002/06/26 23:52:04  jcumming
 * Explicitly specify second template argument for ListInitializationSwitch,
 * rather than relying on the default value.  This eliminates a compilation
 * problem using the xlC compiler.
 *
 * Revision 1.3  2002/05/22 22:39:41  jcumming
 * Added #include of <blitz/tinymatio.cc>.
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

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
template<class T_expr>
class _bz_tinyMatExpr;

template<class T_numtype, int N_rows, int N_columns, int N_rowStride,
    int N_colStride>
class _bz_tinyMatrixRef {

public:
    _bz_tinyMatrixRef(T_numtype* _bz_restrict const data)
        : data_(data)
    { }

    T_numtype * _bz_restrict data()
    { return (T_numtype * _bz_restrict)data_; }

    T_numtype& _bz_restrict operator()(int i, int j)
    { return data_[i * N_rowStride + j * N_colStride]; }

    T_numtype operator()(int i, int j) const
    { return data_[i * N_rowStride + j * N_colStride]; }

protected:
    T_numtype * _bz_restrict const data_;
};

template<class P_numtype, int N_rows, int N_columns>
class TinyMatrix {

public:
    typedef P_numtype T_numtype;
    typedef _bz_tinyMatrixRef<T_numtype, N_rows, N_columns, N_columns, 1> 
        T_reference;
    typedef TinyMatrix<T_numtype, N_rows, N_columns> T_matrix;

    TinyMatrix() { }

    T_numtype* _bz_restrict data()
    { return data_; }

    const T_numtype* _bz_restrict data() const
    { return data_; }

    T_numtype* _bz_restrict dataFirst()
    { return data_; }

    const T_numtype* _bz_restrict dataFirst() const
    { return data_; }

    // NEEDS_WORK -- precondition checks
    T_numtype& _bz_restrict operator()(int i, int j)
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

    template<class T_expr>
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

    T_numtype* _bz_restrict getInitializationIterator()
    { return dataFirst(); }

protected:
    T_numtype data_[N_rows * N_columns];
};

BZ_NAMESPACE_END

#include <blitz/meta/matvec.h>     // Matrix-vector product metaprogram
#include <blitz/meta/matmat.h>     // Matrix-matrix products
#include <blitz/tinymatio.cc>      // I/O operations

#endif // BZ_TINYMAT_H

