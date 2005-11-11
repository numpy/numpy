/***************************************************************************
 * blitz/meta/matmat.h   TinyMatrix matrix-matrix product metaprogram
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
 * Revision 1.2  2002/09/12 07:03:09  eric
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
 * Revision 1.2  2001/01/24 20:22:51  tveldhui
 * Updated copyright date in headers.
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

