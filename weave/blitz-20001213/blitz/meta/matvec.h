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


template<class T_numtype1, class T_numtype2, int N_rows, int N_columns, 
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

    enum {
        _bz_staticLengthCount = 1,
        _bz_dynamicLengthCount = 0,
        _bz_staticLength = N_rows

#ifdef BZ_HAVE_COSTS
     ,
        _bz_costPerEval = 2 * N_columns * costs::memoryAccess
            + (N_columns-1) * costs::add
#endif

    };

    unsigned _bz_suggestLength() const
    {
        return N_rows;
    }

    _bz_bool _bz_hasFastAccess() const
    { return _bz_true; }

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

template<class T_numtype1, class T_numtype2, int N_rows, int N_columns>
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
    enum { go = J < (N_columns-1) };
   
    template<class T_numtype1, class T_numtype2> 
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
    enum { go = I < (N_rows - 1) };

    template<class T_numtype1, class T_numtype2, class T_numtype3>
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

