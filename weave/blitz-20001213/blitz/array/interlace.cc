/***************************************************************************
 * blitz/array/interlace.cc
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
 * Revision 1.2  2002/09/12 07:02:06  eric
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
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:13  tveldhui
 * Imported sources
 *
 *
 */

#ifndef BZ_ARRAYINTERLACE_CC
#define BZ_ARRAYINTERLACE_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/interlace.cc> must be included via <blitz/array.h>
#endif

#ifndef BZ_ARRAYSHAPE_H
 #include <blitz/array/shape.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * This header provides two collections of routines:
 *
 * interlaceArrays(shape, A1, A2, ...);
 * allocateArrays(shape, A1, A2, ...);
 *
 * interlaceArrays allocates a set of arrays so that their data is
 * interlaced.  For example,
 *
 * Array<int,2> A, B;
 * interlaceArrays(shape(10,10), A, B);
 *
 * sets up the array storage so that A(0,0) is followed by B(0,0) in
 * memory; then A(0,1) and B(0,1), and so on.
 *
 * The allocateArrays() routines may or may not interlace the data,
 * depending on whether interlacing is advantageous for the architecture.
 * This is controlled by the setting of BZ_INTERLACE_ARRAYS in
 * <blitz/tuning.h>.
 */

// Warning: Can't instantiate TinyVector<Range,N> because causes
// conflict between TinyVector<T,N>::operator=(T) and
// TinyVector<T,N>::operator=(Range)

// NEEDS_WORK -- also shape for up to N=11
// NEEDS_WORK -- shape(Range r1, Range r2, ...) (return TinyVector<Range,n>)
//               maybe use Domain objects
// NEEDS_WORK -- doesn't make a lot of sense for user to provide a
//               GeneralArrayStorage<N_rank+1>

template<class T_numtype>
void makeInterlacedArray(Array<T_numtype,2>& mainArray,
    Array<T_numtype,1>& subarray, int slice)
{
    Array<T_numtype,1> tmp = mainArray(Range::all(), slice);
    subarray.reference(tmp);
}

template<class T_numtype>
void makeInterlacedArray(Array<T_numtype,3>& mainArray,
    Array<T_numtype,2>& subarray, int slice)
{
    Array<T_numtype,2> tmp = mainArray(Range::all(), Range::all(), 
        slice);
    subarray.reference(tmp);
}

template<class T_numtype>
void makeInterlacedArray(Array<T_numtype,4>& mainArray,
    Array<T_numtype,3>& subarray, int slice)
{
    Array<T_numtype,3> tmp = mainArray(Range::all(), Range::all(), 
        Range::all(), slice);
    subarray.reference(tmp);
}

// These routines always allocate interlaced arrays
template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 2, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 3, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 4, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 5, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
    makeInterlacedArray(array, a5, 4);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 6, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
    makeInterlacedArray(array, a5, 4);
    makeInterlacedArray(array, a6, 5);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 7, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
    makeInterlacedArray(array, a5, 4);
    makeInterlacedArray(array, a6, 5);
    makeInterlacedArray(array, a7, 6);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 8, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
    makeInterlacedArray(array, a5, 4);
    makeInterlacedArray(array, a6, 5);
    makeInterlacedArray(array, a7, 6);
    makeInterlacedArray(array, a8, 7);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8,
    Array<T_numtype,N_rank>& a9)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 9, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
    makeInterlacedArray(array, a5, 4);
    makeInterlacedArray(array, a6, 5);
    makeInterlacedArray(array, a7, 6);
    makeInterlacedArray(array, a8, 7);
    makeInterlacedArray(array, a9, 8);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8,
    Array<T_numtype,N_rank>& a9, Array<T_numtype,N_rank>& a10)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 10, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
    makeInterlacedArray(array, a5, 4);
    makeInterlacedArray(array, a6, 5);
    makeInterlacedArray(array, a7, 6);
    makeInterlacedArray(array, a8, 7);
    makeInterlacedArray(array, a9, 8);
    makeInterlacedArray(array, a10, 9);
}

template<class T_numtype, int N_rank>
void interlaceArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8,
    Array<T_numtype,N_rank>& a9, Array<T_numtype,N_rank>& a10,
    Array<T_numtype,N_rank>& a11)
{
    GeneralArrayStorage<N_rank+1> storage;
    Array<T_numtype, N_rank+1> array(shape, 11, storage);
    makeInterlacedArray(array, a1, 0);
    makeInterlacedArray(array, a2, 1);
    makeInterlacedArray(array, a3, 2);
    makeInterlacedArray(array, a4, 3);
    makeInterlacedArray(array, a5, 4);
    makeInterlacedArray(array, a6, 5);
    makeInterlacedArray(array, a7, 6);
    makeInterlacedArray(array, a8, 7);
    makeInterlacedArray(array, a9, 8);
    makeInterlacedArray(array, a10, 9);
    makeInterlacedArray(array, a11, 10);
}

// NEEDS_WORK -- make `storage' a parameter in these routines
//  Will be tricky: have to convert GeneralArrayStorage<N_rank> to
//  GeneralArrayStorage<N_rank+1>

// These routines may or may not interlace arrays, depending on
// whether it is advantageous for this platform.

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2);
#else
    a1.resize(shape);
    a2.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4, a5);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
    a5.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4, a5, a6);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
    a5.resize(shape);
    a6.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4, a5, a6, a7);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
    a5.resize(shape);
    a6.resize(shape);
    a7.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4, a5, a6, a7, a8);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
    a5.resize(shape);
    a6.resize(shape);
    a7.resize(shape);
    a8.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8,
    Array<T_numtype,N_rank>& a9)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4, a5, a6, a7, a8, a9);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
    a5.resize(shape);
    a6.resize(shape);
    a7.resize(shape);
    a8.resize(shape);
    a9.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8,
    Array<T_numtype,N_rank>& a9, Array<T_numtype,N_rank>& a10)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
    a5.resize(shape);
    a6.resize(shape);
    a7.resize(shape);
    a8.resize(shape);
    a9.resize(shape);
    a10.resize(shape);
#endif
}

template<class T_numtype, int N_rank>
void allocateArrays(const TinyVector<int,N_rank>& shape,
    Array<T_numtype,N_rank>& a1, Array<T_numtype,N_rank>& a2,
    Array<T_numtype,N_rank>& a3, Array<T_numtype,N_rank>& a4,
    Array<T_numtype,N_rank>& a5, Array<T_numtype,N_rank>& a6,
    Array<T_numtype,N_rank>& a7, Array<T_numtype,N_rank>& a8,
    Array<T_numtype,N_rank>& a9, Array<T_numtype,N_rank>& a10,
    Array<T_numtype,N_rank>& a11)
{
#ifdef BZ_INTERLACE_ARRAYS
    interlaceArrays(shape, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
#else
    a1.resize(shape);
    a2.resize(shape);
    a3.resize(shape);
    a4.resize(shape);
    a5.resize(shape);
    a6.resize(shape);
    a7.resize(shape);
    a8.resize(shape);
    a9.resize(shape);
    a10.resize(shape);
    a11.resize(shape);
#endif
}

// NEEDS_WORK -- allocateArrays for TinyVector<Range,N_rank>

// This constructor is used to create interlaced arrays.
template<class T_numtype, int N_rank>
Array<T_numtype,N_rank>::Array(const TinyVector<int,N_rank-1>& shape,
    int lastExtent, const GeneralArrayStorage<N_rank>& storage)
    : storage_(storage)
{
    // Create an array with the given shape, plus an extra dimension
    // for the number of arrays being allocated.  This extra dimension
    // must have minor storage order.

    if (ordering(0) == 0)
    {
        // Column major storage order (or something like it)
        length_[0] = lastExtent;
        storage_.setBase(0,0);
        for (int i=1; i < N_rank; ++i)
            length_[i] = shape[i-1];
    }
    else if (ordering(0) == N_rank-1)
    {
        // Row major storage order (or something like it)
        for (int i=0; i < N_rank-1; ++i)
            length_[i] = shape[i];
        length_[N_rank-1] = lastExtent;
        storage_.setBase(N_rank-1, 0);
    }
    else {
        BZPRECHECK(0, "Used allocateArrays() with a peculiar storage format");
    }

    setupStorage(N_rank-1);
}

// NEEDS_WORK -- see note about TinyVector<Range,N> in <blitz/arrayshape.h>
#if 0
template<class T_numtype, int N_rank>
Array<T_numtype,N_rank>::Array(const TinyVector<Range,N_rank-1>& shape,
    int lastExtent, const GeneralArrayStorage<N_rank>& storage)
    : storage_(storage)
{
#ifdef BZ_DEBUG
    for (int i=0; i < N_rank; ++i)
      BZPRECHECK(shape[i].isAscendingContiguous(),
        "In call to allocateArrays(), a Range object is not ascending" << endl
        << "contiguous: " << shape[i] << endl);
#endif

    if (ordering(0) == 0)
    {
        // Column major storage order (or something like it)
        length_[0] = lastExtent;
        storage_.setBase(0,0);
        for (int i=1; i < N_rank; ++i)
        {
            length_[i] = shape[i-1].length();
            storage_.setBase(i, shape[i-1].first());
        }
    }
    else if (ordering(0) == N_rank-1)
    {
        // Row major storage order (or something like it)
        for (int i=0; i < N_rank-1; ++i)
        {
            length_[i] = shape[i];
            storage_.setBase(i, shape[i].first());
        }
        length_[N_rank-1] = lastExtent;
        storage_.setBase(N_rank-1, 0);
    }
    else {
        BZPRECHECK(0, "Used allocateArrays() with a peculiar storage format");
    }

    setupStorage(N_rank-1);
}
#endif

BZ_NAMESPACE_END

#endif // BZ_ARRAYINTER_CC

