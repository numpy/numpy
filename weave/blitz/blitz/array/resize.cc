/***************************************************************************
 * blitz/array/resize.cc  Resizing of arrays
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
 * Revision 1.4  2002/03/06 15:50:41  patricg
 *
 * for (d=0; d < N_rank; ++d) replaced by for (int d=0; d < N_rank; ++d)
 * (for scoping problem)
 *
 * Revision 1.3  2001/02/11 15:43:39  tveldhui
 * Additions from Julian Cummings:
 *  - StridedDomain class
 *  - more versions of resizeAndPreserve
 *
 * Revision 1.2  2001/01/25 00:25:55  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_ARRAYRESIZE_CC
#define BZ_ARRAYRESIZE_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/resize.cc> must be included via <blitz/array.h>
#endif

#include <blitz/minmax.h>

BZ_NAMESPACE(blitz)

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int length0)
{
    BZPRECONDITION(length0 > 0);
    BZPRECONDITION(N_rank == 1);

    if (length0 != length_[firstRank])
    {
        length_[firstRank] = length0;
        setupStorage(0);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0));
    BZPRECONDITION(N_rank == 2);

    if ((extent0 != length_[0]) || (extent1 != length_[1]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        setupStorage(1);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0));
    BZPRECONDITION(N_rank == 3);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        setupStorage(2);
    }
}


template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0));
    BZPRECONDITION(N_rank == 4);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        setupStorage(3);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0) && (extent4 > 0));
    BZPRECONDITION(N_rank == 5);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3])
        || (extent4 != length_[4]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        length_[4] = extent4;
        setupStorage(4);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0) && (extent4 > 0) && (extent5 > 0));
    BZPRECONDITION(N_rank == 6);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3])
        || (extent4 != length_[4]) || (extent5 != length_[5]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        length_[4] = extent4;
        length_[5] = extent5;
        setupStorage(5);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0) && (extent4 > 0) && (extent5 > 0)
        && (extent6 > 0));
    BZPRECONDITION(N_rank == 7);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3])
        || (extent4 != length_[4]) || (extent5 != length_[5])
        || (extent6 != length_[6]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        length_[4] = extent4;
        length_[5] = extent5;
        length_[6] = extent6;
        setupStorage(6);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0) && (extent4 > 0) && (extent5 > 0)
        && (extent6 > 0) && (extent7 > 0));
    BZPRECONDITION(N_rank == 8);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3])
        || (extent4 != length_[4]) || (extent5 != length_[5])
        || (extent6 != length_[6]) || (extent7 != length_[7]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        length_[4] = extent4;
        length_[5] = extent5;
        length_[6] = extent6;
        length_[7] = extent7;
        setupStorage(7);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7, int extent8)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0) && (extent4 > 0) && (extent5 > 0)
        && (extent6 > 0) && (extent7 > 0) && (extent8 > 0));
    BZPRECONDITION(N_rank == 9);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3])
        || (extent4 != length_[4]) || (extent5 != length_[5])
        || (extent6 != length_[6]) || (extent7 != length_[7])
        || (extent8 != length_[8]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        length_[4] = extent4;
        length_[5] = extent5;
        length_[6] = extent6;
        length_[7] = extent7;
        length_[8] = extent8;
        setupStorage(8);
    }
}


template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7, int extent8, int extent9)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0) && (extent4 > 0) && (extent5 > 0)
        && (extent6 > 0) && (extent7 > 0) && (extent8 > 0)
        && (extent9 > 0));
    BZPRECONDITION(N_rank == 10);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3])
        || (extent4 != length_[4]) || (extent5 != length_[5])
        || (extent6 != length_[6]) || (extent7 != length_[7])
        || (extent8 != length_[8]) || (extent9 != length_[9]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        length_[4] = extent4;
        length_[5] = extent5;
        length_[6] = extent6;
        length_[7] = extent7;
        length_[8] = extent8;
        length_[9] = extent9;
        setupStorage(9);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7, int extent8, int extent9,
    int extent10)
{
    BZPRECONDITION((extent0 > 0) && (extent1 > 0) && (extent2 > 0)
        && (extent3 > 0) && (extent4 > 0) && (extent5 > 0)
        && (extent6 > 0) && (extent7 > 0) && (extent8 > 0)
        && (extent9 > 0) && (extent10 > 0));
    BZPRECONDITION(N_rank == 11);

    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3])
        || (extent4 != length_[4]) || (extent5 != length_[5])
        || (extent6 != length_[6]) || (extent7 != length_[7])
        || (extent8 != length_[8]) || (extent9 != length_[9])
        || (extent10 != length_[10]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        length_[4] = extent4;
        length_[5] = extent5;
        length_[6] = extent6;
        length_[7] = extent7;
        length_[8] = extent8;
        length_[9] = extent9;
        length_[10] = extent10;
        setupStorage(10);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0)
{
    BZPRECONDITION(length0 > 0);
    BZPRECONDITION(N_rank == 1);

    if (length0 != length_[firstRank])
    {
#if defined(__KCC) || defined(__DECCXX)
        // NEEDS_WORK: have to discard the base() parameter for EDG,
        // because it gives the following bizarre error:

/*
 * "blitz/tinyvec.h", line 421: error: the size of an array must be greater
 *         than zero
 *     T_numtype data_[N_length];
 *                     ^
 *         detected during:
 *           instantiation of class "blitz::TinyVector<int, 0>" at line 273 of
 *                     "./../blitz/array/resize.cc"
 *           instantiation of
 *                     "void blitz::Array<int, 1>::resizeAndPreserve(int)" 
 */
        T_array B(length0, storage_);
#else
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0), storage_);  // line 273
#endif
        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
              ubound(0)));
            B(overlap0) = (*this)(overlap0);
        }
        reference(B);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0));
    BZPRECONDITION(N_rank == 2);

    if ((length0 != length_[0]) || (length1 != length_[1]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
                ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), 
                ubound(1)));
            B(overlap0, overlap1) = (*this)(overlap0, overlap1);
        }
        reference(B);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0));
    BZPRECONDITION(N_rank == 3);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, length2), 
            storage_);
        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
                ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), 
                ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), 
                ubound(2)));
            B(overlap0, overlap1, overlap2) = (*this)(overlap0, overlap1, 
                overlap2);
        }
        reference(B);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0));
    BZPRECONDITION(N_rank == 4);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1,
            length2, length3), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), ubound(3)));
            B(overlap0, overlap1, overlap2, overlap3) = (*this)(overlap0,
                overlap1, overlap2, overlap3);
        }
        reference(B);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3, int length4)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0) && (length4 > 0));
    BZPRECONDITION(N_rank == 5);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3])
        || (length4 != length_[4]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, 
            length2, length3, length4), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), ubound(3)));
            Range overlap4 = Range(fromStart, minmax::min(B.ubound(4), ubound(4)));
            B(overlap0, overlap1, overlap2, overlap3, overlap4) = (*this)
                (overlap0, overlap1, overlap2, overlap3, overlap4);
        }
        reference(B);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3, int length4, int length5)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0) && (length4 > 0) && (length5 > 0));
    BZPRECONDITION(N_rank == 6);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3])
        || (length4 != length_[4]) || (length5 != length_[5]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, length2, 
            length3, length4, length5), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), ubound(3)));
            Range overlap4 = Range(fromStart, minmax::min(B.ubound(4), ubound(4)));
            Range overlap5 = Range(fromStart, minmax::min(B.ubound(5), ubound(5)));
            B(overlap0, overlap1, overlap2, overlap3, overlap4, overlap5)
                = (*this)(overlap0, overlap1, overlap2, overlap3, overlap4,
                overlap5);
        }
        reference(B);
    }
}

/* Added by Julian Cummings */
template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3, int length4, int length5, int length6)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0) && (length4 > 0) && (length5 > 0) && (length6 > 0));
    BZPRECONDITION(N_rank == 7);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3])
        || (length4 != length_[4]) || (length5 != length_[5])
        || (length6 != length_[6]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, length2,
            length3, length4, length5, length6), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
               ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), 
               ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), 
               ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), 
               ubound(3)));
            Range overlap4 = Range(fromStart, minmax::min(B.ubound(4), 
               ubound(4)));
            Range overlap5 = Range(fromStart, minmax::min(B.ubound(5), 
               ubound(5)));
            Range overlap6 = Range(fromStart, minmax::min(B.ubound(6), 
               ubound(6)));
            B(overlap0, overlap1, overlap2, overlap3, overlap4, overlap5,
              overlap6)
                = (*this)(overlap0, overlap1, overlap2, overlap3, overlap4,
                overlap5, overlap6);
        }
        reference(B);
    }
}

/* Added by Julian Cummings */
template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3, int length4, int length5, int length6,
    int length7)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0) && (length4 > 0) && (length5 > 0) && (length6 > 0)
        && (length7 > 0));
    BZPRECONDITION(N_rank == 8);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3])
        || (length4 != length_[4]) || (length5 != length_[5])
        || (length6 != length_[6]) || (length7 != length_[7]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, length2,
            length3, length4, length5, length6, length7), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
               ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), 
               ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), 
               ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), 
               ubound(3)));
            Range overlap4 = Range(fromStart, minmax::min(B.ubound(4), 
               ubound(4)));
            Range overlap5 = Range(fromStart, minmax::min(B.ubound(5), 
               ubound(5)));
            Range overlap6 = Range(fromStart, minmax::min(B.ubound(6), 
               ubound(6)));
            Range overlap7 = Range(fromStart, minmax::min(B.ubound(7), 
               ubound(7)));
            B(overlap0, overlap1, overlap2, overlap3, overlap4, overlap5,
              overlap6, overlap7)
                = (*this)(overlap0, overlap1, overlap2, overlap3, overlap4,
                overlap5, overlap6, overlap7);
        }
        reference(B);
    }
}

/* Added by Julian Cummings */
template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3, int length4, int length5, int length6,
    int length7, int length8)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0) && (length4 > 0) && (length5 > 0) && (length6 > 0)
        && (length7 > 0) && (length8 > 0));
    BZPRECONDITION(N_rank == 9);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3])
        || (length4 != length_[4]) || (length5 != length_[5])
        || (length6 != length_[6]) || (length7 != length_[7])
        || (length8 != length_[8]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, length2,
            length3, length4, length5, length6, length7, length8), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
               ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), 
               ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), 
               ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), 
               ubound(3)));
            Range overlap4 = Range(fromStart, minmax::min(B.ubound(4), 
               ubound(4)));
            Range overlap5 = Range(fromStart, minmax::min(B.ubound(5), 
               ubound(5)));
            Range overlap6 = Range(fromStart, minmax::min(B.ubound(6), 
               ubound(6)));
            Range overlap7 = Range(fromStart, minmax::min(B.ubound(7), 
               ubound(7)));
            Range overlap8 = Range(fromStart, minmax::min(B.ubound(8), 
               ubound(8)));
            B(overlap0, overlap1, overlap2, overlap3, overlap4, overlap5,
              overlap6, overlap7, overlap8)
                = (*this)(overlap0, overlap1, overlap2, overlap3, overlap4,
                overlap5, overlap6, overlap7, overlap8);
        }
        reference(B);
    }
}

/* Added by Julian Cummings */
template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3, int length4, int length5, int length6,
    int length7, int length8, int length9)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0) && (length4 > 0) && (length5 > 0) && (length6 > 0)
        && (length7 > 0) && (length8 > 0) && (length9 > 0));
    BZPRECONDITION(N_rank == 10);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3])
        || (length4 != length_[4]) || (length5 != length_[5])
        || (length6 != length_[6]) || (length7 != length_[7])
        || (length8 != length_[8]) || (length9 != length_[9]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, length2,
            length3, length4, length5, length6, length7, length8, length9),
            storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
               ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), 
               ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), 
               ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), 
               ubound(3)));
            Range overlap4 = Range(fromStart, minmax::min(B.ubound(4), 
               ubound(4)));
            Range overlap5 = Range(fromStart, minmax::min(B.ubound(5), 
               ubound(5)));
            Range overlap6 = Range(fromStart, minmax::min(B.ubound(6), 
               ubound(6)));
            Range overlap7 = Range(fromStart, minmax::min(B.ubound(7), 
               ubound(7)));
            Range overlap8 = Range(fromStart, minmax::min(B.ubound(8), 
               ubound(8)));
            Range overlap9 = Range(fromStart, minmax::min(B.ubound(9), 
               ubound(9)));
            B(overlap0, overlap1, overlap2, overlap3, overlap4, overlap5,
              overlap6, overlap7, overlap8, overlap9)
                = (*this)(overlap0, overlap1, overlap2, overlap3, overlap4,
                overlap5, overlap6, overlap7, overlap8, overlap9);
        }
        reference(B);
    }
}

/* Added by Julian Cummings */
template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(int length0, int length1,
    int length2, int length3, int length4, int length5, int length6,
    int length7, int length8, int length9, int length10)
{
    BZPRECONDITION((length0 > 0) && (length1 > 0) && (length2 > 0)
        && (length3 > 0) && (length4 > 0) && (length5 > 0) && (length6 > 0)
        && (length7 > 0) && (length8 > 0) && (length9 > 0) && (length10 > 0));
    BZPRECONDITION(N_rank == 11);

    if ((length0 != length_[0]) || (length1 != length_[1])
        || (length2 != length_[2]) || (length3 != length_[3])
        || (length4 != length_[4]) || (length5 != length_[5])
        || (length6 != length_[6]) || (length7 != length_[7])
        || (length8 != length_[8]) || (length9 != length_[9])
        || (length10 != length_[10]))
    {
        T_array B(base(), BZ_BLITZ_SCOPE(shape)(length0, length1, length2,
            length3, length4, length5, length6, length7, length8, length9,
            length10), storage_);

        if (numElements())
        {
            Range overlap0 = Range(fromStart, minmax::min(B.ubound(0), 
               ubound(0)));
            Range overlap1 = Range(fromStart, minmax::min(B.ubound(1), 
               ubound(1)));
            Range overlap2 = Range(fromStart, minmax::min(B.ubound(2), 
               ubound(2)));
            Range overlap3 = Range(fromStart, minmax::min(B.ubound(3), 
               ubound(3)));
            Range overlap4 = Range(fromStart, minmax::min(B.ubound(4), 
               ubound(4)));
            Range overlap5 = Range(fromStart, minmax::min(B.ubound(5), 
               ubound(5)));
            Range overlap6 = Range(fromStart, minmax::min(B.ubound(6), 
               ubound(6)));
            Range overlap7 = Range(fromStart, minmax::min(B.ubound(7), 
               ubound(7)));
            Range overlap8 = Range(fromStart, minmax::min(B.ubound(8), 
               ubound(8)));
            Range overlap9 = Range(fromStart, minmax::min(B.ubound(9), 
               ubound(9)));
            Range overlap10 = Range(fromStart, minmax::min(B.ubound(10), 
               ubound(10)));
        }
        reference(B);
    }
}

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(const TinyVector<int,N_rank>& extent)
{
// NEEDS_WORK -- don't resize if unnecessary
//    BZPRECONDITION(all(extent > 0));
//    if (any(extent != length_))
//    {
        length_ = extent;
        setupStorage(N_rank);
//    }
}

/* Added by Julian Cummings */
template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resizeAndPreserve(
    const TinyVector<int,N_rank>& extent)
{
// NEEDS_WORK -- don't resize if unnecessary
//    BZPRECONDITION(all(extent > 0));
//    if (any(extent != length_))
//    {
        T_array B(base(), extent, storage_);

        if (numElements())
        {
          TinyVector<int,N_rank> ub;
          for (int d=0; d < N_rank; ++d)
            ub(d) = minmax::min(B.ubound(d),ubound(d));
          RectDomain<N_rank> overlap(lbound(),ub);
          B(overlap) = (*this)(overlap);
        }
        reference(B);
//    }
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYRESIZE_CC
