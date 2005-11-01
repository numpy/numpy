/***************************************************************************
 * blitz/array/resize.cc  Resizing of arrays
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
 ****************************************************************************/
#ifndef BZ_ARRAYRESIZE_CC
#define BZ_ARRAYRESIZE_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/resize.cc> must be included via <blitz/array.h>
#endif

#include <blitz/minmax.h>

BZ_NAMESPACE(blitz)

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0)
{
    BZPRECONDITION(extent0 >= 0);
    BZPRECONDITION(N_rank == 1);

    if (extent0 != length_[0])
    {
        length_[0] = extent0;
        setupStorage(0);
    }
}

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0));
    BZPRECONDITION(N_rank == 2);

    if ((extent0 != length_[0]) || (extent1 != length_[1]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        setupStorage(1);
    }
}

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0));
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


template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0));
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

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0) && (extent4 >= 0));
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

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0) && (extent4 >= 0) && (extent5 >= 0));
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

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0) && (extent4 >= 0) && (extent5 >= 0)
        && (extent6 >= 0));
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

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0) && (extent4 >= 0) && (extent5 >= 0)
        && (extent6 >= 0) && (extent7 >= 0));
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

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7, int extent8)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0) && (extent4 >= 0) && (extent5 >= 0)
        && (extent6 >= 0) && (extent7 >= 0) && (extent8 >= 0));
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


template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7, int extent8, int extent9)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0) && (extent4 >= 0) && (extent5 >= 0)
        && (extent6 >= 0) && (extent7 >= 0) && (extent8 >= 0)
        && (extent9 >= 0));
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

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2, int extent3, int extent4, int extent5,
    int extent6, int extent7, int extent8, int extent9,
    int extent10)
{
    BZPRECONDITION((extent0 >= 0) && (extent1 >= 0) && (extent2 >= 0)
        && (extent3 >= 0) && (extent4 >= 0) && (extent5 >= 0)
        && (extent6 >= 0) && (extent7 >= 0) && (extent8 >= 0)
        && (extent9 >= 0) && (extent10 >= 0));
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


template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0)
{
	BZPRECONDITION(r0.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());

	setupStorage(0);
}

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());

	setupStorage(1);
}

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2)
{ 
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());

	setupStorage(2);
}

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());

	setupStorage(3);
} 

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3, Range r4)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous() && r4.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());
	length_[4] = r4.length();
	storage_.setBase(4, r4.first());

	setupStorage(4);
} 

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3, Range r4, Range r5)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous() && r4.isAscendingContiguous()
			&& r5.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());
	length_[4] = r4.length();
	storage_.setBase(4, r4.first());
	length_[5] = r5.length();
	storage_.setBase(5, r5.first());

	setupStorage(5);
} 

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3, Range r4, Range r5, Range r6)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous() && r4.isAscendingContiguous()
			&& r5.isAscendingContiguous() && r6.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());
	length_[4] = r4.length();
	storage_.setBase(4, r4.first());
	length_[5] = r5.length();
	storage_.setBase(5, r5.first());
	length_[6] = r6.length();
	storage_.setBase(6, r6.first());

	setupStorage(6);
} 

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3, Range r4, Range r5, Range r6, Range r7)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous() && r4.isAscendingContiguous()
			&& r5.isAscendingContiguous() && r6.isAscendingContiguous()
			&& r7.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());
	length_[4] = r4.length();
	storage_.setBase(4, r4.first());
	length_[5] = r5.length();
	storage_.setBase(5, r5.first());
	length_[6] = r6.length();
	storage_.setBase(6, r6.first());
	length_[7] = r7.length();
	storage_.setBase(7, r7.first());

	setupStorage(7);
} 

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3, Range r4, Range r5, Range r6, Range r7, Range r8)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous() && r4.isAscendingContiguous()
			&& r5.isAscendingContiguous() && r6.isAscendingContiguous()
			&& r7.isAscendingContiguous() && r8.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());
	length_[4] = r4.length();
	storage_.setBase(4, r4.first());
	length_[5] = r5.length();
	storage_.setBase(5, r5.first());
	length_[6] = r6.length();
	storage_.setBase(6, r6.first());
	length_[7] = r7.length();
	storage_.setBase(7, r7.first());
	length_[8] = r8.length();
	storage_.setBase(8, r8.first());

	setupStorage(8);
} 

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3, Range r4, Range r5, Range r6, Range r7, Range r8,
		Range r9)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous() && r4.isAscendingContiguous()
			&& r5.isAscendingContiguous() && r6.isAscendingContiguous()
			&& r7.isAscendingContiguous() && r8.isAscendingContiguous()
			&& r9.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());
	length_[4] = r4.length();
	storage_.setBase(4, r4.first());
	length_[5] = r5.length();
	storage_.setBase(5, r5.first());
	length_[6] = r6.length();
	storage_.setBase(6, r6.first());
	length_[7] = r7.length();
	storage_.setBase(7, r7.first());
	length_[8] = r8.length();
	storage_.setBase(8, r8.first());
	length_[9] = r9.length();
	storage_.setBase(9, r9.first());

	setupStorage(9);
} 

template<typename T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
		Range r3, Range r4, Range r5, Range r6, Range r7, Range r8,
		Range r9, Range r10)
{
	BZPRECONDITION(r0.isAscendingContiguous() &&
			r1.isAscendingContiguous() && r2.isAscendingContiguous()
			&& r3.isAscendingContiguous() && r4.isAscendingContiguous()
			&& r5.isAscendingContiguous() && r6.isAscendingContiguous()
			&& r7.isAscendingContiguous() && r8.isAscendingContiguous()
			&& r9.isAscendingContiguous() && r10.isAscendingContiguous());

	length_[0] = r0.length();
	storage_.setBase(0, r0.first());
	length_[1] = r1.length();
	storage_.setBase(1, r1.first());
	length_[2] = r2.length();
	storage_.setBase(2, r2.first());
	length_[3] = r3.length();
	storage_.setBase(3, r3.first());
	length_[4] = r4.length();
	storage_.setBase(4, r4.first());
	length_[5] = r5.length();
	storage_.setBase(5, r5.first());
	length_[6] = r6.length();
	storage_.setBase(6, r6.first());
	length_[7] = r7.length();
	storage_.setBase(7, r7.first());
	length_[8] = r8.length();
	storage_.setBase(8, r8.first());
	length_[9] = r9.length();
	storage_.setBase(9, r9.first());
	length_[10] = r10.length();
	storage_.setBase(10, r10.first());

	setupStorage(10);
} 


template<typename T_numtype, int N_rank>
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

template<typename T_numtype, int N_rank>
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

template<typename T_numtype, int N_rank>
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

template<typename T_numtype, int N_rank>
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

template<typename T_numtype, int N_rank>
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

template<typename T_numtype, int N_rank>
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
template<typename T_numtype, int N_rank>
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
template<typename T_numtype, int N_rank>
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
template<typename T_numtype, int N_rank>
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
template<typename T_numtype, int N_rank>
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
template<typename T_numtype, int N_rank>
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

template<typename T_numtype, int N_rank>
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
template<typename T_numtype, int N_rank>
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
