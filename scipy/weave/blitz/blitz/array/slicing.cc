/***************************************************************************
 * blitz/array/slicing.cc  Slicing of arrays
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
#ifndef BZ_ARRAYSLICING_CC
#define BZ_ARRAYSLICING_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/slicing.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * These routines make the array a view of a portion of another array.
 * They all work by first referencing the other array, and then slicing.
 */

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, const RectDomain<N_rank>& subdomain)
{
    reference(array);
    for (int i=0; i < N_rank; ++i)
        slice(i, subdomain[i]);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, const StridedDomain<N_rank>& subdomain)
{
    reference(array);
    for (int i=0; i < N_rank; ++i)
        slice(i, subdomain[i]);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0)
{
    reference(array);
    slice(0, r0);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3,
    Range r4)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
    slice(4, r4);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3,
    Range r4, Range r5)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
    slice(4, r4);
    slice(5, r5);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3,
    Range r4, Range r5, Range r6)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
    slice(4, r4);
    slice(5, r5);
    slice(6, r6);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3,
    Range r4, Range r5, Range r6, Range r7)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
    slice(4, r4);
    slice(5, r5);
    slice(6, r6);
    slice(7, r7);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3,
    Range r4, Range r5, Range r6, Range r7, Range r8)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
    slice(4, r4);
    slice(5, r5);
    slice(6, r6);
    slice(7, r7);
    slice(8, r8);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3,
    Range r4, Range r5, Range r6, Range r7, Range r8, Range r9)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
    slice(4, r4);
    slice(5, r5);
    slice(6, r6);
    slice(7, r7);
    slice(8, r8);
    slice(9, r9);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::constructSubarray(
    Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2, Range r3,
    Range r4, Range r5, Range r6, Range r7, Range r8, Range r9, Range r10)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
    slice(4, r4);
    slice(5, r5);
    slice(6, r6);
    slice(7, r7);
    slice(8, r8);
    slice(9, r9);
    slice(10, r10);
}

/*
 * This member template is used to implement operator() with any
 * combination of int and Range parameters.  There's room for up
 * to 11 parameters, but any unused parameters have no effect.
 */
template<typename P_numtype, int N_rank> template<int N_rank2, typename R0,
    class R1, typename R2, typename R3, typename R4, typename R5, typename R6, typename R7,
    class R8, typename R9, typename R10>
void Array<P_numtype, N_rank>::constructSlice(Array<T_numtype, N_rank2>& array,
    R0 r0, R1 r1, R2 r2, R3 r3, R4 r4, R5 r5, R6 r6, R7 r7, R8 r8, R9 r9,
    R10 r10)
{
    MemoryBlockReference<T_numtype>::changeBlock(array);

    int setRank = 0;

    TinyVector<int, N_rank2> rankMap;

    slice(setRank, r0, array, rankMap, 0);
    slice(setRank, r1, array, rankMap, 1);
    slice(setRank, r2, array, rankMap, 2);
    slice(setRank, r3, array, rankMap, 3);
    slice(setRank, r4, array, rankMap, 4);
    slice(setRank, r5, array, rankMap, 5);
    slice(setRank, r6, array, rankMap, 6);
    slice(setRank, r7, array, rankMap, 7);
    slice(setRank, r8, array, rankMap, 8);
    slice(setRank, r9, array, rankMap, 9);
    slice(setRank, r10, array, rankMap, 10);

    // Redo the ordering_ array to account for dimensions which
    // have been sliced away.
    int j = 0;
    for (int i=0; i < N_rank2; ++i)
    {
        if (rankMap[array.ordering(i)] != -1)
            storage_.setOrdering(j++, rankMap[array.ordering(i)]);
    }

    calculateZeroOffset();
}

/*
 * This member template is also used in the implementation of
 * operator() with any combination of int and Rank parameters.
 * It's called by constructSlice(), above.  This version handles
 * Range parameters.
 */
template<typename P_numtype, int N_rank> template<int N_rank2>
void Array<P_numtype, N_rank>::slice(int& setRank, Range r,
    Array<T_numtype,N_rank2>& array, TinyVector<int,N_rank2>& rankMap,
    int sourceRank)
{
    // NEEDS WORK: ordering will change completely when some ranks
    // are deleted.

#ifdef BZ_DEBUG_SLICE
cout << "slice(" << setRank << ", [" << r.first(array.lbound(sourceRank))
     << ", " << r.last(array.ubound(sourceRank)) << "], Array<T,"
     << N_rank2 << ">, " << sourceRank << ")" << endl;
#endif

    rankMap[sourceRank] = setRank;
    length_[setRank] = array.length(sourceRank);
    stride_[setRank] = array.stride(sourceRank);
    storage_.setAscendingFlag(setRank, array.isRankStoredAscending(sourceRank));
    storage_.setBase(setRank, array.base(sourceRank));
    slice(setRank, r);
    ++setRank;
}

/*
 * This member template is also used in the implementation of
 * operator() with any combination of int and Rank parameters.
 * It's called by constructSlice(), above.  This version handles
 * int parameters, which reduce the dimensionality by one.
 */
template<typename P_numtype, int N_rank> template<int N_rank2>
void Array<P_numtype, N_rank>::slice(int&, int i,
    Array<T_numtype,N_rank2>& array, TinyVector<int,N_rank2>& rankMap,
    int sourceRank)
{
#ifdef BZ_DEBUG_SLICE
    cout << "slice(" << i
         << ", Array<T," << N_rank2 << ">, " << sourceRank << ")" << endl;
    cout << "Offset by " << (i * array.stride(sourceRank))
         << endl;
#endif
    BZPRECHECK(array.isInRangeForDim(i, sourceRank),
        "Slice is out of range for array: index=" << i << " rank=" << sourceRank
         << endl << "Possible range for index: [" << array.lbound(sourceRank)
         << ", " << array.ubound(sourceRank) << "]");

    rankMap[sourceRank] = -1;
    data_ += i * array.stride(sourceRank);
#ifdef BZ_DEBUG_SLICE
    cout << "data_ = " << data_ << endl;
#endif
}

/*
 * After calling slice(int rank, Range r), the array refers only to the
 * Range r of the original array.
 * e.g. Array<int,1> x(100);
 *      x.slice(firstRank, Range(25,50));
 *      x = 0;       // Sets elements 25..50 of the original array to 0
 */
template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::slice(int rank, Range r)
{
    BZPRECONDITION((rank >= 0) && (rank < N_rank));

    int first = r.first(lbound(rank));
    int last  = r.last(ubound(rank));
    int stride = r.stride();

#ifdef BZ_DEBUG_SLICE
cout << "slice(" << rank << ", Range):" << endl
     << "first = " << first << " last = " << last << "stride = " << stride
     << endl << "length_[rank] = " << length_[rank] << endl;
#endif

    BZPRECHECK(
        ((first <= last) && (stride > 0)
         || (first >= last) && (stride < 0))
        && (first >= base(rank) && (first - base(rank)) < length_[rank])
        && (last >= base(rank) && (last - base(rank)) < length_[rank]),
        "Bad array slice: Range(" << first << ", " << last << ", "
        << stride << ").  Array is Range(" << lbound(rank) << ", "
        << ubound(rank) << ")");

    // Will the storage be non-contiguous?
    // (1) Slice in the minor dimension and the range does not span
    //     the entire index interval (NB: non-unit strides are possible)
    // (2) Slice in a middle dimension and the range is not Range::all()

    length_[rank] = (last - first) / stride + 1;

    // TV 20000312: added second term here, for testsuite/Josef-Wagenhuber
    int offset = (first - base(rank) * stride) * stride_[rank];

    data_ += offset;
    zeroOffset_ += offset;

    stride_[rank] *= stride;
    // JCC: adjust ascending flag if slicing with backwards Range
    if (stride<0)
        storage_.setAscendingFlag(rank, !isRankStoredAscending(rank));
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYSLICING_CC
