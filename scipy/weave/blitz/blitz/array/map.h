// -*- C++ -*-
/***************************************************************************
 * blitz/array/map.h      Declaration of the ArrayIndexMapping class
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

/*
 * ArrayIndexMapping is used to implement tensor array notation.  For
 * example:
 *
 * Array<float, 2> A, B;
 * firstIndex i;
 * secondIndex j;
 * thirdIndex k;
 * Array<float, 3> C = A(i,j) * B(j,k);
 *
 * For expression templates purposes, something like B(j,k) is represented
 * by an instance of class ArrayIndexMapping.  This class maps an array onto
 * the destination array coordinate system, e.g. B(j,k) -> C(i,j,k)
 */

#ifndef BZ_ARRAYMAP_H
#define BZ_ARRAYMAP_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/map.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * _bz_doArrayIndexMapping is a helper class.  It is specialized for
 * ranks 1, 2, 3, ..., 11.
 */

template<int N_rank>
struct _bz_doArrayIndexMapping {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, N_rank>&, 
        const TinyVector<int,N_destRank>&, int, int, int, int, int, int,
        int, int, int, int, int)
    {
        // If you try to use an array index mapping on an array with
        // rank greater than 11, then you'll get a precondition failure
        // here.
        BZPRECONDITION(0);
        return T_numtype();
    }
};

template<>
struct _bz_doArrayIndexMapping<1> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 1>& array,
        const TinyVector<int,N_destRank>& index, int i0, int, int, int, int, 
        int, int, int, int, int, int)
    {
        return array(index[i0]);
    }
};


template<>
struct _bz_doArrayIndexMapping<2> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 2>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int, 
        int, int, int, int, int, int, int, int)
    {
        return array(index[i0], index[i1]);
    }
};

template<>
struct _bz_doArrayIndexMapping<3> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 3>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int, int, int, int, int, int, int, int)
    {
        return array(index[i0], index[i1], index[i2]);
    }
};

template<>
struct _bz_doArrayIndexMapping<4> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 4>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int, int, int, int, int, int, int)
    {
        return array(index[i0], index[i1], index[i2], index[i3]);
    }
};

template<>
struct _bz_doArrayIndexMapping<5> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 5>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int i4, int, int, int, int, int, int)
    {
        return array(index[i0], index[i1], index[i2], index[i3], index[i4]);
    }
};

template<>
struct _bz_doArrayIndexMapping<6> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 6>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int i4, int i5, int, int, int, int, int)
    {
        return array(index[i0], index[i1], index[i2], index[i3], index[i4],
            index[i5]);
    }
};

template<>
struct _bz_doArrayIndexMapping<7> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 7>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int i4, int i5, int i6, int, int, int, int)
    {
        return array(index[i0], index[i1], index[i2], index[i3], index[i4],
            index[i5], index[i6]);
    }
};

template<>
struct _bz_doArrayIndexMapping<8> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 8>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int i4, int i5, int i6, int i7, int, int, int)
    {
        return array(index[i0], index[i1], index[i2], index[i3], index[i4],
            index[i5], index[i6], index[i7]);
    }
};

template<>
struct _bz_doArrayIndexMapping<9> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 9>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int i4, int i5, int i6, int i7, int i8, int, int)
    {
        return array(index[i0], index[i1], index[i2], index[i3], index[i4],
            index[i5], index[i6], index[i7], index[i8]);
    }
};

template<>
struct _bz_doArrayIndexMapping<10> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 10>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int i4, int i5, int i6, int i7, int i8, int i9, int)
    {
        return array(index[i0], index[i1], index[i2], index[i3], index[i4],
            index[i5], index[i6], index[i7], index[i8], index[i9]);
    }
};

template<>
struct _bz_doArrayIndexMapping<11> {
    template<typename T_numtype, int N_destRank>
    static T_numtype map(const Array<T_numtype, 11>& array,
        const TinyVector<int,N_destRank>& index, int i0, int i1, int i2,
        int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10)
    {
        return array(index[i0], index[i1], index[i2], index[i3], index[i4],
            index[i5], index[i6], index[i7], index[i8], index[i9],
            index[i10]);
    }
};

template<typename P_numtype, int N_rank, int N_map0, int N_map1=0, int N_map2=0,
    int N_map3=0, int N_map4=0, int N_map5=0, int N_map6=0, int N_map7=0, 
    int N_map8=0, int N_map9=0, int N_map10=0>
class ArrayIndexMapping {
public:
    typedef P_numtype T_numtype;
    typedef const Array<T_numtype,N_rank>& T_ctorArg1;
    typedef int                            T_ctorArg2;    // dummy

    /*
     * This enum block finds the maximum of the N_map0, N_map1, ..., N_map10
     * parameters and stores it in maxRank10.  The rank of the expression is
     * then maxRank10 + 1, since the IndexPlaceholders start at 0 rather than
     * 1.  
     */
    static const int
        maxRank1 = (N_map0 > N_map1) ? N_map0 : N_map1,
        maxRank2 = (N_map2 > maxRank1) ? N_map2 : maxRank1,
        maxRank3 = (N_map3 > maxRank2) ? N_map3 : maxRank2,
        maxRank4 = (N_map4 > maxRank3) ? N_map4 : maxRank3,
        maxRank5 = (N_map5 > maxRank4) ? N_map5 : maxRank4,
        maxRank6 = (N_map6 > maxRank5) ? N_map6 : maxRank5,
        maxRank7 = (N_map7 > maxRank6) ? N_map7 : maxRank6,
        maxRank8 = (N_map8 > maxRank7) ? N_map8 : maxRank7,
        maxRank9 = (N_map9 > maxRank8) ? N_map9 : maxRank8,
        maxRank10 = (N_map10 > maxRank9) ? N_map10 : maxRank9;

    static const int 
        numArrayOperands = 1, 
        numIndexPlaceholders = 1,
        rank = maxRank10 + 1;

    ArrayIndexMapping(const Array<T_numtype, N_rank>& array)
        : array_(array)
    { 
    }

    ArrayIndexMapping(const ArrayIndexMapping<T_numtype,N_rank,N_map0,
        N_map1,N_map2,N_map3,N_map4,N_map5,N_map6,N_map7,N_map8,N_map9,
        N_map10>& z)
        : array_(z.array_)
    { 
    }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    template<int N_destRank>
    T_numtype operator()(TinyVector<int, N_destRank> i)
    {
        return _bz_doArrayIndexMapping<N_rank>::map(array_, i,
            N_map0, N_map1, N_map2, N_map3, N_map4, N_map5, N_map6,
            N_map7, N_map8, N_map9, N_map10);
    }
#else
    template<int N_destRank>
    T_numtype operator()(const TinyVector<int, N_destRank>& i)
    {
        return _bz_doArrayIndexMapping<N_rank>::map(array_, i,
            N_map0, N_map1, N_map2, N_map3, N_map4, N_map5, N_map6,
            N_map7, N_map8, N_map9, N_map10);
    }
#endif

    int ascending(int rank)
    {
        if (N_map0 == rank)
            return array_.isRankStoredAscending(0);
        else if ((N_map1 == rank) && (N_rank > 1))
            return array_.isRankStoredAscending(1);
        else if ((N_map2 == rank) && (N_rank > 2))
            return array_.isRankStoredAscending(2);
        else if ((N_map3 == rank) && (N_rank > 3))
            return array_.isRankStoredAscending(3);
        else if ((N_map4 == rank) && (N_rank > 4))
            return array_.isRankStoredAscending(4);
        else if ((N_map5 == rank) && (N_rank > 5))
            return array_.isRankStoredAscending(5);
        else if ((N_map6 == rank) && (N_rank > 6))
            return array_.isRankStoredAscending(6);
        else if ((N_map7 == rank) && (N_rank > 7))
            return array_.isRankStoredAscending(7);
        else if ((N_map8 == rank) && (N_rank > 8))
            return array_.isRankStoredAscending(8);
        else if ((N_map9 == rank) && (N_rank > 9))
            return array_.isRankStoredAscending(9);
        else if ((N_map10 == rank) && (N_rank > 10))
            return array_.isRankStoredAscending(10);
        else
            return INT_MIN;   // tiny(int());
    }

    int ordering(int rank)
    {
        if (N_map0 == rank)
            return array_.ordering(0);
        else if ((N_map1 == rank) && (N_rank > 1))
            return array_.ordering(1);
        else if ((N_map2 == rank) && (N_rank > 2))
            return array_.ordering(2);
        else if ((N_map3 == rank) && (N_rank > 3))
            return array_.ordering(3);
        else if ((N_map4 == rank) && (N_rank > 4))
            return array_.ordering(4);
        else if ((N_map5 == rank) && (N_rank > 5))
            return array_.ordering(5);
        else if ((N_map6 == rank) && (N_rank > 6))
            return array_.ordering(6);
        else if ((N_map7 == rank) && (N_rank > 7))
            return array_.ordering(7);
        else if ((N_map8 == rank) && (N_rank > 8))
            return array_.ordering(8);
        else if ((N_map9 == rank) && (N_rank > 9))
            return array_.ordering(9);
        else if ((N_map10 == rank) && (N_rank > 10))
            return array_.ordering(10);
        else
            return INT_MIN;   // tiny(int());
    }

    int lbound(int rank)
    { 
        if (N_map0 == rank)    
            return array_.lbound(0);
        else if ((N_map1 == rank) && (N_rank > 1))
            return array_.lbound(1);
        else if ((N_map2 == rank) && (N_rank > 2))
            return array_.lbound(2);
        else if ((N_map3 == rank) && (N_rank > 3))
            return array_.lbound(3);
        else if ((N_map4 == rank) && (N_rank > 4))
            return array_.lbound(4);
        else if ((N_map5 == rank) && (N_rank > 5))
            return array_.lbound(5);
        else if ((N_map6 == rank) && (N_rank > 6))
            return array_.lbound(6);
        else if ((N_map7 == rank) && (N_rank > 7))
            return array_.lbound(7);
        else if ((N_map8 == rank) && (N_rank > 8))
            return array_.lbound(8);
        else if ((N_map9 == rank) && (N_rank > 9))
            return array_.lbound(9);
        else if ((N_map10 == rank) && (N_rank > 10))
            return array_.lbound(10);
        else
            return INT_MIN;   // tiny(int());
    }

    int ubound(int rank)
    {   
        if (N_map0 == rank)
            return array_.ubound(0);
        else if ((N_map1 == rank) && (N_rank > 1))
            return array_.ubound(1);
        else if ((N_map2 == rank) && (N_rank > 2))
            return array_.ubound(2);
        else if ((N_map3 == rank) && (N_rank > 3))
            return array_.ubound(3);
        else if ((N_map4 == rank) && (N_rank > 4))
            return array_.ubound(4);
        else if ((N_map5 == rank) && (N_rank > 5))
            return array_.ubound(5);
        else if ((N_map6 == rank) && (N_rank > 6))
            return array_.ubound(6);
        else if ((N_map7 == rank) && (N_rank > 7))
            return array_.ubound(7);
        else if ((N_map8 == rank) && (N_rank > 8))
            return array_.ubound(8);
        else if ((N_map9 == rank) && (N_rank > 9))
            return array_.ubound(9);
        else if ((N_map10 == rank) && (N_rank > 10))
            return array_.ubound(10);
        else
            return INT_MAX;   // huge(int());
    }

    // If you have a precondition failure on this routine, it means
    // you are trying to use stack iteration mode on an expression
    // which contains an index placeholder.  You must use index
    // iteration mode instead.
    int operator*()
    {
        BZPRECONDITION(0);
        return 0;
    }

    // See operator*() note
    void push(int)
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void pop(int)
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void advance()
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void advance(int)
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void loadStride(int)
    {
        BZPRECONDITION(0);
    }

    bool isUnitStride(int) const
    {
        BZPRECONDITION(0);
        return false;
    }

    void advanceUnitStride()
    {
        BZPRECONDITION(0);
    }

    bool canCollapse(int,int) const
    {   BZPRECONDITION(0);  return false; }

    T_numtype operator[](int)
    {   
        BZPRECONDITION(0);
        return T_numtype();
    }

    T_numtype fastRead(int)
    {
        BZPRECONDITION(0);
        return T_numtype();
    }

    int suggestStride(int) const
    {
        BZPRECONDITION(0);
        return 0;
    }

    bool isStride(int,int) const
    {
        BZPRECONDITION(0);
        return true;
    }

    template<int N_rank2>
    void moveTo(const TinyVector<int,N_rank2>&)
    {
        BZPRECONDITION(0);
        return ;
    }

    void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat&) const
    {
        // NEEDS_WORK-- do real formatting for reductions
        str += "map[NEEDS_WORK]";
    }

    template<typename T_shape>
    bool shapeCheck(const T_shape&) const
    { 
        // NEEDS_WORK-- do a real shape check (tricky)
        return true; 
    }

private:
    ArrayIndexMapping() : array_( Array<T_numtype, N_rank>() ) { }

    const Array<T_numtype, N_rank>& array_;
};

BZ_NAMESPACE_END

#endif // BZ_ARRAYMAP_H

