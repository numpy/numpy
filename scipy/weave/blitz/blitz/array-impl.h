// -*- C++ -*-
/***************************************************************************
 * blitz/array-impl.h    Definition of the Array<P_numtype, N_rank> class
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
 ***************************************************************************/

/*
 * Wish list for array classes.
 *  - Arrays whose dimensions are unknown at compile time.
 *  - where()/elsewhere()/elsewhere() as in Dan Quinlan's implementation
 *  - block reduction operations
 *  - conversion to/from matrix & vector
 *  - apply(T func(T))
 *  - apply(T func(const T&))
 *  - apply<T func(T)>
 */

#ifndef BZ_ARRAY_H
#define BZ_ARRAY_H

#include <blitz/blitz.h>
#include <blitz/memblock.h>
#include <blitz/range.h>
#include <blitz/tinyvec.h>

#ifdef BZ_ARRAY_SPACE_FILLING_TRAVERSAL
#include <blitz/traversal.h>
#endif

#include <blitz/indexexpr.h>
#include <blitz/prettyprint.h>

#include <blitz/array/slice.h>     // Subarrays and slicing
#include <blitz/array/map.h>       // Tensor index notation
#include <blitz/array/multi.h>     // Multicomponent arrays
#include <blitz/array/domain.h>    // RectDomain class
#include <blitz/array/storage.h>   // GeneralArrayStorage


BZ_NAMESPACE(blitz)

/*
 * Forward declarations
 */

template<typename T_numtype, int N_rank>
class ArrayIterator;

template<typename T_numtype, int N_rank>
class ConstArrayIterator;

template<typename T_numtype, int N_rank>
class FastArrayIterator;

template<typename P_expr>
class _bz_ArrayExpr;

template<typename T_array, typename T_index>
class IndirectArray;

template <typename P_numtype,int N_rank>
void swap(Array<P_numtype,N_rank>&,Array<P_numtype,N_rank>&);

template <typename P_numtype, int N_rank>
void find(Array<TinyVector<int,N_rank>,1>&,const Array<P_numtype,N_rank>&);

/*
 * Declaration of class Array
 */

// NEEDS_WORK: Array should inherit protected from MemoryBlockReference.
// To make this work, need to expose MemoryBlockReference::numReferences()
// and make Array<P,N2> a friend of Array<P,N> for slicing.

template<typename P_numtype, int N_rank>
class Array : public MemoryBlockReference<P_numtype> 
#ifdef BZ_NEW_EXPRESSION_TEMPLATES
    , public ETBase<Array<P_numtype,N_rank> >
#endif
{

private:
    typedef MemoryBlockReference<P_numtype> T_base;
    using T_base::data_;
    using T_base::changeToNullBlock;
    using T_base::numReferences;

public:
    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    /*
     * T_numtype  is the numeric type stored in the array.
     * T_index    is a vector type which can be used to access elements
     *            of many-dimensional arrays.
     * T_array    is the array type itself -- Array<T_numtype, N_rank>
     * T_iterator is a a fast iterator for the array, used for expression
     *            templates
     * iterator   is a STL-style iterator
     * const_iterator is an STL-style const iterator
     */

    typedef P_numtype                T_numtype;
    typedef TinyVector<int, N_rank>  T_index;
    typedef Array<T_numtype, N_rank> T_array;
    typedef FastArrayIterator<T_numtype, N_rank> T_iterator;

    typedef ArrayIterator<T_numtype,N_rank> iterator;
    typedef ConstArrayIterator<T_numtype,N_rank> const_iterator;

    static const int _bz_rank = N_rank;

    //////////////////////////////////////////////
    // Constructors                             //
    //////////////////////////////////////////////

    
    /*
     * Construct an array from an array expression.
     */

    template<typename T_expr>
    explicit Array(_bz_ArrayExpr<T_expr> expr);

    /*
     * Any missing length arguments will have their value taken from the
     * last argument.  For example,
     *   Array<int,3> A(32,64);
     * will create a 32x64x64 array.  This is handled by setupStorage().
     */

    Array(GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        length_ = 0;
        stride_ = 0;
        zeroOffset_ = 0;
    }

    explicit Array(int length0, 
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        length_[0] = length0;
        setupStorage(0);
    }

    Array(int length0, int length1,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 2);
        TAU_TYPE_STRING(p1, "Array<T,N>::Array() [T="
            + CT(T_numtype) + ",N=" + CT(N_rank) + "]");
        TAU_PROFILE(p1, "void (int,int)", TAU_BLITZ);

        length_[0] = length0;
        length_[1] = length1;
        setupStorage(1);
    }

    Array(int length0, int length1, int length2,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 3);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        setupStorage(2);
    }

    Array(int length0, int length1, int length2, int length3,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 4);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        setupStorage(3);
    }

    Array(int length0, int length1, int length2, int length3, int length4,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 5);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        length_[4] = length4;
        setupStorage(4);
    }

    Array(int length0, int length1, int length2, int length3, int length4,
        int length5,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 6);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        length_[4] = length4;
        length_[5] = length5;
        setupStorage(5);
    }

    Array(int length0, int length1, int length2, int length3, int length4,
        int length5, int length6,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 7);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        length_[4] = length4;
        length_[5] = length5;
        length_[6] = length6;
        setupStorage(6);
    }

    Array(int length0, int length1, int length2, int length3, int length4,
        int length5, int length6, int length7,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 8);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        length_[4] = length4;
        length_[5] = length5;
        length_[6] = length6;
        length_[7] = length7;
        setupStorage(7);
    }

    Array(int length0, int length1, int length2, int length3, int length4,
        int length5, int length6, int length7, int length8,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 9);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        length_[4] = length4;
        length_[5] = length5;
        length_[6] = length6;
        length_[7] = length7;
        length_[8] = length8;
        setupStorage(8);
    }

    Array(int length0, int length1, int length2, int length3, int length4,
        int length5, int length6, int length7, int length8, int length9,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 10);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        length_[4] = length4;
        length_[5] = length5;
        length_[6] = length6;
        length_[7] = length7;
        length_[8] = length8;
        length_[9] = length9;
        setupStorage(9);
    }

    Array(int length0, int length1, int length2, int length3, int length4,
        int length5, int length6, int length7, int length8, int length9,
        int length10,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(N_rank >= 11);
        length_[0] = length0;
        length_[1] = length1;
        length_[2] = length2;
        length_[3] = length3;
        length_[4] = length4;
        length_[5] = length5;
        length_[6] = length6;
        length_[7] = length7;
        length_[8] = length8;
        length_[9] = length9;
        length_[10] = length10;
        setupStorage(10);
    }

    /*
     * Construct an array from an existing block of memory.  Ownership
     * is not acquired (this is provided for backwards compatibility).
     */
    Array(T_numtype* restrict dataFirst, TinyVector<int, N_rank> shape,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
      : MemoryBlockReference<T_numtype>(product(shape), dataFirst, 
          neverDeleteData),
        storage_(storage)
    {
        BZPRECONDITION(dataFirst != 0);

        length_ = shape;
        computeStrides();
        data_ += zeroOffset_;
    }

    /*
     * Construct an array from an existing block of memory, with a
     * given set of strides.  Ownership is not acquired (i.e. the memory
     * block will not be freed by Blitz++).
     */
    Array(T_numtype* restrict dataFirst, TinyVector<int, N_rank> shape,
        TinyVector<int, N_rank> stride, 
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
      : MemoryBlockReference<T_numtype>(product(shape), dataFirst, 
          neverDeleteData),
        storage_(storage)
    {
        BZPRECONDITION(dataFirst != 0);

        length_ = shape;
        stride_ = stride;
        calculateZeroOffset();
        data_ += zeroOffset_;
    }

    /*
     * Construct an array from an existing block of memory.
     */
    Array(T_numtype* restrict dataFirst, TinyVector<int, N_rank> shape,
        preexistingMemoryPolicy deletionPolicy,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
      : MemoryBlockReference<T_numtype>(product(shape), dataFirst, 
            deletionPolicy),
        storage_(storage)
    {
        BZPRECONDITION(dataFirst != 0);

        length_ = shape;
        computeStrides();
        data_ += zeroOffset_;

        if (deletionPolicy == duplicateData)
            reference(copy());
    }

    /*
     * Construct an array from an existing block of memory, with a
     * given set of strides.  
     */
    Array(T_numtype* restrict dataFirst, TinyVector<int, N_rank> shape,
        TinyVector<int, N_rank> stride,
        preexistingMemoryPolicy deletionPolicy,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
      : MemoryBlockReference<T_numtype>(product(shape), dataFirst, 
          deletionPolicy),
        storage_(storage)
    {
        BZPRECONDITION(dataFirst != 0);

        length_ = shape;
        stride_ = stride;
        calculateZeroOffset();
        data_ += zeroOffset_;

        if (deletionPolicy == duplicateData)
            reference(copy());
    }

    /*
     * This constructor takes an extent (length) vector and storage format.
     */

    Array(const TinyVector<int, N_rank>& extent, 
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        length_ = extent;
        setupStorage(N_rank - 1);
    }

    /*
     * This construct takes a vector of bases (lbounds) and a vector of
     * extents.
     */

    Array(const TinyVector<int, N_rank>& lbounds,
        const TinyVector<int, N_rank>& extent,
        const GeneralArrayStorage<N_rank>& storage 
           = GeneralArrayStorage<N_rank>());

    /*
     * These constructors allow arbitrary bases (starting indices) to be set.
     * e.g. Array<int,2> A(Range(10,20), Range(20,30))
     * will create an 11x11 array whose indices are 10..20 and 20..30
     */
    Array(Range r0, 
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(r0.isAscendingContiguous());

        length_[0] = r0.length();
        storage_.setBase(0, r0.first());
        setupStorage(0);
    }

    Array(Range r0, Range r1,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
    {
        BZPRECONDITION(r0.isAscendingContiguous() && 
            r1.isAscendingContiguous());

        length_[0] = r0.length();
        storage_.setBase(0, r0.first());
        length_[1] = r1.length();
        storage_.setBase(1, r1.first());

        setupStorage(1);
    }

    Array(Range r0, Range r1, Range r2,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3, Range r4,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3, Range r4, Range r5,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3, Range r4, Range r5,
        Range r6,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3, Range r4, Range r5,
        Range r6, Range r7,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3, Range r4, Range r5,
        Range r6, Range r7, Range r8,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3, Range r4, Range r5,
        Range r6, Range r7, Range r8, Range r9,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    Array(Range r0, Range r1, Range r2, Range r3, Range r4, Range r5,
        Range r6, Range r7, Range r8, Range r9, Range r10,
        GeneralArrayStorage<N_rank> storage = GeneralArrayStorage<N_rank>())
        : storage_(storage)
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

    /*
     * Create a reference of another array
     */
    Array(const Array<T_numtype, N_rank>& array)
#ifdef BZ_NEW_EXPRESSION_TEMPLATES
        : MemoryBlockReference<T_numtype>(),
          ETBase< Array<T_numtype, N_rank> >(array)
#else
        : MemoryBlockReference<T_numtype>()
#endif
    {
        // NEEDS_WORK: this const_cast is a tad ugly.
        reference(const_cast<T_array&>(array));
    }

    /*
     * These constructors are used for creating interlaced arrays (see
     * <blitz/arrayshape.h>
     */
    Array(const TinyVector<int,N_rank-1>& shape,
        int lastExtent, const GeneralArrayStorage<N_rank>& storage);
    //Array(const TinyVector<Range,N_rank-1>& shape,
    //    int lastExtent, const GeneralArrayStorage<N_rank>& storage);

    /*
     * These constructors make the array a view of a subportion of another
     * array.  If there fewer than N_rank Range arguments provided, no
     * slicing is performed in the unspecified ranks.
     * e.g. Array<int,3> A(20,20,20);
     *      Array<int,3> B(A, Range(5,15));
     * is equivalent to:
     *      Array<int,3> B(A, Range(5,15), Range::all(), Range::all());
     */
    Array(Array<T_numtype, N_rank>& array, Range r0)
    {
        constructSubarray(array, r0);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1)
    {
        constructSubarray(array, r0, r1);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2)
    {
        constructSubarray(array, r0, r1, r2);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3)
    {
        constructSubarray(array, r0, r1, r2, r3);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3, Range r4)
    {
        constructSubarray(array, r0, r1, r2, r3, r4);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3, Range r4, Range r5)
    {
        constructSubarray(array, r0, r1, r2, r3, r4, r5);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3, Range r4, Range r5, Range r6)
    {
        constructSubarray(array, r0, r1, r2, r3, r4, r5, r6);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3, Range r4, Range r5, Range r6, Range r7)
    {
        constructSubarray(array, r0, r1, r2, r3, r4, r5, r6, r7);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3, Range r4, Range r5, Range r6, Range r7, Range r8)
    {
        constructSubarray(array, r0, r1, r2, r3, r4, r5, r6, r7, r8);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3, Range r4, Range r5, Range r6, Range r7, Range r8, Range r9)
    {
        constructSubarray(array, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9);
    }

    Array(Array<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3, Range r4, Range r5, Range r6, Range r7, Range r8, Range r9,
        Range r10)
    {
        constructSubarray(array, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10);
    }

    Array(Array<T_numtype, N_rank>& array, 
        const RectDomain<N_rank>& subdomain)
    {
        constructSubarray(array, subdomain);
    }

    /* Constructor added by Julian Cummings */
    Array(Array<T_numtype, N_rank>& array, 
        const StridedDomain<N_rank>& subdomain)
    {
        constructSubarray(array, subdomain);
    }

    /*
     * This constructor is invoked by the operator()'s which take
     * a combination of integer and Range arguments.  It's not intended
     * for end-user use.
     */
    template<int N_rank2, typename R0, typename R1, typename R2, typename R3, typename R4,
        typename R5, typename R6, typename R7, typename R8, typename R9, typename R10>
    Array(Array<T_numtype,N_rank2>& array, R0 r0, R1 r1, R2 r2,
        R3 r3, R4 r4, R5 r5, R6 r6, R7 r7, R8 r8, R9 r9, R10 r10)
    {
        constructSlice(array, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10);
    }

    //////////////////////////////////////////////
    // Member functions
    //////////////////////////////////////////////

    const TinyVector<int, N_rank>&    base() const
    { return storage_.base(); }

    int                               base(int rank) const
    { return storage_.base(rank); }

    iterator                          begin() 
    { return iterator(*this); }

    const_iterator                    begin() const
    { return const_iterator(*this); }

    T_iterator                        beginFast() const
    { return T_iterator(*this); }

    // Deprecated: now extractComponent(...)
    template<typename P_numtype2>
    Array<P_numtype2,N_rank>          chopComponent(P_numtype2 a, int compNum,
                                          int numComponents) const
    { return extractComponent(a, compNum, numComponents); }

    int                               cols() const
    { return length_[1]; }

    int                               columns() const
    { return length_[1]; }

    T_array                           copy() const;

    // data_ always refers to the point (0,0,...,0) which may
    // not be in the array if the base is not zero in each rank.
    // These data() routines return a pointer to the first
    // element in the array (but note that it may not be
    // stored first in memory if some ranks are stored descending).

    int                               dataOffset() const
    {
        return dot(storage_.base(), stride_);
    }

    const T_numtype* restrict     data() const
    { return data_ + dataOffset(); }

    T_numtype* restrict           data() 
    { return data_ + dataOffset(); }

    // These dataZero() routines refer to the point (0,0,...,0)
    // which may not be in the array if the bases are nonzero.
    
    const T_numtype* restrict     dataZero() const
    { return data_; }

    T_numtype* restrict           dataZero()
    { return data_; }

    // These dataFirst() routines refer to the element in the
    // array which falls first in memory.

    int                               dataFirstOffset() const
    {
        int pos = 0;

        // Used to use tinyvector expressions:
        // return data_ + dot(storage_.base()
        //     + (1 - storage_.ascendingFlag()) * (length_ - 1), stride_);

        for (int i=0; i < N_rank; ++i)
           pos += (storage_.base(i) + (1-storage_.isRankStoredAscending(i)) *
              (length_(i)-1)) * stride_(i);

        return pos;
    }
    
    const T_numtype* restrict     dataFirst() const
    {
        return data_ + dataFirstOffset();
    }

    T_numtype* restrict           dataFirst()
    {
        return data_ + dataFirstOffset();
    }

    int                               depth() const
    { return length_[2]; }

    int                               dimensions() const
    { return N_rank; }

    RectDomain<N_rank>                domain() const
    {
        return RectDomain<N_rank>(lbound(), ubound());
    }

    void dumpStructureInformation(ostream& os = cout) const;

    iterator                          end()
    {
        return iterator();
    }

    const_iterator                    end() const
    {
        return const_iterator();
    }

    int                               extent(int rank) const
    { return length_[rank]; }

    const TinyVector<int,N_rank>&     extent() const
    { return length_; }

    template<typename P_numtype2>
    Array<P_numtype2,N_rank>          extractComponent(P_numtype2, int compNum,
                                          int numComponents) const;

    void                              free() 
    {
        changeToNullBlock();
        length_ = 0;
    }
 
    bool isMajorRank(int rank) const { return storage_.ordering(rank) == 0; }
    bool isMinorRank(int rank) const { return storage_.ordering(rank) != 0; }
    bool isRankStoredAscending(int rank) const {
        return storage_.isRankStoredAscending(rank);
    }

    bool isStorageContiguous() const;

    int                    lbound(int rank) const { return base(rank); }
    TinyVector<int,N_rank> lbound()         const { return base(); }

    int                            length(int rank) const { return length_[rank]; }
    const TinyVector<int, N_rank>& length()         const { return length_; }

    void makeUnique();

    int numElements() const { return product(length_); }

    // NEEDS_WORK -- Expose the numReferences() method
    // MemoryBlockReference<T_numtype>::numReferences;

    // The storage_.ordering_ array is a list of dimensions from
    // the most minor (stride 1) to major dimension.  Generally,
    // ordering(0) will return the dimension which has the smallest
    // stride, and ordering(N_rank-1) will return the dimension with
    // the largest stride.
    int                               ordering(int storageRankIndex) const
    { return storage_.ordering(storageRankIndex); }

    const TinyVector<int, N_rank>&    ordering() const
    { return storage_.ordering(); }

    void                              transposeSelf(int r0, int r1, int r2=0, 
        int r3=0, int r4=0, int r5=0, int r6=0, int r7=0, int r8=0, int 
        r9=0, int r10=0);
    T_array                           transpose(int r0, int r1, int r2=0,
        int r3=0, int r4=0, int r5=0, int r6=0, int r7=0, int r8=0, int
        r9=0, int r10=0);

    int                               rank() const
    { return N_rank; }

    void                              reference(const T_array&);

    // Added by Derrick Bass
    T_array                           reindex(const TinyVector<int,N_rank>&);
    void                              reindexSelf(const 
                                                TinyVector<int,N_rank>&);

    void                              resize(int extent);
    void                              resize(int extent1, int extent2);
    void                              resize(int extent1, int extent2,
                                        int extent3);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4, int extent5);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4, int extent5,
                                        int extent6);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4, int extent5,
                                        int extent6, int extent7);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4, int extent5,
                                        int extent6, int extent7, int extent8);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4, int extent5,
                                        int extent6, int extent7, int extent8,
                                        int extent9);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4, int extent5,
                                        int extent6, int extent7, int extent8,
                                        int extent9, int extent10);
    void                              resize(int extent1, int extent2,
                                        int extent3, int extent4, int extent5,
                                        int extent6, int extent7, int extent8,
                                        int extent9, int extent10, 
                                        int extent11);


    void                              resize(Range r1);
    void                              resize(Range r1, Range r2);
    void                              resize(Range r1, Range r2, Range r3);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4, Range r5);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4, Range r5, Range r6);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4, Range r5, Range r6,
                                        Range r7);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4, Range r5, Range r6,
                                        Range r7, Range r8);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4, Range r5, Range r6,
                                        Range r7, Range r8, Range r9);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4, Range r5, Range r6,
                                        Range r7, Range r8, Range r9, 
                                        Range r10);
    void                              resize(Range r1, Range r2, Range r3,
                                        Range r4, Range r5, Range r6,
                                        Range r7, Range r8, Range r9, 
                                        Range r10, Range r11);

    void                              resize(const TinyVector<int,N_rank>&);
 

    void                              resizeAndPreserve(const TinyVector<int,
                                                                   N_rank>&);
    void                              resizeAndPreserve(int extent);
    void                              resizeAndPreserve(int extent1, 
                                        int extent2);
    void                              resizeAndPreserve(int extent1, 
                                        int extent2, int extent3);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4,
                                        int extent5);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4,
                                        int extent5, int extent6);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4,
                                        int extent5, int extent6, int extent7);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4,
                                        int extent5, int extent6, int extent7,
                                        int extent8);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4,
                                        int extent5, int extent6, int extent7,
                                        int extent8, int extent9);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4,
                                        int extent5, int extent6, int extent7,
                                        int extent8, int extent9, 
                                        int extent10);
    void                              resizeAndPreserve(int extent1,
                                        int extent2, int extent3, int extent4,
                                        int extent5, int extent6, int extent7,
                                        int extent8, int extent9, int extent10,
                                        int extent11);

    // NEEDS_WORK -- resizeAndPreserve(Range,...)
    // NEEDS_WORK -- resizeAndPreserve(const Domain<N_rank>&);

    T_array                           reverse(int rank);
    void                              reverseSelf(int rank);

    int                               rows() const
    { return length_[0]; }
    
    void                              setStorage(GeneralArrayStorage<N_rank>);
    
    void                              slice(int rank, Range r);

    const TinyVector<int, N_rank>&    shape() const
    { return length_; }

    int                               size() const
    { return numElements(); }

    const TinyVector<int, N_rank>&    stride() const
    { return stride_; }

    int                               stride(int rank) const
    { return stride_[rank]; }

    int                               ubound(int rank) const
    { return base(rank) + length_(rank) - 1; }

    TinyVector<int, N_rank>           ubound() const
    { 
        TinyVector<int, N_rank> ub;
        for (int i=0; i < N_rank; ++i)
          ub(i) = base(i) + extent(i) - 1;
        // WAS: ub = base() + extent() - 1;
        return ub;
    }

    int                               zeroOffset() const
    { return zeroOffset_; }

    //////////////////////////////////////////////
    // Debugging routines
    //////////////////////////////////////////////

    bool isInRangeForDim(int i, int d) const {
        return i >= base(d) && (i - base(d)) < length_[d];
    }

    bool isInRange(int i0) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0];
    }

    bool isInRange(int i0, int i1) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1];
    }

    bool isInRange(int i0, int i1, int i2) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2];
    }

    bool isInRange(int i0, int i1, int i2, int i3) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3];
    }

    bool isInRange(int i0, int i1, int i2, int i3, int i4) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3]
            && i4 >= base(4) && (i4 - base(4)) < length_[4];
    }

    bool isInRange(int i0, int i1, int i2, int i3, int i4, int i5) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3]
            && i4 >= base(4) && (i4 - base(4)) < length_[4]
            && i5 >= base(5) && (i5 - base(5)) < length_[5];
    }

    bool isInRange(int i0, int i1, int i2, int i3, int i4, int i5, int i6) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3]
            && i4 >= base(4) && (i4 - base(4)) < length_[4]
            && i5 >= base(5) && (i5 - base(5)) < length_[5]
            && i6 >= base(6) && (i6 - base(6)) < length_[6];
    }

    bool isInRange(int i0, int i1, int i2, int i3, int i4,
        int i5, int i6, int i7) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3]
            && i4 >= base(4) && (i4 - base(4)) < length_[4]
            && i5 >= base(5) && (i5 - base(5)) < length_[5]
            && i6 >= base(6) && (i6 - base(6)) < length_[6]
            && i7 >= base(7) && (i7 - base(7)) < length_[7];
    }

    bool isInRange(int i0, int i1, int i2, int i3, int i4,
        int i5, int i6, int i7, int i8) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3]
            && i4 >= base(4) && (i4 - base(4)) < length_[4]
            && i5 >= base(5) && (i5 - base(5)) < length_[5]
            && i6 >= base(6) && (i6 - base(6)) < length_[6]
            && i7 >= base(7) && (i7 - base(7)) < length_[7]
            && i8 >= base(8) && (i8 - base(8)) < length_[8];
    }

    bool isInRange(int i0, int i1, int i2, int i3, int i4,
        int i5, int i6, int i7, int i8, int i9) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3]
            && i4 >= base(4) && (i4 - base(4)) < length_[4]
            && i5 >= base(5) && (i5 - base(5)) < length_[5]
            && i6 >= base(6) && (i6 - base(6)) < length_[6]
            && i7 >= base(7) && (i7 - base(7)) < length_[7]
            && i8 >= base(8) && (i8 - base(8)) < length_[8]
            && i9 >= base(9) && (i9 - base(9)) < length_[9];
    }

    bool isInRange(int i0, int i1, int i2, int i3, int i4,
        int i5, int i6, int i7, int i8, int i9, int i10) const {
        return i0 >= base(0) && (i0 - base(0)) < length_[0]
            && i1 >= base(1) && (i1 - base(1)) < length_[1]
            && i2 >= base(2) && (i2 - base(2)) < length_[2]
            && i3 >= base(3) && (i3 - base(3)) < length_[3]
            && i4 >= base(4) && (i4 - base(4)) < length_[4]
            && i5 >= base(5) && (i5 - base(5)) < length_[5]
            && i6 >= base(6) && (i6 - base(6)) < length_[6]
            && i7 >= base(7) && (i7 - base(7)) < length_[7]
            && i8 >= base(8) && (i8 - base(8)) < length_[8]
            && i9 >= base(9) && (i9 - base(9)) < length_[9]
            && i10 >= base(10) && (i10 - base(10)) < length_[10];
    }

    bool isInRange(const T_index& index) const {
        for (int i=0; i < N_rank; ++i)
            if (index[i] < base(i) || (index[i] - base(i)) >= length_[i])
                return false;

        return true;
    }

    bool assertInRange(const T_index& BZ_DEBUG_PARAM(index)) const {
        BZPRECHECK(isInRange(index), "Array index out of range: " << index
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0)) const {
        BZPRECHECK(isInRange(i0), "Array index out of range: " << i0
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1)) const {
        BZPRECHECK(isInRange(i0,i1), "Array index out of range: (" 
            << i0 << ", " << i1 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2), "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3), "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3),
        int BZ_DEBUG_PARAM(i4)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3,i4), "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3 
            << ", " << i4 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3), int BZ_DEBUG_PARAM(i4),
        int BZ_DEBUG_PARAM(i5)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3,i4,i5), "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3
            << ", " << i4 << ", " << i5 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3), int BZ_DEBUG_PARAM(i4),
        int BZ_DEBUG_PARAM(i5), int BZ_DEBUG_PARAM(i6)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3,i4,i5,i6), 
            "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3
            << ", " << i4 << ", " << i5 << ", " << i6 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3), int BZ_DEBUG_PARAM(i4),
        int BZ_DEBUG_PARAM(i5), int BZ_DEBUG_PARAM(i6),
        int BZ_DEBUG_PARAM(i7)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3,i4,i5,i6,i7),
            "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3
            << ", " << i4 << ", " << i5 << ", " << i6 << ", " << i7 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3), int BZ_DEBUG_PARAM(i4),
        int BZ_DEBUG_PARAM(i5), int BZ_DEBUG_PARAM(i6), int BZ_DEBUG_PARAM(i7),
        int BZ_DEBUG_PARAM(i8)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3,i4,i5,i6,i7,i8),
            "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3
            << ", " << i4 << ", " << i5 << ", " << i6 << ", " << i7 
            << ", " << i8 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3), int BZ_DEBUG_PARAM(i4),
        int BZ_DEBUG_PARAM(i5), int BZ_DEBUG_PARAM(i6), int BZ_DEBUG_PARAM(i7),
        int BZ_DEBUG_PARAM(i8), int BZ_DEBUG_PARAM(i9)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9),
            "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3
            << ", " << i4 << ", " << i5 << ", " << i6 << ", " << i7
            << ", " << i8 << ", " << i9 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    bool assertInRange(int BZ_DEBUG_PARAM(i0), int BZ_DEBUG_PARAM(i1),
        int BZ_DEBUG_PARAM(i2), int BZ_DEBUG_PARAM(i3), int BZ_DEBUG_PARAM(i4),
        int BZ_DEBUG_PARAM(i5), int BZ_DEBUG_PARAM(i6), int BZ_DEBUG_PARAM(i7),
        int BZ_DEBUG_PARAM(i8), int BZ_DEBUG_PARAM(i9),
        int BZ_DEBUG_PARAM(i10)) const
    {
        BZPRECHECK(isInRange(i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10),
            "Array index out of range: ("
            << i0 << ", " << i1 << ", " << i2 << ", " << i3
            << ", " << i4 << ", " << i5 << ", " << i6 << ", " << i7
            << ", " << i8 << ", " << i9 << ", " << i10 << ")"
            << endl << "Lower bounds: " << storage_.base() << endl
            <<         "Length:       " << length_ << endl);
        return true;
    }

    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    template<int N_rank2>
    const T_numtype& restrict operator()(const TinyVector<int,N_rank2>& index) const
    {
        assertInRange(index);
        return data_[dot(index, stride_)];
    }

    template<int N_rank2>
    T_numtype& restrict operator()(const TinyVector<int,N_rank2>& index) 
    {
        assertInRange(index);
        return data_[dot(index, stride_)];
    }

    const T_numtype& restrict operator()(TinyVector<int,1> index) const
    {
        assertInRange(index[0]);
        return data_[index[0] * stride_[0]];
    }

    T_numtype& operator()(TinyVector<int,1> index)
    {
        assertInRange(index[0]);
        return data_[index[0] * stride_[0]];
    }

    const T_numtype& restrict operator()(TinyVector<int,2> index) const
    {
        assertInRange(index[0], index[1]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]];
    }

    T_numtype& operator()(TinyVector<int,2> index)
    {
        assertInRange(index[0], index[1]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]];
    }

    const T_numtype& restrict operator()(TinyVector<int,3> index) const
    {
        assertInRange(index[0], index[1], index[2]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2]];
    }

    T_numtype& operator()(TinyVector<int,3> index)
    {
        assertInRange(index[0], index[1], index[2]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,4>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]];
    }

    T_numtype& operator()(const TinyVector<int,4>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,5>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4]];
    }

    T_numtype& operator()(const TinyVector<int,5>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,6>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]];
    }

    T_numtype& operator()(const TinyVector<int,6>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,7>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6]];
    }

    T_numtype& operator()(const TinyVector<int,7>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,8>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]];
    }

    T_numtype& operator()(const TinyVector<int,8>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,9>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7], index[8]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]
            + index[8] * stride_[8]];
    }

    T_numtype& operator()(const TinyVector<int,9>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7], index[8]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]
            + index[8] * stride_[8]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,10>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7], index[8], index[9]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]
            + index[8] * stride_[8] + index[9] * stride_[9]];
    }

    T_numtype& operator()(const TinyVector<int,10>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7], index[8], index[9]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]
            + index[8] * stride_[8] + index[9] * stride_[9]];
    }

    const T_numtype& restrict operator()(const TinyVector<int,11>& index) const
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7], index[8], index[9],
            index[10]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]
            + index[8] * stride_[8] + index[9] * stride_[9]
            + index[10] * stride_[10]];
    }

    T_numtype& operator()(const TinyVector<int,11>& index)
    {
        assertInRange(index[0], index[1], index[2], index[3],
            index[4], index[5], index[6], index[7], index[8], index[9],
            index[10]);
        return data_[index[0] * stride_[0] + index[1] * stride_[1]
            + index[2] * stride_[2] + index[3] * stride_[3]
            + index[4] * stride_[4] + index[5] * stride_[5]
            + index[6] * stride_[6] + index[7] * stride_[7]
            + index[8] * stride_[8] + index[9] * stride_[9]
            + index[10] * stride_[10]];
    }

    const T_numtype& restrict operator()(int i0) const
    { 
        assertInRange(i0);
        return data_[i0 * stride_[0]]; 
    }

    T_numtype& restrict operator()(int i0) 
    {
        assertInRange(i0);
        return data_[i0 * stride_[0]];
    }

    const T_numtype& restrict operator()(int i0, int i1) const
    { 
        assertInRange(i0, i1);
        return data_[i0 * stride_[0] + i1 * stride_[1]];
    }

    T_numtype& restrict operator()(int i0, int i1)
    {
        assertInRange(i0, i1);
        return data_[i0 * stride_[0] + i1 * stride_[1]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2) const
    {
        assertInRange(i0, i1, i2);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2) 
    {
        assertInRange(i0, i1, i2);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3) const
    {
        assertInRange(i0, i1, i2, i3);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3)
    {
        assertInRange(i0, i1, i2, i3);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4) const
    {
        assertInRange(i0, i1, i2, i3, i4);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4)
    {
        assertInRange(i0, i1, i2, i3, i4);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5) const
    {
        assertInRange(i0, i1, i2, i3, i4, i5);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5)
    {
        assertInRange(i0, i1, i2, i3, i4, i5);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6) const
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6)
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7) const
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7)
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7, int i8) const
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7, i8);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
            + i8 * stride_[8]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7, int i8)
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7, i8);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
            + i8 * stride_[8]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7, int i8, int i9) const
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
            + i8 * stride_[8] + i9 * stride_[9]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7, int i8, int i9)
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
            + i8 * stride_[8] + i9 * stride_[9]];
    }

    const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7, int i8, int i9, int i10) const
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7, i8, 
            i9, i10);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
            + i8 * stride_[8] + i9 * stride_[9] + i10 * stride_[10]];
    }

    T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
        int i4, int i5, int i6, int i7, int i8, int i9, int i10)
    {
        assertInRange(i0, i1, i2, i3, i4, i5, i6, i7, i8, 
            i9, i10);
        return data_[i0 * stride_[0] + i1 * stride_[1]
            + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
            + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
            + i8 * stride_[8] + i9 * stride_[9] + i10 * stride_[10]];
    }

    /*
     * Slicing to produce subarrays.  If the number of Range arguments is
     * fewer than N_rank, then missing arguments are treated like Range::all().
     */

    T_array& noConst() const
    { return const_cast<T_array&>(*this); }

    T_array operator()(const RectDomain<N_rank>& subdomain) const
    {
        return T_array(noConst(), subdomain);
    }

    /* Operator added by Julian Cummings */
    T_array operator()(const StridedDomain<N_rank>& subdomain) const
    {
        return T_array(noConst(), subdomain);
    }

    T_array operator()(Range r0) const
    {
        return T_array(noConst(), r0);
    }

    T_array operator()(Range r0, Range r1) const
    {
        return T_array(noConst(), r0, r1);
    }

    T_array operator()(Range r0, Range r1, Range r2) const
    {
        return T_array(noConst(), r0, r1, r2);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3) const
    {
        return T_array(noConst(), r0, r1, r2, r3);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3, Range r4) const
    {
        return T_array(noConst(), r0, r1, r2, r3, r4);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3, Range r4,
        Range r5) const
    {
        return T_array(noConst(), r0, r1, r2, r3, r4, r5);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3, Range r4,
        Range r5, Range r6) const
    {
        return T_array(noConst(), r0, r1, r2, r3, r4, r5, r6);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3, Range r4,
        Range r5, Range r6, Range r7) const
    {
        return T_array(noConst(), r0, r1, r2, r3, r4, r5, r6, r7);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3, Range r4,
        Range r5, Range r6, Range r7, Range r8) const
    {
        return T_array(noConst(), r0, r1, r2, r3, r4, r5, r6, r7, r8);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3, Range r4,
        Range r5, Range r6, Range r7, Range r8, Range r9) const
    {
        return T_array(noConst(), r0, r1, r2, r3, r4, r5, r6, r7, r8, r9);
    }

    T_array operator()(Range r0, Range r1, Range r2, Range r3, Range r4,
        Range r5, Range r6, Range r7, Range r8, Range r9, Range r10) const
    {
        return T_array(noConst(), r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10);
    }

    // Allow any mixture of Range, int and Vector<int> objects as
    // operands for operator():   A(Range(3,7), 5, Range(2,4))

    /*
     * These versions of operator() allow any combination of int
     * and Range operands to be used.  Each int operand reduces
     * the rank of the resulting array by one.  
     *
     * e.g.  Array<int,4> A(20,20,20,20);
     *       Array<int,2> B = A(Range(5,15), 3, 5, Range(8,9));
     *
     * SliceInfo is a helper class defined in <blitz/arrayslice.h>.
     * It counts the number of Range vs. int arguments and does some
     * other helpful things.
     *
     * Once partial specialization becomes widely implemented, these
     * operators may be expanded to accept Vector<int> arguments
     * and produce ArrayPick<T,N> objects.
     *
     * This operator() is not provided with a single argument because
     * the appropriate cases exist above.
     */

#ifdef BZ_HAVE_PARTIAL_ORDERING

    template<typename T1, typename T2>
    typename SliceInfo<T_numtype,T1,T2>::T_slice
    operator()(T1 r1, T2 r2) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2>::T_slice slice;
        return slice(noConst(), r1, r2, nilArraySection(), nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3>
    typename SliceInfo<T_numtype,T1,T2,T3>::T_slice 
    operator()(T1 r1, T2 r2, T3 r3) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3>::T_slice slice;
        return slice(noConst(), r1, r2, r3, nilArraySection(), nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4>
    typename SliceInfo<T_numtype,T1,T2,T3,T4>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    typename SliceInfo<T_numtype,T1,T2,T3,T4,T5>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4,T5>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, r5, nilArraySection(),
            nilArraySection(), nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, r5, r6, nilArraySection(), nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7>
    typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6, T7 r7) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, r5, r6, r7, nilArraySection(), nilArraySection(),
            nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7, typename T8>
    typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6, T7 r7, T8 r8) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, r5, r6, r7, r8,
            nilArraySection(), nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7, typename T8, typename T9>
    typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8,T9>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6, T7 r7, T8 r8, T9 r9) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8,T9>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, r5, r6, r7, r8, r9, nilArraySection(), nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7, typename T8, typename T9, typename T10>
    typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6, T7 r7, T8 r8, T9 r9, T10 r10) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, nilArraySection());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
        typename T7, typename T8, typename T9, typename T10, typename T11>
    typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11>::T_slice
    operator()(T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6, T7 r7, T8 r8, T9 r9, T10 r10, T11 r11) const
    {
        typedef typename SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11>::T_slice slice;
        return slice(noConst(), r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11);
    }

#endif // BZ_HAVE_PARTIAL_ORDERING

    /*
     * These versions of operator() are provided to support tensor-style
     * array notation, e.g.
     *
     * Array<float, 2> A, B;
     * firstIndex i;
     * secondIndex j;
     * thirdIndex k;
     * Array<float, 3> C = A(i,j) * B(j,k);
     */

    template<int N0>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0> >
    operator()(IndexPlaceholder<N0>) const
    { 
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0> >
            (noConst());
    }

    template<int N0, int N1>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1> >(noConst());
    } 

    template<int N0, int N1, int N2>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2> >(noConst());
    }

    template<int N0, int N1, int N2, int N3>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3> >(noConst());
    }

    template<int N0, int N1, int N2, int N3, int N4>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3, N4> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>, 
        IndexPlaceholder<N4>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3, N4> >(noConst());
    }

    template<int N0, int N1, int N2, int N3, int N4, int N5>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3, 
        N4, N5> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>, IndexPlaceholder<N4>,
        IndexPlaceholder<N5>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3, N4, N5> >(noConst());
    }

    template<int N0, int N1, int N2, int N3, int N4, int N5, int N6>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3,
        N4, N5, N6> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>, IndexPlaceholder<N4>,
        IndexPlaceholder<N5>, IndexPlaceholder<N6>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3, N4, N5, N6> >(noConst());
    }

    template<int N0, int N1, int N2, int N3, int N4, int N5, int N6,
        int N7>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3,
        N4, N5, N6, N7> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>, IndexPlaceholder<N4>,
        IndexPlaceholder<N5>, IndexPlaceholder<N6>, 
        IndexPlaceholder<N7>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3, N4, N5, N6, N7> >(noConst());
    }

    template<int N0, int N1, int N2, int N3, int N4, int N5, int N6,
        int N7, int N8>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3,
        N4, N5, N6, N7, N8> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>, IndexPlaceholder<N4>,
        IndexPlaceholder<N5>, IndexPlaceholder<N6>, IndexPlaceholder<N7>,
        IndexPlaceholder<N8>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3, N4, N5, N6, N7, N8> >(noConst());
    }

    template<int N0, int N1, int N2, int N3, int N4, int N5, int N6,
        int N7, int N8, int N9>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3,
        N4, N5, N6, N7, N8, N9> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>, IndexPlaceholder<N4>,
        IndexPlaceholder<N5>, IndexPlaceholder<N6>, IndexPlaceholder<N7>,
        IndexPlaceholder<N8>, IndexPlaceholder<N9>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3, N4, N5, N6, N7, N8, N9> >(noConst());
    }

    template<int N0, int N1, int N2, int N3, int N4, int N5, int N6,
        int N7, int N8, int N9, int N10>
    _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0, N1, N2, N3,
        N4, N5, N6, N7, N8, N9, N10> >
    operator()(IndexPlaceholder<N0>, IndexPlaceholder<N1>,
        IndexPlaceholder<N2>, IndexPlaceholder<N3>, IndexPlaceholder<N4>,
        IndexPlaceholder<N5>, IndexPlaceholder<N6>, IndexPlaceholder<N7>,
        IndexPlaceholder<N8>, IndexPlaceholder<N9>, 
        IndexPlaceholder<N10>) const
    {
        return _bz_ArrayExpr<ArrayIndexMapping<T_numtype, N_rank, N0,
            N1, N2, N3, N4, N5, N6, N7, N8, N9, N10> >(noConst());
    }

    //////////////////////////////////////////////
    // Support for multicomponent arrays
    //////////////////////////////////////////////

    /*
     * See <blitz/array/multi.h> for an explanation of the traits class
     * multicomponent_traits.
     */

    Array<typename multicomponent_traits<T_numtype>::T_element,N_rank>
    operator[](const unsigned component) {
        typedef typename multicomponent_traits<T_numtype>::T_element T_compType;

        return extractComponent(T_compType(),component,
                                multicomponent_traits<T_numtype>::numComponents);
    }

    const Array<typename multicomponent_traits<T_numtype>::T_element,N_rank>
    operator[](const unsigned component) const {
        typedef typename multicomponent_traits<T_numtype>::T_element T_compType;

        return extractComponent(T_compType(),component,
                                multicomponent_traits<T_numtype>::numComponents);
    }

    Array<typename multicomponent_traits<T_numtype>::T_element,N_rank>
    operator[](const int component) {
        return operator[](static_cast<unsigned>(component));
    }

    const Array<typename multicomponent_traits<T_numtype>::T_element,N_rank>
    operator[](const int component) const {
        return operator[](static_cast<unsigned>(component));
    }

    //////////////////////////////////////////////
    // Indirection
    //////////////////////////////////////////////
 
    template<typename T_indexContainer>
    IndirectArray<T_array, T_indexContainer>
    operator[](const T_indexContainer& index)
    {
        return IndirectArray<T_array, T_indexContainer>(*this,
            const_cast<T_indexContainer&>(index));
    }
 
    //////////////////////////////////////////////
    // Assignment Operators
    //////////////////////////////////////////////

    // Scalar operand
    // NEEDS_WORK : need a precondition check on
    // isStorageContiguous when operator, is used.
    ListInitializationSwitch<T_array,T_numtype*> operator=(T_numtype x)
    {
        return ListInitializationSwitch<T_array,T_numtype*>(*this, x);
    }

    T_array& initialize(T_numtype);

    // Was:
    // T_array& operator=(T_numtype);

#ifdef BZ_NEW_EXPRESSION_TEMPLATES
    template<typename T_expr>
    T_array& operator=(const ETBase<T_expr>&);
    T_array& operator=(const Array<T_numtype,N_rank>&);

    template<typename T> T_array& operator+=(const T&);
    template<typename T> T_array& operator-=(const T&);
    template<typename T> T_array& operator*=(const T&);
    template<typename T> T_array& operator/=(const T&);
    template<typename T> T_array& operator%=(const T&);
    template<typename T> T_array& operator^=(const T&);
    template<typename T> T_array& operator&=(const T&);
    template<typename T> T_array& operator|=(const T&);
    template<typename T> T_array& operator>>=(const T&);
    template<typename T> T_array& operator<<=(const T&);

#else
    T_array& operator+=(T_numtype);
    T_array& operator-=(T_numtype);
    T_array& operator*=(T_numtype);
    T_array& operator/=(T_numtype);
    T_array& operator%=(T_numtype);
    T_array& operator^=(T_numtype);
    T_array& operator&=(T_numtype);
    T_array& operator|=(T_numtype);
    T_array& operator>>=(T_numtype);
    T_array& operator<<=(T_numtype);

    // Array operands
    T_array& operator=(const Array<T_numtype,N_rank>&);

    template<typename P_numtype2> 
    T_array& operator=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator+=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator-=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator*=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator/=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator%=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator^=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator&=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator|=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator>>=(const Array<P_numtype2,N_rank>&);
    template<typename P_numtype2>
    T_array& operator<<=(const Array<P_numtype2,N_rank>&);

    // Array expression operands
    template<typename T_expr>
    inline T_array& operator=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator+=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator-=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator*=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator/=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator%=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator^=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator&=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator|=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator>>=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);
    template<typename T_expr>
    inline T_array& operator<<=(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr);

    // NEEDS_WORK -- Index placeholder operand

    // NEEDS_WORK -- Random operand
#endif

public:
    // Undocumented implementation routines

    template<typename T_expr, typename T_update>
    inline T_array& evaluate(T_expr expr, T_update);

#ifdef BZ_HAVE_STD
#ifdef BZ_ARRAY_SPACE_FILLING_TRAVERSAL
    template<typename T_expr, typename T_update>
    inline T_array& evaluateWithFastTraversal(
        const TraversalOrder<N_rank - 1>& order, 
        T_expr expr, T_update);
#endif // BZ_ARRAY_SPACE_FILLING_TRAVERSAL
#endif

#ifdef BZ_ARRAY_2D_STENCIL_TILING
    template<typename T_expr, typename T_update>
    inline T_array& evaluateWithTiled2DTraversal(
        T_expr expr, T_update);
#endif

    template<typename T_expr, typename T_update>
    inline T_array& evaluateWithIndexTraversal1(
        T_expr expr, T_update);

    template<typename T_expr, typename T_update>
    inline T_array& evaluateWithIndexTraversalN(
        T_expr expr, T_update);

    template<typename T_expr, typename T_update>
    inline T_array& evaluateWithStackTraversal1(
        T_expr expr, T_update);

    template<typename T_expr, typename T_update>
    inline T_array& evaluateWithStackTraversalN(
        T_expr expr, T_update);


    T_numtype* restrict getInitializationIterator() { return dataFirst(); }

    bool canCollapse(int outerRank, int innerRank) const { 
#ifdef BZ_DEBUG_TRAVERSE
        BZ_DEBUG_MESSAGE("stride(" << innerRank << ")=" << stride(innerRank)
          << ", extent()=" << extent(innerRank) << ", stride(outerRank)="
          << stride(outerRank));
#endif
        return (stride(innerRank) * extent(innerRank) == stride(outerRank)); 
    }

protected:
    //////////////////////////////////////////////
    // Implementation routines
    //////////////////////////////////////////////

    _bz_inline2 void computeStrides();
    _bz_inline2 void setupStorage(int rank);
    void constructSubarray(Array<T_numtype, N_rank>& array, 
        const RectDomain<N_rank>&);
    void constructSubarray(Array<T_numtype, N_rank>& array,
        const StridedDomain<N_rank>&);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0, Range r1);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3, Range r4);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3, Range r4, Range r5);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3, Range r4, Range r5, Range r6);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3, Range r4, Range r5, Range r6,
        Range r7);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3, Range r4, Range r5, Range r6,
        Range r7, Range r8);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3, Range r4, Range r5, Range r6,
        Range r7, Range r8, Range r9);
    void constructSubarray(Array<T_numtype, N_rank>& array, Range r0,
        Range r1, Range r2, Range r3, Range r4, Range r5, Range r6,
        Range r7, Range r8, Range r9, Range r10);

    void calculateZeroOffset();

    template<int N_rank2, typename R0, typename R1, typename R2, typename R3, typename R4, 
        typename R5, typename R6, typename R7, typename R8, typename R9, typename R10>
    void constructSlice(Array<T_numtype, N_rank2>& array, R0 r0, R1 r1, R2 r2, 
        R3 r3, R4 r4, R5 r5, R6 r6, R7 r7, R8 r8, R9 r9, R10 r10);

    template<int N_rank2>
    void slice(int& setRank, Range r, Array<T_numtype,N_rank2>& array,
        TinyVector<int,N_rank2>& rankMap, int sourceRank);

    template<int N_rank2>
    void slice(int& setRank, int i, Array<T_numtype,N_rank2>& array,
        TinyVector<int,N_rank2>& rankMap, int sourceRank);

    template<int N_rank2>
    void slice(int&, nilArraySection, Array<T_numtype,N_rank2>&,
        TinyVector<int,N_rank2>&, int)
    { }

    void doTranspose(int destRank, int sourceRank, T_array& array);

protected:
    //////////////////////////////////////////////
    // Data members
    //////////////////////////////////////////////

    // NB: adding new data members may require changes to ctors, reference()

    /*
     * For a description of the storage_ members, see the comments for class 
     * GeneralArrayStorage<N_rank> above.
     *
     * length_[] contains the extent of each rank.  E.g. a 10x20x30 array
     *           would have length_ = { 10, 20, 30}.
     * stride_[] contains the stride to move to the next element along each
     *           rank.
     * zeroOffset_ is the distance from the first element in the array 
     *           to the point (0,0,...,0).  If base_ is zero and all ranks are 
     *           stored ascending, then zeroOffset_ is zero.  This value
     *           is needed because to speed up indexing, the data_ member
     *           (inherited from MemoryBlockReference) always refers to
     *           (0,0,...,0).
     */
    GeneralArrayStorage<N_rank> storage_;
    TinyVector<int, N_rank> length_;
    TinyVector<int, N_rank> stride_;
    int zeroOffset_;
};

/*
 * Rank numbers start with zero, which may be confusing to users coming
 * from Fortran.  To make code more readable, the following constants
 * may help.  Example: instead of
 * 
 * int firstRankExtent = A.extent(0);
 *
 * One can write:
 *
 * int firstRankExtent = A.extent(firstRank);
 */

const int firstRank    = 0;
const int secondRank   = 1;
const int thirdRank    = 2;
const int fourthRank   = 3;
const int fifthRank    = 4;
const int sixthRank    = 5;
const int seventhRank  = 6;
const int eighthRank   = 7;
const int ninthRank    = 8;
const int tenthRank    = 9;
const int eleventhRank = 10;

const int firstDim    = 0;
const int secondDim   = 1;
const int thirdDim    = 2;
const int fourthDim   = 3;
const int fifthDim    = 4;
const int sixthDim    = 5;
const int seventhDim  = 6;
const int eighthDim   = 7;
const int ninthDim    = 8;
const int tenthDim    = 9;
const int eleventhDim = 10;

/*
 * Global Functions
 */

template<typename T_numtype>
ostream& operator<<(ostream&, const Array<T_numtype,1>&);

template<typename T_numtype>
ostream& operator<<(ostream&, const Array<T_numtype,2>&);

template<typename T_numtype, int N_rank>
ostream& operator<<(ostream&, const Array<T_numtype,N_rank>&);

template<typename T_numtype, int N_rank>
istream& operator>>(istream& is, Array<T_numtype,N_rank>& x);

template <typename P_numtype,int N_rank>
void swap(Array<P_numtype,N_rank>& a,Array<P_numtype,N_rank>& b) {
    Array<P_numtype,N_rank> c(a);
    a.reference(b);
    b.reference(c);
}

template <typename P_expr>
void find(Array<TinyVector<int,P_expr::rank>,1>& indices,
          const _bz_ArrayExpr<P_expr>& expr) {
    find(indices,
         static_cast< Array<typename P_expr::T_numtype,P_expr::rank> >(expr));
}

template <typename P_numtype, int N_rank>
void find(Array<TinyVector<int,N_rank>,1>& indices,
          const Array<P_numtype,N_rank>& exprVals) {
    indices.resize(exprVals.size());
    typename Array<P_numtype,N_rank>::const_iterator it, end = exprVals.end();
    int j=0; 
    for (it = exprVals.begin(); it != end; ++it)
        if (*it) 
            indices(j++) = it.position();
    if (j) 
        indices.resizeAndPreserve(j);
    else 
        indices.free();
    return;
}


BZ_NAMESPACE_END

/*
 * Include implementations of the member functions and some additional
 * global functions.
 */

#include <blitz/array/iter.h>       // Array iterators
#include <blitz/array/fastiter.h>   // Fast Array iterators (for et)
#include <blitz/array/expr.h>       // Array expression objects
#include <blitz/array/methods.cc>   // Member functions
#include <blitz/array/eval.cc>      // Array expression evaluation
#include <blitz/array/ops.cc>       // Assignment operators
#include <blitz/array/io.cc>        // Output formatting
#include <blitz/array/et.h>         // Expression templates
#include <blitz/array/reduce.h>     // Array reduction expression templates
#include <blitz/array/interlace.cc> // Allocation of interlaced arrays
#include <blitz/array/resize.cc>    // Array resize, resizeAndPreserve
#include <blitz/array/slicing.cc>   // Slicing and subarrays
#include <blitz/array/cycle.cc>     // Cycling arrays
#include <blitz/array/complex.cc>   // Special support for complex arrays
#include <blitz/array/zip.h>        // Zipping multicomponent types
#include <blitz/array/where.h>      // where(X,Y,Z)
#include <blitz/array/indirect.h>   // Indirection
#include <blitz/array/stencils.h>   // Stencil objects

#endif // BZ_ARRAY_H
