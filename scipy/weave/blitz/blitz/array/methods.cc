#ifndef BZ_ARRAYMETHODS_CC
#define BZ_ARRAYMETHODS_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/methods.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_numtype, int N_rank> template<typename T_expr>
Array<P_numtype,N_rank>::Array(_bz_ArrayExpr<T_expr> expr)
{
    // Determine extent of the array expression

    TinyVector<int,N_rank> lbound, extent, ordering;
    TinyVector<bool,N_rank> ascendingFlag;
    TinyVector<bool,N_rank> in_ordering;
    in_ordering = false;

    int j = 0;
    for (int i=0; i < N_rank; ++i)
    {
        lbound(i) = expr.lbound(i);
        int ubound = expr.ubound(i);
        extent(i) = ubound - lbound(i) + 1;
        int orderingj = expr.ordering(i);
        if (orderingj != INT_MIN && orderingj < N_rank &&
            !in_ordering( orderingj )) { // unique value in ordering array
            in_ordering( orderingj ) = true;
            ordering(j++) = orderingj;
        }
        int ascending = expr.ascending(i);
        ascendingFlag(i) = (ascending == 1);

#ifdef BZ_DEBUG
        if ((lbound(i) == INT_MIN) || (ubound == INT_MAX) 
          || (ordering(i) == INT_MIN) || (ascending == INT_MIN))
        {
          BZPRECHECK(0,
           "Attempted to construct an array from an expression " << endl
           << "which does not have a shape.  To use this constructor, "
           << endl 
           << "the expression must contain at least one array operand.");
          return;
        }
#endif
    }

    // It is possible that ordering is not a permutation of 0,...,N_rank-1.
    // In that case j will be less than N_rank. We fill in ordering with the
    // usused values in decreasing order.
    for (int i = N_rank-1; j < N_rank; ++j) {
        while (in_ordering(i))
          --i;
        ordering(j) = i--;
    }

    Array<T_numtype,N_rank> A(lbound,extent,
        GeneralArrayStorage<N_rank>(ordering,ascendingFlag));
    A = expr;
    reference(A);
}

template<typename P_numtype, int N_rank>
Array<P_numtype,N_rank>::Array(const TinyVector<int, N_rank>& lbounds,
    const TinyVector<int, N_rank>& extent,
    const GeneralArrayStorage<N_rank>& storage)
    : storage_(storage)
{
    length_ = extent;
    storage_.setBase(lbounds);
    setupStorage(N_rank - 1);
}


/*
 * This routine takes the storage information for the array
 * (ascendingFlag_[], base_[], and ordering_[]) and the size
 * of the array (length_[]) and computes the stride vector
 * (stride_[]) and the zero offset (see explanation in array.h).
 */
template<typename P_numtype, int N_rank>
_bz_inline2 void Array<P_numtype, N_rank>::computeStrides()
{
    if (N_rank > 1)
    {
      int stride = 1;

      // This flag simplifies the code in the loop, encouraging
      // compile-time computation of strides through constant folding.
      bool allAscending = storage_.allRanksStoredAscending();

      // BZ_OLD_FOR_SCOPING
      int n;
      for (n=0; n < N_rank; ++n)
      {
          int strideSign = +1;

          // If this rank is stored in descending order, then the stride
          // will be negative.
          if (!allAscending)
          {
            if (!isRankStoredAscending(ordering(n)))
                strideSign = -1;
          }

          // The stride for this rank is the product of the lengths of
          // the ranks minor to it.
          stride_[ordering(n)] = stride * strideSign;

          stride *= length_[ordering(n)];
      }
    }
    else {
        // Specialization for N_rank == 1
        // This simpler calculation makes it easier for the compiler
        // to propagate stride values.

        if (isRankStoredAscending(0))
            stride_[0] = 1;
        else
            stride_[0] = -1;
    }

    calculateZeroOffset();
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::calculateZeroOffset()
{
    // Calculate the offset of (0,0,...,0)
    zeroOffset_ = 0;

    // zeroOffset_ = - sum(where(ascendingFlag_, stride_ * base_,
    //     (length_ - 1 + base_) * stride_))
    for (int n=0; n < N_rank; ++n)
    {
        if (!isRankStoredAscending(n))
            zeroOffset_ -= (length_[n] - 1 + base(n)) * stride_[n];
        else
            zeroOffset_ -= stride_[n] * base(n);
    }
}

template<typename P_numtype, int N_rank>
bool Array<P_numtype, N_rank>::isStorageContiguous() const
{
    // The storage is contiguous if for the set
    // { | stride[i] * extent[i] | }, i = 0..N_rank-1,
    // there is only one value which is not in the set
    // of strides; and if there is one stride which is 1.

    // This algorithm is quadratic in the rank.  It is hard
    // to imagine this being a serious problem.

    int numStridesMissing = 0;
    bool haveUnitStride = false;

    for (int i=0; i < N_rank; ++i)
    {
        int stride = BZ_MATHFN_SCOPE(abs)(stride_[i]);
        if (stride == 1)
            haveUnitStride = true;

        int vi = stride * length_[i];

        int j = 0;
        for (j=0; j < N_rank; ++j)
            if (BZ_MATHFN_SCOPE(abs)(stride_[j]) == vi)
                break;

        if (j == N_rank)
        {
            ++numStridesMissing;
            if (numStridesMissing == 2)
                return false;
        }
    }

    return haveUnitStride;
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::dumpStructureInformation(ostream& os) const
{
    os << "Dump of Array<" << BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(P_numtype) 
       << ", " << N_rank << ">:" << endl
       << "ordering_      = " << storage_.ordering() << endl
       << "ascendingFlag_ = " << storage_.ascendingFlag() << endl
       << "base_          = " << storage_.base() << endl
       << "length_        = " << length_ << endl
       << "stride_        = " << stride_ << endl
       << "zeroOffset_    = " << zeroOffset_ << endl
       << "numElements()  = " << numElements() << endl
       << "isStorageContiguous() = " << isStorageContiguous() << endl;
}

/*
 * Make this array a view of another array's data.
 */
template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::reference(const Array<P_numtype, N_rank>& array)
{
    storage_ = array.storage_;
    length_ = array.length_;
    stride_ = array.stride_;
    zeroOffset_ = array.zeroOffset_;

    MemoryBlockReference<P_numtype>::changeBlock(array.noConst());
}

/*
 * Modify the Array storage.  Array must be unallocated.
 */
template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::setStorage(GeneralArrayStorage<N_rank> x)
{
#ifdef BZ_DEBUG
    if (size() != 0) {
        BZPRECHECK(0,"Cannot modify storage format of an Array that has already been allocated!" << endl);
        return;
    }
#endif
    storage_ = x;
    return;
}

/*
 * This method is called to allocate memory for a new array.  
 */
template<typename P_numtype, int N_rank>
_bz_inline2 void Array<P_numtype, N_rank>::setupStorage(int lastRankInitialized)
{
    TAU_TYPE_STRING(p1, "Array<T,N>::setupStorage() [T="
        + CT(P_numtype) + ",N=" + CT(N_rank) + "]");
    TAU_PROFILE(" ", p1, TAU_BLITZ);

    /*
     * If the length of some of the ranks was unspecified, fill these
     * in using the last specified value.
     *
     * e.g. Array<int,3> A(40) results in a 40x40x40 array.
     */
    for (int i=lastRankInitialized + 1; i < N_rank; ++i)
    {
        storage_.setBase(i, storage_.base(lastRankInitialized));
        length_[i] = length_[lastRankInitialized];
    }

    // Compute strides
    computeStrides();

    // Allocate a block of memory
    int numElem = numElements();
    if (numElem==0)
        MemoryBlockReference<P_numtype>::changeToNullBlock();
    else
        MemoryBlockReference<P_numtype>::newBlock(numElem);

    // Adjust the base of the array to account for non-zero base
    // indices and reversals
    data_ += zeroOffset_;
}

template<typename P_numtype, int N_rank>
Array<P_numtype, N_rank> Array<P_numtype, N_rank>::copy() const
{
    if (numElements())
    {
        Array<T_numtype, N_rank> z(length_, storage_);
        z = *this;
        return z;
    }
    else {
        // Null array-- don't bother allocating an empty block.
        return *this;
    }
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::makeUnique()
{
    if (numReferences() > 1)
    {
        T_array tmp = copy();
        reference(tmp);
    }
}

template<typename P_numtype, int N_rank>
Array<P_numtype, N_rank> Array<P_numtype, N_rank>::transpose(int r0, int r1, 
    int r2, int r3, int r4, int r5, int r6, int r7, int r8, int r9, int r10)
{
    T_array B(*this);
    B.transposeSelf(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10);
    return B;
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::transposeSelf(int r0, int r1, int r2, int r3,
    int r4, int r5, int r6, int r7, int r8, int r9, int r10)
{
    BZPRECHECK(r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10 == N_rank * (N_rank-1) / 2,
        "Invalid array transpose() arguments." << endl
        << "Arguments must be a permutation of the numerals (0,...,"
        << (N_rank - 1) << ")");

    // Create a temporary reference copy of this array
    Array<T_numtype, N_rank> x(*this);

    // Now reorder the dimensions using the supplied permutation
    doTranspose(0, r0, x);
    doTranspose(1, r1, x);
    doTranspose(2, r2, x);
    doTranspose(3, r3, x);
    doTranspose(4, r4, x);
    doTranspose(5, r5, x);
    doTranspose(6, r6, x);
    doTranspose(7, r7, x);
    doTranspose(8, r8, x);
    doTranspose(9, r9, x);
    doTranspose(10, r10, x);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::doTranspose(int destRank, int sourceRank,
    Array<T_numtype, N_rank>& array)
{
    // BZ_NEEDS_WORK: precondition check

    if (destRank >= N_rank)
        return;

    length_[destRank] = array.length_[sourceRank];
    stride_[destRank] = array.stride_[sourceRank];
    storage_.setAscendingFlag(destRank, 
        array.isRankStoredAscending(sourceRank));
    storage_.setBase(destRank, array.base(sourceRank));

    // BZ_NEEDS_WORK: Handling the storage ordering is currently O(N^2)
    // but it can be done fairly easily in linear time by constructing
    // the appropriate permutation.

    // Find sourceRank in array.storage_.ordering_
    int i=0;
    for (; i < N_rank; ++i)
        if (array.storage_.ordering(i) == sourceRank)
            break;

    storage_.setOrdering(i, destRank);
}

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::reverseSelf(int rank)
{
    BZPRECONDITION(rank < N_rank);

    storage_.setAscendingFlag(rank, !isRankStoredAscending(rank));

    int adjustment = stride_[rank] * (lbound(rank) + ubound(rank));
    zeroOffset_ += adjustment;
    data_ += adjustment;
    stride_[rank] *= -1;
}

template<typename P_numtype, int N_rank>
Array<P_numtype, N_rank> Array<P_numtype,N_rank>::reverse(int rank)
{
    T_array B(*this);
    B.reverseSelf(rank);
    return B;
}

template<typename P_numtype, int N_rank> template<typename P_numtype2>
Array<P_numtype2,N_rank> Array<P_numtype,N_rank>::extractComponent(P_numtype2, 
    int componentNumber, int numComponents) const
{
    BZPRECONDITION((componentNumber >= 0) 
        && (componentNumber < numComponents));

    TinyVector<int,N_rank> stride2;
    for (int i=0; i < N_rank; ++i)
      stride2(i) = stride_(i) * numComponents;
    const P_numtype2* dataFirst2 = 
        ((const P_numtype2*)dataFirst()) + componentNumber;
    return Array<P_numtype2,N_rank>(const_cast<P_numtype2*>(dataFirst2), 
        length_, stride2, storage_);
}

/* 
 * These routines reindex the current array to use a new base vector.
 * The first reindexes the array, the second just returns a reindex view
 * of the current array, leaving the current array unmodified.
 * (Contributed by Derrick Bass)
 */
template<typename P_numtype, int N_rank>
_bz_inline2 void Array<P_numtype, N_rank>::reindexSelf(const 
    TinyVector<int, N_rank>& newBase) 
{
    int delta = 0;
    for (int i=0; i < N_rank; ++i)
      delta += (base(i) - newBase(i)) * stride_(i);

    data_ += delta;

    // WAS: dot(base() - newBase, stride_);

    storage_.setBase(newBase);
    calculateZeroOffset();
}

template<typename P_numtype, int N_rank>
_bz_inline2 Array<P_numtype, N_rank> 
Array<P_numtype, N_rank>::reindex(const TinyVector<int, N_rank>& newBase) 
{
    T_array B(*this);
    B.reindexSelf(newBase);
    return B;
}

BZ_NAMESPACE_END

#endif // BZ_ARRAY_CC

