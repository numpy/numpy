#ifndef BZ_ARRAY_STORAGE_H
#define BZ_ARRAY_STORAGE_H

BZ_NAMESPACE(blitz)

/*
 * Declaration of class GeneralStorage<N_rank>
 *
 * This class describes a storage format for an N-dimensional array.
 * The dimensions can be stored in an arbitrary order (for example, as
 * a C-style row major array or Fortran-style column major array, or
 * something else entirely).  Each dimension can be stored in either
 * ascending (the most common) or descending order.  Each dimension
 * can have its own base (starting index value: e.g. 0 for C-style arrays, 
 * 1 for Fortran arrays).
 *
 * GeneralArrayStorage<N> defaults to C-style arrays.  To implement
 * other storage formats, subclass and modify the constructor.  The
 * class FortranArray, below, is an example.
 *
 * Objects inheriting from GeneralArrayStorage<N> can be passed as
 * an optional constructor argument to Array objects.
 * e.g. Array<int,3> A(16,16,16, FortranArray<3>());
 * will create a 3-dimensional 16x16x16 Fortran-style array.
 */

template<int N_rank>
class GeneralArrayStorage {
public:
    class noInitializeFlag { };

    GeneralArrayStorage(noInitializeFlag)
    { }

    GeneralArrayStorage()
    {
        ordering_ = Range(N_rank - 1, 0, -1);
        ascendingFlag_ = 1;
        base_ = 0;
    }

    GeneralArrayStorage(const GeneralArrayStorage<N_rank>& x)
        : ordering_(x.ordering_), ascendingFlag_(x.ascendingFlag_),
          base_(x.base_)
    { 
    }

    GeneralArrayStorage(TinyVector<int,N_rank> ordering,
        TinyVector<bool,N_rank> ascendingFlag)
      : ordering_(ordering), ascendingFlag_(ascendingFlag)
    {
        base_ = 0;
    }

    ~GeneralArrayStorage()
    { }

    TinyVector<int, N_rank>& ordering()
    { return ordering_; }

    const TinyVector<int, N_rank>& ordering() const
    { return ordering_; }

    int ordering(int i) const
    { return ordering_[i]; }

    void setOrdering(int i, int order) 
    { ordering_[i] = order; }

    _bz_bool allRanksStoredAscending() const
    {
        _bz_bool result = _bz_true;
        for (int i=0; i < N_rank; ++i)
            result &= ascendingFlag_[i];
        return result;
    }

    _bz_bool isRankStoredAscending(int i) const
    { return ascendingFlag_[i]; }

    TinyVector<bool, N_rank>& ascendingFlag() 
    { return ascendingFlag_; }

    const TinyVector<bool, N_rank>& ascendingFlag() const
    { return ascendingFlag_; }

    void setAscendingFlag(int i, int ascendingFlag) 
    { ascendingFlag_[i] = ascendingFlag; }

    TinyVector<int, N_rank>& base()
    { return base_; }

    const TinyVector<int, N_rank>& base() const
    { return base_; }

    int base(int i) const
    { return base_[i]; }

    void setBase(int i, int base)
    { base_[i] = base; }

    void setBase(const TinyVector<int, N_rank>& base)
    { base_ = base; }

protected:
    /*
     * ordering_[] specifies the order in which the array is stored in
     * memory.  For a newly allocated array, ordering_(0) will give the
     * rank with unit stride, and ordering_(N_rank-1) will be the rank
     * with largest stride.  An order like [2, 1, 0] corresponds to
     * C-style array storage; an order like [0, 1, 2] corresponds to
     * Fortran array storage.
     *
     * ascendingFlag_[] indicates whether the data in a rank is stored
     * in ascending or descending order.  Most of the time these values
     * will all be true (indicating ascending order).  Some peculiar 
     * formats (e.g. MS-Windows BMP image format) store the data in 
     * descending order.
     *  
     * base_[] gives the first valid index for each rank.  For a C-style
     * array, all the base_ elements will be zero; for a Fortran-style
     * array, they will be one.  base_[] can be set arbitrarily using
     * the Array constructor which takes a Range argument, e.g.
     * Array<float,2> A(Range(30,40),Range(23,33));
     * will create an array with base_[] = { 30, 23 }.
     */
    TinyVector<int,  N_rank> ordering_;
    TinyVector<bool, N_rank> ascendingFlag_;
    TinyVector<int,  N_rank> base_;
};

/*
 * Class FortranArray specializes GeneralArrayStorage to provide Fortran
 * style arrays (column major ordering, base of 1).  The noInitializeFlag()
 * passed to the base constructor indicates that the subclass will take
 * care of initializing the ordering_, ascendingFlag_ and base_ members.
 */

template<int N_rank>
class FortranArray : public GeneralArrayStorage<N_rank> {
public:
    FortranArray()
        : GeneralArrayStorage<N_rank>(noInitializeFlag())
    {
        ordering_ = Range(0, N_rank - 1);
        ascendingFlag_ = 1;
        base_ = 1;
    }
};


// This tag class can be used to provide a nicer notation for
// constructing Fortran-style arrays: instead of
//     Array<int,2> A(3, 3, FortranArray<2>());
// one can simply write:
//     Array<int,2> A(3, 3, fortranArray);
// where fortranArray is an object of type _bz_fortranTag.

class _bz_fortranTag {
public:
    operator GeneralArrayStorage<1>()
    { return FortranArray<1>(); }

    operator GeneralArrayStorage<2>()
    { return FortranArray<2>(); }

    operator GeneralArrayStorage<3>()
    { return FortranArray<3>(); }

    operator GeneralArrayStorage<4>()
    { return FortranArray<4>(); }

    operator GeneralArrayStorage<5>()
    { return FortranArray<5>(); }

    operator GeneralArrayStorage<6>()
    { return FortranArray<6>(); }

    operator GeneralArrayStorage<7>()
    { return FortranArray<7>(); }

    operator GeneralArrayStorage<8>()
    { return FortranArray<8>(); }

    operator GeneralArrayStorage<9>()
    { return FortranArray<9>(); }

    operator GeneralArrayStorage<10>()
    { return FortranArray<10>(); }

    operator GeneralArrayStorage<11>()
    { return FortranArray<11>(); }
};

// A global instance of this class will be placed in
// the blitz library (libblitz.a on unix machines).

_bz_global _bz_fortranTag fortranArray;


/*
 * Class ColumnMajorArray specializes GeneralArrayStorage to provide column
 * major arrays (column major ordering, base of 0).
 */

template<int N_rank>
class ColumnMajorArray : public GeneralArrayStorage<N_rank> {
public:
    ColumnMajorArray()
        : GeneralArrayStorage<N_rank>(noInitializeFlag())
    {
        ordering_ = Range(0, N_rank - 1);
        ascendingFlag_ = 1;
        base_ = 0;
    }
};

BZ_NAMESPACE_END

#endif // BZ_ARRAY_STORAGE_H

