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


// NEEDS_WORK: resizeAndPreserve for N_rank = 7..11

template<class T_numtype, int N_rank>
void Array<T_numtype, N_rank>::resize(const TinyVector<int,N_rank>& extent)
{
// NEEDS_WORK
//    BZPRECONDITION(all(extent > 0));
//    if (any(extent != length_))
//    {
        length_ = extent;
        setupStorage(N_rank);
//    }
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYRESIZE_CC
