#ifndef BZ_ARRAY_H
 #error <blitz/array/io.cc> must be included via <blitz/array.h>
#endif

#ifndef BZ_ARRAYIO_CC
#define BZ_ARRAYIO_CC

BZ_NAMESPACE(blitz)

template<class T_numtype>
ostream& operator<<(ostream& os, const Array<T_numtype,1>& x)
{
    os << x.extent(firstRank) << endl;
    os << " [ ";
    for (int i=x.lbound(firstRank); i <= x.ubound(firstRank); ++i)
    {
        os << setw(9) << x(i) << " ";
        if (!((i+1-x.lbound(firstRank))%7))
            os << endl << "  ";
    }
    os << " ]";
    return os;
}

template<class T_numtype>
ostream& operator<<(ostream& os, const Array<T_numtype,2>& x)
{
    os << x.rows() << " x " << x.columns() << endl;
    os << "[ ";
    for (int i=x.lbound(firstRank); i <= x.ubound(firstRank); ++i)
    {
        for (int j=x.lbound(secondRank); j <= x.ubound(secondRank); ++j)
        {
            os << setw(9) << x(i,j) << " ";
            if (!((j+1-x.lbound(secondRank)) % 7))
                os << endl << "  ";
        }

        if (i != x.ubound(firstRank))
           os << endl << "  ";
    }

    os << "]" << endl;

    return os;
}

template<class T_numtype, int N_rank>
ostream& operator<<(ostream& os, const Array<T_numtype,N_rank>& x)
{
    for (int i=0; i < N_rank; ++i)
    {
        os << x.extent(i);
        if (i != N_rank - 1)
            os << " x ";
    }

    os << endl << "[ ";
   
    typedef _bz_typename Array<T_numtype, N_rank>::const_iterator 
         ArrayConstIterator;
    ArrayConstIterator iter = x.begin(), end = x.end();
    int p = 0;

    while (iter != end) {
        os << setw(9) << (*iter) << " ";
        ++iter;

        // See if we need a linefeed
        ++p;
        if (!(p % 7))
            os << endl << "  ";
    }

    os << "]" << endl;
    return os;
}

/*
 *  Input
 */

template<class T_numtype, int N_rank>
istream& operator>>(istream& is, Array<T_numtype,N_rank>& x)
{
    TinyVector<int,N_rank> extent;
    char sep;
 
    // Read the extent vector: this is separated by 'x's, e.g.
    // 3 x 4 x 5

    for (int i=0; i < N_rank; ++i)
    {
        is >> extent(i);

        BZPRECHECK(!is.bad(), "Premature end of input while scanning array");

        if (i != N_rank - 1)
        {
            is >> sep;
            BZPRECHECK(sep == 'x', "Format error while scanning input array"
                << endl << " (expected 'x' between array extents)");
        }
    }

    is >> sep;
    BZPRECHECK(sep == '[', "Format error while scanning input array"
        << endl << " (expected '[' before beginning of array data)");

    x.resize(extent);

    typedef _bz_typename Array<T_numtype,N_rank>::iterator
         ArrayIterator;
    ArrayIterator iter = x.begin(), end = x.end();

    while (iter != end) {
        BZPRECHECK(!is.bad(), "Premature end of input while scanning array");

        is >> (*iter);
        ++iter;
    }

    is >> sep;
    BZPRECHECK(sep == ']', "Format error while scanning input array"
       << endl << " (expected ']' after end of array data)");

    return is;
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYIO_CC
