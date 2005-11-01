/***************************************************************************
 * blitz/array/io.cc  Input/output of arrays.
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
#ifndef BZ_ARRAYIO_CC
#define BZ_ARRAYIO_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/io.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

template<typename T_numtype>
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

template<typename T_numtype>
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

template<typename T_numtype, int N_rank>
ostream& operator<<(ostream& os, const Array<T_numtype,N_rank>& x)
{
    for (int i=0; i < N_rank; ++i)
    {
        os << x.extent(i);
        if (i != N_rank - 1)
            os << " x ";
    }

    os << endl << "[ ";
    
    _bz_typename Array<T_numtype, N_rank>::const_iterator iter = x.begin();
    _bz_typename Array<T_numtype, N_rank>::const_iterator end = x.end();
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

template<typename T_numtype, int N_rank>
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

    _bz_typename Array<T_numtype,N_rank>::iterator iter = x.begin();
    _bz_typename Array<T_numtype,N_rank>::iterator end = x.end();

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
