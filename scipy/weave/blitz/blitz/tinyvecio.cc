/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_TINYVECIO_CC
#define BZ_TINYVECIO_CC

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

BZ_NAMESPACE(blitz)

// NEEDS_WORK

template<typename P_numtype, int N_length>
ostream& operator<<(ostream& os, const TinyVector<P_numtype, N_length>& x)
{
    os << N_length << " [ ";
    for (int i=0; i < N_length; ++i)
    {
        os << setw(10) << x[i];
        if (!((i+1)%7))
            os << endl << "  ";
    }
    os << " ]";
    return os;
}

// Input of tinyvec contribute by Adam Levar <adaml@mcneilhouse.com>
template <typename T_numtype, int N_length>
istream& operator>>(istream& is, TinyVector<T_numtype, N_length>& x)
{
    int length;
    char sep;
             
    is >> length;
    is >> sep;
    BZPRECHECK(sep == '[', "Format error while scanning input array"
        << endl << " (expected '[' before beginning of array data)");

    BZPRECHECK(length == N_length, "Size mismatch");                    
    for (int i = 0; i < N_length; ++i)
    {
        BZPRECHECK(!is.bad(), "Premature end of input while scanning array");
        is >> x(i);
    }
    is >> sep;
    BZPRECHECK(sep == ']', "Format error while scanning input array"
       << endl << " (expected ']' after end of array data)");
    
    return is;
}

BZ_NAMESPACE_END

#endif // BZ_TINYVECIO_CC
