/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:25:28  ej
 * Looks like I need all the .cc files for blitz also
 *
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_TINYVECIO_CC
#define BZ_TINYVECIO_CC

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

BZ_NAMESPACE(blitz)

// NEEDS_WORK

template<class P_numtype, int N_length>
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
