/*
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_TINYMATIO_CC
#define BZ_TINYMATIO_CC

#ifndef BZ_TINYMAT_H
 #include <blitz/tinymat.h>
#endif

BZ_NAMESPACE(blitz)

template <typename P_numtype, int N_rows, int N_columns>
ostream& operator<<(ostream& os,
    const TinyMatrix<P_numtype, N_rows, N_columns>& x)
{
    os << "(" << N_rows << "," << N_columns << "): " << endl;
    for (int i=0; i < N_rows; ++i)
    {
        os << " [ ";
        for (int j=0; j < N_columns; ++j)
        {
            os << setw(10) << x(i,j);
            if (!((j+1)%7))
                os << endl << "  ";
        }
        os << " ]" << endl;
    }
    return os;
}

template <typename P_numtype, int N_rows, int N_columns>
istream& operator>>(istream& is, 
    TinyMatrix<P_numtype, N_rows, N_columns>& x)
{
    int rows, columns;
    char sep;
             
    is >> rows >> columns;

    BZPRECHECK(rows == N_rows, "Size mismatch in number of rows");
    BZPRECHECK(columns == N_columns, "Size mismatch in number of columns");

    for (int i=0; i < N_rows; ++i) 
    {
        is >> sep;
        BZPRECHECK(sep == '[', "Format error while scanning input matrix"
            << endl << " (expected '[' before beginning of row data)");
        for (int j = 0; j < N_columns; ++j)
        {
            BZPRECHECK(!is.bad(), "Premature end of input while scanning matrix");
            is >> x(i,j);
        }
        is >> sep;
        BZPRECHECK(sep == ']', "Format error while scanning input matrix"
            << endl << " (expected ']' after end of row data)");
    }

    return is;
}

BZ_NAMESPACE_END

#endif // BZ_TINYMATIO_CC

