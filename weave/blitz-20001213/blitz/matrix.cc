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
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.4  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.3  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_MATRIX_CC
#define BZ_MATRIX_CC

#ifndef BZ_MATRIX_H
 #include <blitz/matrix.h>
#endif

BZ_NAMESPACE(blitz)

// Matrix expression operand
template<class P_numtype, class P_structure> template<class P_expr>
Matrix<P_numtype, P_structure>& 
Matrix<P_numtype, P_structure>::operator=(_bz_MatExpr<P_expr> expr)
{
    // Check for compatible structures.

    // Fast evaluation (compatible structures)
    // (not implemented)

    // Slow evaluation
    _bz_typename P_structure::T_iterator iter(rows(), cols());
    while (iter)
    {
        data_[iter.offset()] = expr(iter.row(), iter.col());
        ++iter;
    }

    return *this;
}

template<class P_numtype, class P_structure>
ostream& operator<<(ostream& os, const Matrix<P_numtype, P_structure>& matrix)
{
    os << "[ ";
    for (int i=0; i < matrix.rows(); ++i)
    {
        for (int j=0; j < matrix.columns(); ++j)
        {
            os << setw(10) << matrix(i,j);
            if ((!((j+1)%7)) && (j < matrix.cols()-1))
                os << endl << "         ...";
        }
        if (i != matrix.rows() - 1)
            os << endl  << "  ";
    }
    os << " ]";
    return os;
}

BZ_NAMESPACE_END

#endif // BZ_MATRIX_CC
