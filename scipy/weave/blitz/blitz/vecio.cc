/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_VECIO_CC
#define BZ_VECIO_CC

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

BZ_NAMESPACE(blitz)

// This version of operator<< is provided as a temporary measure
// only.  It will be revised in a future release.
// NEEDS_WORK

template<typename P_numtype>
ostream& operator<<(ostream& os, const Vector<P_numtype>& x)
{
    os << "[ ";
    for (int i=0; i < x.length(); ++i)
    {
        os << setw(10) << x[i];
        if (!((i+1)%7))
            os << endl << "  ";
    }
    os << " ]";
    return os;
}

template<typename P_expr>
ostream& operator<<(ostream& os, _bz_VecExpr<P_expr> expr)
{
    Vector<_bz_typename P_expr::T_numtype> result(expr);
    os << result;
    return os;
}

BZ_NAMESPACE_END

#endif // BZ_VECIO_CC
