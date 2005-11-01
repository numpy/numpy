// -*- C++ -*-
/***************************************************************************
 * blitz/tinymatexpr.h   Tiny Matrix Expressions
 *
 * $Id$
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
 ***************************************************************************/

#ifndef BZ_TINYMATEXPR_H
#define BZ_TINYMATEXPR_H

#ifndef BZ_TINYMAT_H
 #error <blitz/tinymatexpr.h> must be included via <blitz/tinymat.h>
#endif

BZ_NAMESPACE(blitz)

template<typename T_expr>
class _bz_tinyMatExpr {
public:
    typedef _bz_typename T_expr::T_numtype T_numtype;

    static const int
        rows = T_expr::rows,
        columns = T_expr::columns;

    _bz_tinyMatExpr(T_expr expr)
        : expr_(expr)
    { }

    _bz_tinyMatExpr(const _bz_tinyMatExpr<T_expr>& x)
        : expr_(x.expr_)
    { }

    T_numtype operator()(int i, int j) const
    { return expr_(i,j); }

protected:
    T_expr expr_;
};

BZ_NAMESPACE_END

#endif // BZ_TINYMATEXPR_H

