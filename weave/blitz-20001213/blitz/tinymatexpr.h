/***************************************************************************
 * blitz/tinymatexpr.h   Tiny Matrix Expressions
 *
 * $Id$
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 ***************************************************************************
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
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

#ifndef BZ_TINYMATEXPR_H
#define BZ_TINYMATEXPR_H

#ifndef BZ_TINYMAT_H
 #error <blitz/tinymatexpr.h> must be included via <blitz/tinymat.h>
#endif

BZ_NAMESPACE(blitz)

template<class T_expr>
class _bz_tinyMatExpr {
public:
    typedef _bz_typename T_expr::T_numtype T_numtype;

    enum {
        rows = T_expr::rows,
        columns = T_expr::columns
    };

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

