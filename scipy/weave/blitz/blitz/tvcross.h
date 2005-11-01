/***************************************************************************
 * blitz/tvcross.h      Cross product of TinyVector<N,3>'s
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

#ifndef BZ_TVCROSS_H
#define BZ_TVCROSS_H

#ifndef BZ_TINYVEC_H
 #error <blitz/tvcross.h> must be included via <blitz/tinyvec.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * cross product.
 *
 * NEEDS_WORK: - cross product of two different vector types
 *             - cross product involving expressions
 */

template<typename T_numtype>
TinyVector<T_numtype,3> cross(const TinyVector<T_numtype,3>& x, 
    const TinyVector<T_numtype,3>& y)
{
    return TinyVector<T_numtype,3>(x[1]*y[2] - y[1]*x[2],
        y[0]*x[2] - x[0]*y[2], x[0]*y[1] - y[0]*x[1]);
}


BZ_NAMESPACE_END

#endif // BZ_TVCROSS_H
