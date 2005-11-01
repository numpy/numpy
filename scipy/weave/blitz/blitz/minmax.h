/***************************************************************************
 * blitz/minmax.h  Declaration of min and max functions
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

#ifndef BZ_MINMAX_H
#define BZ_MINMAX_H

#include <blitz/promote.h>

BZ_NAMESPACE(blitz)

/*
 * These functions are in their own namespace (blitz::minmax) to avoid
 * conflicts with the array reduction operations min and max.
 */

BZ_NAMESPACE(minmax)

template<typename T1, typename T2>
BZ_PROMOTE(T1,T2) min(const T1& a, const T2& b)
{
    typedef BZ_PROMOTE(T1,T2) T_promote;

    if (a <= b)
        return T_promote(a);
    else
        return T_promote(b);
}

template<typename T1, typename T2>
BZ_PROMOTE(T1,T2) max(const T1& a, const T2& b)
{
    typedef BZ_PROMOTE(T1,T2) T_promote;

    if (a >= b)
        return T_promote(a);
    else
        return T_promote(b);
}

BZ_NAMESPACE_END

BZ_NAMESPACE_END

#endif
