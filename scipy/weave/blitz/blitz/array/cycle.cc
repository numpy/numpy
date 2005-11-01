/***************************************************************************
 * blitz/array/cycle.cc  Cycle arrays for time-stepping of PDEs.
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
#ifndef BZ_ARRAYCYCLE_CC
#define BZ_ARRAYCYCLE_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/cycle.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(tmp);
}

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b,
    Array<T_numtype, N_rank>& c)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(c);
    c.reference(tmp);
}

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b,
    Array<T_numtype, N_rank>& c, Array<T_numtype, N_rank>& d)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(c);
    c.reference(d);
    d.reference(tmp);
}

template<typename T_numtype, int N_rank>
void cycleArrays(Array<T_numtype, N_rank>& a, Array<T_numtype, N_rank>& b,
    Array<T_numtype, N_rank>& c, Array<T_numtype, N_rank>& d,
    Array<T_numtype, N_rank>& e)
{
    Array<T_numtype, N_rank> tmp(a);
    a.reference(b);
    b.reference(c);
    c.reference(d);
    d.reference(e);
    e.reference(tmp);
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYCYCLE_CC
