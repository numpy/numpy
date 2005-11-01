/***************************************************************************
 * blitz/extremum.h      Declaration of the Extremum<T_numtype, T_index> class
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

#ifndef BZ_EXTREMUM_H
#define BZ_EXTREMUM_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

BZ_NAMESPACE(blitz)

// The Extremum class is used for returning extreme values and their
// locations in a numeric container.  It's a simple 2-tuple, with the
// first element being the extreme value, and the send its location.
// An object of type Extremum can be automatically converted to
// the numeric type via operator T_numtype().
template<typename P_numtype, typename P_index>
class Extremum {
public:
    typedef P_numtype T_numtype;
    typedef P_index   T_index;

    Extremum(T_numtype value, T_index index)
        : value_(value), index_(index)
    { }

    T_numtype value() const
    { return value_; }

    T_index index() const
    { return index_; }

    void setValue(T_numtype value)
    { value_ = value; }

    void setIndex(T_index index)
    { index_ = index; }

    operator T_numtype() const
    { return value_; }

protected:
    T_numtype value_;
    T_index index_;
};

BZ_NAMESPACE_END

#endif // BZ_EXTREMUM_H

