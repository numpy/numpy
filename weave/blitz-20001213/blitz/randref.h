/***************************************************************************
 * blitz/randref.h      Random number generators, expression templates
 *                      wrapper
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
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.3  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_RANDREF_H
#define BZ_RANDREF_H

#ifndef BZ_RANDOM_H
 #error <blitz/randref.h> must be included via <blitz/random.h>
#endif // BZ_RANDOM_H

BZ_NAMESPACE(blitz)

template<class P_distribution>
class _bz_VecExprRandom {

public:
    typedef _bz_typename Random<P_distribution>::T_numtype T_numtype;

    _bz_VecExprRandom(Random<P_distribution>& random)
        : random_(random)
    { }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprRandom(_bz_VecExprRandom<P_distribution>& x)
        : random_(x.random_)
    { }
#endif

    T_numtype operator[](unsigned) const
    { return random_.random(); }

    T_numtype operator()(unsigned) const
    { return random_.random(); }

    unsigned length(unsigned recommendedLength) const
    { return recommendedLength; }

    unsigned _bz_suggestLength() const
    { return 0; }

    _bz_bool _bz_hasFastAccess() const
    { return 1; }

    T_numtype _bz_fastAccess(unsigned) const
    { return random_.random(); }

private:
    _bz_VecExprRandom() { }

    Random<P_distribution>& random_;
};

BZ_NAMESPACE_END

#endif // BZ_RANDREF_H

