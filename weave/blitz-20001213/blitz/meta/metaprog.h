/***************************************************************************
 * blitz/meta/metaprog.h   Useful metaprogram declarations
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
 * Revision 1.1  2002/01/03 19:50:38  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:13  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:08:44  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_META_METAPROG_H
#define BZ_META_METAPROG_H

BZ_NAMESPACE(blitz)

// Null Operand

class _bz_meta_nullOperand {
public:
    _bz_meta_nullOperand() { }
};

template<class T> inline T operator+(const T& a, _bz_meta_nullOperand)
{ return a; }
template<class T> inline T operator*(const T& a, _bz_meta_nullOperand)
{ return a; }


// MetaMax

template<int N1, int N2>
class _bz_meta_max {
public:
    enum { max = (N1 > N2) ? N1 : N2 };
};

// MetaMin

template<int N1, int N2>
class _bz_meta_min {
public:
    enum { min = (N1 < N2) ? N1 : N2 };
};

BZ_NAMESPACE_END 

#endif // BZ_META_METAPROG_H
