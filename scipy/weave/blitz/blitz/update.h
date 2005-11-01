// -*- C++ -*-
/***************************************************************************
 * blitz/update.h      Declaration of the _bz_XXXX updater classes
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

#ifndef BZ_UPDATE_H
#define BZ_UPDATE_H

#include <blitz/blitz.h>

BZ_NAMESPACE(blitz)

class _bz_updater_base { };

#define BZ_DECL_UPDATER(name,op,symbol)                     \
  template<typename X, typename Y>                          \
  class name : public _bz_updater_base {                    \
  public:                                                   \
    static inline void update(X& restrict x, Y y)           \
    { x op y; }                                             \
    static void prettyPrint(BZ_STD_SCOPE(string) &str)      \
    { str += symbol; }                                      \
  }

template<typename X, typename Y>
class _bz_update : public _bz_updater_base {
  public:
    static inline void update(X& restrict x, Y y)
    { x = (X)y; }

    static void prettyPrint(BZ_STD_SCOPE(string) &str)
    { str += "="; }
};

BZ_DECL_UPDATER(_bz_plus_update, +=, "+=");
BZ_DECL_UPDATER(_bz_minus_update, -=, "-=");
BZ_DECL_UPDATER(_bz_multiply_update, *=, "*=");
BZ_DECL_UPDATER(_bz_divide_update, /=, "/=");
BZ_DECL_UPDATER(_bz_mod_update, %=, "%=");
BZ_DECL_UPDATER(_bz_xor_update, ^=, "^=");
BZ_DECL_UPDATER(_bz_bitand_update, &=, "&=");
BZ_DECL_UPDATER(_bz_bitor_update, |=, "|=");
BZ_DECL_UPDATER(_bz_shiftl_update, <<=, "<<=");
BZ_DECL_UPDATER(_bz_shiftr_update, >>=, ">>=");

BZ_NAMESPACE_END

#endif // BZ_UPDATE_H

