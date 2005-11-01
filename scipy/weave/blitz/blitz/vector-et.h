/***************************************************************************
 * blitz/vector-et.h      Vector<P_numtype> class + expression templates
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

#ifndef BZ_VECTOR_ET_H
#define BZ_VECTOR_ET_H

#include <blitz/vector.h>

// These are compile-time expensive things not included
// by <blitz/vector.h>, but needed if we want vector expressions.

#include <blitz/vecbops.cc>         // Operators with two operands
#include <blitz/vecuops.cc>         // Functions with one argument
#include <blitz/vecbfn.cc>          // Functions with two arguments

#endif  // BZ_VECTOR_ET_H

