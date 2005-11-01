/***************************************************************************
 * blitz/array/et.h  Include expression templates implementation for arrays
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
#ifndef BZ_ARRAY_ET_H
#define BZ_ARRAY_ET_H

#ifdef BZ_NEW_EXPRESSION_TEMPLATES
 #include <blitz/array/newet.h>     // Expression templates
#else
 #include <blitz/array/bops.cc>     // Expression templates, two operands
 #include <blitz/array/uops.cc>     // Expression templates, math functions
 #include <blitz/array/misc.cc>     // Expression templates, miscellaneous
#endif

#endif
