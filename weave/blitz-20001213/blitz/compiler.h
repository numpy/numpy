/***************************************************************************
 * blitz/compiler.h      Compiler specific directives and kludges
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
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.7  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.6  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.5  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.4  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1996/11/11 17:29:13  tveldhui
 * Periodic RCS update
 *
 *
 */


#ifndef BZ_COMPILER_H
#define BZ_COMPILER_H

// The file <blitz/config.h> is generated automatically by the
// script 'bzconfig', located in the 'compiler' directory.

#include <blitz/config.h>

/*
 * Define some kludges.
 */

#ifndef BZ_TEMPLATES
    #error  In <blitz/config.h>: A working template implementation is required by Blitz++ (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_MEMBER_TEMPLATES
  #error  In <blitz/config.h>: Your compiler does not support member templates.  (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_FULL_SPECIALIZATION_SYNTAX
  #error In <blitz/config.h>: Your compiler does not support template<> full specialization syntax.  You may need to rerun the compiler/bzconfig script.
#endif

#ifndef BZ_PARTIAL_ORDERING
  #error In <blitz/config.h>: Your compiler does not support partial ordering (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_PARTIAL_SPECIALIZATION
  #error In <blitz/config.h>: Your compiler does not support partial specialization (you may need to rerun the compiler/bzconfig script)
#endif

#ifdef BZ_NAMESPACES
    #define BZ_NAMESPACE(X)        namespace X {
    #define BZ_NAMESPACE_END       }
    #define BZ_USING_NAMESPACE(X)  using namespace X;
#else
    #define BZ_NAMESPACE(X)
    #define BZ_NAMESPACE_END
    #define BZ_USING_NAMESPACE(X)
#endif

#ifdef BZ_TEMPLATE_QUALIFIED_RETURN_TYPE
  #define BZ_USE_NUMTRAIT
#endif

#ifdef BZ_DEFAULT_TEMPLATE_PARAMETERS
    #define BZ_TEMPLATE_DEFAULT(X)   = X
#else
    #define BZ_TEMPLATE_DEFAULT
#endif

#ifdef BZ_EXPLICIT
    #define _bz_explicit     explicit
#else
    #define _bz_explicit   
#endif

#ifdef BZ_TYPENAME
    #define _bz_typename     typename
#else
    #define _bz_typename
#endif

#ifdef BZ_MUTABLE
    #define _bz_mutable      mutable
#else
    #define _bz_mutable
#endif

#ifdef BZ_DISABLE_RESTRICT
 #undef BZ_NCEG_RESTRICT
#endif

#ifdef BZ_NCEG_RESTRICT
    #define _bz_restrict     restrict
#elif defined(BZ_NCEG_RESTRICT_EGCS)
    #define _bz_restrict     __restrict__
#else
    #define _bz_restrict
#endif

#ifdef BZ_BOOL
    #define _bz_bool         bool
    #define _bz_true         true
    #define _bz_false        false
#else
    #define _bz_bool         int
    #define _bz_true         1
    #define _bz_false        0
#endif

#ifdef BZ_ENUM_COMPUTATIONS_WITH_CAST
    #define BZ_ENUM_CAST(X)   (int)X
#elif defined(BZ_ENUM_COMPUTATIONS)
    #define BZ_ENUM_CAST(X)   X
#else
    #error In <blitz/config.h>: Your compiler does not support enum computations.  You may have to rerun compiler/bzconfig.
#endif

#if defined(BZ_MATH_FN_IN_NAMESPACE_STD)
  #define BZ_MATHFN_SCOPE(x) std::x
#elif defined(BZ_NAMESPACES)
  #define BZ_MATHFN_SCOPE(x) ::x
#else
  #define BZ_MATHFN_SCOPE(x) x
#endif

#if defined(BZ_COMPLEX_MATH_IN_NAMESPACE_STD)
  #define BZ_CMATHFN_SCOPE(x) std::x
#elif defined(BZ_NAMESPACES)
  #define BZ_CMATHFN_SCOPE(x) ::x
#else
  #define BZ_CMATHFN_SCOPE(x) x
#endif

#if defined(BZ_NAMESPACES)
  #define BZ_IEEEMATHFN_SCOPE(x) ::x
#else
  #define BZ_IEEEMATHFN_SCOPE(x) x
#endif

#if defined(BZ_NAMESPACES)
  #define BZ_BLITZ_SCOPE(x) blitz::x
#else
  #define BZ_BLITZ_SCOPE(x) ::x
#endif

#if defined(BZ_NAMESPACES)
  #define BZ_STD_SCOPE(x) std::x
#else
  #define BZ_STD_SCOPE(x) ::x
#endif

#endif // BZ_COMPILER_H

