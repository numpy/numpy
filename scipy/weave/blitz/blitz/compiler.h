/***************************************************************************
 * blitz/compiler.h      Compiler specific directives and kludges
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


#ifndef BZ_COMPILER_H
#define BZ_COMPILER_H

// The file <blitz/bzconfig.h> is used to select a compiler-specific
// config.h file that is generated automatically by configure.

#include <blitz/bzconfig.h>

/*
 * Define some kludges.
 */

#ifndef BZ_HAVE_TEMPLATES
    #error  In <blitz/config.h>: A working template implementation is required by Blitz++ (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_HAVE_MEMBER_TEMPLATES
  #error  In <blitz/config.h>: Your compiler does not support member templates.  (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_HAVE_FULL_SPECIALIZATION_SYNTAX
  #error In <blitz/config.h>: Your compiler does not support template<> full specialization syntax.  You may need to rerun the compiler/bzconfig script.
#endif

#ifndef BZ_HAVE_PARTIAL_ORDERING
  #error In <blitz/config.h>: Your compiler does not support partial ordering (you may need to rerun the compiler/bzconfig script)
#endif

#ifndef BZ_HAVE_PARTIAL_SPECIALIZATION
  #error In <blitz/config.h>: Your compiler does not support partial specialization (you may need to rerun the compiler/bzconfig script)
#endif

#ifdef BZ_HAVE_NAMESPACES
    #define BZ_NAMESPACE(X)        namespace X {
    #define BZ_NAMESPACE_END       }
    #define BZ_USING_NAMESPACE(X)  using namespace X;
#else
    #define BZ_NAMESPACE(X)
    #define BZ_NAMESPACE_END
    #define BZ_USING_NAMESPACE(X)
#endif

#ifdef BZ_HAVE_TEMPLATE_QUALIFIED_RETURN_TYPE
  #define BZ_USE_NUMTRAIT
#endif

#ifdef BZ_HAVE_DEFAULT_TEMPLATE_PARAMETERS
    #define BZ_TEMPLATE_DEFAULT(X)   = X
#else
    #define BZ_TEMPLATE_DEFAULT
#endif

#ifndef BZ_HAVE_EXPLICIT
    #define explicit   
#endif

#ifdef BZ_HAVE_TYPENAME
    #define _bz_typename     typename
#else
    #define _bz_typename
#endif

#ifndef BZ_HAVE_MUTABLE
    #define mutable
#endif

#ifdef BZ_DISABLE_RESTRICT
 #undef BZ_HAVE_NCEG_RESTRICT
#endif

#ifndef BZ_HAVE_NCEG_RESTRICT
    #if defined(BZ_HAVE_NCEG_RESTRICT_EGCS)
        #define restrict     __restrict__
    #else
        #define restrict
    #endif
#endif

#if !defined(BZ_HAVE_BOOL) && !defined(BZ_NO_BOOL_KLUDGE)
    #define bool    int
    #define true    1
    #define false   0
#endif

#ifdef BZ_HAVE_ENUM_COMPUTATIONS_WITH_CAST
    #define BZ_ENUM_CAST(X)   (int)X
#elif defined(BZ_HAVE_ENUM_COMPUTATIONS)
    #define BZ_ENUM_CAST(X)   X
#else
    #error In <blitz/config.h>: Your compiler does not support enum computations.  You may have to rerun compiler/bzconfig.
#endif

#if defined(BZ_MATH_FN_IN_NAMESPACE_STD)
  #define BZ_MATHFN_SCOPE(x) std::x
#elif defined(BZ_HAVE_NAMESPACES)
  #define BZ_MATHFN_SCOPE(x) ::x
#else
  #define BZ_MATHFN_SCOPE(x) x
#endif

#if defined(BZ_HAVE_COMPLEX_MATH_IN_NAMESPACE_STD)
  #define BZ_CMATHFN_SCOPE(x) std::x
#elif defined(BZ_HAVE_NAMESPACES)
  #define BZ_CMATHFN_SCOPE(x) ::x
#else
  #define BZ_CMATHFN_SCOPE(x) x
#endif

#if defined(BZ_HAVE_NAMESPACES)
  #define BZ_IEEEMATHFN_SCOPE(x) ::x
#else
  #define BZ_IEEEMATHFN_SCOPE(x) x
#endif

#if defined(BZ_HAVE_NAMESPACES)
  #define BZ_BLITZ_SCOPE(x) blitz::x
#else
  #define BZ_BLITZ_SCOPE(x) ::x
#endif

#if defined(BZ_HAVE_NAMESPACES) && defined(BZ_HAVE_STD)
  #define BZ_STD_SCOPE(x) std::x
#else
  #define BZ_STD_SCOPE(x) ::x
#endif

#endif // BZ_COMPILER_H

