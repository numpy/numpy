/***************************************************************************
 * blitz/tuning.h      Platform-specific code tuning
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

#ifndef BZ_TUNING_H
#define BZ_TUNING_H

// These estimates should be conservative (i.e. underestimate the
// cache sizes).
#define BZ_L1_CACHE_ESTIMATED_SIZE    8192
#define BZ_L2_CACHE_ESTIMATED_SIZE    65536


#undef  BZ_PARTIAL_LOOP_UNROLL
#define BZ_PASS_EXPR_BY_VALUE
#undef  BZ_PTR_INC_FASTER_THAN_INDIRECTION
#define BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
#undef  BZ_KCC_COPY_PROPAGATION_KLUDGE
#undef  BZ_ALTERNATE_FORWARD_BACKWARD_TRAVERSALS
#undef  BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
#define BZ_INLINE_GROUP1
#define BZ_INLINE_GROUP2
#define BZ_COLLAPSE_LOOPS
#define BZ_USE_FAST_READ_ARRAY_EXPR
#define BZ_ARRAY_EXPR_USE_COMMON_STRIDE
#undef  BZ_ARRAY_SPACE_FILLING_TRAVERSAL
#undef  BZ_ARRAY_FAST_TRAVERSAL_UNROLL
#undef  BZ_ARRAY_STACK_TRAVERSAL_CSE_AND_ANTIALIAS
#undef  BZ_ARRAY_STACK_TRAVERSAL_UNROLL
#define BZ_ARRAY_2D_STENCIL_TILING
#define BZ_ARRAY_2D_STENCIL_TILE_SIZE       128
#undef  BZ_INTERLACE_ARRAYS
#undef  BZ_ALIGN_BLOCKS_ON_CACHELINE_BOUNDARY
#define BZ_FAST_COMPILE


#ifndef BZ_DISABLE_NEW_ET
 #define BZ_NEW_EXPRESSION_TEMPLATES
#endif

#ifdef BZ_FAST_COMPILE
#define BZ_ETPARMS_CONSTREF
#define BZ_NO_INLINE_ET
#endif

/*
 * Platform-specific tuning
 */

#ifdef _CRAYT3E
 // The backend compiler on the T3E does a better job of
 // loop unrolling.
 #undef BZ_PARTIAL_LOOP_UNROLL
 #undef BZ_ARRAY_FAST_TRAVERSAL_UNROLL
 #undef BZ_ARRAY_STACK_TRAVERSAL_UNROLL
#endif

#ifdef __GNUC__
 // The egcs compiler does a good job of loop unrolling, if
 // -funroll-loops is used.
 #undef BZ_PARTIAL_LOOP_UNROLL
 #undef BZ_ARRAY_FAST_TRAVERSAL_UNROLL
 #undef BZ_ARRAY_STACK_TRAVERSAL_UNROLL
#endif

#ifdef  BZ_DISABLE_KCC_COPY_PROPAGATION_KLUDGE
 #undef BZ_KCC_COPY_PROPAGATION_KLUDGE
#endif

#ifdef  BZ_INLINE_GROUP1
 #define _bz_inline1 inline
#else
 #define _bz_inline1
#endif

#ifdef  BZ_INLINE_GROUP2
 #define _bz_inline2 inline
#else
 #define _bz_inline2
#endif

#ifdef  BZ_NO_INLINE_ET
 #define _bz_inline_et 
#else
 #define _bz_inline_et inline
#endif

#ifdef  BZ_ETPARMS_CONSTREF
 #define BZ_ETPARM(X) const X&
#else
 #define BZ_ETPARM(X) X
#endif

#ifdef __DECCXX
 // The DEC cxx compiler has problems with loop unrolling
 // because of aliasing.  Loop unrolling and anti-aliasing
 // is done by Blitz++.

  #define  BZ_PARTIAL_LOOP_UNROLL
  #define  BZ_ARRAY_STACK_TRAVERSAL_CSE_AND_ANTIALIAS
  #define  BZ_ARRAY_STACK_TRAVERSAL_UNROLL
#endif

/*
 * BZ_NO_PROPAGATE(X) prevents the compiler from performing
 * copy propagation on a variable.  This is used for loop
 * unrolling to prevent KAI C++ from rearranging the
 * ordering of memory accesses.
 */

#define BZ_NO_PROPAGATE(X)   X

#ifdef __KCC
#ifdef BZ_USE_NO_PROPAGATE
    extern "C" int __kai_apply(const char*, ...);

    #undef  BZ_NO_PROPAGATE(X)
    #define BZ_NO_PROPAGATE(X)  __kai_apply("(%a)",&X)
#endif
#endif

#endif // BZ_TUNING_H
