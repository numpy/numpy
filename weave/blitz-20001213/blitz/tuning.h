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
 ***************************************************************************
 * $Log$
 * Revision 1.2  2002/09/12 07:04:04  eric
 * major rewrite of weave.
 *
 * 0.
 * The underlying library code is significantly re-factored and simpler. There used to be a xxx_spec.py and xxx_info.py file for every group of type conversion classes.  The spec file held the python code that handled the conversion and the info file had most of the C code templates that were generated.  This proved pretty confusing in practice, so the two files have mostly been merged into the spec file.
 *
 * Also, there was quite a bit of code duplication running around.  The re-factoring was able to trim the standard conversion code base (excluding blitz and accelerate stuff) by about 40%.  This should be a huge maintainability and extensibility win.
 *
 * 1.
 * With multiple months of using Numeric arrays, I've found some of weave's "magic variable" names unwieldy and want to change them.  The following are the old declarations for an array x of Float32 type:
 *
 *         PyArrayObject* x = convert_to_numpy(...);
 *         float* x_data = (float*) x->data;
 *         int*   _Nx = x->dimensions;
 *         int*   _Sx = x->strides;
 *         int    _Dx = x->nd;
 *
 * The new declaration looks like this:
 *
 *         PyArrayObject* x_array = convert_to_numpy(...);
 *         float* x = (float*) x->data;
 *         int*   Nx = x->dimensions;
 *         int*   Sx = x->strides;
 *         int    Dx = x->nd;
 *
 * This is obviously not backward compatible, and will break some code (including a lot of mine).  It also makes inline() code more readable and natural to write.
 *
 * 2.
 * I've switched from CXX to Gordon McMillan's SCXX for list, tuples, and dictionaries.  I like CXX pretty well, but its use of advanced C++ (templates, etc.) caused some portability problems.  The SCXX library is similar to CXX but doesn't use templates at all.  This, like (1) is not an
 * API compatible change and requires repairing existing code.
 *
 * I have also thought about boost python, but it also makes heavy use of templates.  Moving to SCXX gets rid of almost all template usage for the standard type converters which should help portability.  std::complex and std::string from the STL are the only templates left.  Of course blitz still uses templates in a major way so weave.blitz will continue to be hard on compilers.
 *
 * I've actually considered scrapping the C++ classes for list, tuples, and
 * dictionaries, and just fall back to the standard Python C API because the classes are waaay slower than the raw API in many cases.  They are also more convenient and less error prone in many cases, so I've decided to stick with them.  The PyObject variable will always be made available for variable "x" under the name "py_x" for more speedy operations.  You'll definitely want to use these for anything that needs to be speedy.
 *
 * 3.
 * strings are converted to std::string now.  I found this to be the most useful type in for strings in my code.  Py::String was used previously.
 *
 * 4.
 * There are a number of reference count "errors" in some of the less tested conversion codes such as instance, module, etc.  I've cleaned most of these up.  I put errors in quotes here because I'm actually not positive that objects passed into "inline" really need reference counting applied to them.  The dictionaries passed in by inline() hold references to these objects so it doesn't seem that they could ever be garbage collected inadvertently.  Variables used by ext_tools, though, definitely need the reference counting done.  I don't think this is a major cost in speed, so it probably isn't worth getting rid of the ref count code.
 *
 * 5.
 * Unicode objects are now supported.  This was necessary to support rendering Unicode strings in the freetype wrappers for Chaco.
 *
 * 6.
 * blitz++ was upgraded to the latest CVS.  It compiles about twice as fast as the old blitz and looks like it supports a large number of compilers (though only gcc 2.95.3 is tested).  Compile times now take about 9 seconds on my 850 MHz PIII laptop.
 *
 * Revision 1.3  2001/01/26 18:30:50  tveldhui
 * More source code reorganization to reduce compile times.
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:10  tveldhui
 * Imported sources
 *
 * Revision 1.6  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.5  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.4  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1996/11/11 17:29:13  tveldhui
 * Initial revision
 *
 *
 */

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
