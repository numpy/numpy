/***************************************************************************
 * blitz/blitz.h      Includes all the important header files
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
 * Revision 1.9  2002/07/19 20:40:32  jcumming
 * Put ending semicolon into definition of BZ_MUTEX_* macros so that you
 * don't need to add a semicolon after invoking the macro.
 *
 * Revision 1.8  2001/02/15 13:13:30  tveldhui
 * Fixed problem with BZ_THREADSAFE macros
 *
 * Revision 1.7  2001/02/11 22:03:44  tveldhui
 * Fixed minor typo in blitz.h
 *
 * Revision 1.6  2001/02/04 22:36:41  tveldhui
 * Oops, was including <pthread.h> inside the blitz namespace.
 *
 * Revision 1.5  2001/02/04 16:32:28  tveldhui
 * Made memory block reference counting (optionally) threadsafe when
 * BZ_THREADSAFE is defined.  Currently uses pthread mutex.
 * When compiling with gcc -pthread, _REENTRANT automatically causes
 * BZ_THREADSAFE to be enabled.
 *
 * Revision 1.4  2001/01/26 18:30:50  tveldhui
 * More source code reorganization to reduce compile times.
 *
 * Revision 1.3  2001/01/24 22:51:50  tveldhui
 * Reorganized #include orders to avoid including the huge Vector e.t.
 * implementation when using Array.
 *
 * Revision 1.2  2001/01/24 20:22:49  tveldhui
 * Updated copyright date in headers.
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
 * Revision 1.4  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1996/11/11 17:29:13  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1996/04/14  21:13:00  todd
 * Initial revision
 *
 */

#ifndef BZ_BLITZ_H
#define BZ_BLITZ_H

/*
 * These symbols allow use of the IEEE and System V math libraries
 * (libm.a and libmsaa.a) on some platforms.
 */

#ifdef BZ_ENABLE_XOPEN_SOURCE
 #ifndef _ALL_SOURCE
  #define _ALL_SOURCE
 #endif
 #ifndef _XOPEN_SOURCE
  #define _XOPEN_SOURCE
 #endif
 #ifndef _XOPEN_SOURCE_EXTENDED
  #define _XOPEN_SOURCE_EXTENDED 1
 #endif
#endif

#include <blitz/compiler.h>          // Compiler-specific directives
#include <blitz/tuning.h>            // Performance tuning
#include <blitz/tau.h>               // Profiling

#include <string>
#include <stdio.h>                   // sprintf, etc.

#ifdef BZ_HAVE_STD
  #include <iostream>
  #include <iomanip>
#else
  #include <iostream.h>
  #include <iomanip.h>
#endif

#ifdef BZ_MATH_FN_IN_NAMESPACE_STD 
  #include <cmath>
#endif

#include <math.h>

#ifdef BZ_HAVE_COMPLEX
  #include <complex>
#endif

#define BZ_THROW                     // Needed in <blitz/numinquire.h>

BZ_NAMESPACE(blitz)

#ifdef BZ_HAVE_STD
 BZ_USING_NAMESPACE(std)
#endif

#ifdef BZ_GENERATE_GLOBAL_INSTANCES
 #define _bz_global
 #define BZ_GLOBAL_INIT(X)   =X
#else
 #define _bz_global extern
 #define BZ_GLOBAL_INIT(X) 
#endif

BZ_NAMESPACE_END

/*
 * Thread safety issues.
 * Compiling with -pthread under gcc, or -mt under solaris,
 * should automatically turn on BZ_THREADSAFE.
 */
#ifdef _REENTRANT
 #ifndef BZ_THREADSAFE
  #define BZ_THREADSAFE
 #endif
#endif

/*
 * Which mutex implementation should be used for synchronizing
 * reference counts.   Currently only one option -- pthreads.
 */
#ifdef BZ_THREADSAFE
#define BZ_THREADSAFE_USE_PTHREADS
#endif

#ifdef BZ_THREADSAFE_USE_PTHREADS
 #include <pthread.h>

 #define BZ_MUTEX_DECLARE(name)   pthread_mutex_t name;
 #define BZ_MUTEX_INIT(name)      pthread_mutex_init(&name,NULL);
 #define BZ_MUTEX_LOCK(name)      pthread_mutex_lock(&name);
 #define BZ_MUTEX_UNLOCK(name)    pthread_mutex_unlock(&name);
 #define BZ_MUTEX_DESTROY(name)   pthread_mutex_destroy(&name);
#else
 #define BZ_MUTEX_DECLARE(name)
 #define BZ_MUTEX_INIT(name)
 #define BZ_MUTEX_LOCK(name)
 #define BZ_MUTEX_UNLOCK(name)
 #define BZ_MUTEX_DESTROY(name)
#endif

#include <blitz/bzdebug.h>           // Debugging macros

#endif // BZ_BLITZ_H
