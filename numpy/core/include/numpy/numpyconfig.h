#ifndef _NPY_NUMPYCONFIG_H_
#define _NPY_NUMPYCONFIG_H_

#include "_numpyconfig.h"

/* 
 * On Mac OS X, because there is only one configuration stage for all the archs
 * in universal builds, any macro which depends on the arch needs to be
 * harcoded
 */
#ifdef __APPLE__
	#undef NPY_SIZEOF_LONG
	#undef NPY_SIZEOF_PY_INTPTR_T

	#ifdef __LP64__
		#define NPY_SIZEOF_LONG 		8
		#define NPY_SIZEOF_PY_INTPTR_T 	8
	#else
		#define NPY_SIZEOF_LONG 		4
		#define NPY_SIZEOF_PY_INTPTR_T 	4
	#endif
#endif

#endif
