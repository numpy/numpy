/* Select the compiler-specific config.h header file */
#ifndef BZCONFIG_H
#define BZCONFIG_H

#if defined(__APPLE)
/* IBM xlc compiler for Darwin */
#include <blitz/apple/bzconfig.h>

#elif defined(__ICC)
/* Intel icc compiler */
#include <blitz/intel/bzconfig.h>

#elif defined(_MSC_VER)
/* Microsoft VS.NET compiler */
#include <blitz/ms/bzconfig.h>

#elif defined(__IBM)
/* IBM xlC compiler */
#include <blitz/ibm/bzconfig.h>

#elif defined(__DECCXX)
/* Compaq cxx compiler */
#include <blitz/compaq/bzconfig.h>

#elif defined(__HP_aCC)
/* HP aCC compiler */
#include <blitz/hp/bzconfig.h>

#elif defined(_SGI_COMPILER_VERSION)
/* SGI CC compiler */
#include <blitz/sgi/bzconfig.h>

#elif defined(__SUNPRO_CC)
/* SunPRO CC compiler */
#include <blitz/sun/bzconfig.h>

#elif defined(__GNUC__)
/* GNU gcc compiler */
#include <blitz/gnu/bzconfig.h>

#elif defined(__PGI)
/* PGI pgCC compiler */
#include <blitz/pgi/bzconfig.h>

#elif defined(__KCC)
/* KAI KCC compiler */
#include <blitz/kai/bzconfig.h>

#elif defined(__FUJITSU)
/* Fujitsu FCC compiler */
#include <blitz/fujitsu/bzconfig.h>

#elif defined(__PATHSCALE)
/* Pathscale pathCC compiler */
#include <blitz/pathscale/bzconfig.h>

/* Add other compilers here */

#else
#error Unknown compiler
#endif

#endif /* BZCONFIG_H */
