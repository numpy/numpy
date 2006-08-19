
/* This expects the following variables to be defined (besides
   the usual ones from pyconfig.h

   SIZEOF_LONG_DOUBLE -- sizeof(long double) or sizeof(double) if no
                         long double is present on platform.
   CHAR_BIT       --     number of bits in a char (usually 8)
                         (should be in limits.h)

*/

#ifndef Py_ARRAYOBJECT_H
#define Py_ARRAYOBJECT_H
#include "ndarrayobject.h"
#ifdef NPY_NO_PREFIX
#include "noprefix.h"
#endif


/* Add signal handling macros */

#define NPY_SIG_ON
#define NPY_SIG_OFF
#define NPY_SIG_CHECK


#endif
