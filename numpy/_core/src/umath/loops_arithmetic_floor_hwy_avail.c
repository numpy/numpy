/**
 * Availability check functions for Highway SIMD floor divide.
 *
 * These are compiled once (non-dispatched) as part of the _multiarray_umath
 * module and return whether Highway SIMD is available for a given element size.
 * When NPY_HAVE_HIGHWAY is not defined (Highway disabled or unavailable),
 * always returns 0 so the .c.src template falls through to scalar paths.
 */

#include "loops_arithmetic_floor_hwy.h"

int npy_highway_floor_divide_available(int element_size)
{
#if defined(NPY_HAVE_HIGHWAY)
    return (element_size == 1 || element_size == 2 || element_size == 4) ? 1 : 0;
#else
    (void)element_size;
    return 0;
#endif
}

int npy_highway_floor_divide_unsigned_available(int element_size)
{
#if defined(NPY_HAVE_HIGHWAY)
    return (element_size == 1 || element_size == 2 || element_size == 4) ? 1 : 0;
#else
    (void)element_size;
    return 0;
#endif
}

int npy_highway_floor_divide_scalar_available(int element_size)
{
#if defined(NPY_HAVE_HIGHWAY)
    return (element_size == 1 || element_size == 2 || element_size == 4 || element_size == 8) ? 1 : 0;
#else
    (void)element_size;
    return 0;
#endif
}

int npy_highway_floor_divide_scalar_unsigned_available(int element_size)
{
#if defined(NPY_HAVE_HIGHWAY)
    return (element_size == 4) ? 1 : 0;
#else
    (void)element_size;
    return 0;
#endif
}
