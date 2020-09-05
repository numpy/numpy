#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#include "memory.h" // load*
#include "reorder.h" // npyv_zip*

#ifndef _NPY_SIMD_VSX_INTERLEAVE_H
#define _NPY_SIMD_VSX_INTERLEAVE_H

NPY_FINLINE npyv_f64x2 npyv_load_deinterleave_f64x2(const double *ptr)
{
    npyv_f64 a = npyv_load_f64(ptr);
    npyv_f64 b = npyv_load_f64(ptr + npyv_nlanes_f64);
    return npyv_zip_f64(a, b);
}

NPY_FINLINE void npyv_store_interleave_f64x2(double *ptr, npyv_f64x2 a)
{
    npyv_f64x2 zip = npyv_zip_f64(a[0], a[1]);
    npyv_store_f64(ptr, zip[0]);
    npyv_store_f64(ptr + npyv_nlanes_f64, zip[1]);
}
#endif // _NPY_SIMD_VSX_INTERLEAVE_H
