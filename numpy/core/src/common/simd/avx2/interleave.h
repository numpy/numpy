#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#include "memory.h" // load*
#include "reorder.h" // npyv_zip*

#ifndef _NPY_SIMD_AVX2_INTERLEAVE_H
#define _NPY_SIMD_AVX2_INTERLEAVE_H

NPY_FINLINE npyv_f64x2 npyv_load_deinterleave_f64x2(const double *ptr)
{
    npyv_f64 ab0 = npyv_load_f64(ptr);
    npyv_f64 ab1 = npyv_load_f64(ptr + npyv_nlanes_f64);
    npyv_f64x2 ab, swap_halves = npyv_combine_f64(ab0, ab1);
    ab.val[0] = _mm256_unpacklo_pd(swap_halves.val[0], swap_halves.val[1]);
    ab.val[1] = _mm256_unpackhi_pd(swap_halves.val[0], swap_halves.val[1]);
    return ab;
}

NPY_FINLINE void npyv_store_interleave_f64x2(double *ptr, npyv_f64x2 a)
{
    npyv_f64x2 zip = npyv_zip_f64(a.val[0], a.val[1]);
    npyv_store_f64(ptr, zip.val[0]);
    npyv_store_f64(ptr + npyv_nlanes_f64, zip.val[1]);
}
#endif // _NPY_SIMD_AVX2_INTERLEAVE_H
