#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_INTERLEAVE_H
#define _NPY_SIMD_NEON_INTERLEAVE_H

#if NPY_SIMD_F64
NPY_FINLINE npyv_f64x2 npyv_load_deinterleave_f64x2(const double *ptr)
{ return vld2q_f64(ptr); }

NPY_FINLINE void npyv_store_interleave_f64x2(double *ptr, npyv_f64x2 a)
{ vst2q_f64(ptr, a); }
#endif // NPY_SIMD_F64

#endif
