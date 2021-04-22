#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SSE_MATH_H
#define _NPY_SIMD_SSE_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 _mm_sqrt_ps
#define npyv_sqrt_f64 _mm_sqrt_pd

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return _mm_div_ps(_mm_set1_ps(1.0f), a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return _mm_div_pd(_mm_set1_pd(1.0), a); }

// Absolute
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{
    return _mm_and_ps(
        a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))
    );
}
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{
    return _mm_and_pd(
        a, _mm_castsi128_pd(npyv_setall_s64(0x7fffffffffffffffLL))
    );
}

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return _mm_mul_ps(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return _mm_mul_pd(a, a); }

#endif
