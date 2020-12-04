#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VSX_MATH_H
#define _NPY_SIMD_VSX_MATH_H
/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 vec_sqrt
#define npyv_sqrt_f64 vec_sqrt

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    return vec_div(one, a);
}
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{
    const npyv_f64 one = npyv_setall_f64(1.0);
    return vec_div(one, a);
}

// Absolute
#define npyv_abs_f32 vec_abs
#define npyv_abs_f64 vec_abs

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return vec_mul(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return vec_mul(a, a); }

#endif // _NPY_SIMD_VSX_MATH_H
