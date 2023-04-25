#ifndef _NPY_UMATH_ESTRIN_H_
#define _NPY_UMATH_ESTRIN_H_

/*
 * Helper macros for Estrin polynomial evaluation.
 */

#define simd_estrin_1_f32(x, c, i) \
    npyv_muladd_f32(x, npyv_setall_f32(c[1 + i]), npyv_setall_f32(c[i]))
#define simd_estrin_2_f32(x, x2, c, i) \
    npyv_muladd_f32(x2, npyv_setall_f32(c[2 + i]), simd_estrin_1_f32(x, c, i))
#define simd_estrin_3_f32(x, x2, c, i)                  \
    npyv_muladd_f32(x2, simd_estrin_1_f32(x, c, 2 + i), \
                    simd_estrin_1_f32(x, c, i))
#define simd_estrin_4_f32(x, x2, x4, c, i)         \
    npyv_muladd_f32(x4, npyv_setall_f32(c[4 + i]), \
                    simd_estrin_3_f32(x, x2, c, i))
#define simd_estrin_5_f32(x, x2, x4, c, i)              \
    npyv_muladd_f32(x4, simd_estrin_1_f32(x, c, 4 + i), \
                    simd_estrin_3_f32(x, x2, c, i))

#define simd_estrin_1_f64(x, poly, i) \
    npyv_muladd_f64(x, npyv_setall_f64(poly[1 + i]), npyv_setall_f64(poly[i]))
#define simd_estrin_2_f64(x, x2, poly, i)             \
    npyv_muladd_f64(x2, npyv_setall_f64(poly[2 + i]), \
                    simd_estrin_1_f64(x, c, i))
#define simd_estrin_3_f64(x, x2, poly, i)                  \
    npyv_muladd_f64(x2, simd_estrin_1_f64(x, poly, 2 + i), \
                    simd_estrin_1_f64(x, poly, i))
#define simd_estrin_4_f64(x, x2, x4, poly, i)         \
    npyv_muladd_f64(x4, npyv_setall_f64(poly[4 + i]), \
                    simd_estrin_3_f64(x, x2, poly, i))
#define simd_estrin_5_f64(x, x2, x4, poly, i)              \
    npyv_muladd_f64(x4, simd_estrin_1_f64(x, poly, 4 + i), \
                    simd_estrin_3_f64(x, x2, poly, i))
#define simd_estrin_6_f64(x, x2, x4, poly, i)                  \
    npyv_muladd_f64(x4, simd_estrin_2_f64(x, x2, poly, 4 + i), \
                    simd_estrin_3_f64(x, x2, poly, i))
#define simd_estrin_7_f64(x, x2, x4, poly, i)                  \
    npyv_muladd_f64(x4, simd_estrin_3_f64(x, x2, poly, 4 + i), \
                    simd_estrin_3_f64(x, x2, poly, i))

#endif
