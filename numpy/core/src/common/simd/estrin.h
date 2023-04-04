/*
 * Helper macros for double-precision Estrin polynomial evaluation.
 */

#define npyv_estrin_1_f64(x, poly, i) npyv_muladd_f64(npyv_setall_f64(poly[1 + i]), npyv_setall_f64(poly[i]), x)
#define npyv_estrin_2_f64(x, x2, poly, i) npyv_muladd_f64(npyv_setall_f64(poly[2 + i]), npyv_estrin_1_f64(x, c, i), x2)
#define npyv_estrin_3_f64(x, x2, poly, i) \
    npyv_muladd_f64(npyv_estrin_1_f64(x, poly, 2 + i), npyv_estrin_1_f64(x, poly, i), x2)
#define NPYV_ESTRIN_4_(x, x2, x4, poly, i) \
    npyv_muladd_f64(npyv_setall_f64(poly[4 + i]), npyv_estrin_3_f64(x, x2, poly, i), x4)
#define npyv_estrin_5_f64(x, x2, x4, poly, i) \
    npyv_muladd_f64(npyv_estrin_1_f64(x, poly, 4 + i), npyv_estrin_3_f64(x, x2, poly, i), x4)
#define npyv_estrin_6_f64(x, x2, x4, poly, i) \
    npyv_muladd_f64(npyv_estrin_2_f64(x, x2, poly, 4 + i), npyv_estrin_3_f64(x, x2, poly, i), x4)
#define npyv_estrin_7_f64(x, x2, x4, poly, i) \
    npyv_muladd_f64(npyv_estrin_3_f64(x, x2, poly, 4 + i), npyv_estrin_3_f64(x, x2, poly, i), x4)
