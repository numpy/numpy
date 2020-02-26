#ifndef __NPY_MATH_C99_H_
#define __NPY_MATH_C99_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#ifdef __SUNPRO_CC
#include <sunmath.h>
#endif
#ifdef HAVE_NPY_CONFIG_H
#include <npy_config.h>
#endif
#include <numpy/npy_common.h>

/* By adding static inline specifiers to npy_math function definitions when
   appropriate, compiler is given the opportunity to optimize */
#if NPY_INLINE_MATH
#define NPY_INPLACE NPY_INLINE static
#else
#define NPY_INPLACE
#endif


/*
 * NAN and INFINITY like macros (same behavior as glibc for NAN, same as C99
 * for INFINITY)
 *
 * XXX: I should test whether INFINITY and NAN are available on the platform
 */
NPY_INLINE static float __npy_inff(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7f800000UL};
    return __bint.__f;
}

NPY_INLINE static float __npy_nanf(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x7fc00000UL};
    return __bint.__f;
}

NPY_INLINE static float __npy_pzerof(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x00000000UL};
    return __bint.__f;
}

NPY_INLINE static float __npy_nzerof(void)
{
    const union { npy_uint32 __i; float __f;} __bint = {0x80000000UL};
    return __bint.__f;
}

#define NPY_INFINITYF __npy_inff()
#define NPY_NANF __npy_nanf()
#define NPY_PZEROF __npy_pzerof()
#define NPY_NZEROF __npy_nzerof()

#define NPY_INFINITY ((npy_double)NPY_INFINITYF)
#define NPY_NAN ((npy_double)NPY_NANF)
#define NPY_PZERO ((npy_double)NPY_PZEROF)
#define NPY_NZERO ((npy_double)NPY_NZEROF)

#define NPY_INFINITYL ((npy_longdouble)NPY_INFINITYF)
#define NPY_NANL ((npy_longdouble)NPY_NANF)
#define NPY_PZEROL ((npy_longdouble)NPY_PZEROF)
#define NPY_NZEROL ((npy_longdouble)NPY_NZEROF)

/*
 * Useful constants
 */
#define NPY_E         2.718281828459045235360287471352662498  /* e */
#define NPY_LOG2E     1.442695040888963407359924681001892137  /* log_2 e */
#define NPY_LOG10E    0.434294481903251827651128918916605082  /* log_10 e */
#define NPY_LOGE2     0.693147180559945309417232121458176568  /* log_e 2 */
#define NPY_LOGE10    2.302585092994045684017991454684364208  /* log_e 10 */
#define NPY_PI        3.141592653589793238462643383279502884  /* pi */
#define NPY_PI_2      1.570796326794896619231321691639751442  /* pi/2 */
#define NPY_PI_4      0.785398163397448309615660845819875721  /* pi/4 */
#define NPY_1_PI      0.318309886183790671537767526745028724  /* 1/pi */
#define NPY_2_PI      0.636619772367581343075535053490057448  /* 2/pi */
#define NPY_EULER     0.577215664901532860606512090082402431  /* Euler constant */
#define NPY_SQRT2     1.414213562373095048801688724209698079  /* sqrt(2) */
#define NPY_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */

#define NPY_Ef        2.718281828459045235360287471352662498F /* e */
#define NPY_LOG2Ef    1.442695040888963407359924681001892137F /* log_2 e */
#define NPY_LOG10Ef   0.434294481903251827651128918916605082F /* log_10 e */
#define NPY_LOGE2f    0.693147180559945309417232121458176568F /* log_e 2 */
#define NPY_LOGE10f   2.302585092994045684017991454684364208F /* log_e 10 */
#define NPY_PIf       3.141592653589793238462643383279502884F /* pi */
#define NPY_PI_2f     1.570796326794896619231321691639751442F /* pi/2 */
#define NPY_PI_4f     0.785398163397448309615660845819875721F /* pi/4 */
#define NPY_1_PIf     0.318309886183790671537767526745028724F /* 1/pi */
#define NPY_2_PIf     0.636619772367581343075535053490057448F /* 2/pi */
#define NPY_EULERf    0.577215664901532860606512090082402431F /* Euler constant */
#define NPY_SQRT2f    1.414213562373095048801688724209698079F /* sqrt(2) */
#define NPY_SQRT1_2f  0.707106781186547524400844362104849039F /* 1/sqrt(2) */

#define NPY_El        2.718281828459045235360287471352662498L /* e */
#define NPY_LOG2El    1.442695040888963407359924681001892137L /* log_2 e */
#define NPY_LOG10El   0.434294481903251827651128918916605082L /* log_10 e */
#define NPY_LOGE2l    0.693147180559945309417232121458176568L /* log_e 2 */
#define NPY_LOGE10l   2.302585092994045684017991454684364208L /* log_e 10 */
#define NPY_PIl       3.141592653589793238462643383279502884L /* pi */
#define NPY_PI_2l     1.570796326794896619231321691639751442L /* pi/2 */
#define NPY_PI_4l     0.785398163397448309615660845819875721L /* pi/4 */
#define NPY_1_PIl     0.318309886183790671537767526745028724L /* 1/pi */
#define NPY_2_PIl     0.636619772367581343075535053490057448L /* 2/pi */
#define NPY_EULERl    0.577215664901532860606512090082402431L /* Euler constant */
#define NPY_SQRT2l    1.414213562373095048801688724209698079L /* sqrt(2) */
#define NPY_SQRT1_2l  0.707106781186547524400844362104849039L /* 1/sqrt(2) */

/*
 * Constants used in vector implementation of exp(x)
 */
#define NPY_RINT_CVT_MAGICf 0x1.800000p+23f
#define NPY_CODY_WAITE_LOGE_2_HIGHf -6.93145752e-1f
#define NPY_CODY_WAITE_LOGE_2_LOWf -1.42860677e-6f
#define NPY_COEFF_P0_EXPf 9.999999999980870924916e-01f
#define NPY_COEFF_P1_EXPf 7.257664613233124478488e-01f
#define NPY_COEFF_P2_EXPf 2.473615434895520810817e-01f
#define NPY_COEFF_P3_EXPf 5.114512081637298353406e-02f
#define NPY_COEFF_P4_EXPf 6.757896990527504603057e-03f
#define NPY_COEFF_P5_EXPf 5.082762527590693718096e-04f
#define NPY_COEFF_Q0_EXPf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_EXPf -2.742335390411667452936e-01f
#define NPY_COEFF_Q2_EXPf 2.159509375685829852307e-02f

/*
 * Constants used in vector implementation of float64 exp(x)
 */
#define NPY_RINT_CVT_MAGIC 0x1.8p52
#define NPY_INV_LN2_MUL_32 0x1.71547652b82fep+5
#define NPY_TANG_NEG_L1 -0x1.62e42fefp-6
#define NPY_TANG_NEG_L2 -0x1.473de6af278edp-39
#define NPY_TANG_A1 0x1p-1
#define NPY_TANG_A2 0x1.5555555548f7cp-3
#define NPY_TANG_A3 0x1.5555555545d4ep-5
#define NPY_TANG_A4 0x1.11115b7aa905ep-7
#define NPY_TANG_A5 0x1.6c1728d739765p-10

/* Lookup table for 2^(j/32) */
static npy_uint64 EXP_Table_top[32] = {
    0x3FF0000000000000,
    0x3FF059B0D3158540,
    0x3FF0B5586CF98900,
    0x3FF11301D0125B40,
    0x3FF172B83C7D5140,
    0x3FF1D4873168B980,
    0x3FF2387A6E756200,
    0x3FF29E9DF51FDEC0,
    0x3FF306FE0A31B700,
    0x3FF371A7373AA9C0,
    0x3FF3DEA64C123400,
    0x3FF44E0860618900,
    0x3FF4BFDAD5362A00,
    0x3FF5342B569D4F80,
    0x3FF5AB07DD485400,
    0x3FF6247EB03A5580,
    0x3FF6A09E667F3BC0,
    0x3FF71F75E8EC5F40,
    0x3FF7A11473EB0180,
    0x3FF82589994CCE00,
    0x3FF8ACE5422AA0C0,
    0x3FF93737B0CDC5C0,
    0x3FF9C49182A3F080,
    0x3FFA5503B23E2540,
    0x3FFAE89F995AD380,
    0x3FFB7F76F2FB5E40,
    0x3FFC199BDD855280,
    0x3FFCB720DCEF9040,
    0x3FFD5818DCFBA480,
    0x3FFDFC97337B9B40,
    0x3FFEA4AFA2A490C0,
    0x3FFF50765B6E4540,
};

static npy_uint64 EXP_Table_tail[32] = {
    0x0000000000000000,
    0x3D0A1D73E2A475B4,
    0x3CEEC5317256E308,
    0x3CF0A4EBBF1AED93,
    0x3D0D6E6FBE462876,
    0x3D053C02DC0144C8,
    0x3D0C3360FD6D8E0B,
    0x3D009612E8AFAD12,
    0x3CF52DE8D5A46306,
    0x3CE54E28AA05E8A9,
    0x3D011ADA0911F09F,
    0x3D068189B7A04EF8,
    0x3D038EA1CBD7F621,
    0x3CBDF0A83C49D86A,
    0x3D04AC64980A8C8F,
    0x3CD2C7C3E81BF4B7,
    0x3CE921165F626CDD,
    0x3D09EE91B8797785,
    0x3CDB5F54408FDB37,
    0x3CF28ACF88AFAB35,
    0x3CFB5BA7C55A192D,
    0x3D027A280E1F92A0,
    0x3CF01C7C46B071F3,
    0x3CFC8B424491CAF8,
    0x3D06AF439A68BB99,
    0x3CDBAA9EC206AD4F,
    0x3CFC2220CB12A092,
    0x3D048A81E5E8F4A5,
    0x3CDC976816BAD9B8,
    0x3CFEB968CAC39ED3,
    0x3CF9858F73A18F5E,
    0x3C99D3E12DD8A18B,
};


/*
 * Constants used in vector implementation of log(x)
 */
#define NPY_COEFF_P0_LOGf 0.000000000000000000000e+00f
#define NPY_COEFF_P1_LOGf 9.999999999999998702752e-01f
#define NPY_COEFF_P2_LOGf 2.112677543073053063722e+00f
#define NPY_COEFF_P3_LOGf 1.480000633576506585156e+00f
#define NPY_COEFF_P4_LOGf 3.808837741388407920751e-01f
#define NPY_COEFF_P5_LOGf 2.589979117907922693523e-02f
#define NPY_COEFF_Q0_LOGf 1.000000000000000000000e+00f
#define NPY_COEFF_Q1_LOGf 2.612677543073109236779e+00f
#define NPY_COEFF_Q2_LOGf 2.453006071784736363091e+00f
#define NPY_COEFF_Q3_LOGf 9.864942958519418960339e-01f
#define NPY_COEFF_Q4_LOGf 1.546476374983906719538e-01f
#define NPY_COEFF_Q5_LOGf 5.875095403124574342950e-03f
/*
 * Constants used in vector implementation of sinf/cosf(x)
 */
#define NPY_TWO_O_PIf 0x1.45f306p-1f
#define NPY_CODY_WAITE_PI_O_2_HIGHf -0x1.921fb0p+00f
#define NPY_CODY_WAITE_PI_O_2_MEDf -0x1.5110b4p-22f
#define NPY_CODY_WAITE_PI_O_2_LOWf -0x1.846988p-48f
#define NPY_COEFF_INVF0_COSINEf 0x1.000000p+00f
#define NPY_COEFF_INVF2_COSINEf -0x1.000000p-01f
#define NPY_COEFF_INVF4_COSINEf 0x1.55553cp-05f
#define NPY_COEFF_INVF6_COSINEf -0x1.6c06dcp-10f
#define NPY_COEFF_INVF8_COSINEf 0x1.98e616p-16f
#define NPY_COEFF_INVF3_SINEf -0x1.555556p-03f
#define NPY_COEFF_INVF5_SINEf 0x1.11119ap-07f
#define NPY_COEFF_INVF7_SINEf -0x1.a06bbap-13f
#define NPY_COEFF_INVF9_SINEf 0x1.7d3bbcp-19f
/*
 * Integer functions.
 */
NPY_INPLACE npy_uint npy_gcdu(npy_uint a, npy_uint b);
NPY_INPLACE npy_uint npy_lcmu(npy_uint a, npy_uint b);
NPY_INPLACE npy_ulong npy_gcdul(npy_ulong a, npy_ulong b);
NPY_INPLACE npy_ulong npy_lcmul(npy_ulong a, npy_ulong b);
NPY_INPLACE npy_ulonglong npy_gcdull(npy_ulonglong a, npy_ulonglong b);
NPY_INPLACE npy_ulonglong npy_lcmull(npy_ulonglong a, npy_ulonglong b);

NPY_INPLACE npy_int npy_gcd(npy_int a, npy_int b);
NPY_INPLACE npy_int npy_lcm(npy_int a, npy_int b);
NPY_INPLACE npy_long npy_gcdl(npy_long a, npy_long b);
NPY_INPLACE npy_long npy_lcml(npy_long a, npy_long b);
NPY_INPLACE npy_longlong npy_gcdll(npy_longlong a, npy_longlong b);
NPY_INPLACE npy_longlong npy_lcmll(npy_longlong a, npy_longlong b);

NPY_INPLACE npy_ubyte npy_rshiftuhh(npy_ubyte a, npy_ubyte b);
NPY_INPLACE npy_ubyte npy_lshiftuhh(npy_ubyte a, npy_ubyte b);
NPY_INPLACE npy_ushort npy_rshiftuh(npy_ushort a, npy_ushort b);
NPY_INPLACE npy_ushort npy_lshiftuh(npy_ushort a, npy_ushort b);
NPY_INPLACE npy_uint npy_rshiftu(npy_uint a, npy_uint b);
NPY_INPLACE npy_uint npy_lshiftu(npy_uint a, npy_uint b);
NPY_INPLACE npy_ulong npy_rshiftul(npy_ulong a, npy_ulong b);
NPY_INPLACE npy_ulong npy_lshiftul(npy_ulong a, npy_ulong b);
NPY_INPLACE npy_ulonglong npy_rshiftull(npy_ulonglong a, npy_ulonglong b);
NPY_INPLACE npy_ulonglong npy_lshiftull(npy_ulonglong a, npy_ulonglong b);

NPY_INPLACE npy_byte npy_rshifthh(npy_byte a, npy_byte b);
NPY_INPLACE npy_byte npy_lshifthh(npy_byte a, npy_byte b);
NPY_INPLACE npy_short npy_rshifth(npy_short a, npy_short b);
NPY_INPLACE npy_short npy_lshifth(npy_short a, npy_short b);
NPY_INPLACE npy_int npy_rshift(npy_int a, npy_int b);
NPY_INPLACE npy_int npy_lshift(npy_int a, npy_int b);
NPY_INPLACE npy_long npy_rshiftl(npy_long a, npy_long b);
NPY_INPLACE npy_long npy_lshiftl(npy_long a, npy_long b);
NPY_INPLACE npy_longlong npy_rshiftll(npy_longlong a, npy_longlong b);
NPY_INPLACE npy_longlong npy_lshiftll(npy_longlong a, npy_longlong b);

/*
 * avx function has a common API for both sin & cos. This enum is used to
 * distinguish between the two
 */
typedef enum {
    npy_compute_sin,
    npy_compute_cos
} NPY_TRIG_OP;

/*
 * C99 double math funcs
 */
NPY_INPLACE double npy_sin(double x);
NPY_INPLACE double npy_cos(double x);
NPY_INPLACE double npy_tan(double x);
NPY_INPLACE double npy_sinh(double x);
NPY_INPLACE double npy_cosh(double x);
NPY_INPLACE double npy_tanh(double x);

NPY_INPLACE double npy_asin(double x);
NPY_INPLACE double npy_acos(double x);
NPY_INPLACE double npy_atan(double x);

NPY_INPLACE double npy_log(double x);
NPY_INPLACE double npy_log10(double x);
NPY_INPLACE double npy_exp(double x);
NPY_INPLACE double npy_sqrt(double x);
NPY_INPLACE double npy_cbrt(double x);

NPY_INPLACE double npy_fabs(double x);
NPY_INPLACE double npy_ceil(double x);
NPY_INPLACE double npy_fmod(double x, double y);
NPY_INPLACE double npy_floor(double x);

NPY_INPLACE double npy_expm1(double x);
NPY_INPLACE double npy_log1p(double x);
NPY_INPLACE double npy_hypot(double x, double y);
NPY_INPLACE double npy_acosh(double x);
NPY_INPLACE double npy_asinh(double xx);
NPY_INPLACE double npy_atanh(double x);
NPY_INPLACE double npy_rint(double x);
NPY_INPLACE double npy_trunc(double x);
NPY_INPLACE double npy_exp2(double x);
NPY_INPLACE double npy_log2(double x);

NPY_INPLACE double npy_atan2(double x, double y);
NPY_INPLACE double npy_pow(double x, double y);
NPY_INPLACE double npy_modf(double x, double* y);
NPY_INPLACE double npy_frexp(double x, int* y);
NPY_INPLACE double npy_ldexp(double n, int y);

NPY_INPLACE double npy_copysign(double x, double y);
double npy_nextafter(double x, double y);
double npy_spacing(double x);

/*
 * IEEE 754 fpu handling. Those are guaranteed to be macros
 */

/* use builtins to avoid function calls in tight loops
 * only available if npy_config.h is available (= numpys own build) */
#if HAVE___BUILTIN_ISNAN
    #define npy_isnan(x) __builtin_isnan(x)
#else
    #ifndef NPY_HAVE_DECL_ISNAN
        #define npy_isnan(x) ((x) != (x))
    #else
        #if defined(_MSC_VER) && (_MSC_VER < 1900)
            #define npy_isnan(x) _isnan((x))
        #else
            #define npy_isnan(x) isnan(x)
        #endif
    #endif
#endif


/* only available if npy_config.h is available (= numpys own build) */
#if HAVE___BUILTIN_ISFINITE
    #define npy_isfinite(x) __builtin_isfinite(x)
#else
    #ifndef NPY_HAVE_DECL_ISFINITE
        #ifdef _MSC_VER
            #define npy_isfinite(x) _finite((x))
        #else
            #define npy_isfinite(x) !npy_isnan((x) + (-x))
        #endif
    #else
        #define npy_isfinite(x) isfinite((x))
    #endif
#endif

/* only available if npy_config.h is available (= numpys own build) */
#if HAVE___BUILTIN_ISINF
    #define npy_isinf(x) __builtin_isinf(x)
#else
    #ifndef NPY_HAVE_DECL_ISINF
        #define npy_isinf(x) (!npy_isfinite(x) && !npy_isnan(x))
    #else
        #if defined(_MSC_VER) && (_MSC_VER < 1900)
            #define npy_isinf(x) (!_finite((x)) && !_isnan((x)))
        #else
            #define npy_isinf(x) isinf((x))
        #endif
    #endif
#endif

#ifndef NPY_HAVE_DECL_SIGNBIT
    int _npy_signbit_f(float x);
    int _npy_signbit_d(double x);
    int _npy_signbit_ld(long double x);
    #define npy_signbit(x) \
        (sizeof (x) == sizeof (long double) ? _npy_signbit_ld (x) \
         : sizeof (x) == sizeof (double) ? _npy_signbit_d (x) \
         : _npy_signbit_f (x))
#else
    #define npy_signbit(x) signbit((x))
#endif

/*
 * float C99 math functions
 */
NPY_INPLACE float npy_sinf(float x);
NPY_INPLACE float npy_cosf(float x);
NPY_INPLACE float npy_tanf(float x);
NPY_INPLACE float npy_sinhf(float x);
NPY_INPLACE float npy_coshf(float x);
NPY_INPLACE float npy_tanhf(float x);
NPY_INPLACE float npy_fabsf(float x);
NPY_INPLACE float npy_floorf(float x);
NPY_INPLACE float npy_ceilf(float x);
NPY_INPLACE float npy_rintf(float x);
NPY_INPLACE float npy_truncf(float x);
NPY_INPLACE float npy_sqrtf(float x);
NPY_INPLACE float npy_cbrtf(float x);
NPY_INPLACE float npy_log10f(float x);
NPY_INPLACE float npy_logf(float x);
NPY_INPLACE float npy_expf(float x);
NPY_INPLACE float npy_expm1f(float x);
NPY_INPLACE float npy_asinf(float x);
NPY_INPLACE float npy_acosf(float x);
NPY_INPLACE float npy_atanf(float x);
NPY_INPLACE float npy_asinhf(float x);
NPY_INPLACE float npy_acoshf(float x);
NPY_INPLACE float npy_atanhf(float x);
NPY_INPLACE float npy_log1pf(float x);
NPY_INPLACE float npy_exp2f(float x);
NPY_INPLACE float npy_log2f(float x);

NPY_INPLACE float npy_atan2f(float x, float y);
NPY_INPLACE float npy_hypotf(float x, float y);
NPY_INPLACE float npy_powf(float x, float y);
NPY_INPLACE float npy_fmodf(float x, float y);

NPY_INPLACE float npy_modff(float x, float* y);
NPY_INPLACE float npy_frexpf(float x, int* y);
NPY_INPLACE float npy_ldexpf(float x, int y);

NPY_INPLACE float npy_copysignf(float x, float y);
float npy_nextafterf(float x, float y);
float npy_spacingf(float x);

/*
 * long double C99 math functions
 */
NPY_INPLACE npy_longdouble npy_sinl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_cosl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_tanl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_sinhl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_coshl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_tanhl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_fabsl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_floorl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_ceill(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_rintl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_truncl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_sqrtl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_cbrtl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_log10l(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_logl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_expl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_expm1l(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_asinl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_acosl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_atanl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_asinhl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_acoshl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_atanhl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_log1pl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_exp2l(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_log2l(npy_longdouble x);

NPY_INPLACE npy_longdouble npy_atan2l(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_hypotl(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_powl(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_fmodl(npy_longdouble x, npy_longdouble y);

NPY_INPLACE npy_longdouble npy_modfl(npy_longdouble x, npy_longdouble* y);
NPY_INPLACE npy_longdouble npy_frexpl(npy_longdouble x, int* y);
NPY_INPLACE npy_longdouble npy_ldexpl(npy_longdouble x, int y);

NPY_INPLACE npy_longdouble npy_copysignl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_nextafterl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_spacingl(npy_longdouble x);

/*
 * Non standard functions
 */
NPY_INPLACE double npy_deg2rad(double x);
NPY_INPLACE double npy_rad2deg(double x);
NPY_INPLACE double npy_logaddexp(double x, double y);
NPY_INPLACE double npy_logaddexp2(double x, double y);
NPY_INPLACE double npy_divmod(double x, double y, double *modulus);
NPY_INPLACE double npy_heaviside(double x, double h0);

NPY_INPLACE float npy_deg2radf(float x);
NPY_INPLACE float npy_rad2degf(float x);
NPY_INPLACE float npy_logaddexpf(float x, float y);
NPY_INPLACE float npy_logaddexp2f(float x, float y);
NPY_INPLACE float npy_divmodf(float x, float y, float *modulus);
NPY_INPLACE float npy_heavisidef(float x, float h0);

NPY_INPLACE npy_longdouble npy_deg2radl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_rad2degl(npy_longdouble x);
NPY_INPLACE npy_longdouble npy_logaddexpl(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_logaddexp2l(npy_longdouble x, npy_longdouble y);
NPY_INPLACE npy_longdouble npy_divmodl(npy_longdouble x, npy_longdouble y,
                           npy_longdouble *modulus);
NPY_INPLACE npy_longdouble npy_heavisidel(npy_longdouble x, npy_longdouble h0);

#define npy_degrees npy_rad2deg
#define npy_degreesf npy_rad2degf
#define npy_degreesl npy_rad2degl

#define npy_radians npy_deg2rad
#define npy_radiansf npy_deg2radf
#define npy_radiansl npy_deg2radl

/*
 * Complex declarations
 */

/*
 * C99 specifies that complex numbers have the same representation as
 * an array of two elements, where the first element is the real part
 * and the second element is the imaginary part.
 */
#define __NPY_CPACK_IMP(x, y, type, ctype)   \
    union {                                  \
        ctype z;                             \
        type a[2];                           \
    } z1;;                                   \
                                             \
    z1.a[0] = (x);                           \
    z1.a[1] = (y);                           \
                                             \
    return z1.z;

static NPY_INLINE npy_cdouble npy_cpack(double x, double y)
{
    __NPY_CPACK_IMP(x, y, double, npy_cdouble);
}

static NPY_INLINE npy_cfloat npy_cpackf(float x, float y)
{
    __NPY_CPACK_IMP(x, y, float, npy_cfloat);
}

static NPY_INLINE npy_clongdouble npy_cpackl(npy_longdouble x, npy_longdouble y)
{
    __NPY_CPACK_IMP(x, y, npy_longdouble, npy_clongdouble);
}
#undef __NPY_CPACK_IMP

/*
 * Same remark as above, but in the other direction: extract first/second
 * member of complex number, assuming a C99-compatible representation
 *
 * Those are defineds as static inline, and such as a reasonable compiler would
 * most likely compile this to one or two instructions (on CISC at least)
 */
#define __NPY_CEXTRACT_IMP(z, index, type, ctype)   \
    union {                                         \
        ctype z;                                    \
        type a[2];                                  \
    } __z_repr;                                     \
    __z_repr.z = z;                                 \
                                                    \
    return __z_repr.a[index];

static NPY_INLINE double npy_creal(npy_cdouble z)
{
    __NPY_CEXTRACT_IMP(z, 0, double, npy_cdouble);
}

static NPY_INLINE double npy_cimag(npy_cdouble z)
{
    __NPY_CEXTRACT_IMP(z, 1, double, npy_cdouble);
}

static NPY_INLINE float npy_crealf(npy_cfloat z)
{
    __NPY_CEXTRACT_IMP(z, 0, float, npy_cfloat);
}

static NPY_INLINE float npy_cimagf(npy_cfloat z)
{
    __NPY_CEXTRACT_IMP(z, 1, float, npy_cfloat);
}

static NPY_INLINE npy_longdouble npy_creall(npy_clongdouble z)
{
    __NPY_CEXTRACT_IMP(z, 0, npy_longdouble, npy_clongdouble);
}

static NPY_INLINE npy_longdouble npy_cimagl(npy_clongdouble z)
{
    __NPY_CEXTRACT_IMP(z, 1, npy_longdouble, npy_clongdouble);
}
#undef __NPY_CEXTRACT_IMP

/*
 * Double precision complex functions
 */
double npy_cabs(npy_cdouble z);
double npy_carg(npy_cdouble z);

npy_cdouble npy_cexp(npy_cdouble z);
npy_cdouble npy_clog(npy_cdouble z);
npy_cdouble npy_cpow(npy_cdouble x, npy_cdouble y);

npy_cdouble npy_csqrt(npy_cdouble z);

npy_cdouble npy_ccos(npy_cdouble z);
npy_cdouble npy_csin(npy_cdouble z);
npy_cdouble npy_ctan(npy_cdouble z);

npy_cdouble npy_ccosh(npy_cdouble z);
npy_cdouble npy_csinh(npy_cdouble z);
npy_cdouble npy_ctanh(npy_cdouble z);

npy_cdouble npy_cacos(npy_cdouble z);
npy_cdouble npy_casin(npy_cdouble z);
npy_cdouble npy_catan(npy_cdouble z);

npy_cdouble npy_cacosh(npy_cdouble z);
npy_cdouble npy_casinh(npy_cdouble z);
npy_cdouble npy_catanh(npy_cdouble z);

/*
 * Single precision complex functions
 */
float npy_cabsf(npy_cfloat z);
float npy_cargf(npy_cfloat z);

npy_cfloat npy_cexpf(npy_cfloat z);
npy_cfloat npy_clogf(npy_cfloat z);
npy_cfloat npy_cpowf(npy_cfloat x, npy_cfloat y);

npy_cfloat npy_csqrtf(npy_cfloat z);

npy_cfloat npy_ccosf(npy_cfloat z);
npy_cfloat npy_csinf(npy_cfloat z);
npy_cfloat npy_ctanf(npy_cfloat z);

npy_cfloat npy_ccoshf(npy_cfloat z);
npy_cfloat npy_csinhf(npy_cfloat z);
npy_cfloat npy_ctanhf(npy_cfloat z);

npy_cfloat npy_cacosf(npy_cfloat z);
npy_cfloat npy_casinf(npy_cfloat z);
npy_cfloat npy_catanf(npy_cfloat z);

npy_cfloat npy_cacoshf(npy_cfloat z);
npy_cfloat npy_casinhf(npy_cfloat z);
npy_cfloat npy_catanhf(npy_cfloat z);


/*
 * Extended precision complex functions
 */
npy_longdouble npy_cabsl(npy_clongdouble z);
npy_longdouble npy_cargl(npy_clongdouble z);

npy_clongdouble npy_cexpl(npy_clongdouble z);
npy_clongdouble npy_clogl(npy_clongdouble z);
npy_clongdouble npy_cpowl(npy_clongdouble x, npy_clongdouble y);

npy_clongdouble npy_csqrtl(npy_clongdouble z);

npy_clongdouble npy_ccosl(npy_clongdouble z);
npy_clongdouble npy_csinl(npy_clongdouble z);
npy_clongdouble npy_ctanl(npy_clongdouble z);

npy_clongdouble npy_ccoshl(npy_clongdouble z);
npy_clongdouble npy_csinhl(npy_clongdouble z);
npy_clongdouble npy_ctanhl(npy_clongdouble z);

npy_clongdouble npy_cacosl(npy_clongdouble z);
npy_clongdouble npy_casinl(npy_clongdouble z);
npy_clongdouble npy_catanl(npy_clongdouble z);

npy_clongdouble npy_cacoshl(npy_clongdouble z);
npy_clongdouble npy_casinhl(npy_clongdouble z);
npy_clongdouble npy_catanhl(npy_clongdouble z);


/*
 * Functions that set the floating point error
 * status word.
 */

/*
 * platform-dependent code translates floating point
 * status to an integer sum of these values
 */
#define NPY_FPE_DIVIDEBYZERO  1
#define NPY_FPE_OVERFLOW      2
#define NPY_FPE_UNDERFLOW     4
#define NPY_FPE_INVALID       8

int npy_clear_floatstatus_barrier(char*);
int npy_get_floatstatus_barrier(char*);
/*
 * use caution with these - clang and gcc8.1 are known to reorder calls
 * to this form of the function which can defeat the check. The _barrier
 * form of the call is preferable, where the argument is
 * (char*)&local_variable
 */
int npy_clear_floatstatus(void);
int npy_get_floatstatus(void);

void npy_set_floatstatus_divbyzero(void);
void npy_set_floatstatus_overflow(void);
void npy_set_floatstatus_underflow(void);
void npy_set_floatstatus_invalid(void);

#ifdef __cplusplus
}
#endif

#if NPY_INLINE_MATH
#include "npy_math_internal.h"
#endif

#endif
