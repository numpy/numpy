#ifndef __NPY_MATH_C99_H_
#define __NPY_MATH_C99_H_

#include <math.h>
#include <numpy/npy_common.h>

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
#define NPY_E         2.7182818284590452353602874713526625  /* e */
#define NPY_LOG2E     1.4426950408889634073599246810018921  /* log_2 e */
#define NPY_LOG10E    0.4342944819032518276511289189166051  /* log_10 e */
#define NPY_LOGE2     0.6931471805599453094172321214581766  /* log_e 2 */
#define NPY_LOGE10    2.3025850929940456840179914546843642  /* log_e 10 */
#define NPY_PI        3.1415926535897932384626433832795029  /* pi */
#define NPY_PI_2      1.5707963267948966192313216916397514  /* pi/2 */
#define NPY_PI_4      0.7853981633974483096156608458198757  /* pi/4 */
#define NPY_1_PI      0.3183098861837906715377675267450287  /* 1/pi */
#define NPY_2_PI      0.6366197723675813430755350534900574  /* 2/pi */
#define NPY_EULER     0.5772156649015328606065120900824024  /* Euler constant */

#define NPY_Ef        2.7182818284590452353602874713526625F /* e */
#define NPY_LOG2Ef    1.4426950408889634073599246810018921F /* log_2 e */
#define NPY_LOG10Ef   0.4342944819032518276511289189166051F /* log_10 e */
#define NPY_LOGE2f    0.6931471805599453094172321214581766F /* log_e 2 */
#define NPY_LOGE10f   2.3025850929940456840179914546843642F /* log_e 10 */
#define NPY_PIf       3.1415926535897932384626433832795029F /* pi */
#define NPY_PI_2f     1.5707963267948966192313216916397514F /* pi/2 */
#define NPY_PI_4f     0.7853981633974483096156608458198757F /* pi/4 */
#define NPY_1_PIf     0.3183098861837906715377675267450287F /* 1/pi */
#define NPY_2_PIf     0.6366197723675813430755350534900574F /* 2/pi */
#define NPY_EULERf    0.5772156649015328606065120900824024F /* Euler constant */

#define NPY_El        2.7182818284590452353602874713526625L /* e */
#define NPY_LOG2El    1.4426950408889634073599246810018921L /* log_2 e */
#define NPY_LOG10El   0.4342944819032518276511289189166051L /* log_10 e */
#define NPY_LOGE2l    0.6931471805599453094172321214581766L /* log_e 2 */
#define NPY_LOGE10l   2.3025850929940456840179914546843642L /* log_e 10 */
#define NPY_PIl       3.1415926535897932384626433832795029L /* pi */
#define NPY_PI_2l     1.5707963267948966192313216916397514L /* pi/2 */
#define NPY_PI_4l     0.7853981633974483096156608458198757L /* pi/4 */
#define NPY_1_PIl     0.3183098861837906715377675267450287L /* 1/pi */
#define NPY_2_PIl     0.6366197723675813430755350534900574L /* 2/pi */
#define NPY_EULERl    0.5772156649015328606065120900824024L /* Euler constant */

/*
 * C99 double math funcs
 */
double npy_sin(double x);
double npy_cos(double x);
double npy_tan(double x);
double npy_sinh(double x);
double npy_cosh(double x);
double npy_tanh(double x);

double npy_asin(double x);
double npy_acos(double x);
double npy_atan(double x);
double npy_aexp(double x);
double npy_alog(double x);
double npy_asqrt(double x);
double npy_afabs(double x);

double npy_log(double x);
double npy_log10(double x);
double npy_exp(double x);
double npy_sqrt(double x);

double npy_fabs(double x);
double npy_ceil(double x);
double npy_fmod(double x, double y);
double npy_floor(double x);

double npy_expm1(double x);
double npy_log1p(double x);
double npy_hypot(double x, double y);
double npy_acosh(double x);
double npy_asinh(double xx);
double npy_atanh(double x);
double npy_rint(double x);
double npy_trunc(double x);
double npy_exp2(double x);
double npy_log2(double x);

double npy_atan2(double x, double y);
double npy_pow(double x, double y);
double npy_modf(double x, double* y);

/*
 * IEEE 754 fpu handling. Those are guaranteed to be macros
 */
#ifndef NPY_HAVE_DECL_ISNAN
        #define npy_isnan(x) ((x) != (x))
#else
        #define npy_isnan(x) isnan((x))
#endif

#ifndef NPY_HAVE_DECL_ISFINITE
        #define npy_isfinite(x) !npy_isnan((x) + (-x))
#else
        #define npy_isfinite(x) isfinite((x))
#endif

#ifndef NPY_HAVE_DECL_ISINF
        #define npy_isinf(x) (!npy_isfinite(x) && !npy_isnan(x))
#else
        #define npy_isinf(x) isinf((x))
#endif

#ifndef NPY_HAVE_DECL_SIGNBIT
	int _npy_signbit_f(float x);
	int _npy_signbit_d(double x);
	int _npy_signbit_ld(npy_longdouble x);
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

float npy_sinf(float x);
float npy_cosf(float x);
float npy_tanf(float x);
float npy_sinhf(float x);
float npy_coshf(float x);
float npy_tanhf(float x);
float npy_fabsf(float x);
float npy_floorf(float x);
float npy_ceilf(float x);
float npy_rintf(float x);
float npy_truncf(float x);
float npy_sqrtf(float x);
float npy_log10f(float x);
float npy_logf(float x);
float npy_expf(float x);
float npy_expm1f(float x);
float npy_asinf(float x);
float npy_acosf(float x);
float npy_atanf(float x);
float npy_asinhf(float x);
float npy_acoshf(float x);
float npy_atanhf(float x);
float npy_log1pf(float x);
float npy_exp2f(float x);
float npy_log2f(float x);

float npy_atan2f(float x, float y);
float npy_hypotf(float x, float y);
float npy_powf(float x, float y);
float npy_fmodf(float x, float y);

float npy_modff(float x, float* y);

/*
 * float C99 math functions
 */

npy_longdouble npy_sinl(npy_longdouble x);
npy_longdouble npy_cosl(npy_longdouble x);
npy_longdouble npy_tanl(npy_longdouble x);
npy_longdouble npy_sinhl(npy_longdouble x);
npy_longdouble npy_coshl(npy_longdouble x);
npy_longdouble npy_tanhl(npy_longdouble x);
npy_longdouble npy_fabsl(npy_longdouble x);
npy_longdouble npy_floorl(npy_longdouble x);
npy_longdouble npy_ceill(npy_longdouble x);
npy_longdouble npy_rintl(npy_longdouble x);
npy_longdouble npy_truncl(npy_longdouble x);
npy_longdouble npy_sqrtl(npy_longdouble x);
npy_longdouble npy_log10l(npy_longdouble x);
npy_longdouble npy_logl(npy_longdouble x);
npy_longdouble npy_expl(npy_longdouble x);
npy_longdouble npy_expm1l(npy_longdouble x);
npy_longdouble npy_asinl(npy_longdouble x);
npy_longdouble npy_acosl(npy_longdouble x);
npy_longdouble npy_atanl(npy_longdouble x);
npy_longdouble npy_asinhl(npy_longdouble x);
npy_longdouble npy_acoshl(npy_longdouble x);
npy_longdouble npy_atanhl(npy_longdouble x);
npy_longdouble npy_log1pl(npy_longdouble x);
npy_longdouble npy_exp2l(npy_longdouble x);
npy_longdouble npy_log2l(npy_longdouble x);

npy_longdouble npy_atan2l(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_hypotl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_powl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_fmodl(npy_longdouble x, npy_longdouble y);

npy_longdouble npy_modfl(npy_longdouble x, npy_longdouble* y);

/*
 * Non standard functions
 */
double npy_deg2rad(double x);
double npy_rad2deg(double x);
double npy_logaddexp(double x, double y);
double npy_logaddexp2(double x, double y);

float npy_deg2radf(float x);
float npy_rad2degf(float x);
float npy_logaddexpf(float x, float y);
float npy_logaddexp2f(float x, float y);

npy_longdouble npy_deg2radl(npy_longdouble x);
npy_longdouble npy_rad2degl(npy_longdouble x);
npy_longdouble npy_logaddexpl(npy_longdouble x, npy_longdouble y);
npy_longdouble npy_logaddexp2l(npy_longdouble x, npy_longdouble y);

#define npy_degrees npy_rad2deg
#define npy_degreesf npy_rad2degf
#define npy_degreesl npy_rad2degl

#define npy_radians npy_deg2rad
#define npy_radiansf npy_deg2radf
#define npy_radiansl npy_deg2radl

#endif
