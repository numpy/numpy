#ifndef __NPY_MATH_C99_H_
#define __NPY_MATH_C99_H_

#include <math.h>
#include <numpy/npy_common.h>
/*
 * C99 double math funcs
 */
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

/*
 * IEEE 754 fpu handling. Those are guaranteed to be macros
 */
#ifndef NPY_HAVE_DECL_ISNAN
        #define npy_isnan(x) _npy_isnan((x))
#else
        #define npy_isnan(x) isnan((x))
#endif

#ifndef NPY_HAVE_DECL_ISFINITE
        #define npy_isfinite(x) _npy_isfinite((x))
#else
        #define npy_isfinite(x) isfinite((x))
#endif

#ifndef NPY_HAVE_DECL_ISFINITE
        #define npy_isinf(x) _npy_isinf((x))
#else
        #define npy_isinf(x) isinf((x))
#endif

#ifndef NPY_HAVE_DECL_SIGNBIT
        #define npy_signbit(x) _npy_signbit((x))
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

#endif
