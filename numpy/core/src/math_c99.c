/*
 * A small module to implement missing C99 math capabilities required by numpy
 *
 * Please keep this independant of python as much as possible !
 */

/*
 * Include python.h because it may modify math.h configuration, but we won't
 * use any python code at all here
 */
#include "Python.h"
#include "config.h"
#include <math.h>

/*
 * Basic functions, double version. Some old/weird platforms may not have those
 *
 * Original code by Konrad Hinsen.
 */
#ifndef HAVE_EXPM1
double expm1(double x)
{
    double u = exp(x);
    if (u == 1.0) {
        return x;
    } else if (u-1.0 == -1.0) {
        return -1;
    } else {
        return (u-1.0) * x/log(u);
    }
}
#endif

#ifndef HAVE_LOG1P
double log1p(double x)
{
    double u = 1. + x;
    if (u == 1.0) {
        return x;
    } else {
        return log(u) * x / (u-1.);
    }
}
#endif

#ifndef HAVE_HYPOT
double hypot(double x, double y)
{
    double yx;

    x = fabs(x);
    y = fabs(y);
    if (x < y) {
        double temp = x;
        x = y;
        y = temp;
    }
    if (x == 0.)
        return 0.;
    else {
        yx = y/x;
        return x*sqrt(1.+yx*yx);
    }
}
#endif

#ifndef HAVE_ACOSH
double acosh(double x)
{
    return 2*log(sqrt((x+1.0)/2)+sqrt((x-1.0)/2));
}
#endif

#ifndef HAVE_ASINH
double asinh(double xx)
{
    double x, d;
    int sign;
    if (xx < 0.0) {
        sign = -1;
        x = -xx;
    }
    else {
        sign = 1;
        x = xx;
    }
    if (x > 1e8) {
        d = x;
    } else {
        d = sqrt(x*x + 1);
    }
    return sign*log1p(x*(1.0 + x/(d+1)));
}
#endif

#ifndef HAVE_ATANH
static double atanh(double x)
{
    return 0.5*log1p(2.0*x/(1.0-x));
}
#endif

#ifndef HAVE_RINT
double rint(double x)
{
    double y, r;

    y = floor(x);
    r = x - y;

    if (r > 0.5) goto rndup;

    /* Round to nearest even */
    if (r==0.5) {
        r = y - 2.0*floor(0.5*y);
        if (r==1.0) {
        rndup:
            y+=1.0;
        }
    }
    return y;
}
#endif

#ifndef HAVE_TRUNC
double trunc(double x)
{
    if (x < 0) {
    	return ceil(x);
    }
    else {
        return floor(x);
    }

}
#endif

/*
 * if C99 extensions not available then define dummy functions that use the
 * double versions for
 *
 * sin, cos, tan
 * sinh, cosh, tanh,
 * fabs, floor, ceil, fmod, sqrt, log10, log, exp, fabs
 * asin, acos, atan,
 * asinh, acosh, atanh
 *
 * hypot, atan2, pow, expm1
 *
 * We assume the above are always available in their double versions.
 */

/*
 * Long double versions
 */
#ifndef HAVE_SINL
longdouble sinl(longdouble x)
{
    return (longdouble) sin((double)x);
}
#endif

#ifndef HAVE_COSL
longdouble cosl(longdouble x)
{
    return (longdouble) cos((double)x);
}
#endif

#ifndef HAVE_TANL
longdouble tanl(longdouble x)
{
    return (longdouble) tan((double)x);
}
#endif

#ifndef HAVE_SINHL
longdouble sinhl(longdouble x)
{
    return (longdouble) sinh((double)x);
}
#endif

#ifndef HAVE_COSHL
longdouble coshl(longdouble x)
{
    return (longdouble) cosh((double)x);
}
#endif

#ifndef HAVE_TANHL
longdouble tanhl(longdouble x)
{
    return (longdouble) tanh((double)x);
}
#endif

#ifndef HAVE_FABSL
longdouble fabsl(longdouble x)
{
    return (longdouble) fabs((double)x);
}
#endif

#ifndef HAVE_FLOORL
longdouble floorl(longdouble x)
{
    return (longdouble) floor((double)x);
}
#endif

#ifndef HAVE_CEILL
longdouble ceill(longdouble x)
{
    return (longdouble) ceil((double)x);
}
#endif

#ifndef HAVE_SQRTL
longdouble sqrtl(longdouble x)
{
    return (longdouble) sqrt((double)x);
}
#endif

#ifndef HAVE_LOG10L
longdouble log10l(longdouble x)
{
    return (longdouble) log10((double)x);
}
#endif

#ifndef HAVE_LOGL
longdouble logl(longdouble x)
{
    return (longdouble) log((double)x);
}
#endif

#ifndef HAVE_EXPL
longdouble expl(longdouble x)
{
    return (longdouble) exp((double)x);
}
#endif

#ifndef HAVE_EXPM1L
longdouble expm1l(longdouble x)
{
    return (longdouble) expm1((double)x);
}
#endif

#ifndef HAVE_ASINL
longdouble asinl(longdouble x)
{
    return (longdouble) asin((double)x);
}
#endif

#ifndef HAVE_ACOSL
longdouble acosl(longdouble x)
{
    return (longdouble) acos((double)x);
}
#endif

#ifndef HAVE_ATANL
longdouble atanl(longdouble x)
{
    return (longdouble) atan((double)x);
}
#endif

#ifndef HAVE_RINTL
longdouble rintl(longdouble x)
{
    return (longdouble) rint((double)x);
}
#endif

#ifndef HAVE_EXPML
longdouble expml(longdouble x)
{
    return (longdouble) expm((double)x);
}
#endif

#ifndef HAVE_ATAN2L
longdouble atan2l(longdouble x, longdouble y)
{
    return (longdouble) atan2((double)x, (double) y);
}
#endif

#ifndef HAVE_HYPOTL
longdouble hypotl(longdouble x, longdouble y)
{
    return (longdouble) hypot((double)x, (double) y);
}
#endif

#ifndef HAVE_POWL
longdouble powl(longdouble x, longdouble y)
{
    return (longdouble) pow((double)x, (double) y);
}
#endif

#ifndef HAVE_FMODL
longdouble fmodl(longdouble x, longdouble y)
{
    return (longdouble) fmod((double)x, (double) y);
}
#endif

#ifndef HAVE_MODFL
longdouble modfl(longdouble x, longdouble *iptr)
{
    double nx, niptr, y;
    nx = (double) x;
    y = modf(nx, &niptr);
    *iptr = (longdouble) niptr;
    return (longdouble) y;
}
#endif

/*
 * float versions
 */
#ifndef HAVE_SINF
float sinf(float x)
{
    return (float) sin((double)x);
}
#endif

#ifndef HAVE_COSF
float cosf(float x)
{
    return (float) cos((double)x);
}
#endif

#ifndef HAVE_TANF
float tanf(float x)
{
    return (float) tan((double)x);
}
#endif

#ifndef HAVE_SINHF
float sinhf(float x)
{
    return (float) sinh((double)x);
}
#endif

#ifndef HAVE_COSHF
float coshf(float x)
{
    return (float) cosh((double)x);
}
#endif

#ifndef HAVE_TANHF
float tanhf(float x)
{
    return (float) tanh((double)x);
}
#endif

#ifndef HAVE_FABSF
float fabsf(float x)
{
    return (float) fabs((double)x);
}
#endif

#ifndef HAVE_FLOORF
float floorf(float x)
{
    return (float) floor((double)x);
}
#endif

#ifndef HAVE_CEILF
float ceilf(float x)
{
    return (float) ceil((double)x);
}
#endif

#ifndef HAVE_SQRTF
float sqrtf(float x)
{
    return (float) sqrt((double)x);
}
#endif

#ifndef HAVE_LOG10F
float log10f(float x)
{
    return (float) log10((double)x);
}
#endif

#ifndef HAVE_LOGF
float logf(float x)
{
    return (float) log((double)x);
}
#endif

#ifndef HAVE_EXPF
float expf(float x)
{
    return (float) exp((double)x);
}
#endif

#ifndef HAVE_EXPM1F
float expm1f(float x)
{
    return (float) expm1((double)x);
}
#endif

#ifndef HAVE_ASINF
float asinf(float x)
{
    return (float) asin((double)x);
}
#endif

#ifndef HAVE_ACOSF
float acosf(float x)
{
    return (float) acos((double)x);
}
#endif

#ifndef HAVE_ATANF
float atanf(float x)
{
    return (float) atan((double)x);
}
#endif

#ifndef HAVE_RINTF
float rintf(float x)
{
    return (float) rint((double)x);
}
#endif

#ifndef HAVE_EXPMF
float expmf(float x)
{
    return (float) expm((double)x);
}
#endif

#ifndef HAVE_ATAN2F
float atan2f(float x, float y)
{
    return (float) atan2((double)x, (double) y);
}
#endif

#ifndef HAVE_HYPOTF
float hypotf(float x, float y)
{
    return (float) hypot((double)x, (double) y);
}
#endif

#ifndef HAVE_POWF
float powf(float x, float y)
{
    return (float) pow((double)x, (double) y);
}
#endif

#ifndef HAVE_FMODF
float fmodf(float x, float y)
{
    return (float) fmod((double)x, (double) y);
}
#endif

#ifndef HAVE_MODFF
float modff(float x, float *iptr)
{
    double nx, niptr, y;
    nx = (double) x;
    y = modf(nx, &niptr);
    *iptr = (float) niptr;
    return (float) y;
}
#endif

