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

