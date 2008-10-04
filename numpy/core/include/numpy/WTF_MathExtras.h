/*
 * Copyright (C) 2006, 2007 Apple Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions

 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright

 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE COMPUTER, INC. ``AS IS'' AND ANY

 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE COMPUTER, INC. OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,

 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY

 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 */


#ifndef WTF_MathExtras_h
#define WTF_MathExtras_h

#include <math.h>
#include <stdlib.h>
#include <time.h>

#if defined (__SVR4) && defined (__sun) && defined (__GNUC__)

#include <ieeefp.h>

#endif

#if defined(_MSC_VER)

#include <xmath.h>
#include <limits.h>

#if HAVE(FLOAT_H)
#include <float.h>
#endif

#endif

#ifndef M_PI
const double piDouble = 3.14159265358979323846;
const float piFloat = 3.14159265358979323846f;
#else
const double piDouble = M_PI;
const float piFloat = (float)M_PI;
#endif

#ifndef M_PI_4
const double piOverFourDouble = 0.785398163397448309616;
const float piOverFourFloat = 0.785398163397448309616f;
#else
const double piOverFourDouble = M_PI_4;
const float piOverFourFloat = (float)M_PI_4;
#endif

#if defined (__SVR4) && defined (__sun) && defined (__GNUC__)

#ifndef isfinite
#define isfinite(x) (finite(x) && !isnand(x))
#endif
#ifndef isinf
#define isinf(x) (!finite(x) && !isnand(x))
#endif
#ifndef signbit
#define signbit(x) (x < 0.0) /* FIXME: Wrong for negative 0. */
#endif

#endif

#if defined(_MSC_VER)

#define isinf(num) ( !_finite(num) && !_isnan(num) )
#define isnan(num) ( !!_isnan(num) )
/*
#define lround(num) ( (long)(num > 0 ? num + 0.5 : ceil(num - 0.5)) )
#define lroundf(num) ( (long)(num > 0 ? num + 0.5f : ceilf(num - 0.5f)) )
#define round(num) ( num > 0 ? floor(num + 0.5) : ceil(num - 0.5) )
#define roundf(num) ( num > 0 ? floorf(num + 0.5f) : ceilf(num - 0.5f) )
*/
#define signbit(num) ( _copysign(1.0, num) < 0 )
#define trunc(num) ( num > 0 ? floor(num) : ceil(num) )
#define nextafter(x, y) ( _nextafter(x, y) )
#define nextafterf(x, y) ( x > y ? x - FLT_EPSILON : x + FLT_EPSILON )
#define copysign(x, y) ( _copysign(x, y) )
#define isfinite(x) ( _finite(x) )

/*
 * Work around a bug in Win, where atan2(+-infinity, +-infinity)
 * yields NaN instead of specific values.
 */
/*
double
wtf_atan2(double x, double y)
{
    static double posInf = std::numeric_limits<double>::infinity();
    static double negInf = -std::numeric_limits<double>::infinity();
    static double nan = std::numeric_limits<double>::quiet_NaN();


    double result = nan;

    if (x == posInf && y == posInf)
        result = piOverFourDouble;
    else if (x == posInf && y == negInf)
        result = 3 * piOverFourDouble;
    else if (x == negInf && y == posInf)

        result = -piOverFourDouble;
    else if (x == negInf && y == negInf)
        result = -3 * piOverFourDouble;
    else
        result = ::atan2(x, y);

    return result;
}
*/

/*
 * Work around a bug in the Microsoft CRT, where fmod(x, +-infinity)
 * yields NaN instead of x.
 */
#define wtf_fmod(x, y) ( (!isinf(x) && isinf(y)) ? x : fmod(x, y) )

/*
 * Work around a bug in the Microsoft CRT, where pow(NaN, 0)
 * yields NaN instead of 1.
 */
#define wtf_pow(x, y) ( y == 0 ? 1 : pow(x, y) )

/*
#define atan2(x, y) wtf_atan2(x, y)
*/
#define fmod(x, y) wtf_fmod(x, y)
#define pow(x, y) wtf_pow(x, y)

#endif /* COMPILER(MSVC) */


#define deg2rad(d)  ( d * piDouble / 180.0 )
#define rad2deg(r)  ( r * 180.0 / piDouble )
#define deg2grad(d) ( d * 400.0 / 360.0 )
#define grad2deg(g) ( g * 360.0 / 400.0 )
#define rad2grad(r) ( r * 200.0 / piDouble )
#define grad2rad(g) ( g * piDouble / 200.0 )

#define deg2radf(d)  ( d * piFloat / 180.0f )
#define rad2degf(r)  ( r * 180.0f / piFloat )
#define deg2gradf(d) ( d * 400.0f / 360.0f )
#define grad2degf(g) ( g * 360.0f / 400.0f )
#define rad2gradf(r) ( r * 200.0f / piFloat )
#define grad2radf(g) ( g * piFloat / 200.0f )


#endif /* #ifndef WTF_MathExtras_h */
