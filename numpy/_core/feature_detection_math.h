/*
 * Prototypes for math functions probed as a batch by meson.build.
 *
 * Intentionally omits <math.h> and <stdlib.h> to avoid conflicting with
 * calling-convention-modified declarations (e.g. MSVC's __cdecl) or
 * macro-based declarations in system headers.
 */

double sin(double);
double cos(double);
double tan(double);
double sinh(double);
double cosh(double);
double tanh(double);
double fabs(double);
double floor(double);
double ceil(double);
double sqrt(double);
double log10(double);
double log(double);
double exp(double);
double asin(double);
double acos(double);
double atan(double);
double fmod(double, double);
double modf(double, double*);
double frexp(double, int*);
double ldexp(double, int);
double expm1(double);
double log1p(double);
double acosh(double);
double asinh(double);
double atanh(double);
double rint(double);
double trunc(double);
double exp2(double);
double copysign(double, double);
double nextafter(double, double);
double cbrt(double);
double log2(double);
double pow(double, double);
double hypot(double, double);
double atan2(double, double);

long long strtoll(const char*, char**, int);
unsigned long long strtoull(const char*, char**, int);
