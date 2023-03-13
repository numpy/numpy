#include <complex.h>

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#define cfloat _Fcomplex
#define cdouble _Dcomplex
#define cldouble _Lcomplex
#else
#define cfloat complex float
#define cdouble complex double
#define cldouble complex long double
#endif

// musl libc defines the following macros which breaks the declarations.
// See https://git.musl-libc.org/cgit/musl/tree/include/complex.h#n108
#undef crealf
#undef cimagf
#undef conjf
#undef creal
#undef cimag
#undef conj
#undef creall
#undef cimagl
#undef cconjl

cfloat csinf(cfloat);
cfloat ccosf(cfloat);
cfloat ctanf(cfloat);
cfloat csinhf(cfloat);
cfloat ccoshf(cfloat);
cfloat ctanhf(cfloat);
float crealf(cfloat);
float cimagf(cfloat);
cfloat conjf(cfloat);

cdouble csin(cdouble);
cdouble ccos(cdouble);
cdouble ctan(cdouble);
cdouble csinh(cdouble);
cdouble ccosh(cdouble);
cdouble ctanh(cdouble);
double creal(cdouble);
double cimag(cdouble);
cdouble conj(cdouble);

cldouble csinl(cldouble);
cldouble ccosl(cldouble);
cldouble ctanl(cldouble);
cldouble csinhl(cldouble);
cldouble ccoshl(cldouble);
cldouble ctanhl(cldouble);
long double creall(cldouble);
long double cimagl(cldouble);
cldouble conjl(cldouble);
