/* This header is designed to be copy-pasted into downstream packages, since it provides
   a compatibility layer between the old C struct complex types and the new native C99
   complex types. The new macros are in libnpymath/npy_math.h, which is why it is included here. */
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_NPYMATHCOMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_NPYMATHCOMPAT_H_

#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION

#include <libnpymath/npy_math.h>

#define NPYMATH_GCDU npymath_gcdu
#define NPYMATH_LCMU npymath_lcmu
#define NPYMATH_GCDUL npymath_gcdul
#define NPYMATH_LCMUL npymath_lcmul
#define NPYMATH_GCDULL npymath_gcdull
#define NPYMATH_LCMULL npymath_lcmull

#define NPYMATH_GCD npymath_gcd
#define NPYMATH_LCM npymath_lcm
#define NPYMATH_GCDL npymath_gcdl
#define NPYMATH_LCML npymath_lcml
#define NPYMATH_GCDLL npymath_gcdll
#define NPYMATH_LCMLL npymath_lcmll

#define NPYMATH_RSHIFTUHH npymath_rshiftuhh
#define NPYMATH_LSHIFTUHH npymath_lshiftuhh
#define NPYMATH_RSHIFTUH npymath_rshiftuh
#define NPYMATH_LSHIFTUH npymath_lshiftuh
#define NPYMATH_RSHIFTU npymath_rshiftu
#define NPYMATH_LSHIFTU npymath_lshiftu
#define NPYMATH_RSHIFTUL npymath_rshiftul
#define NPYMATH_LSHIFTUL npymath_lshiftul
#define NPYMATH_RSHIFTULL npymath_rshiftull
#define NPYMATH_LSHIFTULL npymath_lshiftull

#define NPYMATH_RSHIFTHH npymath_rshifthh
#define NPYMATH_LSHIFTHH npymath_lshifthh
#define NPYMATH_RSHIFTH npymath_rshifth
#define NPYMATH_LSHIFTH npymath_lshifth
#define NPYMATH_RSHIFT npymath_rshift
#define NPYMATH_LSHIFT npymath_lshift
#define NPYMATH_RSHIFTL npymath_rshiftl
#define NPYMATH_LSHIFTL npymath_lshiftl
#define NPYMATH_RSHIFTLL npymath_rshiftll
#define NPYMATH_LSHIFTLL npymath_lshiftll

#define NPYMATH_POPCOUNTUHH npymath_popcountuhh
#define NPYMATH_POPCOUNTUH npymath_popcountuh
#define NPYMATH_POPCOUNTU npymath_popcountu
#define NPYMATH_POPCOUNTUL npymath_popcountul
#define NPYMATH_POPCOUNTULL npymath_popcountull
#define NPYMATH_POPCOUNTHH npymath_popcounthh
#define NPYMATH_POPCOUNTH npymath_popcounth
#define NPYMATH_POPCOUNT npymath_popcount
#define NPYMATH_POPCOUNTL npymath_popcountl
#define NPYMATH_POPCOUNTLL npymath_popcountll

#define NPYMATH_SIN npymath_sin
#define NPYMATH_COS npymath_cos
#define NPYMATH_TAN npymath_tan
#define NPYMATH_HYPOT npymath_hypot
#define NPYMATH_LOG2 npymath_log2
#define NPYMATH_ATAN2 npymath_atan2

#define NPYMATH_SPACING npymath_spacing

#define NPYMATH_SINF npymath_sinf
#define NPYMATH_COSF npymath_cosf
#define NPYMATH_TANF npymath_tanf
#define NPYMATH_EXPF npymath_expf
#define NPYMATH_SQRTF npymath_sqrtf
#define NPYMATH_HYPOTF npymath_hypotf
#define NPYMATH_LOG2F npymath_log2f
#define NPYMATH_ATAN2F npymath_atan2f
#define NPYMATH_POWF npymath_powf
#define NPYMATH_MODFF npymath_modff

#define NPYMATH_SPACINGF npymath_spacingf

#define NPYMATH_SINL npymath_sinl
#define NPYMATH_COSL npymath_cosl
#define NPYMATH_TANL npymath_tanl
#define NPYMATH_EXPL npymath_expl
#define NPYMATH_SQRTL npymath_sqrtl
#define NPYMATH_HYPOTL npymath_hypotl
#define NPYMATH_LOG2L npymath_log2l
#define NPYMATH_ATAN2L npymath_atan2l
#define NPYMATH_POWL npymath_powl
#define NPYMATH_MODFL npymath_modfl

#define NPYMATH_SPACINGL npymath_spacingl

#define NPYMATH_DEG2RAD npymath_deg2rad
#define NPYMATH_RAD2DEG npymath_rad2deg
#define NPYMATH_LOGADDEXP npymath_logaddexp
#define NPYMATH_LOGADDEXP2 npymath_logaddexp2
#define NPYMATH_DIVMOD npymath_divmod
#define NPYMATH_HEAVISIDE npymath_heaviside

#define NPYMATH_DEG2RADF npymath_deg2radf
#define NPYMATH_RAD2DEGF npymath_rad2degf
#define NPYMATH_LOGADDEXPF npymath_logaddexpf
#define NPYMATH_LOGADDEXP2F npymath_logaddexp2f
#define NPYMATH_DIVMODF npymath_divmodf
#define NPYMATH_HEAVISIDEF npymath_heavisidef

#define NPYMATH_DEG2RADL npymath_deg2radl
#define NPYMATH_RAD2DEGL npymath_rad2degl
#define NPYMATH_LOGADDEXPL npymath_logaddexpl
#define NPYMATH_LOGADDEXP2L npymath_logaddexp2l
#define NPYMATH_DIVMODL npymath_divmodl
#define NPYMATH_HEAVISIDEL npymath_heavisidel

#define NPYMATH_CREAL npymath_creal
#define NPYMATH_CSETREAL npymath_csetreal
#define NPYMATH_CIMAG npymath_cimag
#define NPYMATH_CSETIMAG npymath_csetimag
#define NPYMATH_CREALF npymath_crealf
#define NPYMATH_CSETREALF npymath_csetrealf
#define NPYMATH_CIMAGF npymath_cimagf
#define NPYMATH_CSETIMAGF npymath_csetimagf
#define NPYMATH_CREALL npymath_creall
#define NPYMATH_CSETREALL npymath_csetreall
#define NPYMATH_CIMAGL npymath_cimagl
#define NPYMATH_CSETIMAGL npymath_csetimagl

#define NPYMATH_CPACK npymath_cpack
#define NPYMATH_CPACKF npymath_cpackf
#define NPYMATH_CPACKL npymath_cpackl

#define NPYMATH_CABS npymath_cabs
#define NPYMATH_CARG npymath_carg

#define NPYMATH_CEXP npymath_cexp
#define NPYMATH_CLOG npymath_clog
#define NPYMATH_CPOW npymath_cpow

#define NPYMATH_CSQRT npymath_csqrt

#define NPYMATH_CCOS npymath_ccos
#define NPYMATH_CSIN npymath_csin
#define NPYMATH_CTAN npymath_ctan

#define NPYMATH_CCOSH npymath_ccosh
#define NPYMATH_CSINH npymath_csinh
#define NPYMATH_CTANH npymath_ctanh

#define NPYMATH_CACOS npymath_cacos
#define NPYMATH_CASIN npymath_casin
#define NPYMATH_CATAN npymath_catan

#define NPYMATH_CACOSH npymath_cacosh
#define NPYMATH_CASINH npymath_casinh
#define NPYMATH_CATANH npymath_catanh

#define NPYMATH_CABSF npymath_cabsf
#define NPYMATH_CARGF npymath_cargf

#define NPYMATH_CEXPF npymath_cexpf
#define NPYMATH_CLOGF npymath_clogf
#define NPYMATH_CPOWF npymath_cpowf

#define NPYMATH_CSQRTF npymath_csqrtf

#define NPYMATH_CCOSF npymath_ccosf
#define NPYMATH_CSINF npymath_csinf
#define NPYMATH_CTANF npymath_ctanf

#define NPYMATH_CCOSHF npymath_ccoshf
#define NPYMATH_CSINHF npymath_csinhf
#define NPYMATH_CTANHF npymath_ctanhf

#define NPYMATH_CACOSF npymath_cacosf
#define NPYMATH_CASINF npymath_casinf
#define NPYMATH_CATANF npymath_catanf

#define NPYMATH_CACOSHF npymath_cacoshf
#define NPYMATH_CASINHF npymath_casinhf
#define NPYMATH_CATANHF npymath_catanhf

#define NPYMATH_CABSL npymath_cabsl
#define NPYMATH_CARGL npymath_cargl

#define NPYMATH_CEXPL npymath_cexpl
#define NPYMATH_CLOGL npymath_clogl
#define NPYMATH_CPOWL npymath_cpowl

#define NPYMATH_CSQRTL npymath_csqrtl

#define NPYMATH_CCOSL npymath_ccosl
#define NPYMATH_CSINL npymath_csinl
#define NPYMATH_CTANL npymath_ctanl

#define NPYMATH_CCOSHL npymath_ccoshl
#define NPYMATH_CSINHL npymath_csinhl
#define NPYMATH_CTANHL npymath_ctanhl

#define NPYMATH_CACOSL npymath_cacosl
#define NPYMATH_CASINL npymath_casinl
#define NPYMATH_CATANL npymath_catanl

#define NPYMATH_CACOSHL npymath_cacoshl
#define NPYMATH_CASINHL npymath_casinhl
#define NPYMATH_CATANHL npymath_catanhl

#define NPYMATH_CLEAR_FLOATSTATUS_BARRIER npymath_clear_floatstatus_barrier
#define NPYMATH_GET_FLOATSTATUS_BARRIER npymath_get_floatstatus_barrier

#define NPYMATH_CLEAR_FLOATSTATUS npymath_clear_floatstatus
#define NPYMATH_GET_FLOATSTATUS npymath_get_floatstatus

#define NPYMATH_SET_FLOATSTATUS_DIVBYZERO npymath_set_floatstatus_divbyzero
#define NPYMATH_SET_FLOATSTATUS_OVERFLOW npymath_set_floatstatus_overflow
#define NPYMATH_SET_FLOATSTATUS_UNDERFLOW npymath_set_floatstatus_underflow
#define NPYMATH_SET_FLOATSTATUS_INVALID npymath_set_floatstatus_invalid

#define NPYMATH_HALF_TO_FLOAT npymath_half_to_float
#define NPYMATH_HALF_TO_DOUBLE npymath_half_to_double
#define NPYMATH_FLOAT_TO_HALF npymath_float_to_half
#define NPYMATH_DOUBLE_TO_HALF npymath_double_to_half

#define NPYMATH_HALF_EQ npymath_half_eq
#define NPYMATH_HALF_NE npymath_half_ne
#define NPYMATH_HALF_LE npymath_half_le
#define NPYMATH_HALF_LT npymath_half_lt
#define NPYMATH_HALF_GE npymath_half_ge
#define NPYMATH_HALF_GT npymath_half_gt

#define NPYMATH_HALF_EQ_NONAN npymath_half_eq_nonan
#define NPYMATH_HALF_LT_NONAN npymath_half_lt_nonan
#define NPYMATH_HALF_LE_NONAN npymath_half_le_nonan

#define NPYMATH_HALF_ISZERO npymath_half_iszero
#define NPYMATH_HALF_ISNAN npymath_half_isnan
#define NPYMATH_HALF_ISINF npymath_half_isinf
#define NPYMATH_HALF_ISFINITE npymath_half_isfinite
#define NPYMATH_HALF_SIGNBIT npymath_half_signbit
#define NPYMATH_HALF_COPYSIGN npymath_half_copysign
#define NPYMATH_HALF_SPACING npymath_half_spacing
#define NPYMATH_HALF_NEXTAFTER npymath_half_nextafter
#define NPYMATH_HALF_DIVMOD npymath_half_divmod

#define NPYMATH_FLOATBITS_TO_HALFBITS npymath_floatbits_to_halfbits
#define NPYMATH_DOUBLEBITS_TO_HALFBITS npymath_doublebits_to_halfbits
#define NPYMATH_HALFBITS_TO_FLOATBITS npymath_halfbits_to_floatbits
#define NPYMATH_HALFBITS_TO_DOUBLEBITS npymath_halfbits_to_doublebits

#else /* NPY_FEATURE_VERSION < NPY_2_0_API_VERSION */

#include <numpy/npymath.h>

#define NPYMATH_INFINITYF NPY_INFINITYF
#define NPYMATH_NANF NPY_NANF
#define NPYMATH_PZEROF NPY_PZEROF
#define NPYMATH_NZEROF NPY_NZEROF

#define NPYMATH_INFINITY NPY_INFINITY
#define NPYMATH_NAN NPY_NAN
#define NPYMATH_PZERO NPY_PZERO
#define NPYMATH_NZERO NPY_NZERO

#define NPYMATH_INFINITYL NPY_INFINITYL
#define NPYMATH_NANL NPY_NANL
#define NPYMATH_PZEROL NPY_PZEROL
#define NPYMATH_NZEROL NPY_NZEROL

#define NPYMATH_E NPY_E
#define NPYMATH_LOG2E NPY_LOG2E
#define NPYMATH_LOG10E NPY_LOG10E
#define NPYMATH_LOGE2 NPY_LOGE2
#define NPYMATH_LOGE10 NPY_LOGE10
#define NPYMATH_PI NPY_PI
#define NPYMATH_PI_2 NPY_PI_2
#define NPYMATH_PI_4 NPY_PI_4
#define NPYMATH_1_PI NPY_1_PI
#define NPYMATH_2_PI NPY_2_PI
#define NPYMATH_EULER NPY_EULER
#define NPYMATH_SQRT2 NPY_SQRT2
#define NPYMATH_SQRT1_2 NPY_SQRT1_2

#define NPYMATH_Ef NPY_Ef
#define NPYMATH_LOG2Ef NPY_LOG2Ef
#define NPYMATH_LOG10Ef NPY_LOG10Ef
#define NPYMATH_LOGE2f NPY_LOGE2f
#define NPYMATH_LOGE10f NPY_LOGE10f
#define NPYMATH_PIf NPY_PIf
#define NPYMATH_PI_2f NPY_PI_2f
#define NPYMATH_PI_4f NPY_PI_4f
#define NPYMATH_1_PIf NPY_1_PIf
#define NPYMATH_2_PIf NPY_2_PIf
#define NPYMATH_EULERf NPY_EULERf
#define NPYMATH_SQRT2f NPY_SQRT2f
#define NPYMATH_SQRT1_2f NPY_SQRT1_2f

#define NPYMATH_El NPY_El
#define NPYMATH_LOG2El NPY_LOG2El
#define NPYMATH_LOG10El NPY_LOG10El
#define NPYMATH_LOGE2l NPY_LOGE2l
#define NPYMATH_LOGE10l NPY_LOGE10l
#define NPYMATH_PIl NPY_PIl
#define NPYMATH_PI_2l NPY_PI_2l
#define NPYMATH_PI_4l NPY_PI_4l
#define NPYMATH_1_PIl NPY_1_PIl
#define NPYMATH_2_PIl NPY_2_PIl
#define NPYMATH_EULERl NPY_EULERl
#define NPYMATH_SQRT2l NPY_SQRT2l
#define NPYMATH_SQRT1_2l NPY_SQRT1_2l

#define NPYMATH_GCDU npy_gcdu
#define NPYMATH_LCMU npy_lcmu
#define NPYMATH_GCDUL npy_gcdul
#define NPYMATH_LCMUL npy_lcmul
#define NPYMATH_GCDULL npy_gcdull
#define NPYMATH_LCMULL npy_lcmull

#define NPYMATH_GCD npy_gcd
#define NPYMATH_LCM npy_lcm
#define NPYMATH_GCDL npy_gcdl
#define NPYMATH_LCML npy_lcml
#define NPYMATH_GCDLL npy_gcdll
#define NPYMATH_LCMLL npy_lcmll

#define NPYMATH_RSHIFTUHH npy_rshiftuhh
#define NPYMATH_LSHIFTUHH npy_lshiftuhh
#define NPYMATH_RSHIFTUH npy_rshiftuh
#define NPYMATH_LSHIFTUH npy_lshiftuh
#define NPYMATH_RSHIFTU npy_rshiftu
#define NPYMATH_LSHIFTU npy_lshiftu
#define NPYMATH_RSHIFTUL npy_rshiftul
#define NPYMATH_LSHIFTUL npy_lshiftul
#define NPYMATH_RSHIFTULL npy_rshiftull
#define NPYMATH_LSHIFTULL npy_lshiftull

#define NPYMATH_RSHIFTHH npy_rshifthh
#define NPYMATH_LSHIFTHH npy_lshifthh
#define NPYMATH_RSHIFTH npy_rshifth
#define NPYMATH_LSHIFTH npy_lshifth
#define NPYMATH_RSHIFT npy_rshift
#define NPYMATH_LSHIFT npy_lshift
#define NPYMATH_RSHIFTL npy_rshiftl
#define NPYMATH_LSHIFTL npy_lshiftl
#define NPYMATH_RSHIFTLL npy_rshiftll
#define NPYMATH_LSHIFTLL npy_lshiftll

#define NPYMATH_POPCOUNTUHH npy_popcountuhh
#define NPYMATH_POPCOUNTUH npy_popcountuh
#define NPYMATH_POPCOUNTU npy_popcountu
#define NPYMATH_POPCOUNTUL npy_popcountul
#define NPYMATH_POPCOUNTULL npy_popcountull
#define NPYMATH_POPCOUNTHH npy_popcounthh
#define NPYMATH_POPCOUNTH npy_popcounth
#define NPYMATH_POPCOUNT npy_popcount
#define NPYMATH_POPCOUNTL npy_popcountl
#define NPYMATH_POPCOUNTLL npy_popcountll

#define NPYMATH_SIN npy_sin
#define NPYMATH_COS npy_cos
#define NPYMATH_TAN npy_tan
#define NPYMATH_HYPOT npy_hypot
#define NPYMATH_LOG2 npy_log2
#define NPYMATH_ATAN2 npy_atan2

#define npymath_sinh npy_sinh
#define npymath_cosh npy_cosh
#define npymath_tanh npy_tanh
#define npymath_asin npy_asin
#define npymath_acos npy_acos
#define npymath_atan npy_atan
#define npymath_log npy_log
#define npymath_log10 npy_log10
#define npymath_cbrt npy_cbrt
#define npymath_fabs npy_fabs
#define npymath_ceil npy_ceil
#define npymath_fmod npy_fmod
#define npymath_floor npy_floor
#define npymath_expm1 npy_expm1
#define npymath_log1p npy_log1p
#define npymath_acosh npy_acosh
#define npymath_asinh npy_asinh
#define npymath_atanh npy_atanh
#define npymath_rint npy_rint
#define npymath_trunc npy_trunc
#define npymath_exp2 npy_exp2
#define npymath_frexp npy_frexp
#define npymath_ldexp npy_ldexp
#define npymath_copysign npy_copysign
#define npymath_exp npy_exp
#define npymath_sqrt npy_sqrt
#define npymath_pow npy_pow
#define npymath_modf npy_modf
#define npymath_nextafter npy_nextafter

#define NPYMATH_SPACING npy_spacing

#define npymath_isnan npy_isnan
#define npymath_isfinite npy_isfinite
#define npymath_isinf npy_isinf
#define npymath_signbit npy_signbit

#define NPYMATH_SINF npy_sinf
#define NPYMATH_COSF npy_cosf
#define NPYMATH_TANF npy_tanf
#define NPYMATH_EXPF npy_expf
#define NPYMATH_SQRTF npy_sqrtf
#define NPYMATH_HYPOTF npy_hypotf
#define NPYMATH_LOG2F npy_log2f
#define NPYMATH_ATAN2F npy_atan2f
#define NPYMATH_POWF npy_powf
#define NPYMATH_MODFF npy_modff

#define npymath_sinhf npy_sinhf
#define npymath_coshf npy_coshf
#define npymath_tanhf npy_tanhf
#define npymath_asinf npy_asinf
#define npymath_acosf npy_acosf
#define npymath_atanf npy_atanf
#define npymath_logf npy_logf
#define npymath_log10f npy_log10f
#define npymath_cbrtf npy_cbrtf
#define npymath_fabsf npy_fabsf
#define npymath_ceilf npy_ceilf
#define npymath_fmodf npy_fmodf
#define npymath_floorf npy_floorf
#define npymath_expm1f npy_expm1f
#define npymath_log1pf npy_log1pf
#define npymath_asinhf npy_asinhf
#define npymath_acoshf npy_acoshf
#define npymath_atanhf npy_atanhf
#define npymath_rintf npy_rintf
#define npymath_truncf npy_truncf
#define npymath_exp2f npy_exp2f
#define npymath_frexpf npy_frexpf
#define npymath_ldexpf npy_ldexpf
#define npymath_copysignf npy_copysignf
#define npymath_nextafterf npy_nextafterf

#define NPYMATH_SPACINGF npy_spacingf

#define NPYMATH_SINL npy_sinl
#define NPYMATH_COSL npy_cosl
#define NPYMATH_TANL npy_tanl
#define NPYMATH_EXPL npy_expl
#define NPYMATH_SQRTL npy_sqrtl
#define NPYMATH_HYPOTL npy_hypotl
#define NPYMATH_LOG2L npy_log2l
#define NPYMATH_ATAN2L npy_atan2l
#define NPYMATH_POWL npy_powl
#define NPYMATH_MODFL npy_modfl

#define npymath_sinhl npy_sinhl
#define npymath_coshl npy_coshl
#define npymath_tanhl npy_tanhl
#define npymath_fabsl npy_fabsl
#define npymath_floorl npy_floorl
#define npymath_ceill npy_ceill
#define npymath_rintl npy_rintl
#define npymath_truncl npy_truncl
#define npymath_cbrtl npy_cbrtl
#define npymath_log10l npy_log10l
#define npymath_logl npy_logl
#define npymath_expm1l npy_expm1l
#define npymath_asinl npy_asinl
#define npymath_acosl npy_acosl
#define npymath_atanl npy_atanl
#define npymath_asinhl npy_asinhl
#define npymath_acoshl npy_acoshl
#define npymath_atanhl npy_atanhl
#define npymath_log1pl npy_log1pl
#define npymath_exp2l npy_exp2l
#define npymath_fmodl npy_fmodl
#define npymath_frexpl npy_frexpl
#define npymath_ldexpl npy_ldexpl
#define npymath_copysignl npy_copysignl
#define npymath_nextafterl npy_nextafterl

#define NPYMATH_SPACINGL npy_spacingl

#define NPYMATH_DEG2RAD npy_deg2rad
#define NPYMATH_RAD2DEG npy_rad2deg
#define NPYMATH_LOGADDEXP npy_logaddexp
#define NPYMATH_LOGADDEXP2 npy_logaddexp2
#define NPYMATH_DIVMOD npy_divmod
#define NPYMATH_HEAVISIDE npy_heaviside

#define NPYMATH_DEG2RADF npy_deg2radf
#define NPYMATH_RAD2DEGF npy_rad2degf
#define NPYMATH_LOGADDEXPF npy_logaddexpf
#define NPYMATH_LOGADDEXP2F npy_logaddexp2f
#define NPYMATH_DIVMODF npy_divmodf
#define NPYMATH_HEAVISIDEF npy_heavisidef

#define NPYMATH_DEG2RADL npy_deg2radl
#define NPYMATH_RAD2DEGL npy_rad2degl
#define NPYMATH_LOGADDEXPL npy_logaddexpl
#define NPYMATH_LOGADDEXP2L npy_logaddexp2l
#define NPYMATH_DIVMODL npy_divmodl
#define NPYMATH_HEAVISIDEL npy_heavisidel

#define npymath_degrees npy_rad2deg
#define npymath_degreesf npy_rad2degf
#define npymath_degreesl npy_rad2degl

#define npymath_radians npy_deg2rad
#define npymath_radiansf npy_deg2radf
#define npymath_radiansl npy_deg2radl

#define NPYMATH_CREAL npy_creal
#define NPYMATH_CSETREAL(c, i) (c)->real = (i)
#define NPYMATH_CIMAG npy_cimag
#define NPYMATH_CSETIMAG(c, i) (c)->imag = (i)
#define NPYMATH_CREALF npy_crealf
#define NPYMATH_CSETREALF(c, i) (c)->real = (i)
#define NPYMATH_CIMAGF npy_cimagf
#define NPYMATH_CSETIMAGF(c, i) (c)->imag = (i)
#define NPYMATH_CREALL npy_creall
#define NPYMATH_CSETREALL(c, i) (c)->real = (i)
#define NPYMATH_CIMAGL npy_cimagl
#define NPYMATH_CSETIMAGL(c, i) (c)->imag = (i)

#define NPYMATH_CPACK npy_cpack
#define NPYMATH_CPACKF npy_cpackf
#define NPYMATH_CPACKL npy_cpackl

#define NPYMATH_CABS npy_cabs
#define NPYMATH_CARG npy_carg

#define NPYMATH_CEXP npy_cexp
#define NPYMATH_CLOG npy_clog
#define NPYMATH_CPOW npy_cpow

#define NPYMATH_CSQRT npy_csqrt

#define NPYMATH_CCOS npy_ccos
#define NPYMATH_CSIN npy_csin
#define NPYMATH_CTAN npy_ctan

#define NPYMATH_CCOSH npy_ccosh
#define NPYMATH_CSINH npy_csinh
#define NPYMATH_CTANH npy_ctanh

#define NPYMATH_CACOS npy_cacos
#define NPYMATH_CASIN npy_casin
#define NPYMATH_CATAN npy_catan

#define NPYMATH_CACOSH npy_cacosh
#define NPYMATH_CASINH npy_casinh
#define NPYMATH_CATANH npy_catanh

#define NPYMATH_CABSF npy_cabsf
#define NPYMATH_CARGF npy_cargf

#define NPYMATH_CEXPF npy_cexpf
#define NPYMATH_CLOGF npy_clogf
#define NPYMATH_CPOWF npy_cpowf

#define NPYMATH_CSQRTF npy_csqrtf

#define NPYMATH_CCOSF npy_ccosf
#define NPYMATH_CSINF npy_csinf
#define NPYMATH_CTANF npy_ctanf

#define NPYMATH_CCOSHF npy_ccoshf
#define NPYMATH_CSINHF npy_csinhf
#define NPYMATH_CTANHF npy_ctanhf

#define NPYMATH_CACOSF npy_cacosf
#define NPYMATH_CASINF npy_casinf
#define NPYMATH_CATANF npy_catanf

#define NPYMATH_CACOSHF npy_cacoshf
#define NPYMATH_CASINHF npy_casinhf
#define NPYMATH_CATANHF npy_catanhf

#define NPYMATH_CABSL npy_cabsl
#define NPYMATH_CARGL npy_cargl

#define NPYMATH_CEXPL npy_cexpl
#define NPYMATH_CLOGL npy_clogl
#define NPYMATH_CPOWL npy_cpowl

#define NPYMATH_CSQRTL npy_csqrtl

#define NPYMATH_CCOSL npy_ccosl
#define NPYMATH_CSINL npy_csinl
#define NPYMATH_CTANL npy_ctanl

#define NPYMATH_CCOSHL npy_ccoshl
#define NPYMATH_CSINHL npy_csinhl
#define NPYMATH_CTANHL npy_ctanhl

#define NPYMATH_CACOSL npy_cacosl
#define NPYMATH_CASINL npy_casinl
#define NPYMATH_CATANL npy_catanl

#define NPYMATH_CACOSHL npy_cacoshl
#define NPYMATH_CASINHL npy_casinhl
#define NPYMATH_CATANHL npy_catanhl

#define NPYMATH_FPE_DIVIDEBYZERO  NPY_FPE_DIVIDEBYZERO
#define NPYMATH_FPE_OVERFLOW      NPY_FPE_OVERFLOW
#define NPYMATH_FPE_UNDERFLOW     NPY_FPE_UNDERFLOW
#define NPYMATH_FPE_INVALID       NPY_FPE_INVALID

#define NPYMATH_CLEAR_FLOATSTATUS_BARRIER npy_clear_floatstatus_barrier
#define NPYMATH_GET_FLOATSTATUS_BARRIER npy_get_floatstatus_barrier

#define NPYMATH_CLEAR_FLOATSTATUS npy_clear_floatstatus
#define NPYMATH_GET_FLOATSTATUS npy_get_floatstatus

#define NPYMATH_SET_FLOATSTATUS_DIVBYZERO npy_set_floatstatus_divbyzero
#define NPYMATH_SET_FLOATSTATUS_OVERFLOW npy_set_floatstatus_overflow
#define NPYMATH_SET_FLOATSTATUS_UNDERFLOW npy_set_floatstatus_underflow
#define NPYMATH_SET_FLOATSTATUS_INVALID npy_set_floatstatus_invalid

#define NPYMATH_HALF_TO_FLOAT npy_half_to_float
#define NPYMATH_HALF_TO_DOUBLE npy_half_to_double
#define NPYMATH_FLOAT_TO_HALF npy_float_to_half
#define NPYMATH_DOUBLE_TO_HALF npy_double_to_half

#define NPYMATH_HALF_EQ npy_half_eq
#define NPYMATH_HALF_NE npy_half_ne
#define NPYMATH_HALF_LE npy_half_le
#define NPYMATH_HALF_LT npy_half_lt
#define NPYMATH_HALF_GE npy_half_ge
#define NPYMATH_HALF_GT npy_half_gt

#define NPYMATH_HALF_EQ_NONAN npy_half_eq_nonan
#define NPYMATH_HALF_LT_NONAN npy_half_lt_nonan
#define NPYMATH_HALF_LE_NONAN npy_half_le_nonan

#define NPYMATH_HALF_ISZERO npy_half_iszero
#define NPYMATH_HALF_ISNAN npy_half_isnan
#define NPYMATH_HALF_ISINF npy_half_isinf
#define NPYMATH_HALF_ISFINITE npy_half_isfinite
#define NPYMATH_HALF_SIGNBIT npy_half_signbit
#define NPYMATH_HALF_COPYSIGN npy_half_copysign
#define NPYMATH_HALF_SPACING npy_half_spacing
#define NPYMATH_HALF_NEXTAFTER npy_half_nextafter
#define NPYMATH_HALF_DIVMOD npy_half_divmod

#define NPYMATH_HALF_ZERO   NPY_HALF_ZERO
#define NPYMATH_HALF_PZERO  NPY_HALF_PZERO
#define NPYMATH_HALF_NZERO  NPY_HALF_NZERO
#define NPYMATH_HALF_ONE    NPY_HALF_ONE
#define NPYMATH_HALF_NEGONE NPY_HALF_NEGONE
#define NPYMATH_HALF_PINF   NPY_HALF_PINF
#define NPYMATH_HALF_NINF   NPY_HALF_NINF
#define NPYMATH_HALF_NAN    NPY_HALF_NAN

#define NPYMATH_MAX_HALF    NPYMATH_MAX_HALF

#define NPYMATH_FLOATBITS_TO_HALFBITS npy_floatbits_to_halfbits
#define NPYMATH_DOUBLEBITS_TO_HALFBITS npy_doublebits_to_halfbits
#define NPYMATH_HALFBITS_TO_FLOATBITS npy_halfbits_to_floatbits
#define NPYMATH_HALFBITS_TO_DOUBLEBITS npy_halfbits_to_doublebits

#endif /* NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION */

#endif /* NUMPY_CORE_INCLUDE_NUMPY_NPY_2_NPYMATHCOMPAT_H_ */
