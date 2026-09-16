/* #undef SIZEOF_PY_INTPTR_T */
/* #undef SIZEOF_OFF_T */
/* #undef SIZEOF_PY_LONG_LONG */

#define HAVE_BACKTRACE 1
#define HAVE_MADVISE 1
/* #undef HAVE_FTELLO */
/* #undef HAVE_FSEEKO */
#define HAVE_FALLOCATE 1
#define HAVE_STRTOLD_L 1
/* #undef HAVE_THREAD_LOCAL */
#define HAVE__THREAD_LOCAL 1
#define HAVE__THREAD 1
/* #undef HAVE___DECLSPEC_THREAD_ */

/* Optional headers */
#define HAVE_FEATURES_H 1
/* #undef HAVE_XLOCALE_H */
#define HAVE_DLFCN_H 1
#define HAVE_EXECINFO_H 1
#define HAVE_LIBUNWIND_H 1
#define HAVE_SYS_MMAN_H 1
/* #undef HAVE_XMMINTRIN_H */
/* #undef HAVE_EMMINTRIN_H */
/* #undef HAVE_IMMINTRIN_H */

/* Optional intrinsics */
#define HAVE___BUILTIN_ISNAN 1
#define HAVE___BUILTIN_ISINF 1
#define HAVE___BUILTIN_ISFINITE 1
#define HAVE___BUILTIN_BSWAP32 1
#define HAVE___BUILTIN_BSWAP64 1
#define HAVE___BUILTIN_EXPECT 1
#define HAVE___BUILTIN_MUL_OVERFLOW 1
#define HAVE___BUILTIN_ADD_OVERFLOW 1
#define HAVE___BUILTIN_SUB_OVERFLOW 1
#define HAVE___BUILTIN_PREFETCH 1

#define HAVE_ATTRIBUTE_OPTIMIZE_UNROLL_LOOPS 1
#define HAVE_ATTRIBUTE_OPTIMIZE_OPT_3 1
/* #undef HAVE_ATTRIBUTE_OPTIMIZE_OPT_2 */
#define HAVE_ATTRIBUTE_NONNULL 1

/* C99 complex support and complex.h are not universal */
#define HAVE_CABS 1
#define HAVE_CACOS 1
#define HAVE_CACOSH 1
#define HAVE_CARG 1
#define HAVE_CASIN 1
#define HAVE_CASINH 1
#define HAVE_CATAN 1
#define HAVE_CATANH 1
#define HAVE_CEXP 1
#define HAVE_CLOG 1
#define HAVE_CPOW 1
#define HAVE_CSQRT 1
#define HAVE_CABSF 1
#define HAVE_CACOSF 1
#define HAVE_CACOSHF 1
#define HAVE_CARGF 1
#define HAVE_CASINF 1
#define HAVE_CASINHF 1
#define HAVE_CATANF 1
#define HAVE_CATANHF 1
#define HAVE_CEXPF 1
#define HAVE_CLOGF 1
#define HAVE_CPOWF 1
#define HAVE_CSQRTF 1
#define HAVE_CABSL 1
#define HAVE_CACOSL 1
#define HAVE_CACOSHL 1
#define HAVE_CARGL 1
#define HAVE_CASINL 1
#define HAVE_CASINHL 1
#define HAVE_CATANL 1
#define HAVE_CATANHL 1
#define HAVE_CEXPL 1
#define HAVE_CLOGL 1
#define HAVE_CPOWL 1
#define HAVE_CSQRTL 1
/* FreeBSD */
#define HAVE_CSINF 1
#define HAVE_CSINHF 1
#define HAVE_CCOSF 1
#define HAVE_CCOSHF 1
#define HAVE_CTANF 1
#define HAVE_CTANHF 1
#define HAVE_CSIN 1
#define HAVE_CSINH 1
#define HAVE_CCOS 1
#define HAVE_CCOSH 1
#define HAVE_CTAN 1
#define HAVE_CTANH 1
#define HAVE_CSINL 1
#define HAVE_CSINHL 1
#define HAVE_CCOSL 1
#define HAVE_CCOSHL 1
#define HAVE_CTANL 1
#define HAVE_CTANHL 1

#define NPY_CAN_LINK_SVML 1

#define HAVE_LDOUBLE_INTEL_EXTENDED_16_BYTES_LE 1
/* #undef HAVE_LDOUBLE_INTEL_EXTENDED_12_BYTES_LE */
/* #undef HAVE_LDOUBLE_MOTOROLA_EXTENDED_12_BYTES_BE */
/* #undef HAVE_LDOUBLE_IEEE_DOUBLE_LE */
/* #undef HAVE_LDOUBLE_IEEE_DOUBLE_BE */
/* #undef HAVE_LDOUBLE_IEEE_QUAD_LE */
/* #undef HAVE_LDOUBLE_IEEE_QUAD_BE */
/* #undef HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_LE */
/* #undef HAVE_LDOUBLE_IBM_DOUBLE_DOUBLE_BE */

#define HAVE_EXTERNAL_LAPACK 1

#ifndef __cplusplus
/* #undef inline */
#endif

#ifndef NUMPY_CORE_SRC_COMMON_NPY_CONFIG_H_
#error config.h should never be included directly, include npy_config.h instead
#endif
