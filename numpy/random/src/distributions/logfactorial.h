
#ifndef LOGFACTORIAL_H
#define LOGFACTORIAL_H

#ifdef __VMS
#if __CRTL_VER > 80400000
#include <stdint.h>
#else
#include <inttypes.h>
typedef signed char             int8_t;
typedef signed short            int16_t;
typedef signed int              int32_t;
typedef signed char             int_least8_t;
typedef signed short            int_least16_t;
typedef signed int              int_least32_t;
typedef signed long long int    int_least64_t;
typedef signed char             int_fast8_t;
typedef signed int              int_fast16_t;
typedef signed int              int_fast32_t;
typedef signed long long int    int_fast64_t;
typedef signed long long int    intmax_t;
typedef unsigned char           uint8_t;
typedef unsigned short          uint16_t;
typedef unsigned int            uint32_t;
typedef unsigned char           uint_least8_t;
typedef unsigned short          uint_least16_t;
typedef unsigned int            uint_least32_t;
typedef unsigned long long int  uint_least64_t;
typedef unsigned char           uint_fast8_t;
typedef unsigned int            uint_fast16_t;
typedef unsigned int            uint_fast32_t;
typedef unsigned long long int  uint_fast64_t;
typedef unsigned long long int  uintmax_t;
#endif
#else
#include <stdint.h>
#endif


double logfactorial(int64_t k);

#endif
