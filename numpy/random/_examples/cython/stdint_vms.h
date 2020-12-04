#ifndef __STDINT_LOADED
#define __STDINT_LOADED 1
/*************************************************************************
**
**  <stdint.h> - Fixed-width integer types
**
**************************************************************************
**  Header introduced by C99 standard (ISO/IEC 9899:1999)
*****************************************************************************
**                                                                          *
** HPE CONFIDENTIAL. This software is confidential proprietary software     *
** licensed by Hewlett Packard Enterprise Development, LP, and is not       *
** authorized to be used, duplicated or disclosed to anyone without the     *
** prior written permission of HPE.                                         *
** Copyright 2019 Hewlett Packard Enterprise Development, LP                *
**                                                                          *
** VMS SOFTWARE, INC. CONFIDENTIAL. This software is confidential           *
** proprietary software licensed by VMS Software, Inc., and is not          *
** authorized to be used, duplicated or disclosed to anyone without         *
** the prior written permission of VMS Software, Inc.                       *
** Copyright 2019 VMS Software, Inc.                                        *
**                                                                          *
*****************************************************************************
*/

#pragma __nostandard

#ifdef __cplusplus
extern "C" {
#endif

/*
**  ISO/IEC 9899:1999 Cor. 3:2007(E) Section 7.18
*/

/*
**  Signed integral types
*/
typedef signed char             int8_t;
typedef signed short            int16_t;
typedef signed int              int32_t;
typedef signed __int64          int64_t;

typedef signed char             int_least8_t;
typedef signed short            int_least16_t;
typedef signed int              int_least32_t;
typedef signed long long int    int_least64_t;

typedef signed char             int_fast8_t;
typedef signed int              int_fast16_t;
typedef signed int              int_fast32_t;
typedef signed long long int    int_fast64_t;
typedef signed long long int    intmax_t;

/*
**  Unsigned integral types
*/
typedef unsigned char           uint8_t;
typedef unsigned short          uint16_t;
typedef unsigned int            uint32_t;
typedef unsigned __int64        uint64_t;

typedef unsigned char           uint_least8_t;
typedef unsigned short          uint_least16_t;
typedef unsigned int            uint_least32_t;
typedef unsigned long long int  uint_least64_t;

typedef unsigned char           uint_fast8_t;
typedef unsigned int            uint_fast16_t;
typedef unsigned int            uint_fast32_t;
typedef unsigned long long int  uint_fast64_t;
typedef unsigned long long int  uintmax_t;

/*
**  Min constants
*/
#define INT8_MIN        (-127-1)
#define INT16_MIN       (-32767-1)
#define INT32_MIN       (-2147483647-1)
#define INT64_MIN       (-9223372036854775807LL-1)
#define INTMAX_MIN      INT64_MIN

#define INT_LEAST8_MIN  INT8_MIN
#define INT_LEAST16_MIN INT16_MIN
#define INT_LEAST32_MIN INT32_MIN
#define INT_LEAST64_MIN INT64_MIN

#define INT_FAST8_MIN   INT8_MIN
#define INT_FAST16_MIN  INT32_MIN
#define INT_FAST32_MIN  INT32_MIN
#define INT_FAST64_MIN  INT64_MIN

/*
**  Max constants
*/
#define INT8_MAX        (127)
#define INT16_MAX       (32767)
#define INT32_MAX       (2147483647)
#define INT64_MAX       (9223372036854775807LL)
#define INTMAX_MAX      INT64_MAX

#define INT_LEAST8_MAX  INT8_MAX
#define INT_LEAST16_MAX INT16_MAX
#define INT_LEAST32_MAX INT32_MAX
#define INT_LEAST64_MAX INT64_MAX

#define INT_FAST8_MAX   INT8_MAX
#define INT_FAST16_MAX  INT32_MAX
#define INT_FAST32_MAX  INT32_MAX
#define INT_FAST64_MAX  INT64_MAX

#define UINT8_MAX       (255)
#define UINT16_MAX      (65535)
#define UINT32_MAX      (4294967295UL)
#define UINT64_MAX      (18446744073709551615ULL)
#define UINTMAX_MAX     UINT64_MAX

#define UINT_LEAST8_MAX     UINT8_MAX
#define UINT_LEAST16_MAX    UINT16_MAX
#define UINT_LEAST32_MAX    UINT32_MAX
#define UINT_LEAST64_MAX    UINT64_MAX

#define UINT_FAST8_MAX      UINT8_MAX
#define UINT_FAST16_MAX     UINT32_MAX
#define UINT_FAST32_MAX     UINT32_MAX
#define UINT_FAST64_MAX     UINT64_MAX

/*
**  Pointer types
*/
typedef int64_t                 intptr_t;
typedef uint64_t                uintptr_t;

#define INTPTR_MIN      INT64_MIN
#define INTPTR_MAX      INT64_MAX
#define UINTPTR_MAX     UINT64_MAX

/*
**  Function Macros
*/
#define INT8_C(c)       (c)
#define INT16_C(c)      (c)
#define INT32_C(c)      (c)
#define INT64_C(c)      (c##LL)
#define INTMAX_C(c)     INT64_C(c)
#define UINT8_C(c)      (c)
#define UINT16_C(c)     (c)
#define UINT32_C(c)     (c##UL)
#define UINT64_C(c)     (c##ULL)
#define UINTMAX_C( c)    UINT64_C(c)

/*
**  Numeric Limits
*/
#define PTRDIFF_MIN     INT32_MIN
#define PTRDIFF_MAX     INT32_MAX
#define SIZE_MAX        UINT32_MAX
#define SIG_ATOMIC_MIN  INT32_MIN
#define SIG_ATOMIC_MAX  INT32_MAX

#ifdef __cplusplus
}
#endif

#pragma __standard

#endif
