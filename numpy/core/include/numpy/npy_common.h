#ifndef _NPY_COMMON_H_
#define _NPY_COMMON_H_

/* This is auto-generated */
#include "numpyconfig.h"

/* enums for detected endianness */
enum {
	NPY_CPU_UNKNOWN_ENDIAN,
	NPY_CPU_LITTLE,
	NPY_CPU_BIG,
};

/* Some platforms don't define bool, long long, or long double.
   Handle that here.
*/

#define NPY_BYTE_FMT "hhd"
#define NPY_UBYTE_FMT "hhu"
#define NPY_SHORT_FMT "hd"
#define NPY_USHORT_FMT "hu"
#define NPY_INT_FMT "d"
#define NPY_UINT_FMT "u"
#define NPY_LONG_FMT "ld"
#define NPY_ULONG_FMT "lu"
#define NPY_FLOAT_FMT "g"
#define NPY_DOUBLE_FMT "g"

#ifdef PY_LONG_LONG
typedef PY_LONG_LONG npy_longlong;
typedef unsigned PY_LONG_LONG npy_ulonglong;
#  ifdef _MSC_VER
#    define NPY_LONGLONG_FMT         "I64d"
#    define NPY_ULONGLONG_FMT        "I64u"
#    define NPY_LONGLONG_SUFFIX(x)   (x##i64)
#    define NPY_ULONGLONG_SUFFIX(x)  (x##Ui64)
#  else
        /* #define LONGLONG_FMT   "lld"      Another possible variant
           #define ULONGLONG_FMT  "llu"

           #define LONGLONG_FMT   "qd"   -- BSD perhaps?
           #define ULONGLONG_FMT   "qu"
        */
#    define NPY_LONGLONG_FMT         "Ld"
#    define NPY_ULONGLONG_FMT        "Lu"
#    define NPY_LONGLONG_SUFFIX(x)   (x##LL)
#    define NPY_ULONGLONG_SUFFIX(x)  (x##ULL)
#  endif
#else
typedef long npy_longlong;
typedef unsigned long npy_ulonglong;
#  define NPY_LONGLONG_SUFFIX(x)  (x##L)
#  define NPY_ULONGLONG_SUFFIX(x) (x##UL)
#endif


typedef unsigned char npy_bool;
#define NPY_FALSE 0
#define NPY_TRUE 1


#if NPY_SIZEOF_LONGDOUBLE == NPY_SIZEOF_DOUBLE
        typedef double npy_longdouble;
        #define NPY_LONGDOUBLE_FMT "g"
#else
        typedef long double npy_longdouble;
        #define NPY_LONGDOUBLE_FMT "Lg"
#endif

#ifndef Py_USING_UNICODE
#error Must use Python with unicode enabled.
#endif


typedef signed char npy_byte;
typedef unsigned char npy_ubyte;
typedef unsigned short npy_ushort;
typedef unsigned int npy_uint;
typedef unsigned long npy_ulong;

#endif
