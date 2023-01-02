#ifndef __NPY_SORT_COMMON_H__
#define __NPY_SORT_COMMON_H__

#include <stdlib.h>
#include <numpy/ndarraytypes.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 *****************************************************************************
 **                        SWAP MACROS                                      **
 *****************************************************************************
 */

#define BOOL_SWAP(a,b) {npy_bool tmp = (b); (b)=(a); (a) = tmp;}
#define BYTE_SWAP(a,b) {npy_byte tmp = (b); (b)=(a); (a) = tmp;}
#define UBYTE_SWAP(a,b) {npy_ubyte tmp = (b); (b)=(a); (a) = tmp;}
#define SHORT_SWAP(a,b) {npy_short tmp = (b); (b)=(a); (a) = tmp;}
#define USHORT_SWAP(a,b) {npy_ushort tmp = (b); (b)=(a); (a) = tmp;}
#define INT_SWAP(a,b) {npy_int tmp = (b); (b)=(a); (a) = tmp;}
#define UINT_SWAP(a,b) {npy_uint tmp = (b); (b)=(a); (a) = tmp;}
#define LONG_SWAP(a,b) {npy_long tmp = (b); (b)=(a); (a) = tmp;}
#define ULONG_SWAP(a,b) {npy_ulong tmp = (b); (b)=(a); (a) = tmp;}
#define LONGLONG_SWAP(a,b) {npy_longlong tmp = (b); (b)=(a); (a) = tmp;}
#define ULONGLONG_SWAP(a,b) {npy_ulonglong tmp = (b); (b)=(a); (a) = tmp;}
#define HALF_SWAP(a,b) {npy_half tmp = (b); (b)=(a); (a) = tmp;}
#define FLOAT_SWAP(a,b) {npy_float tmp = (b); (b)=(a); (a) = tmp;}
#define DOUBLE_SWAP(a,b) {npy_double tmp = (b); (b)=(a); (a) = tmp;}
#define LONGDOUBLE_SWAP(a,b) {npy_longdouble tmp = (b); (b)=(a); (a) = tmp;}
#define CFLOAT_SWAP(a,b) {npy_cfloat tmp = (b); (b)=(a); (a) = tmp;}
#define CDOUBLE_SWAP(a,b) {npy_cdouble tmp = (b); (b)=(a); (a) = tmp;}
#define CLONGDOUBLE_SWAP(a,b) {npy_clongdouble tmp = (b); (b)=(a); (a) = tmp;}
#define DATETIME_SWAP(a,b) {npy_datetime tmp = (b); (b)=(a); (a) = tmp;}
#define TIMEDELTA_SWAP(a,b) {npy_timedelta tmp = (b); (b)=(a); (a) = tmp;}

/* Need this for the argsort functions */
#define INTP_SWAP(a,b) {npy_intp tmp = (b); (b)=(a); (a) = tmp;}

/*
 *****************************************************************************
 **                        COMPARISON FUNCTIONS                             **
 *****************************************************************************
 */

static inline int
BOOL_LT(npy_bool a, npy_bool b)
{
    return a < b;
}


static inline int
BYTE_LT(npy_byte a, npy_byte b)
{
    return a < b;
}


static inline int
UBYTE_LT(npy_ubyte a, npy_ubyte b)
{
    return a < b;
}


static inline int
SHORT_LT(npy_short a, npy_short b)
{
    return a < b;
}


static inline int
USHORT_LT(npy_ushort a, npy_ushort b)
{
    return a < b;
}


static inline int
INT_LT(npy_int a, npy_int b)
{
    return a < b;
}


static inline int
UINT_LT(npy_uint a, npy_uint b)
{
    return a < b;
}


static inline int
LONG_LT(npy_long a, npy_long b)
{
    return a < b;
}


static inline int
ULONG_LT(npy_ulong a, npy_ulong b)
{
    return a < b;
}


static inline int
LONGLONG_LT(npy_longlong a, npy_longlong b)
{
    return a < b;
}


static inline int
ULONGLONG_LT(npy_ulonglong a, npy_ulonglong b)
{
    return a < b;
}


static inline int
FLOAT_LT(npy_float a, npy_float b)
{
    return a < b || (b != b && a == a);
}


static inline int
DOUBLE_LT(npy_double a, npy_double b)
{
    return a < b || (b != b && a == a);
}


static inline int
LONGDOUBLE_LT(npy_longdouble a, npy_longdouble b)
{
    return a < b || (b != b && a == a);
}


static inline int
_npy_half_isnan(npy_half h)
{
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) != 0x0000u);
}


static inline int
_npy_half_lt_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            return (h1&0x7fffu) > (h2&0x7fffu);
        }
        else {
            /* Signed zeros are equal, have to check for it */
            return (h1 != 0x8000u) || (h2 != 0x0000u);
        }
    }
    else {
        if (h2&0x8000u) {
            return 0;
        }
        else {
            return (h1&0x7fffu) < (h2&0x7fffu);
        }
    }
}


static inline int
HALF_LT(npy_half a, npy_half b)
{
    int ret;

    if (_npy_half_isnan(b)) {
        ret = !_npy_half_isnan(a);
    }
    else {
        ret = !_npy_half_isnan(a) && _npy_half_lt_nonan(a, b);
    }

    return ret;
}

/*
 * For inline functions SUN recommends not using a return in the then part
 * of an if statement. It's a SUN compiler thing, so assign the return value
 * to a variable instead.
 */
static inline int
CFLOAT_LT(npy_cfloat a, npy_cfloat b)
{
    int ret;

    if (a.real < b.real) {
        ret = a.imag == a.imag || b.imag != b.imag;
    }
    else if (a.real > b.real) {
        ret = b.imag != b.imag && a.imag == a.imag;
    }
    else if (a.real == b.real || (a.real != a.real && b.real != b.real)) {
        ret =  a.imag < b.imag || (b.imag != b.imag && a.imag == a.imag);
    }
    else {
        ret = b.real != b.real;
    }

    return ret;
}


static inline int
CDOUBLE_LT(npy_cdouble a, npy_cdouble b)
{
    int ret;

    if (a.real < b.real) {
        ret = a.imag == a.imag || b.imag != b.imag;
    }
    else if (a.real > b.real) {
        ret = b.imag != b.imag && a.imag == a.imag;
    }
    else if (a.real == b.real || (a.real != a.real && b.real != b.real)) {
        ret =  a.imag < b.imag || (b.imag != b.imag && a.imag == a.imag);
    }
    else {
        ret = b.real != b.real;
    }

    return ret;
}


static inline int
CLONGDOUBLE_LT(npy_clongdouble a, npy_clongdouble b)
{
    int ret;

    if (a.real < b.real) {
        ret = a.imag == a.imag || b.imag != b.imag;
    }
    else if (a.real > b.real) {
        ret = b.imag != b.imag && a.imag == a.imag;
    }
    else if (a.real == b.real || (a.real != a.real && b.real != b.real)) {
        ret =  a.imag < b.imag || (b.imag != b.imag && a.imag == a.imag);
    }
    else {
        ret = b.real != b.real;
    }

    return ret;
}


static inline void
STRING_COPY(char *s1, char const*s2, size_t len)
{
    memcpy(s1, s2, len);
}


static inline void
STRING_SWAP(char *s1, char *s2, size_t len)
{
    while(len--) {
        const char t = *s1;
        *s1++ = *s2;
        *s2++ = t;
    }
}


static inline int
STRING_LT(const char *s1, const char *s2, size_t len)
{
    const unsigned char *c1 = (const unsigned char *)s1;
    const unsigned char *c2 = (const unsigned char *)s2;
    size_t i;
    int ret = 0;

    for (i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            ret = c1[i] < c2[i];
            break;
        }
    }
    return ret;
}


static inline void
UNICODE_COPY(npy_ucs4 *s1, npy_ucs4 const *s2, size_t len)
{
    while(len--) {
        *s1++ = *s2++;
    }
}


static inline void
UNICODE_SWAP(npy_ucs4 *s1, npy_ucs4 *s2, size_t len)
{
    while(len--) {
        const npy_ucs4 t = *s1;
        *s1++ = *s2;
        *s2++ = t;
    }
}


static inline int
UNICODE_LT(const npy_ucs4 *s1, const npy_ucs4 *s2, size_t len)
{
    size_t i;
    int ret = 0;

    for (i = 0; i < len; ++i) {
        if (s1[i] != s2[i]) {
            ret = s1[i] < s2[i];
            break;
        }
    }
    return ret;
}


static inline int
DATETIME_LT(npy_datetime a, npy_datetime b)
{
    if (a == NPY_DATETIME_NAT) {
        return 0;
    }

    if (b == NPY_DATETIME_NAT) {
        return 1;
    }

    return a < b;
}


static inline int
TIMEDELTA_LT(npy_timedelta a, npy_timedelta b)
{
    if (a == NPY_DATETIME_NAT) {
        return 0;
    }

    if (b == NPY_DATETIME_NAT) {
        return 1;
    }

    return a < b;
}


static inline void
GENERIC_COPY(char *a, char *b, size_t len)
{
    memcpy(a, b, len);
}


static inline void
GENERIC_SWAP(char *a, char *b, size_t len)
{
    while(len--) {
        const char t = *a;
        *a++ = *b;
        *b++ = t;
    }
}

#ifdef __cplusplus
}
#endif

#endif
