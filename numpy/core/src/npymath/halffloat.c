#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/halffloat.h"

/*
 * This chooses between 'ties to even' and 'ties away from zero'.
 */
#define NPY_HALF_ROUND_TIES_TO_EVEN 1
/*
 * If these are 1, the conversions try to trigger underflow,
 * overflow, and invalid exceptions in the FP system when needed.
 */
#define NPY_HALF_GENERATE_OVERFLOW 1
#define NPY_HALF_GENERATE_UNDERFLOW 1
#define NPY_HALF_GENERATE_INVALID 1

/*
 ********************************************************************
 *                   HALF-PRECISION ROUTINES                        *
 ********************************************************************
 */

float npy_half_to_float(npy_half h)
{
    union { float ret; npy_uint32 retbits; } conv;
    conv.retbits = npy_halfbits_to_floatbits(h);
    return conv.ret;
}

double npy_half_to_double(npy_half h)
{
    union { double ret; npy_uint64 retbits; } conv;
    conv.retbits = npy_halfbits_to_doublebits(h);
    return conv.ret;
}

npy_half npy_float_to_half(float f)
{
    union { float f; npy_uint32 fbits; } conv;
    conv.f = f;
    return npy_floatbits_to_halfbits(conv.fbits);
}

npy_half npy_double_to_half(double d)
{
    union { double d; npy_uint64 dbits; } conv;
    conv.d = d;
    return npy_doublebits_to_halfbits(conv.dbits);
}

int npy_half_iszero(npy_half h)
{
    return (h&0x7fff) == 0;
}

int npy_half_isnan(npy_half h)
{
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) != 0x0000u);
}

int npy_half_isinf(npy_half h)
{
    return ((h&0x7fffu) == 0x7c00u);
}

int npy_half_isfinite(npy_half h)
{
    return ((h&0x7c00u) != 0x7c00u);
}

int npy_half_signbit(npy_half h)
{
    return (h&0x8000u) != 0;
}

npy_half npy_half_spacing(npy_half h)
{
    npy_half ret;
    npy_uint16 h_exp = h&0x7c00u;
    npy_uint16 h_sig = h&0x03ffu;
    if (h_exp == 0x7c00u) {
#if NPY_HALF_GENERATE_INVALID
        npy_set_floatstatus_invalid();
#endif
        ret = NPY_HALF_NAN;
    } else if (h == 0x7bffu) {
#if NPY_HALF_GENERATE_OVERFLOW
        npy_set_floatstatus_overflow();
#endif
        ret = NPY_HALF_PINF;
    } else if ((h&0x8000u) && h_sig == 0) { /* Negative boundary case */
        if (h_exp > 0x2c00u) { /* If result is normalized */
            ret = h_exp - 0x2c00u;
        } else if(h_exp > 0x0400u) { /* The result is a subnormal, but not the smallest */
            ret = 1 << ((h_exp >> 10) - 2);
        } else {
            ret = 0x0001u; /* Smallest subnormal half */
        }
    } else if (h_exp > 0x2800u) { /* If result is still normalized */
        ret = h_exp - 0x2800u;
    } else if (h_exp > 0x0400u) { /* The result is a subnormal, but not the smallest */
        ret = 1 << ((h_exp >> 10) - 1);
    } else {
        ret = 0x0001u;
    }

    return ret;
}

npy_half npy_half_copysign(npy_half x, npy_half y)
{
    return (x&0x7fffu) | (y&0x8000u);
}

npy_half npy_half_nextafter(npy_half x, npy_half y)
{
    npy_half ret;

    if (!npy_half_isfinite(x) || npy_half_isnan(y)) {
#if NPY_HALF_GENERATE_INVALID
        npy_set_floatstatus_invalid();
#endif
        ret = NPY_HALF_NAN;
    } else if (npy_half_eq_nonan(x, y)) {
        ret = x;
    } else if (npy_half_iszero(x)) {
        ret = (y&0x8000u) + 1; /* Smallest subnormal half */
    } else if (!(x&0x8000u)) { /* x > 0 */
        if ((npy_int16)x > (npy_int16)y) { /* x > y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    } else {
        if (!(y&0x8000u) || (x&0x7fffu) > (y&0x7fffu)) { /* x < y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    }
#if NPY_HALF_GENERATE_OVERFLOW
    if (npy_half_isinf(ret)) {
        npy_set_floatstatus_overflow();
    }
#endif

    return ret;
}

int npy_half_eq_nonan(npy_half h1, npy_half h2)
{
    return (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int npy_half_eq(npy_half h1, npy_half h2)
{
    /*
     * The equality cases are as follows:
     *   - If either value is NaN, never equal.
     *   - If the values are equal, equal.
     *   - If the values are both signed zeros, equal.
     */
    return (!npy_half_isnan(h1) && !npy_half_isnan(h2)) &&
           (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int npy_half_ne(npy_half h1, npy_half h2)
{
    return !npy_half_eq(h1, h2);
}

int npy_half_lt_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            return (h1&0x7fffu) > (h2&0x7fffu);
        } else {
            /* Signed zeros are equal, have to check for it */
            return (h1 != 0x8000u) || (h2 != 0x0000u);
        }
    } else {
        if (h2&0x8000u) {
            return 0;
        } else {
            return (h1&0x7fffu) < (h2&0x7fffu);
        }
    }
}

int npy_half_lt(npy_half h1, npy_half h2)
{
    return (!npy_half_isnan(h1) && !npy_half_isnan(h2)) && npy_half_lt_nonan(h1, h2);
}

int npy_half_gt(npy_half h1, npy_half h2)
{
    return npy_half_lt(h2, h1);
}

int npy_half_le_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            return (h1&0x7fffu) >= (h2&0x7fffu);
        } else {
            return 1;
        }
    } else {
        if (h2&0x8000u) {
            /* Signed zeros are equal, have to check for it */
            return (h1 == 0x0000u) && (h2 == 0x8000u);
        } else {
            return (h1&0x7fffu) <= (h2&0x7fffu);
        }
    }
}

int npy_half_le(npy_half h1, npy_half h2)
{
    return (!npy_half_isnan(h1) && !npy_half_isnan(h2)) && npy_half_le_nonan(h1, h2);
}

int npy_half_ge(npy_half h1, npy_half h2)
{
    return npy_half_le(h2, h1);
}



/*
 ********************************************************************
 *                     BIT-LEVEL CONVERSIONS                        *
 ********************************************************************
 */

npy_uint16 npy_floatbits_to_halfbits(npy_uint32 f)
{
    npy_uint32 f_exp, f_sig;
    npy_uint16 h_sgn, h_exp, h_sig;

    h_sgn = (npy_uint16) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                npy_uint16 ret = (npy_uint16) (0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (npy_uint16) (h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
            npy_set_floatstatus_overflow();
#endif
            return (npy_uint16) (h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (f_exp < 0x33000000u) {
#if NPY_HALF_GENERATE_UNDERFLOW
            /* If f != 0, it underflowed to 0 */
            if ((f&0x7fffffff) != 0) {
                npy_set_floatstatus_underflow();
            }
#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));
#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        if ((f_sig&(((npy_uint32)1 << (126 - f_exp)) - 1)) != 0) {
            npy_set_floatstatus_underflow();
        }
#endif
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if NPY_HALF_ROUND_TIES_TO_EVEN
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((f_sig&0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
#else
        f_sig += 0x00001000u;
#endif
        h_sig = (npy_uint16) (f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (npy_uint16) (h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (npy_uint16) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f&0x007fffffu);
#if NPY_HALF_ROUND_TIES_TO_EVEN
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((f_sig&0x00003fffu) != 0x00001000u) {
        f_sig += 0x00001000u;
    }
#else
    f_sig += 0x00001000u;
#endif
    h_sig = (npy_uint16) (f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if NPY_HALF_GENERATE_OVERFLOW
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
        npy_set_floatstatus_overflow();
    }
    return h_sgn + h_sig;
#else
    return h_sgn + h_exp + h_sig;
#endif
}

npy_uint16 npy_doublebits_to_halfbits(npy_uint64 d)
{
    npy_uint64 d_exp, d_sig;
    npy_uint16 h_sgn, h_exp, h_sig;

    h_sgn = (d&0x8000000000000000ULL) >> 48;
    d_exp = (d&0x7ff0000000000000ULL);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (d_exp >= 0x40f0000000000000ULL) {
        if (d_exp == 0x7ff0000000000000ULL) {
            /* Inf or NaN */
            d_sig = (d&0x000fffffffffffffULL);
            if (d_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                npy_uint16 ret = (npy_uint16) (0x7c00u + (d_sig >> 42));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return h_sgn + 0x7c00u;
            }
        } else {
            /* overflow to signed inf */
#if NPY_HALF_GENERATE_OVERFLOW
            npy_set_floatstatus_overflow();
#endif
            return h_sgn + 0x7c00u;
        }
    }

    /* Exponent underflow converts to subnormal half or signed zero */
    if (d_exp <= 0x3f00000000000000ULL) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (d_exp < 0x3e60000000000000ULL) {
#if NPY_HALF_GENERATE_UNDERFLOW
            /* If d != 0, it underflowed to 0 */
            if ((d&0x7fffffffffffffffULL) != 0) {
                npy_set_floatstatus_underflow();
            }
#endif
            return h_sgn;
        }
        /* Make the subnormal significand */
        d_exp >>= 52;
        d_sig = (0x0010000000000000ULL + (d&0x000fffffffffffffULL));
#if NPY_HALF_GENERATE_UNDERFLOW
        /* If it's not exactly represented, it underflowed */
        if ((d_sig&(((npy_uint64)1 << (1051 - d_exp)) - 1)) != 0) {
            npy_set_floatstatus_underflow();
        }
#endif
        d_sig >>= (1009 - d_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if NPY_HALF_ROUND_TIES_TO_EVEN
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((d_sig&0x000007ffffffffffULL) != 0x0000020000000000ULL) {
            d_sig += 0x0000020000000000ULL;
        }
#else
        d_sig += 0x0000020000000000ULL;
#endif
        h_sig = (npy_uint16) (d_sig >> 42);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return h_sgn + h_sig;
    }

    /* Regular case with no overflow or underflow */
    h_exp = (npy_uint16) ((d_exp - 0x3f00000000000000ULL) >> 42);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    d_sig = (d&0x000fffffffffffffULL);
#if NPY_HALF_ROUND_TIES_TO_EVEN
    /*
     * If the last bit in the half significand is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half significand.  In all other cases, we do.
     */
    if ((d_sig&0x000007ffffffffffULL) != 0x0000020000000000ULL) {
        d_sig += 0x0000020000000000ULL;
    }
#else
    d_sig += 0x0000020000000000ULL;
#endif
    h_sig = (npy_uint16) (d_sig >> 42);

    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if NPY_HALF_GENERATE_OVERFLOW
    h_sig += h_exp;
    if (h_sig == 0x7c00u) {
        npy_set_floatstatus_overflow();
    }
    return h_sgn + h_sig;
#else
    return h_sgn + h_exp + h_sig;
#endif
}

npy_uint32 npy_halfbits_to_floatbits(npy_uint16 h)
{
    npy_uint16 h_exp, h_sig;
    npy_uint32 f_sgn, f_exp, f_sig;

    h_exp = (h&0x7c00u);
    f_sgn = ((npy_uint32)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h&0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return f_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((npy_uint32)(127 - 15 - h_exp)) << 23;
            f_sig = ((npy_uint32)(h_sig&0x03ffu)) << 13;
            return f_sgn + f_exp + f_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return f_sgn + 0x7f800000u + (((npy_uint32)(h&0x03ffu)) << 13);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return f_sgn + (((npy_uint32)(h&0x7fffu) + 0x1c000u) << 13);
    }
}

npy_uint64 npy_halfbits_to_doublebits(npy_uint16 h)
{
    npy_uint16 h_exp, h_sig;
    npy_uint64 d_sgn, d_exp, d_sig;

    h_exp = (h&0x7c00u);
    d_sgn = ((npy_uint64)h&0x8000u) << 48;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h&0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return d_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            d_exp = ((npy_uint64)(1023 - 15 - h_exp)) << 52;
            d_sig = ((npy_uint64)(h_sig&0x03ffu)) << 42;
            return d_sgn + d_exp + d_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return d_sgn + 0x7ff0000000000000ULL +
                                (((npy_uint64)(h&0x03ffu)) << 42);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return d_sgn + (((npy_uint64)(h&0x7fffu) + 0xfc000u) << 42);
    }
}
