#ifndef NUMPY_CORE_SRC_COMMON_HALF_PRIVATE_HPP
#define NUMPY_CORE_SRC_COMMON_HALF_PRIVATE_HPP

#include "npstd.hpp"
#include "float_status.hpp"

/*
 * The following functions that emulating float/double/half conversions
 * are copied from npymath without any changes to its functionalty.
 */
namespace np { namespace half_private {

template<bool gen_overflow=true, bool gen_underflow=true, bool round_even=true>
inline uint16_t FromFloatBits(uint32_t f)
{
    uint32_t f_exp, f_sig;
    uint16_t h_sgn, h_exp, h_sig;

    h_sgn = (uint16_t) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /* Inf or NaN */
            f_sig = (f&0x007fffffu);
            if (f_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                uint16_t ret = (uint16_t) (0x7c00u + (f_sig >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (uint16_t) (h_sgn + 0x7c00u);
            }
        } else {
            if constexpr (gen_overflow) {
                /* overflow to signed inf */
                FloatStatus::RaiseOverflow();
            }
            return (uint16_t) (h_sgn + 0x7c00u);
        }
    }

    /* Exponent underflow converts to a subnormal half or signed zero */
    if (f_exp <= 0x38000000u) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (f_exp < 0x33000000u) {
            if constexpr (gen_underflow) {
                /* If f != 0, it underflowed to 0 */
                if ((f&0x7fffffff) != 0) {
                    FloatStatus::RaiseUnderflow();
                }
            }
            return h_sgn;
        }
        /* Make the subnormal significand */
        f_exp >>= 23;
        f_sig = (0x00800000u + (f&0x007fffffu));
        if constexpr (gen_underflow) {
            /* If it's not exactly represented, it underflowed */
            if ((f_sig&(((uint32_t)1 << (126 - f_exp)) - 1)) != 0) {
                FloatStatus::RaiseUnderflow();
            }
        }
        /*
         * Usually the significand is shifted by 13. For subnormals an
         * additional shift needs to occur. This shift is one for the largest
         * exponent giving a subnormal `f_exp = 0x38000000 >> 23 = 112`, which
         * offsets the new first bit. At most the shift can be 1+10 bits.
         */
        f_sig >>= (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        if constexpr (round_even) {
            /*
             * If the last bit in the half significand is 0 (already even), and
             * the remaining bit pattern is 1000...0, then we do not add one
             * to the bit after the half significand. However, the (113 - f_exp)
             * shift can lose up to 11 bits, so the || checks them in the original.
             * In all other cases, we can just add one.
             */
            if (((f_sig&0x00003fffu) != 0x00001000u) || (f&0x000007ffu)) {
                f_sig += 0x00001000u;
            }
        }
        else {
            f_sig += 0x00001000u;
        }
        h_sig = (uint16_t) (f_sig >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return (uint16_t) (h_sgn + h_sig);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (uint16_t) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_sig = (f&0x007fffffu);
    if constexpr (round_even) {
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((f_sig&0x00003fffu) != 0x00001000u) {
            f_sig += 0x00001000u;
        }
    }
    else {
        f_sig += 0x00001000u;
    }
    h_sig = (uint16_t) (f_sig >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    if constexpr (gen_overflow) {
        h_sig += h_exp;
        if (h_sig == 0x7c00u) {
            FloatStatus::RaiseOverflow();
        }
        return h_sgn + h_sig;
    }
    else {
        return h_sgn + h_exp + h_sig;
    }
}

template<bool gen_overflow=true, bool gen_underflow=true, bool round_even=true>
inline uint16_t FromDoubleBits(uint64_t d)
{
    uint64_t d_exp, d_sig;
    uint16_t h_sgn, h_exp, h_sig;

    h_sgn = (d&0x8000000000000000ULL) >> 48;
    d_exp = (d&0x7ff0000000000000ULL);

    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (d_exp >= 0x40f0000000000000ULL) {
        if (d_exp == 0x7ff0000000000000ULL) {
            /* Inf or NaN */
            d_sig = (d&0x000fffffffffffffULL);
            if (d_sig != 0) {
                /* NaN - propagate the flag in the significand... */
                uint16_t ret = (uint16_t) (0x7c00u + (d_sig >> 42));
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
            if constexpr (gen_overflow) {
                FloatStatus::RaiseOverflow();
            }
            return h_sgn + 0x7c00u;
        }
    }

    /* Exponent underflow converts to subnormal half or signed zero */
    if (d_exp <= 0x3f00000000000000ULL) {
        /*
         * Signed zeros, subnormal floats, and floats with small
         * exponents all convert to signed zero half-floats.
         */
        if (d_exp < 0x3e60000000000000ULL) {
            if constexpr (gen_underflow) {
                /* If d != 0, it underflowed to 0 */
                if ((d&0x7fffffffffffffffULL) != 0) {
                    FloatStatus::RaiseUnderflow();
                }
            }
            return h_sgn;
        }
        /* Make the subnormal significand */
        d_exp >>= 52;
        d_sig = (0x0010000000000000ULL + (d&0x000fffffffffffffULL));
        if constexpr (gen_underflow) {
            /* If it's not exactly represented, it underflowed */
            if ((d_sig&(((uint64_t)1 << (1051 - d_exp)) - 1)) != 0) {
                FloatStatus::RaiseUnderflow();
            }
        }
        /*
         * Unlike floats, doubles have enough room to shift left to align
         * the subnormal significand leading to no loss of the last bits.
         * The smallest possible exponent giving a subnormal is:
         * `d_exp = 0x3e60000000000000 >> 52 = 998`. All larger subnormals are
         * shifted with respect to it. This adds a shift of 10+1 bits the final
         * right shift when comparing it to the one in the normal branch.
         */
        assert(d_exp - 998 >= 0);
        d_sig <<= (d_exp - 998);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        if constexpr (round_even) {
            /*
             * If the last bit in the half significand is 0 (already even), and
             * the remaining bit pattern is 1000...0, then we do not add one
             * to the bit after the half significand.  In all other cases, we do.
             */
            if ((d_sig&0x003fffffffffffffULL) != 0x0010000000000000ULL) {
                d_sig += 0x0010000000000000ULL;
            }
        }
        else {
            d_sig += 0x0010000000000000ULL;
        }
        h_sig = (uint16_t) (d_sig >> 53);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_sig will be zero.
         * This is the correct result.
         */
        return h_sgn + h_sig;
    }

    /* Regular case with no overflow or underflow */
    h_exp = (uint16_t) ((d_exp - 0x3f00000000000000ULL) >> 42);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    d_sig = (d&0x000fffffffffffffULL);
    if constexpr (round_even) {
        /*
         * If the last bit in the half significand is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half significand.  In all other cases, we do.
         */
        if ((d_sig&0x000007ffffffffffULL) != 0x0000020000000000ULL) {
            d_sig += 0x0000020000000000ULL;
        }
    }
    else {
        d_sig += 0x0000020000000000ULL;
    }
    h_sig = (uint16_t) (d_sig >> 42);

    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_sig will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    if constexpr (gen_overflow) {
        h_sig += h_exp;
        if (h_sig == 0x7c00u) {
            FloatStatus::RaiseOverflow();
        }
        return h_sgn + h_sig;
    }
    else {
        return h_sgn + h_exp + h_sig;
    }
}

constexpr uint32_t ToFloatBits(uint16_t h)
{
    uint16_t h_exp = (h&0x7c00u);
    uint32_t f_sgn = ((uint32_t)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: { // 0 or subnormal
            uint16_t h_sig = (h&0x03ffu);
            // Signed zero
            if (h_sig == 0) {
                return f_sgn;
            }
            // Subnormal
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            uint32_t f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
            uint32_t f_sig = ((uint32_t)(h_sig&0x03ffu)) << 13;
            return f_sgn + f_exp + f_sig;
        }
        case 0x7c00u: // inf or NaN
            // All-ones exponent and a copy of the significand
            return f_sgn + 0x7f800000u + (((uint32_t)(h&0x03ffu)) << 13);
        default: // normalized
            // Just need to adjust the exponent and shift
            return f_sgn + (((uint32_t)(h&0x7fffu) + 0x1c000u) << 13);
    }
}

constexpr uint64_t ToDoubleBits(uint16_t h)
{
    uint16_t h_exp = (h&0x7c00u);
    uint64_t d_sgn = ((uint64_t)h&0x8000u) << 48;
    switch (h_exp) {
        case 0x0000u: { // 0 or subnormal
            uint16_t h_sig = (h&0x03ffu);
            // Signed zero
            if (h_sig == 0) {
                return d_sgn;
            }
            // Subnormal
            h_sig <<= 1;
            while ((h_sig&0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            uint64_t d_exp = ((uint64_t)(1023 - 15 - h_exp)) << 52;
            uint64_t d_sig = ((uint64_t)(h_sig&0x03ffu)) << 42;
            return d_sgn + d_exp + d_sig;
        }
        case 0x7c00u: // inf or NaN
            // All-ones exponent and a copy of the significand
            return d_sgn + 0x7ff0000000000000ULL + (((uint64_t)(h&0x03ffu)) << 42);
        default: // normalized
            // Just need to adjust the exponent and shift
            return d_sgn + (((uint64_t)(h&0x7fffu) + 0xfc000u) << 42);
    }
}

}} // namespace np::half_private
#endif // NUMPY_CORE_SRC_COMMON_HALF_PRIVATE_HPP
