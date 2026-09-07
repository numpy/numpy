#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "fast_loop_macros.h"
#include "loops.h"
#include "loops_utils.h"
#include "lowlevel_strided_loops.h"

#include "simd/simd.h"
#include "simd/simd.hpp"
#include <hwy/highway.h>

#include <cstdlib>  // llabs

namespace hn = hwy::HWY_NAMESPACE;

/*
 * Timedelta (int64) division by an invariant scalar divisor via the
 * Granlund-Montgomery magic-multiplier method. NaT (NPY_DATETIME_NAT) is
 * propagated. The caller guarantees the divisor is neither 0 nor NaT.
 */

// 64-bit SIMD division is slower than scalar on these targets; use scalar.
#if (defined(NPY_HAVE_VSX) && !defined(NPY_HAVE_VSX4)) || defined(NPY_HAVE_NEON) || defined(NPY_HAVE_LSX)
    #define SIMD_DISABLE_DIV64_OPT
#endif

#if NPY_SIMD && !defined(SIMD_DISABLE_DIV64_OPT)

// Highway needs the fixed-width lane type; npy_int64 is layout-compatible.
using DTag = hn::ScalableTag<int64_t>;
using VI64 = hn::Vec<DTag>;

// Magic-division parameters. The caller guarantees d != 0.
struct s64_div_params {
    int64_t m;      // magic multiplier
    int     sh;     // arithmetic right-shift count
    int64_t dsign;  // -1 if divisor < 0 else 0
};

static HWY_INLINE HWY_ATTR s64_div_params
compute_s64_div_params(int64_t d)
{
    s64_div_params p;
    int64_t d1 = llabs((long long)d);
    // Handle abs overflow (d == INT64_MIN)
    if ((uint64_t)d == 0x8000000000000000ULL) {
        p.m  = (int64_t)0x8000000000000001ULL;
        p.sh = 62;
    }
    else if (d1 > 1) {
        int sh = (int)npyv__bitscan_revnz_u64((uint64_t)(d1 - 1));
        p.sh = sh;
        p.m  = (int64_t)(npyv__divh128_u64(1ULL << sh, (uint64_t)d1) + 1);
    }
    else { // d1 == 1
        p.sh = 0;
        p.m  = 1;
    }
    p.dsign = (d < 0) ? -1 : 0;
    return p;
}

// Truncated (toward-zero) quotient of a signed-64 vector by the invariant divisor.
static HWY_INLINE HWY_ATTR VI64
simd_trunc_divide_s64(VI64 a, VI64 mv, int sh, VI64 dsignv)
{
    VI64 q = hn::Add(a, hn::MulHigh(a, mv));
    q = hn::ShiftRightSame(q, sh);
    q = hn::Sub(q, hn::BroadcastSignBit(a));
    q = hn::Sub(hn::Xor(q, dsignv), dsignv);
    return q;
}

static void HWY_ATTR
simd_divide_by_scalar_contig_timedelta(char **args, npy_intp len)
{
    const int64_t *src = (const int64_t *)args[0];
    int64_t scalar     = *(const int64_t *)args[1];
    int64_t *dst       = (int64_t *)args[2];

    const DTag d;
    const npy_intp vstep   = (npy_intp)hn::Lanes(d);
    const s64_div_params p = compute_s64_div_params(scalar);
    const VI64 mv          = hn::Set(d, p.m);
    const VI64 dsignv      = hn::Set(d, p.dsign);
    const VI64 vnat        = hn::Set(d, NPY_DATETIME_NAT);

    npy_intp i = 0;
    for (; i + vstep <= len; i += vstep) {
        VI64 a  = hn::LoadU(d, src + i);
        auto nat = hn::Eq(a, vnat);
        VI64 q  = simd_trunc_divide_s64(a, mv, p.sh, dsignv);
        q = hn::IfThenElse(nat, vnat, q);
        hn::StoreU(q, d, dst + i);
    }

    for (; i < len; ++i) {
        const npy_int64 a = src[i];
        dst[i] = (a == NPY_DATETIME_NAT) ? NPY_DATETIME_NAT : a / scalar;
    }
}

/*
 * Floor division by an invariant scalar. When propagate_nat is true (m // q),
 * NaT dividends become NaT. Otherwise (m // m -> q), NaT becomes 0 and sets
 * the invalid float status, matching TIMEDELTA_mm_q_floor_divide.
 */
static void HWY_ATTR
simd_floor_divide_by_scalar_contig_timedelta(char **args, npy_intp len,
                                             bool propagate_nat)
{
    const int64_t *src = (const int64_t *)args[0];
    int64_t scalar     = *(const int64_t *)args[1];
    int64_t *dst       = (int64_t *)args[2];

    const DTag d;
    const npy_intp vstep   = (npy_intp)hn::Lanes(d);
    const s64_div_params p = compute_s64_div_params(scalar);
    const VI64 mv          = hn::Set(d, p.m);
    const VI64 dsignv      = hn::Set(d, p.dsign);
    const VI64 vnat        = hn::Set(d, NPY_DATETIME_NAT);
    const VI64 vzero       = hn::Zero(d);
    const VI64 vone        = hn::Set(d, 1);
    const VI64 nsign_d     = hn::Set(d, (npy_int64)(scalar < 0)); // 0 or 1
    const VI64 nat_out     = propagate_nat ? vnat : vzero;
    bool any_nat = false;

    npy_intp i = 0;
    for (; i + vstep <= len; i += vstep) {
        VI64 a   = hn::LoadU(d, src + i);
        auto nat = hn::Eq(a, vnat);
        if (!propagate_nat && !hn::AllFalse(d, nat)) {
            any_nat = true;
        }
        VI64 nsign_a   = hn::IfThenElse(hn::Lt(a, nsign_d), vone, vzero);
        VI64 diff_sign = hn::Sub(nsign_a, nsign_d);
        VI64 to_ninf   = hn::Xor(nsign_a, nsign_d);
        VI64 trunc     = simd_trunc_divide_s64(hn::Add(a, diff_sign), mv, p.sh, dsignv);
        VI64 floor     = hn::Sub(trunc, to_ninf);
        floor = hn::IfThenElse(nat, nat_out, floor);
        hn::StoreU(floor, d, dst + i);
    }
    if (any_nat) {
        npy_set_floatstatus_invalid();
    }

    for (; i < len; ++i) {
        const npy_int64 a = src[i];
        if (a == NPY_DATETIME_NAT) {
            if (propagate_nat) {
                dst[i] = NPY_DATETIME_NAT;
            }
            else {
                npy_set_floatstatus_invalid();
                dst[i] = 0;
            }
        }
        else {
            npy_int64 r = a / scalar;
            // Negative quotients needs to be rounded down
            if (((a > 0) != (scalar > 0)) && ((r * scalar) != a)) {
                r--;
            }
            dst[i] = r;
        }
    }
}

#endif // NPY_SIMD && !SIMD_DISABLE_DIV64_OPT

/*
 * Floor quotient of two nonzero, non-NaT int64 values.
 * Matches Python/NumPy integer floor division (round toward -inf).
 */
static NPY_INLINE npy_int64
timedelta_floor_quotient(npy_int64 a, npy_int64 b)
{
    npy_int64 quo = a / b;
    if (((a > 0) != (b > 0)) && (quo * b != a)) {
        quo -= 1;
    }
    return quo;
}

/*
 * Shared implementation for timedelta floor division by an int64-like divisor.
 *
 * propagate_nat true  (m // q -> m): NaT and divide-by-zero become NaT.
 * propagate_nat false (m // m -> q): NaT becomes 0 with invalid status,
 *                                    divide-by-zero becomes 0 with divbyzero.
 * For !propagate_nat, a NaT divisor is also treated as invalid -> 0.
 */
static void
timedelta_floor_divide_by_int64(char **args, npy_intp const *dimensions,
                                npy_intp const *steps, bool propagate_nat)
{
    BINARY_DEFS

    if (steps[1] == 0) {
        if (n == 0) {
            return;
        }

        const npy_int64 in2 = *(npy_int64 *)ip2;

        if (in2 == 0) {
            npy_set_floatstatus_divbyzero();
            BINARY_LOOP_SLIDING {
                *((npy_int64 *)op1) = propagate_nat ? NPY_DATETIME_NAT : 0;
            }
        }
        else if (!propagate_nat && in2 == NPY_DATETIME_NAT) {
            npy_set_floatstatus_invalid();
            BINARY_LOOP_SLIDING {
                *((npy_int64 *)op1) = 0;
            }
        }
        else {
#if NPY_SIMD && !defined(SIMD_DISABLE_DIV64_OPT)
            if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(npy_timedelta), NPY_SIMD_WIDTH)) {
                simd_floor_divide_by_scalar_contig_timedelta(args, n, propagate_nat);
                return;
            }
#endif
            BINARY_LOOP_SLIDING {
                const npy_int64 in1 = *(npy_int64 *)ip1;
                if (in1 == NPY_DATETIME_NAT) {
                    if (propagate_nat) {
                        *((npy_int64 *)op1) = NPY_DATETIME_NAT;
                    }
                    else {
                        npy_set_floatstatus_invalid();
                        *((npy_int64 *)op1) = 0;
                    }
                }
                else {
                    *((npy_int64 *)op1) = timedelta_floor_quotient(in1, in2);
                }
            }
        }
    }
    else {
        BINARY_LOOP_SLIDING {
            const npy_int64 in1 = *(npy_int64 *)ip1;
            const npy_int64 in2 = *(npy_int64 *)ip2;
            if (in1 == NPY_DATETIME_NAT ||
                    (!propagate_nat && in2 == NPY_DATETIME_NAT)) {
                if (propagate_nat) {
                    *((npy_int64 *)op1) = NPY_DATETIME_NAT;
                }
                else {
                    npy_set_floatstatus_invalid();
                    *((npy_int64 *)op1) = 0;
                }
            }
            else if (in2 == 0) {
                npy_set_floatstatus_divbyzero();
                *((npy_int64 *)op1) = propagate_nat ? NPY_DATETIME_NAT : 0;
            }
            else {
                *((npy_int64 *)op1) = timedelta_floor_quotient(in1, in2);
            }
        }
    }
}

/********************************************************************************
 ** Dispatched loops
 ********************************************************************************/

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TIMEDELTA_mq_m_divide)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    BINARY_DEFS

    /* When the divisor is a scalar, we can vectorize the division */
    if (steps[1] == 0) {
        /* In case of empty array, just return */
        if (n == 0) {
            return;
        }

        const npy_int64 in2 = *(npy_int64 *)ip2;

        /* If divisor is 0, we need not compute anything */
        if (in2 == 0) {
            npy_set_floatstatus_divbyzero();
            BINARY_LOOP_SLIDING {
                *((npy_timedelta *)op1) = NPY_DATETIME_NAT;
            }
        }
        else {
#if NPY_SIMD && !defined(SIMD_DISABLE_DIV64_OPT)
            /* contiguous block of memory with a non-zero scalar divisor */
            if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(npy_timedelta), NPY_SIMD_WIDTH)) {
                simd_divide_by_scalar_contig_timedelta(args, n);
                return;
            }
#endif
            BINARY_LOOP_SLIDING {
                const npy_timedelta in1 = *(npy_timedelta *)ip1;
                if (in1 == NPY_DATETIME_NAT) {
                    *((npy_timedelta *)op1) = NPY_DATETIME_NAT;
                }
                else {
                    *((npy_timedelta *)op1) = in1 / in2;
                }
            }
        }
    }
    else {
        BINARY_LOOP_SLIDING {
            const npy_timedelta in1 = *(npy_timedelta *)ip1;
            const npy_int64 in2 = *(npy_int64 *)ip2;
            if (in1 == NPY_DATETIME_NAT || in2 == 0) {
                *((npy_timedelta *)op1) = NPY_DATETIME_NAT;
            }
            else {
                *((npy_timedelta *)op1) = in1 / in2;
            }
        }
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TIMEDELTA_mm_q_floor_divide)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    timedelta_floor_divide_by_int64(args, dimensions, steps, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TIMEDELTA_mq_m_floor_divide)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    timedelta_floor_divide_by_int64(args, dimensions, steps, true);
}
