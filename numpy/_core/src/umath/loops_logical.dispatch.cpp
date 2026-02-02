#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"
#include <functional>
#include "simd/simd.hpp"
#include <hwy/highway.h>

struct logical_and_t {};
struct logical_or_t {};
struct absolute_t {};
struct logical_not_t {};

namespace {
using namespace np::simd;

/*******************************************************************************
 ** Defining the SIMD kernels
 ******************************************************************************/
/*
 * convert any bit set to boolean true so vectorized and normal operations are
 * consistent, should not be required if bool is used correctly everywhere but
 * you never know
 */
#if NPY_HWY
HWY_INLINE HWY_ATTR Vec<uint8_t> byte_to_true(Vec<uint8_t> v)
{
    return hn::IfThenZeroElse(hn::Eq(v, Zero<uint8_t>()), Set(uint8_t(1)));
}

/*
 * convert mask vector (0xff/0x00) to boolean true.  similar to byte_to_true(),
 * but we've already got a mask and can skip negation.
 */
HWY_INLINE HWY_ATTR Vec<uint8_t> mask_to_true(Vec<uint8_t> v)
{
    return hn::IfThenElseZero(hn::Ne(v, Zero<uint8_t>()), Set(uint8_t(1)));
}

/*
 * For logical_and, we have to be careful to handle non-bool inputs where
 * bits of each operand might not overlap. Example: a = 0x01, b = 0x80
 * Both evaluate to boolean true, however, a & b is false.  Return value
 * should be consistent with byte_to_true().
 */
HWY_INLINE HWY_ATTR Vec<uint8_t> simd_logical_and_u8(Vec<uint8_t> a, Vec<uint8_t> b)
{
    return hn::IfThenZeroElse(
        hn::Eq(Zero<uint8_t>(), hn::Min(a, b)),
        Set(uint8_t(1))
    );
}
/*
 * We don't really need the following, but it simplifies the templating code
 * below since it is paired with simd_logical_and_u8() above.
 */
HWY_INLINE HWY_ATTR Vec<uint8_t> simd_logical_or_u8(Vec<uint8_t> a, Vec<uint8_t> b)
{
    auto r = hn::Or(a, b);
    return byte_to_true(r);
}

HWY_INLINE HWY_ATTR bool simd_any_u8(Vec<uint8_t> v)
{
    return hn::ReduceMax(_Tag<uint8_t>(), v) != 0;
}

HWY_INLINE HWY_ATTR bool simd_all_u8(Vec<uint8_t> v)
{
    return hn::ReduceMin(_Tag<uint8_t>(), v) != 0;
}
#endif

template<typename Op>
struct BinaryLogicalTraits;

template<>
struct BinaryLogicalTraits<logical_or_t> {
    static constexpr bool is_and     = false;
    static constexpr auto scalar_op  = std::logical_or<bool>{};
    static constexpr auto scalar_cmp = std::not_equal_to<bool>{};
#if NPY_HWY
    static constexpr auto anyall = simd_any_u8;

    static HWY_INLINE HWY_ATTR Vec<uint8_t> simd_op(Vec<uint8_t> a, Vec<uint8_t> b) {
        return simd_logical_or_u8(a, b);
    }
#endif
};

template<>
struct BinaryLogicalTraits<logical_and_t> {
    static constexpr bool is_and     = true;
    static constexpr auto scalar_op  = std::logical_and<bool>{};
    static constexpr auto scalar_cmp = std::equal_to<bool>{};
#if NPY_HWY
    static constexpr auto anyall = simd_all_u8;

    static HWY_INLINE HWY_ATTR Vec<uint8_t> simd_op(Vec<uint8_t> a, Vec<uint8_t> b) {
        return simd_logical_and_u8(a, b);
    }
#endif
};

template<typename Op>
struct UnaryLogicalTraits;

template<>
struct UnaryLogicalTraits<logical_not_t> {
    static constexpr auto scalar_op = std::equal_to<bool>{};

#if NPY_HWY
    static HWY_INLINE HWY_ATTR Vec<uint8_t> simd_op(Vec<uint8_t> v) {
        const auto zero = Zero<uint8_t>();
        return mask_to_true(hn::VecFromMask(_Tag<uint8_t>(), hn::Eq(v, zero)));
    }
#endif
};

template<>
struct UnaryLogicalTraits<absolute_t> {
    static constexpr auto scalar_op = std::not_equal_to<bool>{};

#if NPY_HWY
    static HWY_INLINE HWY_ATTR Vec<uint8_t> simd_op(Vec<uint8_t> v) {
        return byte_to_true(v);
    }
#endif
};

#if NPY_HWY
template<typename Op>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_binary_logical_BOOL(npy_bool* op, npy_bool* ip1, npy_bool* ip2, npy_intp len) {
    using Traits = BinaryLogicalTraits<Op>;
    constexpr int UNROLL = 16;
    HWY_LANES_CONSTEXPR int vstep = Lanes<uint8_t>();
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip1 += wstep, ip2 += wstep, op += wstep) {
        for(int i = 0; i < UNROLL; i++) {
            auto a = LoadU(ip1 + vstep * i);
            auto b = LoadU(ip2 + vstep * i);
            auto r = Traits::simd_op(a, b);
            StoreU(r, op + vstep * i);
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip1 += vstep, ip2 += vstep, op += vstep) {
        auto a = LoadU(ip1);
        auto b = LoadU(ip2);
        auto r = Traits::simd_op(a, b);
        StoreU(r, op);
    }

    // Scalar loop to finish off
    for (; len > 0; len--, ip1++, ip2++, op++) {
        *op = Traits::scalar_op(*ip1, *ip2);
    }
}

template<typename Op>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_reduce_logical_BOOL(npy_bool* op, npy_bool* ip, npy_intp len) {
    using Traits = BinaryLogicalTraits<Op>;
    constexpr int UNROLL = 8;
    HWY_LANES_CONSTEXPR int vstep = Lanes<uint8_t>();
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip += wstep) {
        #if defined(NPY_HAVE_SSE2)
            NPY_PREFETCH(reinterpret_cast<const char *>(ip + wstep), 0, 3);
        #endif
        auto v0 = LoadU(ip);
        auto v1 = LoadU(ip + vstep);
        auto v2 = LoadU(ip + vstep * 2);
        auto v3 = LoadU(ip + vstep * 3);
        auto v4 = LoadU(ip + vstep * 4);
        auto v5 = LoadU(ip + vstep * 5);
        auto v6 = LoadU(ip + vstep * 6);
        auto v7 = LoadU(ip + vstep * 7);

        auto m01 = Traits::simd_op(v0, v1);
        auto m23 = Traits::simd_op(v2, v3);
        auto m45 = Traits::simd_op(v4, v5);
        auto m67 = Traits::simd_op(v6, v7);

        auto m0123 = Traits::simd_op(m01, m23);
        auto m4567 = Traits::simd_op(m45, m67);

        auto mv = Traits::simd_op(m0123, m4567);

        if(Traits::anyall(mv) == !Traits::is_and) {
            *op = !Traits::is_and;
            return;
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip += vstep) {
        auto v = LoadU(ip);
        if(Traits::anyall(v) == !Traits::is_and) {
            *op = !Traits::is_and;
            return;
        }
    }

    // Scalar loop to finish off
    for (; len > 0; --len, ++ip) {
        *op = Traits::scalar_op(*op, *ip);
        if (Traits::scalar_cmp(*op, 0)) {
            return;
        }
    }
}

template<typename Op>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_logical_BOOL(npy_bool* op, npy_bool* ip, npy_intp len) {
    using Traits = UnaryLogicalTraits<Op>;
    constexpr int UNROLL = 16;
    HWY_LANES_CONSTEXPR int vstep = Lanes<uint8_t>();
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for(int i = 0; i < UNROLL; i++) {
            auto v = LoadU(ip + vstep * i);
            auto r = Traits::simd_op(v);
            StoreU(r, op + vstep * i);
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        auto v = LoadU(ip);
        auto r = Traits::simd_op(v);
        StoreU(r, op);
    }

    // Scalar loop to finish off
    for (; len > 0; --len, ++ip, ++op) {
        *op = Traits::scalar_op(*ip, 0);
    }
}

#endif  //NPY_HWY
} // namespace anonymous

/*******************************************************************************
 ** Defining ufunc inner functions
 ******************************************************************************/
template<typename Op>
static NPY_INLINE int run_binary_simd_logical_BOOL(
    char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_HWY
    if (sizeof(npy_bool) == 1 && IS_BLOCKABLE_BINARY(sizeof(npy_bool), kMaxLanes<uint8_t>)) {
        simd_binary_logical_BOOL<Op>((npy_bool*)args[2], (npy_bool*)args[0], (npy_bool*)args[1], dimensions[0]);
        return 1;
    }
#endif
    return 0;
}

template<typename Op>
static NPY_INLINE int run_reduce_simd_logical_BOOL(
    char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_HWY
    if (sizeof(npy_bool) == 1 && IS_BLOCKABLE_REDUCE(sizeof(npy_bool), kMaxLanes<uint8_t>)) {
        simd_reduce_logical_BOOL<Op>((npy_bool*)args[0], (npy_bool*)args[1], dimensions[0]);
        return 1;
    }
#endif
    return 0;
}

template<typename Op>
static NPY_INLINE int run_unary_simd_logical_BOOL(
    char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_HWY
    if (sizeof(npy_bool) == 1 && IS_BLOCKABLE_UNARY(sizeof(npy_bool), kMaxLanes<uint8_t>)) {
        simd_unary_logical_BOOL<Op>((npy_bool*)args[1], (npy_bool*)args[0], dimensions[0]);
        return 1;
    }
#endif
    return 0;
}

template <typename Op>
void BOOL_binary_func_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps) {
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];
    npy_intp n = dimensions[0];
    using Traits = BinaryLogicalTraits<Op>;

#if NPY_HWY
    if (run_binary_simd_logical_BOOL<Op>(args, dimensions, steps)) {
        return;
    }
#endif

    for(npy_intp i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) {
        const npy_bool in1 = *(npy_bool*)ip1;
        const npy_bool in2 = *(npy_bool*)ip2;
        *((npy_bool*)op1)  = Traits::scalar_op(in1, in2);
    }
}

template <typename Op>
void BOOL_binary_reduce_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps) {
    char *iop1 = args[0];
    npy_bool io1 = *(npy_bool *)iop1;
    char *ip2    = args[1];
    npy_intp is2 = steps[1];
    npy_intp n   = dimensions[0];
    npy_intp i;
    using Traits = BinaryLogicalTraits<Op>;
#if NPY_HWY
    if (run_reduce_simd_logical_BOOL<Op>(args, dimensions, steps)) {
        return;
    }
#else
    /* for now only use libc on 32-bit/non-x86 */
    if (steps[1] == 1) {
        npy_bool * op = (npy_bool *)args[0];
        if constexpr (Traits::is_and) {

            /* np.all(), search for a zero (false) */
            if (*op) {
                *op = memchr(args[1], 0, dimensions[0]) == NULL;
            }
        }
        else {
            /*
             * np.any(), search for a non-zero (true) via comparing against
             * zero blocks, memcmp is faster than memchr on SSE4 machines
             * with glibc >= 2.12 and memchr can only check for equal 1
             */
            static const npy_bool zero[4096]={0}; /* zero by C standard */

            for (i = 0; !*op && i < n - (n % sizeof(zero)); i += sizeof(zero)) {
                *op = memcmp(&args[1][i], zero, sizeof(zero)) != 0;
            }
            if (!*op && n - i > 0) {
                *op = memcmp(&args[1][i], zero, n - i) != 0;
            }
        }
        return;
    }
#endif

    for(i = 0; i < n; i++, ip2 += is2) {
        const npy_bool in2 = *(npy_bool*)ip2;
        io1 = Traits::scalar_op(io1, in2);
        if ((Traits::is_and && !io1) || (!Traits::is_and && io1))
            break;
    }
    *((npy_bool*)iop1) = io1;
}

template <typename Op>
void BOOL_logical_op_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps) {
    if (IS_BINARY_REDUCE) {
        BOOL_binary_reduce_wrapper<Op>(args, dimensions, steps);
    }
    else {
        BOOL_binary_func_wrapper<Op>(args, dimensions, steps);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BOOL_logical_and)(
    char** args, npy_intp const* dimensions, npy_intp const* steps, void* NPY_UNUSED(func))
{
    BOOL_logical_op_wrapper<logical_and_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BOOL_logical_or)(
    char** args, npy_intp const* dimensions, npy_intp const* steps, void* NPY_UNUSED(func))
{
    BOOL_logical_op_wrapper<logical_or_t>(args, dimensions, steps);
}

template <typename Op>
void BOOL_func_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps)
{
    char *ip1 = args[0], *op1 = args[1];
    npy_intp is1 = steps[0], os1 = steps[1];
    npy_intp n = dimensions[0];
    using Traits = UnaryLogicalTraits<Op>;

    if (run_unary_simd_logical_BOOL<Op>(args, dimensions, steps)) {
        return;
    }

    for(npy_intp i = 0; i < n; i++, ip1 += is1, op1 += os1) {
        npy_bool in1 = *(npy_bool*)ip1;
        *((npy_bool*)op1) = Traits::scalar_op(in1, 0);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BOOL_logical_not)(
    char** args, npy_intp const* dimensions, npy_intp const* steps, void* NPY_UNUSED(func))
{
    BOOL_func_wrapper<logical_not_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BOOL_absolute)(
    char** args, npy_intp const* dimensions, npy_intp const* steps, void* NPY_UNUSED(func))
{
    BOOL_func_wrapper<absolute_t>(args, dimensions, steps);
}
