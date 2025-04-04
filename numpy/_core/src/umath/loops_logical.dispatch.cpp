#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"
#include <functional>

#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

struct logical_and_t {};
struct logical_or_t {};
struct absolute_t {};
struct logical_not_t {};

const hn::ScalableTag<uint8_t> u8;
using vec_u8 = hn::Vec<decltype(u8)>;

/*******************************************************************************
 ** Defining the SIMD kernels
 ******************************************************************************/
/*
 * convert any bit set to boolean true so vectorized and normal operations are
 * consistent, should not be required if bool is used correctly everywhere but
 * you never know
 */

HWY_INLINE HWY_ATTR vec_u8 byte_to_true(vec_u8 v)
{
    return hn::IfThenZeroElse(hn::Eq(v, hn::Zero(u8)), hn::Set(u8, 1));
}
/*
 * convert mask vector (0xff/0x00) to boolean true.  similar to byte_to_true(),
 * but we've already got a mask and can skip negation.
 */
HWY_INLINE HWY_ATTR vec_u8 mask_to_true(vec_u8 v)
{
    const vec_u8 truemask = hn::Set(u8, 1 == 1);
    return hn::And(truemask, v);
}
/*
 * For logical_and, we have to be careful to handle non-bool inputs where
 * bits of each operand might not overlap. Example: a = 0x01, b = 0x80
 * Both evaluate to boolean true, however, a & b is false.  Return value
 * should be consistent with byte_to_true().
 */
HWY_INLINE HWY_ATTR vec_u8 simd_logical_and_u8(vec_u8 a, vec_u8 b)
{
    return hn::IfThenZeroElse(
        hn::Eq(hn::Zero(u8), hn::Min(a, b)), 
        hn::Set(u8, 1)
    );
}
/*
 * We don't really need the following, but it simplifies the templating code
 * below since it is paired with simd_logical_and_u8() above.
 */
HWY_INLINE HWY_ATTR vec_u8 simd_logical_or_u8(vec_u8 a, vec_u8 b)
{
    vec_u8 r = hn::Or(a, b);
    return byte_to_true(r);
}

HWY_INLINE HWY_ATTR npy_bool simd_any_u8(vec_u8 v)
{
    return hn::ReduceMax(u8, v) != 0;
}

HWY_INLINE HWY_ATTR npy_bool simd_all_u8(vec_u8 v)
{
    return hn::ReduceMin(u8, v) != 0;
}

template<typename Op>
struct BinaryLogicalTraits;

template<>
struct BinaryLogicalTraits<logical_or_t> {
    static constexpr bool is_and = false;
    static constexpr auto scalar_op = std::logical_or<bool>{};
    static constexpr auto scalar_cmp = std::not_equal_to<bool>{};
    static constexpr auto anyall = simd_any_u8;

    HWY_INLINE HWY_ATTR vec_u8 simd_op(vec_u8 a, vec_u8 b) {
        return simd_logical_or_u8(a, b);
    }

    HWY_INLINE HWY_ATTR vec_u8 reduce(vec_u8 a, vec_u8 b) {
        return simd_logical_or_u8(a, b);
    }
};

template<>
struct BinaryLogicalTraits<logical_and_t> {
    static constexpr bool is_and = true;
    static constexpr auto scalar_op = std::logical_and<bool>{};
    static constexpr auto scalar_cmp = std::equal_to<bool>{};
    static constexpr auto anyall = simd_all_u8;

    HWY_INLINE HWY_ATTR vec_u8 simd_op(vec_u8 a, vec_u8 b) {
        return simd_logical_and_u8(a, b);
    }

    HWY_INLINE HWY_ATTR vec_u8 reduce(vec_u8 a, vec_u8 b) {
        return simd_logical_and_u8(a, b);
    }
};

template<typename Op>
struct UnaryLogicalTraits;

template<>
struct UnaryLogicalTraits<logical_not_t> {
    static constexpr bool is_not = true;
    static constexpr auto scalar_op = std::equal_to<bool>{};

    HWY_INLINE HWY_ATTR vec_u8 simd_op(vec_u8 v) {
        const vec_u8 zero = hn::Zero(u8);
        return mask_to_true(hn::VecFromMask(u8, hn::Eq(v, zero)));
    }
};

template<>
struct UnaryLogicalTraits<absolute_t> {
    static constexpr bool is_not = false;
    static constexpr auto scalar_op = std::not_equal_to<bool>{};

    HWY_INLINE HWY_ATTR vec_u8 simd_op(vec_u8 v) {
        return byte_to_true(v);
    }
};


template<typename Op>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_binary_logical_BOOL(npy_bool* op, npy_bool* ip1, npy_bool* ip2, npy_intp len) {
    using Traits = BinaryLogicalTraits<Op>;
    Traits traits;
    constexpr int UNROLL = 16;
    const int vstep = hn::Lanes(u8);
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip1 += wstep, ip2 += wstep, op += wstep) {

        for(int i = 0; i < UNROLL; i++) {
            vec_u8 a = hn::LoadU(u8, ip1 + vstep * i);
            vec_u8 b = hn::LoadU(u8, ip2 + vstep * i);
            vec_u8 r = traits.simd_op(a, b);
            hn::StoreU(r, u8, op + vstep * i);
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip1 += vstep, ip2 += vstep, op += vstep) {
        vec_u8 a = hn::LoadU(u8, ip1);
        vec_u8 b = hn::LoadU(u8, ip2);
        vec_u8 r = traits.simd_op(a, b);
        hn::StoreU(r, u8, op);
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
    Traits traits;
    constexpr int UNROLL = 8;
    const int vstep = hn::Lanes(u8);
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip += wstep) {
        #if defined(NPY_HAVE_SSE2)
            NPY_PREFETCH(reinterpret_cast<const char *>(ip + wstep), 0, 3);
        #endif
        vec_u8 v0 = hn::LoadU(u8, ip);
        vec_u8 v1 = hn::LoadU(u8, ip + vstep);
        vec_u8 v2 = hn::LoadU(u8, ip + vstep * 2);
        vec_u8 v3 = hn::LoadU(u8, ip + vstep * 3);
        vec_u8 v4 = hn::LoadU(u8, ip + vstep * 4);
        vec_u8 v5 = hn::LoadU(u8, ip + vstep * 5);
        vec_u8 v6 = hn::LoadU(u8, ip + vstep * 6);
        vec_u8 v7 = hn::LoadU(u8, ip + vstep * 7);

        vec_u8 m01 = traits.reduce(v0, v1);
        vec_u8 m23 = traits.reduce(v2, v3);
        vec_u8 m45 = traits.reduce(v4, v5);
        vec_u8 m67 = traits.reduce(v6, v7);

        vec_u8 m0123 = traits.reduce(m01, m23);
        vec_u8 m4567 = traits.reduce(m45, m67);

        vec_u8 mv = traits.reduce(m0123, m4567);

        if(Traits::anyall(mv) == !Traits::is_and) {
            *op = !Traits::is_and;
            return;
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip += vstep) {
        vec_u8 v = hn::LoadU(u8, ip);
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
    Traits traits;
    constexpr int UNROLL = 16;
    const int vstep = hn::Lanes(u8);
    const int wstep = vstep * UNROLL;

    // Unrolled vectors loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for(int i = 0; i < UNROLL; i++) {
            vec_u8 v = hn::LoadU(u8, ip + vstep * i);
            vec_u8 r = traits.simd_op(v);
            hn::StoreU(r, u8, op + vstep * i);
        }
    }

    // Single vectors loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        vec_u8 v = hn::LoadU(u8, ip);
        vec_u8 r = traits.simd_op(v);
        hn::StoreU(r, u8, op);
    }

    // Scalar loop to finish off
    for (; len > 0; --len, ++ip, ++op) {
        *op = Traits::scalar_op(*ip, 0);
    }
}

/*******************************************************************************
 ** Defining ufunc inner functions
 ******************************************************************************/
template<typename Op>
static NPY_INLINE int run_binary_simd_logical_BOOL(
    char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_SIMD
    if (sizeof(npy_bool) == 1 &&
            IS_BLOCKABLE_BINARY(sizeof(npy_bool), NPY_SIMD_WIDTH)) {
        simd_binary_logical_BOOL<Op>((npy_bool*)args[2], (npy_bool*)args[0],
                                    (npy_bool*)args[1], dimensions[0]
        );
        return 1;
    }
#endif
    return 0;
}

template<typename Op>
static NPY_INLINE int run_reduce_simd_logical_BOOL(
    char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_SIMD
    if (sizeof(npy_bool) == 1 &&
            IS_BLOCKABLE_REDUCE(sizeof(npy_bool), NPY_SIMD_WIDTH)) {
        simd_reduce_logical_BOOL<Op>((npy_bool*)args[0], (npy_bool*)args[1],
            dimensions[0]
        );
        return 1;
    }
#endif
    return 0;
}

template<typename Op>
static NPY_INLINE int run_unary_simd_logical_BOOL(
    char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_SIMD
    if (sizeof(npy_bool) == 1 &&
            IS_BLOCKABLE_UNARY(sizeof(npy_bool), NPY_SIMD_WIDTH)) {
        simd_unary_logical_BOOL<Op>((npy_bool*)args[1], (npy_bool*)args[0], dimensions[0]);
        return 1;
    }
#endif
    return 0;
}

template <typename Op>
void BOOL_binary_func_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps) {
    using Traits = BinaryLogicalTraits<Op>;
    
    if (run_binary_simd_logical_BOOL<Op>(args, dimensions, steps)) {
        return;
    }
    else {
        BINARY_LOOP {
            const npy_bool in1 = *(npy_bool*)ip1;
            const npy_bool in2 = *(npy_bool*)ip2;
            *((npy_bool*)op1) = Traits::scalar_op(in1, in2);
        }
    }
}

template <typename Op>
void BOOL_binary_reduce_wrapper(char** args, npy_intp const* dimensions, npy_intp const* steps) {
    using Traits = BinaryLogicalTraits<Op>;
#if NPY_SIMD
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
            npy_uintp i, n = dimensions[0];

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
    else {
        BINARY_REDUCE_LOOP(npy_bool) {
            const npy_bool in2 = *(npy_bool*)ip2;
            io1 = Traits::scalar_op(io1, in2);
            if ((Traits::is_and && !io1) || (!Traits::is_and && io1)) break;
        }
        *((npy_bool*)iop1) = io1;
    }
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
    using Traits = UnaryLogicalTraits<Op>;
    if (run_unary_simd_logical_BOOL<Op>(args, dimensions, steps)) {
        return;
    }
    else {
        UNARY_LOOP {
            npy_bool in1 = *(npy_bool*)ip1;
            *((npy_bool*)op1) = Traits::scalar_op(in1, 0);
        }
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
