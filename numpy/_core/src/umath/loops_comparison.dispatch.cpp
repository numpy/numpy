#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "lowlevel_strided_loops.h"
// Provides the various *_LOOP macros
#include "fast_loop_macros.h"

#include <hwy/highway.h>
#include <type_traits>
#include <array>

namespace hn = hwy::HWY_NAMESPACE;

const hn::ScalableTag<uint8_t> u8;
const hn::ScalableTag<int8_t> s8;
const hn::ScalableTag<uint16_t> u16;
const hn::ScalableTag<int16_t> s16;
const hn::ScalableTag<uint32_t> u32;
const hn::ScalableTag<int32_t> s32;
const hn::ScalableTag<uint64_t> u64;
const hn::ScalableTag<int64_t> s64;
const hn::ScalableTag<float> f32;
const hn::ScalableTag<double> f64;
using vec_u8  = hn::Vec<decltype(u8)>;
using vec_s8  = hn::Vec<decltype(s8)>;
using vec_u16 = hn::Vec<decltype(u16)>;
using vec_s16 = hn::Vec<decltype(s16)>;
using vec_u32 = hn::Vec<decltype(u32)>;
using vec_s32 = hn::Vec<decltype(s32)>;
using vec_u64 = hn::Vec<decltype(u64)>;
using vec_s64 = hn::Vec<decltype(s64)>;
using vec_f32 = hn::Vec<decltype(f32)>;
using vec_f64 = hn::Vec<decltype(f64)>;

HWY_INLINE HWY_ATTR vec_u8 simd_pack_b8_b16(vec_u16 a, vec_u16 b) {
    return hn::OrderedDemote2To(u8, a, b);
}

HWY_INLINE HWY_ATTR vec_u8 simd_pack_b8_b32(vec_u32 a, vec_u32 b, vec_u32 c, vec_u32 d) {
    auto ab = hn::OrderedDemote2To(u16, a, b);
    auto cd = hn::OrderedDemote2To(u16, c, d);
    return simd_pack_b8_b16(ab, cd);
}

HWY_INLINE HWY_ATTR vec_u8 simd_pack_b8_b64(vec_u64 a, vec_u64 b, vec_u64 c, vec_u64 d,
                                     vec_u64 e, vec_u64 f, vec_u64 g, vec_u64 h) {
    auto ab = hn::OrderedDemote2To(u32, a, b);
    auto cd = hn::OrderedDemote2To(u32, c, d);
    auto ef = hn::OrderedDemote2To(u32, e, f);
    auto gh = hn::OrderedDemote2To(u32, g, h);
    return simd_pack_b8_b32(ab, cd, ef, gh);
}

HWY_INLINE HWY_ATTR vec_u8 simd_xnor_b8(vec_u8 a, vec_u8 b) {
    return hn::Not(hn::Xor(a, b));
}

HWY_INLINE HWY_ATTR vec_u8 simd_xor_b8(vec_u8 a, vec_u8 b) {
    return hn::Xor(a, b);
}

HWY_INLINE HWY_ATTR vec_u8 simd_andc_b8(vec_u8 a, vec_u8 b) {
    return hn::AndNot(b, a);
}

HWY_INLINE HWY_ATTR vec_u8 simd_orc_b8(vec_u8 a, vec_u8 b) {
    return hn::Or(a, hn::Not(b));
}

template<typename T>
struct TypeTraits;

template<>
struct TypeTraits<uint8_t> {
    using ScalarType = npyv_lanetype_u8;
    using ScalarType2 = npy_ubyte;
    using VecType = vec_u8;
    static constexpr auto Tag = u8;
    static constexpr int  Len = 8;
    static constexpr bool IsSigned = false;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<int8_t> {
    using ScalarType = npyv_lanetype_s8;
    using ScalarType2 = npy_byte;
    using VecType = vec_s8;
    static constexpr auto Tag = s8;
    static constexpr int  Len = 8;
    static constexpr bool IsSigned = true;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<uint16_t> {
    using ScalarType = npyv_lanetype_u16;
    using ScalarType2 = npy_ushort;
    using VecType = vec_u16;
    static constexpr auto Tag = u16;
    static constexpr int  Len = 16;
    static constexpr bool IsSigned = false;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<int16_t> {
    using ScalarType = npyv_lanetype_s16;
    using ScalarType2 = npy_short;
    using VecType = vec_s16;
    static constexpr auto Tag = s16;
    static constexpr int  Len = 16;
    static constexpr bool IsSigned = true;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<uint32_t> {
    using ScalarType = npyv_lanetype_u32;
    using ScalarType2 = npy_uint;
    using VecType = vec_u32;
    static constexpr auto Tag = u32;
    static constexpr int  Len = 32;
    static constexpr bool IsSigned = false;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<int32_t> {
    using ScalarType = npyv_lanetype_s32;
    using ScalarType2 = npy_int;
    using VecType = vec_s32;
    static constexpr auto Tag = s32;
    static constexpr int  Len = 32;
    static constexpr bool IsSigned = true;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<uint64_t> {
    using ScalarType = npyv_lanetype_u64;
    using ScalarType2 = npy_ulonglong;
    using VecType = vec_u64;
    static constexpr auto Tag = u64;
    static constexpr int  Len = 64;
    static constexpr bool IsSigned = false;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<int64_t> {
    using ScalarType = npyv_lanetype_s64;
    using ScalarType2 = npy_longlong;
    using VecType = vec_s64;
    static constexpr auto Tag = s64;
    static constexpr int  Len = 64;
    static constexpr bool IsSigned = true;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<float> {
    using ScalarType = npyv_lanetype_f32;
    using ScalarType2 = npy_float;
    using VecType = vec_f32;
    static constexpr auto Tag = f32;
    static constexpr int  Len = 32;
    static constexpr bool IsSigned = false;
    static constexpr bool IsFloat  = true;
    static constexpr int  HasSIMD  = NPY_SIMD_F32;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<double> {
    using ScalarType = npyv_lanetype_f64;
    using ScalarType2 = npy_double;
    using VecType = vec_f64;
    static constexpr auto Tag = f64;
    static constexpr int  Len = 64;
    static constexpr bool IsSigned = false;
    static constexpr bool IsFloat  = true;
    static constexpr int  HasSIMD  = NPY_SIMD_F64;
    static constexpr bool IsBool   = false;
};

template<>
struct TypeTraits<bool> {
    using ScalarType = npyv_lanetype_u8;
    using ScalarType2 = npy_ubyte;
    using VecType = vec_u8;
    static constexpr auto Tag = u8;
    static constexpr int  Len = 8;
    static constexpr bool IsSigned = false;
    static constexpr bool IsFloat  = false;
    static constexpr int  HasSIMD  = NPY_SIMD;
    static constexpr bool IsBool   = true;
};

enum class CompareOp {
    Equal,
    NotEqual,
    Less,
    LessEqual
};

template<CompareOp Op>
struct CompareOpTraits;

template<>
struct CompareOpTraits<CompareOp::Equal> {
    static constexpr bool IsEq = true;
    static constexpr bool IsNeq = false;
    static constexpr auto boolOp = simd_xnor_b8;

    template<typename T>
    static auto compare(const T& a, const T& b) {
        return hn::Eq(a, b);
    }

    template<typename T>
    static bool scalarCompare(T a, T b) {
        return a == b;
    }
};

template<>
struct CompareOpTraits<CompareOp::NotEqual> {
    static constexpr bool IsEq = false;
    static constexpr bool IsNeq = true;
    static constexpr auto boolOp = simd_xor_b8;

    template<typename T>
    static auto compare(const T& a, const T& b) {
        return hn::Ne(a, b);
    }

    template<typename T>
    static bool scalarCompare(T a, T b) {
        return a != b;
    }
};

template<>
struct CompareOpTraits<CompareOp::Less> {
    static constexpr bool IsEq = false;
    static constexpr bool IsNeq = false;
    static constexpr auto boolOp = simd_andc_b8;

    template<typename T>
    static auto compare(const T& a, const T& b) {
        return hn::Lt(a, b);
    }

    template<typename T>
    static bool scalarCompare(T a, T b) {
        return a < b;
    }
};

template<>
struct CompareOpTraits<CompareOp::LessEqual> {
    static constexpr bool IsEq = false;
    static constexpr bool IsNeq = false;
    static constexpr auto boolOp = simd_orc_b8;

    template<typename T>
    static auto compare(const T& a, const T& b) {
        return hn::Le(a, b);
    }

    template<typename T>
    static bool scalarCompare(T a, T b) {
        return a <= b;
    }
};

template<typename Traits, typename Op>
static auto process_simd_compare(const typename Traits::VecType& a, 
                               const typename Traits::VecType& b,
                               Op&& compare_op) {
    auto c = compare_op(a, b);
    return hn::VecFromMask(Traits::Tag, c);
}

template<typename Traits, typename Op>
static vec_u8 process_simd_block(const typename Traits::ScalarType* src1,
                                const typename Traits::ScalarType* src2,
                                Op&& compare_op) {
    if constexpr (Traits::Len == 8) {
        auto a1 = hn::LoadU(Traits::Tag, src1);
        auto b1 = hn::LoadU(Traits::Tag, src2);
        auto c1_vec = process_simd_compare<Traits>(a1, b1, compare_op);
        return hn::BitCast(u8, c1_vec);
    } 
    else if constexpr (Traits::Len == 16) {
        auto a1 = hn::LoadU(Traits::Tag, src1);
        auto b1 = hn::LoadU(Traits::Tag, src2);
        auto c1_vec = process_simd_compare<Traits>(a1, b1, compare_op);

        auto a2 = hn::LoadU(Traits::Tag, src1 + hn::Lanes(Traits::Tag));
        auto b2 = hn::LoadU(Traits::Tag, src2 + hn::Lanes(Traits::Tag));
        auto c2_vec = process_simd_compare<Traits>(a2, b2, compare_op);

        return simd_pack_b8_b16(hn::BitCast(u16, c1_vec), hn::BitCast(u16, c2_vec));
    }
    else if constexpr (Traits::Len == 32) {
        std::array<typename Traits::VecType, 4> results;
        for (int i = 0; i < 4; ++i) {
            auto a = hn::LoadU(Traits::Tag, src1 + hn::Lanes(Traits::Tag) * i);
            auto b = hn::LoadU(Traits::Tag, src2 + hn::Lanes(Traits::Tag) * i);
            results[i] = process_simd_compare<Traits>(a, b, compare_op);
        }

        return simd_pack_b8_b32(
            hn::BitCast(u32, results[0]),
            hn::BitCast(u32, results[1]),
            hn::BitCast(u32, results[2]),
            hn::BitCast(u32, results[3])
        );
    }
    else if constexpr (Traits::Len == 64) {
        std::array<typename Traits::VecType, 8> results;
        for (int i = 0; i < 8; ++i) {
            auto a = hn::LoadU(Traits::Tag, src1 + hn::Lanes(Traits::Tag) * i);
            auto b = hn::LoadU(Traits::Tag, src2 + hn::Lanes(Traits::Tag) * i);
            results[i] = process_simd_compare<Traits>(a, b, compare_op);
        }

        return simd_pack_b8_b64(
            hn::BitCast(u64, results[0]),
            hn::BitCast(u64, results[1]),
            hn::BitCast(u64, results[2]),
            hn::BitCast(u64, results[3]),
            hn::BitCast(u64, results[4]),
            hn::BitCast(u64, results[5]),
            hn::BitCast(u64, results[6]),
            hn::BitCast(u64, results[7])
        );
    }
}

template<typename T, CompareOp Op>
static void simd_binary_compare(char **args, npy_intp len) {
    using Traits = TypeTraits<T>;
    using Traits_Op = CompareOpTraits<Op>;

    typename Traits::ScalarType *src1 = (typename Traits::ScalarType *) args[0];
    typename Traits::ScalarType *src2 = (typename Traits::ScalarType *) args[1];
    npyv_lanetype_u8 *dst = (npyv_lanetype_u8 *) args[2];
    const vec_u8 truemask = hn::Set(u8, 0x1);
    const int vstep = hn::Lanes(u8);

    for (; len >= vstep; len -= vstep, src1 += vstep, src2 += vstep, dst += vstep) {
        vec_u8 r = process_simd_block<Traits>(
            src1, src2,
            [](const auto& a, const auto& b) { return Traits_Op::template compare(a, b); });
        hn::StoreU(hn::And(r, truemask), u8, dst);
    }

    for (; len > 0; --len, ++src1, ++src2, ++dst) {
        *dst = Traits_Op::template scalarCompare(*src1, *src2);
    }
}

template<typename T, CompareOp Op>
static void simd_binary_scalar1_compare(char **args, npy_intp len) {
    using Traits = TypeTraits<T>;
    using Traits_Op = CompareOpTraits<Op>;

    typename Traits::ScalarType scalar = *(typename Traits::ScalarType *) args[0];
    typename Traits::ScalarType *src = (typename Traits::ScalarType *) args[1];
    npyv_lanetype_u8 *dst = (npyv_lanetype_u8 *) args[2];
    const typename Traits::VecType a = hn::Set(Traits::Tag, scalar);
    const vec_u8 truemask = hn::Set(u8, 0x1);
    const int vstep = hn::Lanes(u8);

    for (; len >= vstep; len -= vstep, src += vstep, dst += vstep) {
        vec_u8 r = process_simd_block<Traits>(
            &scalar, src,
            [&a](const auto&, const auto& b) { return Traits_Op::template compare(a, b); }
        );
        hn::StoreU(hn::And(r, truemask), u8, dst);
    }

    for (; len > 0; --len, ++src, ++dst) {
        *dst = Traits_Op::template scalarCompare(scalar, *src);
    }
}

template<typename T, CompareOp Op>
static void simd_binary_scalar2_compare(char **args, npy_intp len) {
    using Traits = TypeTraits<T>;
    using Traits_Op = CompareOpTraits<Op>;

    typename Traits::ScalarType *src = (typename Traits::ScalarType *) args[0];
    typename Traits::ScalarType scalar = *(typename Traits::ScalarType *) args[1];
    npyv_lanetype_u8 *dst = (npyv_lanetype_u8 *) args[2];
    const typename Traits::VecType b = hn::Set(Traits::Tag, scalar);
    const vec_u8 truemask = hn::Set(u8, 0x1);
    const int vstep = hn::Lanes(u8);

    for (; len >= vstep; len -= vstep, src += vstep, dst += vstep) {
        vec_u8 r = process_simd_block<Traits>(
            src, &scalar,
            [&b](const auto& a, const auto&) { return Traits_Op::template compare(a, b); }
        );
        hn::StoreU(hn::And(r, truemask), u8, dst);
    }

    for (; len > 0; --len, ++src, ++dst) {
        *dst = Traits_Op::template scalarCompare(*src, scalar);
    }
}

template<CompareOp Op>
static void simd_binary_compare_b8(char **args, npy_intp len) {
    using Op_Traits = CompareOpTraits<Op>;

    npyv_lanetype_u8 *src1 = (npyv_lanetype_u8 *) args[0];
    npyv_lanetype_u8 *src2 = (npyv_lanetype_u8 *) args[1];
    npyv_lanetype_u8 *dst = (npyv_lanetype_u8 *) args[2];
    const vec_u8 truemask = hn::Set(u8, 0x1);
    const vec_u8 vzero = hn::Set(u8, 0x0);
    const int vstep = hn::Lanes(u8);

    for (; len >= vstep; len -= vstep, src1 += vstep, src2 += vstep, dst += vstep) {
        auto a = hn::Eq(hn::LoadU(u8, src1), vzero);
        auto b = hn::Eq(hn::LoadU(u8, src2), vzero);
        vec_u8 c = Op_Traits::Op(hn::VecFromMask(u8, a), hn::VecFromMask(u8, b));
        hn::StoreU(hn::And(c, truemask), u8, dst);
    }

    for (; len > 0; --len, ++src1, ++src2, ++dst) {
        const npyv_lanetype_u8 a = *src1 != 0;
        const npyv_lanetype_u8 b = *src2 != 0;
        *dst = Op_Traits::scalarCompare(a, b);
    }
}

template<CompareOp Op>
static void simd_binary_scalar1_compare_b8(char **args, npy_intp len) {
    using Op_Traits = CompareOpTraits<Op>;

    npyv_lanetype_u8 scalar = *(npyv_lanetype_u8 *) args[0];
    npyv_lanetype_u8 *src = (npyv_lanetype_u8 *) args[1];
    npyv_lanetype_u8 *dst = (npyv_lanetype_u8 *) args[2];
    const vec_u8 vzero = hn::Set(u8, 0x0);
    const vec_u8 vscalar = hn::Set(u8, scalar);
    const auto a = hn::Eq(vscalar, vzero);
    const vec_u8 truemask = hn::Set(u8, 0x1);
    const int vstep = hn::Lanes(u8);

    for (; len >= vstep; len -= vstep, src += vstep, dst += vstep) {
        auto b = hn::Eq(hn::LoadU(u8, src), vzero);
        vec_u8 c = Op_Traits::Op(hn::VecFromMask(u8, a), hn::VecFromMask(u8, b));
        hn::StoreU(hn::And(c, truemask), u8, dst);
    }

    for (; len > 0; --len, ++src, ++dst) {
        const npyv_lanetype_u8 b = *src != 0;
        *dst = Op_Traits::scalarCompare(scalar, b);
    }
}

template<CompareOp Op>
static void simd_binary_scalar2_compare_b8(char **args, npy_intp len) {
    using Op_Traits = CompareOpTraits<Op>;

    npyv_lanetype_u8 *src = (npyv_lanetype_u8 *) args[0];
    npyv_lanetype_u8 scalar = *(npyv_lanetype_u8 *) args[1];
    npyv_lanetype_u8 *dst = (npyv_lanetype_u8 *) args[2];
    const vec_u8 vzero = hn::Set(u8, 0x0);
    const vec_u8 vscalar = hn::Set(u8, scalar);
    const auto b = hn::Eq(vscalar, vzero);
    const vec_u8 truemask = hn::Set(u8, 0x1);
    const int vstep = hn::Lanes(u8);

    for (; len >= vstep; len -= vstep, src += vstep, dst += vstep) {
        auto a = hn::Eq(hn::LoadU(u8, src), vzero);
        vec_u8 c = Op_Traits::Op(hn::VecFromMask(u8, a), hn::VecFromMask(u8, b));
        hn::StoreU(hn::And(c, truemask), u8, dst);
    }

    for (; len > 0; --len, ++src, ++dst) {
        const npyv_lanetype_u8 a = *src != 0;
        *dst = Op_Traits::scalarCompare(a, scalar);
    }
}

template<typename T, CompareOp Op>
static inline void run_binary_simd_compare(char **args, npy_intp const *dimensions, npy_intp const *steps) {
    using Traits = TypeTraits<T>;
    using Traits_Op = CompareOpTraits<Op>;

    if constexpr (Traits::HasSIMD) {
        if (!is_mem_overlap(args[0], steps[0], args[2], steps[2], dimensions[0]) &&
            !is_mem_overlap(args[1], steps[1], args[2], steps[2], dimensions[0])) {
            /* argument one scalar */
            if (IS_BINARY_CONT_S1(typename Traits::ScalarType2, npy_bool)) {
                simd_binary_scalar1_compare<T, Op>(args, dimensions[0]);
                return;
            }
            /* argument two scalar */
            else if (IS_BINARY_CONT_S2(typename Traits::ScalarType2, npy_bool)) {
                simd_binary_scalar2_compare<T, Op>(args, dimensions[0]);
                return;
            }
            else if (IS_BINARY_CONT(typename Traits::ScalarType2, npy_bool)) {
                simd_binary_compare<T, Op>(args, dimensions[0]);
                return;
            }
        }
    }

    BINARY_LOOP {
        if constexpr (Traits::IsBool) {
            npy_bool in1 = *((npy_bool *)ip1) != 0;
            npy_bool in2 = *((npy_bool *)ip2) != 0;
            *((npy_bool *)op1) = Traits_Op::template scalarCompare(in1, in2);
        } else {
            typename Traits::ScalarType2 in1 = *(typename Traits::ScalarType2 *)ip1;
            typename Traits::ScalarType2 in2 = *(typename Traits::ScalarType2 *)ip2;
            *((npy_bool *)op1) = Traits_Op::template scalarCompare(in1, in2);
        }
    }
}

/********************************************************************************
 ** Defining ufunc inner functions
 ********************************************************************************/
template<size_t Bits, bool IsSigned>
struct BitTraits;

template<>
struct BitTraits<8, true> {
    using Type = int8_t;
};

template<>
struct BitTraits<8, false> {
    using Type = uint8_t;
};

template<>
struct BitTraits<16, true> {
    using Type = int16_t;
};

template<>
struct BitTraits<16, false> {
    using Type = uint16_t;
};

template<>
struct BitTraits<32, true> {
    using Type = int32_t;
};

template<>
struct BitTraits<32, false> {
    using Type = uint32_t;
};

template<>
struct BitTraits<64, true> {
    using Type = int64_t;
};

template<>
struct BitTraits<64, false> {
    using Type = uint64_t;
};

#define DEFINE_COMPARE_FUNCTIONS(TYPE_UPPER, STYPE, is_signed) \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_greater) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    char *nargs[3] = {args[1], args[0], args[2]}; \
    npy_intp nsteps[3] = {steps[1], steps[0], steps[2]}; \
    using Traits = BitTraits<NPY_BITSOF_##STYPE, is_signed>; \
    run_binary_simd_compare<typename Traits::Type, CompareOp::Less>(nargs, dimensions, nsteps); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_greater_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    char *nargs[3] = {args[1], args[0], args[2]}; \
    npy_intp nsteps[3] = {steps[1], steps[0], steps[2]}; \
    using Traits = BitTraits<NPY_BITSOF_##STYPE, is_signed>; \
    run_binary_simd_compare<typename Traits::Type, CompareOp::LessEqual>(nargs, dimensions, nsteps); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_less) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    using Traits = BitTraits<NPY_BITSOF_##STYPE, is_signed>; \
    run_binary_simd_compare<typename Traits::Type, CompareOp::Less>(args, dimensions, steps); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_less_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    using Traits = BitTraits<NPY_BITSOF_##STYPE, is_signed>; \
    run_binary_simd_compare<typename Traits::Type, CompareOp::LessEqual>(args, dimensions, steps); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    using Traits = BitTraits<NPY_BITSOF_##STYPE, false>; \
    run_binary_simd_compare<typename Traits::Type, CompareOp::Equal>(args, dimensions, steps); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_not_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    using Traits = BitTraits<NPY_BITSOF_##STYPE, false>; \
    run_binary_simd_compare<typename Traits::Type, CompareOp::NotEqual>(args, dimensions, steps); \
}

DEFINE_COMPARE_FUNCTIONS(UBYTE,     BYTE,     false)
DEFINE_COMPARE_FUNCTIONS(USHORT,    SHORT,    false)
DEFINE_COMPARE_FUNCTIONS(UINT,      INT,      false)
DEFINE_COMPARE_FUNCTIONS(ULONG,     LONG,     false)
DEFINE_COMPARE_FUNCTIONS(ULONGLONG, LONGLONG, false)
DEFINE_COMPARE_FUNCTIONS(BYTE,      BYTE,     true)
DEFINE_COMPARE_FUNCTIONS(SHORT,     SHORT,    true)
DEFINE_COMPARE_FUNCTIONS(INT,       INT,      true)
DEFINE_COMPARE_FUNCTIONS(LONG,      LONG,     true)
DEFINE_COMPARE_FUNCTIONS(LONGLONG,  LONGLONG, true)


#define DEFINE_COMPARE_FUNCTIONS2(TYPE_UPPER, type, fp) \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_greater) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    char *nargs[3] = {args[1], args[0], args[2]}; \
    npy_intp nsteps[3] = {steps[1], steps[0], steps[2]}; \
    run_binary_simd_compare<type, CompareOp::Less>(nargs, dimensions, nsteps); \
    if (fp) npy_clear_floatstatus_barrier((char*)dimensions); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_greater_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    char *nargs[3] = {args[1], args[0], args[2]}; \
    npy_intp nsteps[3] = {steps[1], steps[0], steps[2]}; \
    run_binary_simd_compare<type, CompareOp::LessEqual>(nargs, dimensions, nsteps); \
    if (fp) npy_clear_floatstatus_barrier((char*)dimensions); \
}\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    run_binary_simd_compare<type, CompareOp::Equal>(args, dimensions, steps); \
    if (fp) npy_clear_floatstatus_barrier((char*)dimensions); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_not_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    run_binary_simd_compare<type, CompareOp::NotEqual>(args, dimensions, steps); \
    if (fp) npy_clear_floatstatus_barrier((char*)dimensions); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_less) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    run_binary_simd_compare<type, CompareOp::Less>(args, dimensions, steps); \
    if (fp) npy_clear_floatstatus_barrier((char*)dimensions); \
} \
\
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE_UPPER##_less_equal) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) { \
    run_binary_simd_compare<type, CompareOp::LessEqual>(args, dimensions, steps); \
    if (fp) npy_clear_floatstatus_barrier((char*)dimensions); \
}

DEFINE_COMPARE_FUNCTIONS2(BOOL,   bool,   false)
DEFINE_COMPARE_FUNCTIONS2(FLOAT,  float,  true)
DEFINE_COMPARE_FUNCTIONS2(DOUBLE, double, true)
