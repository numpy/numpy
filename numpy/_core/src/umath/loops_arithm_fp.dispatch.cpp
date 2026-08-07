#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "simd/simd.hpp"

namespace {
using namespace np::simd;

template <typename T>
struct OpAdd {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    NPY_FINLINE V operator()(const V &a, const V &b) const {
        return hn::Add(a, b);
    }
#endif
    NPY_FINLINE T operator()(T a, T b) const {
        return a + b; 
    }
};

template <typename T>
struct OpSub {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    NPY_FINLINE V operator()(const V &a, const V &b) const {
        return hn::Sub(a, b);
    }
#endif
    NPY_FINLINE T operator()(T a, T b) const {
        return a - b; 
    }
};

template <typename T>
struct OpMul {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    NPY_FINLINE V operator()(const V &a, const V &b) const {
        return hn::Mul(a, b);
    }
#endif
    NPY_FINLINE T operator()(T a, T b) const {
        return a * b; 
    }
};

template <typename T>
struct OpDiv {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    NPY_FINLINE V operator()(const V &a, const V &b) const {
        return hn::Div(a, b);
    }
#endif
    NPY_FINLINE T operator()(T a, T b) const {
        return a / b; 
    }
};

template <typename T>
struct OpMulComplex {
#if NPY_HWY
    template <typename V, typename = std::enable_if_t<kSupportLane<T>>>
    NPY_FINLINE V operator()(const V &a, const V &b) const {
        return hn::MulComplex(a, b);
    }
#endif
    NPY_FINLINE T operator()(T a, T b) const {
        return a * b; 
    }
};

template <typename T, typename Op>
NPY_FINLINE void
real_single_double_func(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    Op op_func;
    npy_intp len = dimensions[0];
    char *src0 = args[0], *src1 = args[1], *dst = args[2];
    npy_intp ssrc0 = steps[0], ssrc1 = steps[1], sdst = steps[2];

    // reduce
    if (ssrc0 == 0 && ssrc0 == sdst && src0 == dst) {
        if constexpr (std::is_same_v<Op, OpAdd<T>>) {
            if constexpr (std::is_same_v<T, float>){
                *((T*)src0) += FLOAT_pairwise_sum(src1, len, ssrc1);
            } else {
                *((T*)src0) += DOUBLE_pairwise_sum(src1, len, ssrc1);
            }
        }else{
            T acc = *((T*)src0);
            if (ssrc1 == sizeof(T)) {
                for (; len > 0; --len, src1 += sizeof(T)) {
                    acc = op_func(acc, *(T *)src1);
                }
            } else {
                for (; len > 0; --len, src1 += ssrc1) {
                    acc = op_func(acc, *(T *)src1);
                }
            }
            *((T*)src0) = acc;
        }
        return;
    }

#if HWY_HAVE_NEON_FP && !NPY_HWY_F64
    if constexpr (std::is_same_v<Op, OpDiv<T>>) {
        goto loop_scalar;
    }
#endif
    /**
     * The SIMD branch is disabled on armhf(armv7) due to the absence of native SIMD
     * support for single-precision floating-point division. Only scalar division is
     * supported natively, and without hardware for performance and accuracy comparison,
     * it's challenging to evaluate the benefits of emulated SIMD intrinsic versus
     * native scalar division.
     *
     * The `npyv_div_f32` universal intrinsic emulates the division operation using an
     * approximate reciprocal combined with 3 Newton-Raphson iterations for enhanced
     * precision. However, this approach has limitations:
     *
     * - It can cause unexpected floating-point overflows in special cases, such as when
     *   the divisor is subnormal (refer: https://github.com/numpy/numpy/issues/25097).
     *
     * - The precision may vary between the emulated SIMD and scalar division due to
     *   non-uniform branches (non-contiguous) in the code, leading to precision
     *   inconsistencies.
     *
     * - Considering the necessity of multiple Newton-Raphson iterations, the performance
     *   gain may not sufficiently offset these drawbacks.
     */

#if NPY_HWY
    using V = Vec<T>;
    if constexpr (kSupportLane<T>) {
        if (static_cast<size_t>(len) > Lanes<T>()*2 &&
            !is_mem_overlap(src0, ssrc0, dst, sdst, len) &&
            !is_mem_overlap(src1, ssrc1, dst, sdst, len)
        ) {
            const T *t_src0 = reinterpret_cast<const T*>(src0);
            const T *t_src1 = reinterpret_cast<const T*>(src1);
            T *t_dst = reinterpret_cast<T*>(dst);
            HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
            const int wstep = vstep * 2;
            // lots of specializations, to squeeze out max performance
            if (ssrc0 == sizeof(T) && ssrc0 == ssrc1 && ssrc0 == sdst) {
                for (; len >= wstep; len -= wstep, t_src0 += wstep, t_src1 += wstep, t_dst += wstep) {
                    V a0 = LoadU(t_src0);
                    V a1 = LoadU(t_src0 + vstep);
                    V b0 = LoadU(t_src1);
                    V b1 = LoadU(t_src1 + vstep);
                    V r0 = op_func(a0, b0);
                    V r1 = op_func(a1, b1);
                    StoreU(r0, t_dst);
                    StoreU(r1, t_dst + vstep);
                }
                for (; len > 0; len -= vstep, t_src0 += vstep, t_src1 += vstep, t_dst += vstep) {
                    if constexpr(std::is_same_v<Op, OpDiv<T>>){
                        V a = LoadNOr(Set<T>(1.0), t_src0, len);
                        V b = LoadNOr(Set<T>(1.0), t_src1, len);
                        V r = op_func(a, b);
                        StoreN(r, t_dst, len);
                     } else {
                        V a = LoadN(t_src0, len);
                        V b = LoadN(t_src1, len);
                        V r = op_func(a, b);
                        StoreN(r, t_dst, len);
                    }
                }
            }
            else if (ssrc0 == 0 && ssrc1 == sizeof(T) && sdst == ssrc1) {
                V a = Set(*t_src0);
                for (; len >= wstep; len -= wstep, t_src1 += wstep, t_dst += wstep) {
                    V b0 = LoadU(t_src1);
                    V b1 = LoadU(t_src1 + vstep);
                    V r0 = op_func(a, b0);
                    V r1 = op_func(a, b1);
                    StoreU(r0, t_dst);
                    StoreU(r1, t_dst + vstep);
                }
                for (; len > 0; len -= vstep, t_src1 += vstep, t_dst += vstep) {
                    if constexpr (std::is_same_v<Op, OpDiv<T>> || std::is_same_v<Op, OpMul<T>>) {
                        V b = LoadNOr(Set<T>(1.0), t_src1, len);
                        V r = op_func(a, b);
                        StoreN(r, t_dst, len);
                    } else {
                        V b = LoadN(t_src1, len);
                        V r = op_func(a, b);
                        StoreN(r, t_dst, len);
                    }
                }
            }
            else if (ssrc1 == 0 && ssrc0 == sizeof(T) && sdst == ssrc0) {
                V b = Set(*t_src1);
                for (; len >= wstep; len -= wstep, t_src0 += wstep, t_dst += wstep) {
                    V a0 = LoadU(t_src0);
                    V a1 = LoadU(t_src0 + vstep);
                    V r0 = op_func(a0, b);
                    V r1 = op_func(a1, b);
                    StoreU(r0, t_dst);
                    StoreU(r1, t_dst + vstep);
                }
                for (; len > 0; len -= vstep, t_src0 += vstep, t_dst += vstep) {
                    if constexpr (std::is_same_v<Op, OpMul<T>>) {
                        V a = LoadNOr(Set<T>(1.0), t_src0, len);
                        V r = op_func(a, b);
                        StoreN(r, t_dst, len);
                    } else if constexpr (std::is_same_v<Op, OpDiv<T>>) {
                        V a = LoadNOr(Set<T>(NPY_NAN), t_src0, len);
                        V r = op_func(a, b);
                        StoreN(r, t_dst, len);
                    } else {
                        V a = LoadN(t_src0, len);
                        V r = op_func(a, b);
                        StoreN(r, t_dst, len);
                    }
                }
            } else {
                goto loop_scalar;
            }
            return;
        }
    }
#endif    // NPY_HWY

loop_scalar:
    for (; len > 0; --len, src0 += ssrc0, src1 += ssrc1, dst += sdst) {
        const T a = *((T*)src0);
        const T b = *((T*)src1);
        *((T*)dst) = op_func(a, b);
    }
}

template <typename T, typename Op>
NPY_FINLINE int
real_single_double_indexed_func(char * const*args, npy_intp const *dimensions, npy_intp const *steps)
{
    Op op_func;
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];
    npy_intp i;
    T *indexed;
    for(i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        indexed  = (T *)(ip1 + is1 * indx);
        *indexed = op_func(*indexed, *(T *)value);
    }
    return 0;
}

template <typename T, typename Op>
NPY_FINLINE void
complex_single_double_func(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    Op op_func;
    using D = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
    npy_intp len = dimensions[0];
    char *b_src0 = args[0], *b_src1 = args[1], *b_dst = args[2];
    npy_intp b_ssrc0 = steps[0], b_ssrc1 = steps[1], b_sdst = steps[2];
    if constexpr (std::is_same_v<Op, OpAdd<T>>) {
        // reduce
        if (b_ssrc0 == 0 && b_ssrc0 == b_sdst && b_src0 == b_dst &&
            b_ssrc1 % (sizeof(T)*2) == 0
        ) {
            T *rl_im = (T *)b_src0;
            T rr, ri;
            if constexpr (std::is_same_v<T, float>){
                CFLOAT_pairwise_sum(&rr, &ri, b_src1, len * 2, b_ssrc1 / 2);
            } else {
                CDOUBLE_pairwise_sum(&rr, &ri, b_src1, len * 2, b_ssrc1 / 2);
            }
            rl_im[0] = op_func(rl_im[0], rr);
            rl_im[1] = op_func(rl_im[1], ri);
            return;
        }
    }

#if NPY_HWY
    using V = Vec<T>;
    using D_ = Vec<D>;
    if constexpr (kSupportLane<T>) {
        // Certain versions of Apple clang (commonly used in CI images) produce
        // non-deterministic output in the mul path with AVX2 enabled on x86_64.
        // Work around by scalarising.
        #if defined(NPY_CPU_AMD64) && defined(__clang__) \
                && defined(__apple_build_version__) \
                && __apple_build_version__ >= 14000000 \
                && __apple_build_version__ < 14030000
            if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                goto loop_scalar;
            }
        #endif  // end affected Apple clang.

        if (!is_mem_overlap(b_src0, b_ssrc0, b_dst, b_sdst, len) &&
            !is_mem_overlap(b_src1, b_ssrc1, b_dst, b_sdst, len) &&
            sizeof(T) == alignof(T) && b_ssrc0 % sizeof(T) == 0 && b_ssrc1 % sizeof(T) == 0 && 
            b_sdst % sizeof(T) == 0 && b_sdst != 0) {
            const T *src0 = (T*)b_src0;
            const T *src1 = (T*)b_src1;
                  T *dst  = (T*)b_dst;

            // Use signed type for strides to handle negative strides
            const npy_intp ssrc0 = b_ssrc0 / sizeof(T);
            const npy_intp ssrc1 = b_ssrc1 / sizeof(T);
            const npy_intp sdst  = b_sdst  / sizeof(T);

            HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
            const int wstep = vstep * 2;
            const int hstep = vstep / 2;

            // lots of specializations, to squeeze out max performance
            // contiguous
            if (ssrc0 == 2 && ssrc0 == ssrc1 && ssrc0 == sdst) {
                for (; len >= vstep; len -= vstep, src0 += wstep, src1 += wstep, dst += wstep) {
                    V a0 = LoadU(src0);
                    V a1 = LoadU(src0 + vstep);
                    V b0 = LoadU(src1);
                    V b1 = LoadU(src1 + vstep);
                    V r0 = op_func(a0, b0);
                    V r1 = op_func(a1, b1);
                    StoreU(r0, dst);
                    StoreU(r1, dst + vstep);
                }
                for (; len > 0; len -= hstep, src0 += vstep, src1 += vstep, dst += vstep) {
                    V a = LoadN(src0, len*2);
                    V b = LoadN(src1, len*2);
                    V r = op_func(a, b);
                    StoreN(r, dst, len*2);
                }
            }
            // scalar 0
            else if (ssrc0 == 0) {
                V a = hn::OddEven(Set(src0[1]), Set(src0[0]));
                // contiguous
                if (ssrc1 == 2 && sdst == ssrc1) {
                    for (; len >= vstep; len -= vstep, src1 += wstep, dst += wstep) {
                        V b0 = LoadU(src1);
                        V b1 = LoadU(src1 + vstep);
                        V r0 = op_func(a, b0);
                        V r1 = op_func(a, b1);
                        StoreU(r0, dst);
                        StoreU(r1, dst + vstep);
                    }
                    for (; len > 0; len -= hstep, src1 += vstep, dst += vstep) {
                        if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                            V b = LoadNOr(Set<T>(1.0), src1, len*2);
                            V r = op_func(a, b);
                            StoreN(r, dst, len*2);
                        } else {
                            V b = LoadN(src1, len*2);
                            V r = op_func(a, b);
                            StoreN(r, dst, len*2);
                        }
                    }
                }
                // non-contiguous
                else if (static_cast<D>(ssrc1) >= 0 && static_cast<D>(sdst) >= 0) {
                    if constexpr (sizeof(T) == 4) {
                        using V64 = Vec<double>;
                        using D64 = Vec<int64_t>;

                        bool can_vectorize = true;

                        if (!(ssrc1 % 2 == 0 && sdst % 2 == 0)) {
                            can_vectorize = false;
                        }
                        
                        if (can_vectorize) {
                            const int vstep64 = Lanes<double>();
                            D64 i1 = Mul(hn::Iota(_Tag<int64_t>(), 0), Set<int64_t>(ssrc1 / 2));
                            D64 i2 = Mul(hn::Iota(_Tag<int64_t>(), 0), Set<int64_t>(sdst / 2));

                            // Reinterpret the scalar as 64-bit for broadcasting
                            double a_as_double = *reinterpret_cast<const double*>(src0);
                            V64 a64 = Set(a_as_double);

                            for (; len >= vstep64; len -= vstep64, src1 += ssrc1 * vstep64, dst += sdst * vstep64)
                            {
                                V64 b64 = hn::GatherIndex(_Tag<double>(), reinterpret_cast<const double*>(src1), i1);
                                V b = hn::BitCast(_Tag<T>(), b64);
                                V a_broadcast = hn::BitCast(_Tag<T>(), a64);
                                V r = op_func(a_broadcast, b);
                                V64 r64 = hn::BitCast(_Tag<double>(), r);
                                hn::ScatterIndex(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i2);
                            }
                            // Handle remainder
                            if (len > 0) {
                                if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                                    V64 b64 = hn::MaskedGatherIndexOr(Set<double>(1.0),  hn::FirstN(_Tag<double>(), len), _Tag<double>(), 
                                        reinterpret_cast<const double*>(src1), i1);
                                    V b = hn::BitCast(_Tag<T>(), b64);
                                    V a_broadcast = hn::BitCast(_Tag<T>(), a64);
                                    V r = op_func(a_broadcast, b);
                                    V64 r64 = hn::BitCast(_Tag<double>(), r);
                                    hn::ScatterIndexN(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i2, len);
                                }else{
                                    V64 b64 = hn::GatherIndexN(_Tag<double>(), reinterpret_cast<const double*>(src1), i1, len);
                                    V b = hn::BitCast(_Tag<T>(), b64);
                                    V a_broadcast = hn::BitCast(_Tag<T>(), a64);
                                    V r = op_func(a_broadcast, b);
                                    V64 r64 = hn::BitCast(_Tag<double>(), r);
                                    hn::ScatterIndexN(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i2, len);
                                }
                            }
                        } else {
                            goto loop_scalar;
                        }
                    } else {
                        D_ i0 = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(ssrc1));
                        D_ i1 = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(sdst));
                        i0 = hn::OddEven(Add(i0, Set<D>(1)), i0);
                        i1 = hn::OddEven(Add(i1, Set<D>(1)), i1);
                        
                        for (; len >= vstep; len -= vstep, src1 += ssrc1*vstep, dst += sdst*vstep) {
                            V b0 = hn::GatherIndex(_Tag<T>(), src1, i0);
                            V b1 = hn::GatherIndex(_Tag<T>(), src1 + ssrc1*hstep, i0);
                            V r0 = op_func(a, b0);
                            V r1 = op_func(a, b1);
                            hn::ScatterIndex(r0, _Tag<T>(), dst, i1);
                            hn::ScatterIndex(r1, _Tag<T>(), dst + sdst*hstep, i1);
                        }
                        for (; len > 0; len -= hstep, src1 += ssrc1*hstep, dst += sdst*hstep) {
                            if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                                V b = hn::MaskedGatherIndexOr(Set<T>(1.0), hn::FirstN(_Tag<T>(), len*2), _Tag<T>(), src1, i0);
                                V r = op_func(a, b);
                                hn::ScatterIndexN(r, _Tag<T>(), dst, i1, len*2);
                            } else {
                                V b = hn::GatherIndexN(_Tag<T>(), src1, i0, len*2);
                                V r = op_func(a, b);
                                hn::ScatterIndexN(r, _Tag<T>(), dst, i1, len*2);
                            }
                        }
                    }
                }
                else {
                    goto loop_scalar;
                }
            }
            // scalar 1
            else if (ssrc1 == 0) {
                V b = hn::OddEven(Set<T>(src1[1]), Set<T>(src1[0]));
                if (ssrc0 == 2 && sdst == ssrc0) {
                    for (; len >= vstep; len -= vstep, src0 += wstep, dst += wstep) {
                        V a0 = LoadU(src0);
                        V a1 = LoadU(src0 + vstep);
                        V r0 = op_func(a0, b);
                        V r1 = op_func(a1, b);
                        StoreU<T>(r0, dst);
                        StoreU<T>(r1, dst + vstep);
                    }
                    for (; len > 0; len -= hstep, src0 += vstep, dst += vstep) {
                        if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                            V a = LoadNOr(Set<T>(1.0), src0, len*2);
                            V r = op_func(a, b);
                            StoreN(r, dst, len*2);
                        } else {
                            V a = LoadN(src0, len*2);
                            V r = op_func(a, b);
                            StoreN(r, dst, len*2);
                        }
                    }
                }
                // non-contiguous
                else if (static_cast<D>(ssrc0) >= 0 && static_cast<D>(sdst) >= 0) {
                    // For 32-bit float complex, use 64-bit gather/scatter
                    if constexpr (sizeof(T) == 4) {
                        using V64 = Vec<double>;
                        using D64 = Vec<int64_t>;

                        bool can_vectorize = true;

                        if (!(ssrc0 % 2 == 0 && sdst % 2 == 0)) {
                            can_vectorize = false;
                        }

                        if (can_vectorize) {
                            const int vstep64 = Lanes<double>();
                            D64 i0 = Mul(hn::Iota(_Tag<int64_t>(), 0), Set<int64_t>(ssrc0 / 2));
                            D64 i1 = Mul(hn::Iota(_Tag<int64_t>(), 0), Set<int64_t>(sdst / 2));

                            double b_as_double = *reinterpret_cast<const double*>(src1);
                            V64 b64 = Set(b_as_double);

                            for (; len >= vstep64; len -= vstep64, src0 += ssrc0 * vstep64, dst += sdst * vstep64) {
                                V64 a64 = hn::GatherIndex(_Tag<double>(), reinterpret_cast<const double*>(src0), i0);
                                V a = hn::BitCast(_Tag<T>(), a64);
                                V b_broadcast = hn::BitCast(_Tag<T>(), b64);
                                V r = op_func(a, b_broadcast);
                                V64 r64 = hn::BitCast(_Tag<double>(), r);
                                hn::ScatterIndex(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i1);
                            }
                            if (len > 0) {
                                if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                                    V64 a64 = hn::MaskedGatherIndexOr(Set<double>(1.0), hn::FirstN(_Tag<double>(), len), _Tag<double>(), 
                                        reinterpret_cast<const double*>(src0), i0);
                                    V a = hn::BitCast(_Tag<T>(), a64);
                                    V b_broadcast = hn::BitCast(_Tag<T>(), b64);
                                    V r = op_func(a, b_broadcast);
                                    V64 r64 = hn::BitCast(_Tag<double>(), r);
                                    hn::ScatterIndexN(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i1, len);
                                }else{
                                    V64 a64 = hn::GatherIndexN(_Tag<double>(), reinterpret_cast<const double*>(src0), i0, len);
                                    V a = hn::BitCast(_Tag<T>(), a64);
                                    V b_broadcast = hn::BitCast(_Tag<T>(), b64);
                                    V r = op_func(a, b_broadcast);
                                    V64 r64 = hn::BitCast(_Tag<double>(), r);
                                    hn::ScatterIndexN(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i1, len);
                                }
                            }
                        } else {
                            goto loop_scalar;
                        }
                    } else {
                        D_ i0 = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(ssrc0));
                        D_ i1 = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(sdst));
                        i0 = hn::OddEven(Add(i0, Set<D>(1)), i0);
                        i1 = hn::OddEven(Add(i1, Set<D>(1)), i1);
                        for (; len >= vstep; len -= vstep, src0 += ssrc0*vstep, dst += sdst*vstep) {
                            V a0 = hn::GatherIndex(_Tag<T>(), src0, i0);
                            V a1 = hn::GatherIndex(_Tag<T>(), src0 + ssrc0*hstep, i0);
                            V r0 = op_func(a0, b);
                            V r1 = op_func(a1, b);
                            hn::ScatterIndex(r0, _Tag<T>(), dst, i1);
                            hn::ScatterIndex(r1, _Tag<T>(), dst + sdst*hstep, i1);
                        }
                        for (; len > 0; len -= hstep, src0 += ssrc0*hstep, dst += sdst*hstep) {
                            if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                                V a = hn::MaskedGatherIndexOr(Set<T>(1.0), hn::FirstN(_Tag<T>(), len*2), _Tag<T>(), src0, i0);
                                V r = op_func(a, b);
                                hn::ScatterIndexN(r, _Tag<T>(), dst, i1, len*2);
                            } else {
                                V a = hn::GatherIndexN(_Tag<T>(), src0, i0, len*2);
                                V r = op_func(a, b);
                                hn::ScatterIndexN(r, _Tag<T>(), dst, i1, len*2);
                            }
                        }
                    }
                }
                else {
                    goto loop_scalar;
                }
            }
            // non-contiguous
            else if (static_cast<D>(ssrc0) >= 0 && static_cast<D>(ssrc1) >= 0 && static_cast<D>(sdst) >= 0) {
                if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
                    if constexpr (sizeof(T) == 4) {
                        using V64 = Vec<double>;
                        using D64 = Vec<int64_t>;

                        bool can_vectorize = true;

                        if (!(ssrc0 % 2 == 0 && ssrc1 % 2 == 0 && sdst % 2 == 0)) {
                            can_vectorize = false;
                        }

                        if (can_vectorize) {
                            const int vstep64 = Lanes<double>();
                            D64 i0 = Mul(hn::Iota(_Tag<int64_t>(), 0), Set<int64_t>(ssrc0/2));
                            D64 i1 = Mul(hn::Iota(_Tag<int64_t>(), 0), Set<int64_t>(ssrc1/2));
                            D64 i2 = Mul(hn::Iota(_Tag<int64_t>(), 0), Set<int64_t>(sdst/2));

                            for (; len >= vstep64; len -= vstep64, src0 += ssrc0  * vstep64,
                                 src1 += ssrc1 * vstep64, dst += sdst * vstep64) {
                                V64 a64 = hn::GatherIndex(_Tag<double>(), reinterpret_cast<const double*>(src0), i0);
                                V64 b64 = hn::GatherIndex(_Tag<double>(), reinterpret_cast<const double*>(src1), i1);
                                V a = hn::BitCast(_Tag<T>(), a64);
                                V b = hn::BitCast(_Tag<T>(), b64);
                                V r = op_func(a, b);
                                V64 r64 = hn::BitCast(_Tag<double>(), r);
                                hn::ScatterIndex(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i2);
                            }
                            if (len > 0) {
                                V64 a64 = hn::MaskedGatherIndexOr(Set<double>(1.0), hn::FirstN(_Tag<double>(), len), _Tag<double>(), 
                                    reinterpret_cast<const double*>(src0), i0);
                                V64 b64 = hn::MaskedGatherIndexOr(Set<double>(1.0), hn::FirstN(_Tag<double>(), len), _Tag<double>(), 
                                    reinterpret_cast<const double*>(src1), i1);
                                V a = hn::BitCast(_Tag<T>(), a64);
                                V b = hn::BitCast(_Tag<T>(), b64);
                                V r = op_func(a, b);
                                V64 r64 = hn::BitCast(_Tag<double>(), r);
                                hn::ScatterIndexN(r64, _Tag<double>(), reinterpret_cast<double*>(dst), i2, len);
                            }
                        } else {
                            goto loop_scalar;
                        }
                    } else {
                        D_ i0 = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(ssrc0));
                        D_ i1 = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(ssrc1));
                        D_ i2 = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(sdst));
                        i0 = hn::OddEven(Add(i0, Set<D>(1)), i0);
                        i1 = hn::OddEven(Add(i1, Set<D>(1)), i1);
                        i2 = hn::OddEven(Add(i2, Set<D>(1)), i2);
                        for (; len >= vstep; len -= vstep, src0 += ssrc0*vstep,
                            src1 += ssrc1*vstep, dst += sdst*vstep) {
                            V a0 = hn::GatherIndex(_Tag<T>(), src0, i0);
                            V a1 = hn::GatherIndex(_Tag<T>(), src0 + ssrc0*hstep, i0);
                            V b0 = hn::GatherIndex(_Tag<T>(), src1, i1);
                            V b1 = hn::GatherIndex(_Tag<T>(), src1 + ssrc1*hstep, i1);
                            V r0 = op_func(a0, b0);
                            V r1 = op_func(a1, b1);
                            hn::ScatterIndex(r0, _Tag<T>(), dst, i2);
                            hn::ScatterIndex(r1, _Tag<T>(), dst + sdst*hstep, i2);
                        }
                        for (; len > 0; len -= hstep, src0 += ssrc0*hstep,
                            src1 += ssrc1*hstep, dst += sdst*hstep) {
                            V a = hn::MaskedGatherIndexOr(Set<T>(1.0), hn::FirstN(_Tag<T>(), len*2), _Tag<T>(), src0, i0);
                            V b = hn::MaskedGatherIndexOr(Set<T>(1.0), hn::FirstN(_Tag<T>(), len*2), _Tag<T>(), src1, i1);
                            V r = op_func(a, b);
                            hn::ScatterIndexN(r, _Tag<T>(), dst, i2, len*2);
                        }
                    }
                } else {
                    goto loop_scalar;
                }
            }
            else {
                // Only multiply is vectorized for the generic non-contig case.
                goto loop_scalar;
            }
            return;
        }
    }
#endif    // NPY_HWY

loop_scalar:
    for (; len > 0; --len, b_src0 += b_ssrc0, b_src1 += b_ssrc1, b_dst += b_sdst) {
        const T a_r = ((T *)b_src0)[0];
        const T a_i = ((T *)b_src0)[1];
        const T b_r = ((T *)b_src1)[0];
        const T b_i = ((T *)b_src1)[1];
        if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
            ((T *)b_dst)[0] = a_r*b_r - a_i*b_i;
            ((T *)b_dst)[1] = a_r*b_i + a_i*b_r;
        } else {
            ((T *)b_dst)[0] = op_func(a_r, b_r);
            ((T *)b_dst)[1] = op_func(a_i, b_i);
        }
    }
}

template <typename T, typename Op>
NPY_FINLINE int
complex_single_double_indexed_func(char * const*args, npy_intp const *dimensions, npy_intp const *steps)
{
    Op op_func;
    char *ip1   = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1   = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];
    npy_intp i;
    T *indexed;
    for(i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        indexed = (T *)(ip1 + is1 * indx);
        const T b_r = ((T *)value)[0];
        const T b_i = ((T *)value)[1];
        if constexpr (std::is_same_v<Op, OpMulComplex<T>>) {
            const T a_r = indexed[0];
            const T a_i = indexed[1];
            indexed[0]  = a_r*b_r - a_i*b_i;
            indexed[1]  = a_r*b_i + a_i*b_r;
        } else {
            indexed[0]  = op_func(indexed[0], b_r);
            indexed[1]  = op_func(indexed[1], b_i);
        }
    }
    return 0;
}

template <typename T, int is_square>
NPY_FINLINE void
complex_conjugate_square_func(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    using D = std::conditional_t<sizeof(T) == 8, int64_t, int32_t>;
    npy_intp len = dimensions[0];
    char *b_src = args[0], *b_dst = args[1];
    npy_intp b_ssrc = steps[0], b_sdst = steps[1];

#if NPY_HWY
    using V = Vec<T>;
    using D_ = Vec<D>;
    if constexpr (kSupportLane<T>) {
        if (!is_mem_overlap(b_src, b_ssrc, b_dst, b_sdst, len) && sizeof(T) == alignof(T) &&
            b_ssrc % sizeof(T) == 0 && b_sdst % sizeof(T) == 0) {
            const T *src  = (T*)b_src;
                  T *dst  = (T*)b_dst;
            const npy_intp ssrc = b_ssrc / sizeof(T);
            const npy_intp sdst = b_sdst / sizeof(T);

            HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
            const int wstep = vstep * 2;
            const int hstep = vstep / 2;

            if (ssrc == 2 && ssrc == sdst) {
                for (; len >= vstep; len -= vstep, src += wstep, dst += wstep) {
                    V a0 = LoadU(src);
                    V a1 = LoadU(src + vstep);
                    if constexpr (is_square) {
                        V r0 = hn::MulComplex(a0, a0);
                        V r1 = hn::MulComplex(a1, a1);
                        StoreU<T>(r0, dst);
                        StoreU<T>(r1, dst + vstep);
                    } else {
                        V r0 = hn::ComplexConj(a0);
                        V r1 = hn::ComplexConj(a1);
                        StoreU<T>(r0, dst);
                        StoreU<T>(r1, dst + vstep);
                    }
                }
                for (; len > 0; len -= hstep, src += vstep, dst += vstep) {
                    V a = LoadN(src, len*2);
                    if constexpr (is_square) {
                        V r = hn::MulComplex(a, a);
                        StoreN(r, dst, len*2);
                    } else {
                        V r = hn::ComplexConj(a);
                        StoreN(r, dst, len*2);
                    }
                }
            }
            else if (ssrc == 2 && static_cast<D>(sdst) >= 0) {
                D_ i = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(sdst));
                i = hn::OddEven(Add(i, Set<D>(1)), i);
                for (; len >= vstep; len -= vstep, src += wstep, dst += sdst*vstep) {
                    V a0 = LoadU(src);
                    V a1 = LoadU(src + vstep);
                    if constexpr (is_square) {
                        V r0 = hn::MulComplex(a0, a0);
                        V r1 = hn::MulComplex(a1, a1);
                        hn::ScatterIndex(r0, _Tag<T>(), dst, i);
                        hn::ScatterIndex(r1, _Tag<T>(), dst + sdst*hstep, i);
                    } else {
                        V r0 = hn::ComplexConj(a0);
                        V r1 = hn::ComplexConj(a1);
                        hn::ScatterIndex(r0, _Tag<T>(), dst, i);
                        hn::ScatterIndex(r1, _Tag<T>(), dst + sdst*hstep, i);
                    }
                }
                for (; len > 0; len -= hstep, src += vstep, dst += sdst*hstep) {
                    V a = LoadN(src, len*2);
                    if constexpr (is_square) {
                        V r = hn::MulComplex(a, a);
                        hn::ScatterIndexN(r, _Tag<T>(), dst, i, len*2);
                    } else {
                        V r = hn::ComplexConj(a);
                        hn::ScatterIndexN(r, _Tag<T>(), dst, i, len*2);
                    }
                }
            }
            else if (sdst == 2 && static_cast<D>(ssrc) >= 0) {
                D_ i = Mul(hn::ShiftRight<1>(hn::Iota(_Tag<D>(), 0)), Set<D>(ssrc));
                i = hn::OddEven(Add(i, Set<D>(1)), i);
                for (; len >= vstep; len -= vstep, src += ssrc*vstep, dst += wstep) {
                    V a0 = hn::GatherIndex(_Tag<T>(), src, i);
                    V a1 = hn::GatherIndex(_Tag<T>(), src + ssrc*hstep, i);
                    if constexpr (is_square) {
                        V r0 = hn::MulComplex(a0, a0);
                        V r1 = hn::MulComplex(a1, a1);
                        StoreU<T>(r0, dst);
                        StoreU<T>(r1, dst + vstep);
                    } else {
                        V r0 = hn::ComplexConj(a0);
                        V r1 = hn::ComplexConj(a1);
                        StoreU<T>(r0, dst);
                        StoreU<T>(r1, dst + vstep);
                    }
                }
                for (; len > 0; len -= hstep, src += ssrc*hstep, dst += vstep) {
                    V a = hn::GatherIndexN(_Tag<T>(), src, i, len*2);
                    if constexpr (is_square) {
                        V r = hn::MulComplex(a, a);
                        StoreN(r, dst, len*2);
                    } else {
                        V r = hn::ComplexConj(a);
                        StoreN(r, dst, len*2);
                    }
                }
            }
            else {
                goto loop_scalar;
            }
            return;
        }
    }

loop_scalar:
#endif   // NPY_HWY

    for (; len > 0; --len, b_src += b_ssrc, b_dst += b_sdst) {
        const T rl = ((T *)b_src)[0];
        const T im = ((T *)b_src)[1];
        if constexpr (is_square) {
            ((T *)b_dst)[0] = rl*rl - im*im;
            ((T *)b_dst)[1] = rl*im + im*rl;
        } else {
            ((T *)b_dst)[0] = rl;
            ((T *)b_dst)[1] = -im;
        }
    }
}
} // namespace anonymous

//###############################################################################
//## Real Single/Double precision
//###############################################################################
/********************************************************************************
 ** Defining ufunc inner functions
 ********************************************************************************/
#define DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(TYPE, KIND, INTR, T) \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    real_single_double_func<T, Op##INTR<T>>(args, dimensions, steps); \
}

DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(FLOAT, add,       Add, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(FLOAT, subtract,  Sub, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(FLOAT, multiply,  Mul, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(FLOAT, divide,    Div, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(DOUBLE, add,      Add, npy_double)
DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(DOUBLE, subtract, Sub, npy_double)
DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(DOUBLE, multiply, Mul, npy_double)
DEFINE_REAL_SINGLE_DOUBLE_FUNCTION(DOUBLE, divide,   Div, npy_double)

#define DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(TYPE, KIND, INTR, T) \
NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND##_indexed) \
(PyArrayMethod_Context *NPY_UNUSED(context), char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) \
{ \
    return real_single_double_indexed_func<T, Op##INTR<T>>(args, dimensions, steps); \
}

DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(FLOAT, add,       Add, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(FLOAT, subtract,  Sub, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(FLOAT, multiply,  Mul, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(FLOAT, divide,    Div, npy_float)
DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(DOUBLE, add,      Add, npy_double)
DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(DOUBLE, subtract, Sub, npy_double)
DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(DOUBLE, multiply, Mul, npy_double)
DEFINE_REAL_SINGLE_DOUBLE_INDEXED_FUNCTION(DOUBLE, divide,   Div, npy_double)


//###############################################################################
//## Complex Single/Double precision
//###############################################################################
/********************************************************************************
 ** Defining ufunc inner functions
 ********************************************************************************/
#define DEFINE_COMPLEX_SINGLE_DOUBLE_FUNCTION(TYPE, KIND, INTR, T) \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    complex_single_double_func<T, Op##INTR<T>>(args, dimensions, steps); \
}

DEFINE_COMPLEX_SINGLE_DOUBLE_FUNCTION(CFLOAT,  add,      Add,        npy_float)
DEFINE_COMPLEX_SINGLE_DOUBLE_FUNCTION(CFLOAT,  subtract, Sub,        npy_float)
DEFINE_COMPLEX_SINGLE_DOUBLE_FUNCTION(CFLOAT,  multiply, MulComplex, npy_float)
DEFINE_COMPLEX_SINGLE_DOUBLE_FUNCTION(CDOUBLE, add,      Add,        npy_double)
DEFINE_COMPLEX_SINGLE_DOUBLE_FUNCTION(CDOUBLE, subtract, Sub,        npy_double)
DEFINE_COMPLEX_SINGLE_DOUBLE_FUNCTION(CDOUBLE, multiply, MulComplex, npy_double)


#define DEFINE_COMPLEX_SINGLE_DOUBLE_INDEXED_FUNCTION(TYPE, KIND, INTR, T) \
NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND##_indexed) \
(PyArrayMethod_Context *NPY_UNUSED(context), char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) \
{ \
    return complex_single_double_indexed_func<T, Op##INTR<T>>(args, dimensions, steps); \
}

DEFINE_COMPLEX_SINGLE_DOUBLE_INDEXED_FUNCTION(CFLOAT,  add,      Add,        npy_float)
DEFINE_COMPLEX_SINGLE_DOUBLE_INDEXED_FUNCTION(CFLOAT,  subtract, Sub,        npy_float)
DEFINE_COMPLEX_SINGLE_DOUBLE_INDEXED_FUNCTION(CFLOAT,  multiply, MulComplex, npy_float)
DEFINE_COMPLEX_SINGLE_DOUBLE_INDEXED_FUNCTION(CDOUBLE, add,      Add,        npy_double)
DEFINE_COMPLEX_SINGLE_DOUBLE_INDEXED_FUNCTION(CDOUBLE, subtract, Sub,        npy_double)
DEFINE_COMPLEX_SINGLE_DOUBLE_INDEXED_FUNCTION(CDOUBLE, multiply, MulComplex, npy_double)


#define DEFINE_COMPLEX_CONJUGATE_SQUARE_FUNCTION(TYPE, KIND, IS_AQUARE, T) \
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##KIND) \
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) \
{ \
    complex_conjugate_square_func<T, IS_AQUARE>(args, dimensions, steps); \
}

DEFINE_COMPLEX_CONJUGATE_SQUARE_FUNCTION(CFLOAT,  conjugate,  0,  npy_float)
DEFINE_COMPLEX_CONJUGATE_SQUARE_FUNCTION(CFLOAT,  square,     1,  npy_float)
DEFINE_COMPLEX_CONJUGATE_SQUARE_FUNCTION(CDOUBLE, conjugate,  0,  npy_double)
DEFINE_COMPLEX_CONJUGATE_SQUARE_FUNCTION(CDOUBLE, square,     1,  npy_double)
