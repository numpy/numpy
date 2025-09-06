#include "numpy/npy_common.h"  // npy_intp
#include "loops.h"        // forward declarations
#include "loops_utils.h"  // is_mem_overlap

#include "simd/simd.hpp"  // Highway & NumPy SIMD Routines
#include <type_traits>    // std::enable_if_t
#include <algorithm>      // std::min
#include <cmath>          // std::sin, std::cos

// annonymous namespace for ODR since this source compiled multiple times
// based on the dispatch targets
namespace {
using namespace np::simd;

template <typename T>
struct OPsin {
    using LaneType = T;
    NPY_FINLINE T operator()(T x) const { return std::sin(x); }
#if NPY_HWY
    template <class X = T, typename = std::enable_if_t<kSupportLane<X>>>
    NPY_FINLINE Vec<T> operator()(const Vec<T>& x) {
      return sr::Sin(prec, x);
    }  
    Precise<T> prec;
#endif
};

template <typename T>
struct OPcos {
    using LaneType = T;
    NPY_FINLINE T operator()(T x) const { return std::cos(x); }
#if NPY_HWY
    template <class X = T, typename = std::enable_if_t<kSupportLane<X>>>
    NPY_FINLINE Vec<T> operator()(const Vec<T>& x) {
      return sr::Cos(prec, x);
    }  
    Precise<T> prec;
#endif
};

template <typename OP, typename T = typename OP::LaneType>
NPY_NOINLINE void
UnaryContig(OP &op, const T *src, T *dst, npy_intp len)
{
#if NPY_HWY
    if constexpr (kSupportLane<T>) {
        const npy_intp nlanes = Lanes<T>();
        for (; len > 0; len -= nlanes, src += nlanes, dst += nlanes) {
            Vec<T> x = LoadN(src, len);
            Vec<T> r = op(x);
            StoreN(r, dst, len);
        }
    }
#else
    if constexpr (0) {
    }
#endif
    else {
        for (const T *end = src + len; src < end; ++src, ++dst) {
            *dst = op(*src);
        }
    }
};

template <typename OP, typename T = typename OP::LaneType>
NPY_FINLINE void
UnaryOP(OP &op, char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    npy_intp len = dimensions[0];
    npy_intp sin = steps[0], sout = steps[1];
    char *in = args[0], *out = args[1];
    // We are dealing with large math kernels allowing inlining
    // or specialized implementations for non-contiguous, overlapping cases, or even
    // providing a separate iteration for contiguous tailing would easily bloat the
    // binary size.
    // Especially if we may need to support runtime dispatching for precision control.
    // If you really think that providing such specialization for these kinds of
    // kernels will increase performance, please make sure to benchmark it first and
    // compare the binary size before and after the change.
    // Note: the intrinsics of NumPy SIMD routines are inlined by default.
    if (NPY_UNLIKELY(is_mem_overlap(args[0], sin, args[1], sout, len) ||
                     sin != sizeof(T) || sout != sizeof(T))) {
        // this for non-contiguous or overlapping case
        constexpr npy_intp kBufferLen = 4096 / sizeof(T);
#if NPY_HWY
        HWY_ALIGN
#endif
        T buffer[kBufferLen];
        for (npy_intp processed = 0; processed < len;) {
            npy_intp chunk = std::min(kBufferLen, len - processed);
            for (npy_intp i = 0; i < chunk; ++i, in += sin) {
                buffer[i] = *reinterpret_cast<const T *>(in);
            }
            UnaryContig(op, buffer, buffer, chunk);
            for (npy_intp i = 0; i < chunk; ++i, out += sout) {
                *reinterpret_cast<T *>(out) = buffer[i];
            }
            processed += chunk;
        }
    }
    else {
        UnaryContig(op, (T *)in, (T *)out, len);
    }
}
}  // namespace

#define UMATH_UNARY_SR_IMPL(TYPE, type, op)                                 \
    NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_##op)(                 \
            char **args, npy_intp const *dimensions, npy_intp const *steps, \
            void *NPY_UNUSED(data))                                         \
    {                                                                       \
        OP##op<type> math_op;                                               \
        UnaryOP(math_op, args, dimensions, steps);                          \
    }

UMATH_UNARY_SR_IMPL(FLOAT, float, sin)
UMATH_UNARY_SR_IMPL(DOUBLE, double, sin)
UMATH_UNARY_SR_IMPL(FLOAT, float, cos)
UMATH_UNARY_SR_IMPL(DOUBLE, double, cos)
