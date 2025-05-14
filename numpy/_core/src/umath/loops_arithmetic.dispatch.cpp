#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/npy_common.h"
#include "numpy/npy_math.h"

#include "loops_utils.h"
#include "loops.h"
#include "fast_loop_macros.h"
#include "simd/simd.h"
#include "lowlevel_strided_loops.h"
#include "common.hpp"

#include <cstring> // for memcpy
#include <limits>
#include <cstdio>

#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {

// Helper function to set float status
inline void set_float_status(bool overflow, bool divbyzero) {
    if (overflow) {
        npy_set_floatstatus_overflow();
    }
    if (divbyzero) {
        npy_set_floatstatus_divbyzero();
    }
}
#if NPY_SIMD
// Signed integer division
template <typename T>
void simd_divide_by_scalar_contig_signed(T* src, T scalar, T* dst, npy_intp len) {
    using D = hn::ScalableTag<T>;
    const D d;
    const size_t N = hn::Lanes(d);

    bool raise_overflow = false;
    bool raise_divbyzero = false;

    if (scalar == 0) {
        // Handle division by zero
        std::fill(dst, dst + len, static_cast<T>(0));
        raise_divbyzero = true;
    }
    else if (scalar == 1) {
        // Special case for division by 1
        if (src != dst) {
            std::memcpy(dst, src, len * sizeof(T));
        }
    }
    else if (scalar == static_cast<T>(-1)) {
        const auto vec_min_val = hn::Set(d, std::numeric_limits<T>::min());
        size_t i = 0;
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = hn::LoadU(d, src + i);
            const auto is_min_val = hn::Eq(vec_src, vec_min_val);
            const auto vec_res = hn::IfThenElse(is_min_val, vec_min_val, hn::Neg(vec_src));
            hn::StoreU(vec_res, d, dst + i);
            if (!raise_overflow && !hn::AllFalse(d, is_min_val)) {
                raise_overflow = true;
            }
        }
        // Handle remaining elements
        for (; i < static_cast<size_t>(len); i++) {
            T val = src[i];
            if (val == std::numeric_limits<T>::min()) {
                dst[i] = std::numeric_limits<T>::min();
                raise_overflow = true;
            } else {
                dst[i] = -val;
            }
        }
    }
    else {
        // General case with floor division semantics
        const auto vec_scalar = hn::Set(d, scalar);
        const auto vec_zero = hn::Zero(d);
        const auto one = hn::Set(d, static_cast<T>(1));
        size_t i = 0;
        
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = hn::LoadU(d, src + i);
            auto vec_div = hn::Div(vec_src, vec_scalar);
            const auto vec_mul = hn::Mul(vec_div, vec_scalar);
            const auto eq_mask = hn::Eq(vec_src, vec_mul);
            const auto diff_signs = hn::Lt(hn::Xor(vec_src, vec_scalar), vec_zero);
            const auto adjust = hn::AndNot(eq_mask, diff_signs);
            
            vec_div = hn::MaskedSubOr(vec_div, adjust, vec_div, one);
            hn::StoreU(vec_div, d, dst + i);
        }
        
        // Handle remaining elements with scalar code
        for (; i < static_cast<size_t>(len); i++) {
            T n = src[i];
            T r = n / scalar;
            if (((n > 0) != (scalar > 0)) && ((r * scalar) != n)) {
                --r;
            }
            dst[i] = r;
        }
    }
    set_float_status(raise_overflow, raise_divbyzero);
}

// Unsigned integer division
template <typename T>
void simd_divide_by_scalar_contig_unsigned(T* src, T scalar, T* dst, npy_intp len) {
    using D = hn::ScalableTag<T>;
    const D d;
    const size_t N = hn::Lanes(d);

    bool raise_divbyzero = false;

    if (scalar == 0) {
        // Handle division by zero
        std::fill(dst, dst + len, static_cast<T>(0));
        raise_divbyzero = true;
    }
    else if (scalar == 1) {
        // Special case for division by 1
        if (src != dst) {
            std::memcpy(dst, src, len * sizeof(T));
        }
    }
    else {
        const auto vec_scalar = hn::Set(d, scalar);
        size_t i = 0;
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = hn::LoadU(d, src + i);
            const auto vec_res = hn::Div(vec_src, vec_scalar);
            hn::StoreU(vec_res, d, dst + i);
        }
        // Handle remaining elements
        for (; i < static_cast<size_t>(len); i++) {
            dst[i] = src[i] / scalar;
        }
    }

    set_float_status(false, raise_divbyzero);
}
#endif // NPY_SIMD
// Floor division for signed integers
template <typename T>
T floor_div(T n, T d) {
    if (HWY_UNLIKELY(d == 0 || (n == std::numeric_limits<T>::min() && d == -1))) {
        if (d == 0) {
            npy_set_floatstatus_divbyzero();
            return 0;
        }
        else {
            npy_set_floatstatus_overflow();
            return std::numeric_limits<T>::min();
        }
    }
    T r = n / d;
    if (((n > 0) != (d > 0)) && ((r * d) != n)) {
        --r;
    }
    return r;
}

// Dispatch functions for signed integer division
template <typename T>
void TYPE_divide(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) {
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(T) {
            const T divisor = *reinterpret_cast<T*>(ip2);
            if (HWY_UNLIKELY(divisor == 0)) {
                npy_set_floatstatus_divbyzero();
                io1 = 0;
            } else if (HWY_UNLIKELY(io1 == std::numeric_limits<T>::min() && divisor == -1)) {
                npy_set_floatstatus_overflow();
                io1 = std::numeric_limits<T>::min();
            } else {
                io1 = floor_div(io1, divisor);
            }
        }
        *reinterpret_cast<T*>(iop1) = io1;
        return;
    }
#if NPY_SIMD   
    if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(T), NPY_SIMD_WIDTH) &&
        *reinterpret_cast<T*>(args[1]) != 0)
    {
        bool no_overlap = nomemoverlap(args[2], steps[2], args[0], steps[0], dimensions[0]);
        if (no_overlap) {
            T* src1 = reinterpret_cast<T*>(args[0]);
            T* src2 = reinterpret_cast<T*>(args[1]);
            T* dst = reinterpret_cast<T*>(args[2]);
            simd_divide_by_scalar_contig_signed(src1, *src2, dst, dimensions[0]);
            return;
        }
    }
#endif // NPY_SIMD

    // Fallback for non-blockable, in-place, or zero divisor cases
    BINARY_LOOP {
        const T dividend = *reinterpret_cast<T*>(ip1);
        const T divisor = *reinterpret_cast<T*>(ip2);
        T* result = reinterpret_cast<T*>(op1);

        if (HWY_UNLIKELY(divisor == 0)) {
            npy_set_floatstatus_divbyzero();
            *result = 0;
        } else if (HWY_UNLIKELY(dividend == std::numeric_limits<T>::min() && divisor == -1)) {
            npy_set_floatstatus_overflow();
            *result = std::numeric_limits<T>::min();
        } else {
            *result = floor_div(dividend, divisor);
        }
    }
}

// Dispatch functions for unsigned integer division
template <typename T>
void TYPE_divide_unsigned(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func)) {
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(T) {
            const T d = *reinterpret_cast<T*>(ip2);
            if (HWY_UNLIKELY(d == 0)) {
                npy_set_floatstatus_divbyzero();
                io1 = 0;
            } else {
                io1 = io1 / d;
            }
        }
        *reinterpret_cast<T*>(iop1) = io1;
        return;
    }
#if NPY_SIMD
    if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(T), NPY_SIMD_WIDTH) &&
        *reinterpret_cast<T*>(args[1]) != 0)
    {
        bool no_overlap = nomemoverlap(args[2], steps[2], args[0], steps[0], dimensions[0]);
        if (no_overlap) {
            T* src1 = reinterpret_cast<T*>(args[0]);
            T* src2 = reinterpret_cast<T*>(args[1]);
            T* dst  = reinterpret_cast<T*>(args[2]);
            simd_divide_by_scalar_contig_unsigned(src1, *src2, dst, dimensions[0]);
            return;
        }
    }
#endif // NPY_SIMD

    // Fallback for non-blockable, in-place, or zero divisor cases
    BINARY_LOOP {
        const T in1 = *reinterpret_cast<T*>(ip1);
        const T in2 = *reinterpret_cast<T*>(ip2);
        if (HWY_UNLIKELY(in2 == 0)) {
            npy_set_floatstatus_divbyzero();
            *reinterpret_cast<T*>(op1) = 0;
        } else {
            *reinterpret_cast<T*>(op1) = in1 / in2;
        }
    }
}

// Indexed division for signed integers
template <typename T>
int TYPE_divide_indexed(PyArrayMethod_Context *NPY_UNUSED(context), 
                       char * const*args, npy_intp const *dimensions, 
                       npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) {
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];

    for(npy_intp i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        T* indexed = (T*)(ip1 + is1 * indx);
        T divisor = *(T*)value;
        *indexed = floor_div(*indexed, divisor);
    }
    return 0;
}

// Indexed division for unsigned integers
template <typename T>
int TYPE_divide_unsigned_indexed(PyArrayMethod_Context *NPY_UNUSED(context), 
                               char * const*args, npy_intp const *dimensions, 
                               npy_intp const *steps, NpyAuxData *NPY_UNUSED(func)) {
    char *ip1 = args[0];
    char *indxp = args[1];
    char *value = args[2];
    npy_intp is1 = steps[0], isindex = steps[1], isb = steps[2];
    npy_intp shape = steps[3];
    npy_intp n = dimensions[0];

    for(npy_intp i = 0; i < n; i++, indxp += isindex, value += isb) {
        npy_intp indx = *(npy_intp *)indxp;
        if (indx < 0) {
            indx += shape;
        }
        T* indexed = (T*)(ip1 + is1 * indx);
        T divisor = *(T*)value;

        if (HWY_UNLIKELY(divisor == 0)) {
            npy_set_floatstatus_divbyzero();
            *indexed = 0;
        } else {
            *indexed = *indexed / divisor;
        }
    }
    return 0;
}

// Macro to define the dispatch functions for signed types
#define DEFINE_DIVIDE_FUNCTION(TYPE, SCALAR_TYPE) \
    extern "C" { \
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_divide)(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func) { \
            TYPE_divide<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
        NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_divide_indexed)(PyArrayMethod_Context *context, char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *func) { \
            return TYPE_divide_indexed<SCALAR_TYPE>(context, args, dimensions, steps, func); \
        } \
    } // extern "C"

// Macro to define the dispatch functions for unsigned types
#define DEFINE_DIVIDE_FUNCTION_UNSIGNED(TYPE, SCALAR_TYPE) \
    extern "C" { \
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_divide)(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func) { \
            TYPE_divide_unsigned<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
        NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_divide_indexed)(PyArrayMethod_Context *context, char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *func) { \
            return TYPE_divide_unsigned_indexed<SCALAR_TYPE>(context, args, dimensions, steps, func); \
        } \
    } // extern "C"


#ifdef NPY_CPU_DISPATCH_CURFX
    DEFINE_DIVIDE_FUNCTION(BYTE, int8_t)
    DEFINE_DIVIDE_FUNCTION(SHORT, int16_t)
    DEFINE_DIVIDE_FUNCTION(INT, int32_t)
    #if NPY_SIZEOF_LONG == 4
        DEFINE_DIVIDE_FUNCTION(LONG, int32_t)
    #elif NPY_SIZEOF_LONG == 8
        DEFINE_DIVIDE_FUNCTION(LONG, int64_t)
    #endif
    DEFINE_DIVIDE_FUNCTION(LONGLONG, int64_t)
#endif

#ifdef NPY_CPU_DISPATCH_CURFX
    DEFINE_DIVIDE_FUNCTION_UNSIGNED(UBYTE, uint8_t)
    DEFINE_DIVIDE_FUNCTION_UNSIGNED(USHORT, uint16_t)
    DEFINE_DIVIDE_FUNCTION_UNSIGNED(UINT, uint32_t)
    #if NPY_SIZEOF_LONG == 4
        DEFINE_DIVIDE_FUNCTION_UNSIGNED(ULONG, uint32_t)
    #elif NPY_SIZEOF_LONG == 8
        DEFINE_DIVIDE_FUNCTION_UNSIGNED(ULONG, uint64_t)
    #endif
    DEFINE_DIVIDE_FUNCTION_UNSIGNED(ULONGLONG, uint64_t)
#endif

#undef DEFINE_DIVIDE_FUNCTION
#undef DEFINE_DIVIDE_FUNCTION_UNSIGNED

} // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
