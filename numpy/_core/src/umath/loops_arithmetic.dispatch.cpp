#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_cpu_dispatch.h"

#include "numpy/npy_common.h"
#include "numpy/npy_math.h"

#include "loops_utils.h"
#include "loops.h"
#include "fast_loop_macros.h"
#include "simd/simd.hpp"
#include "lowlevel_strided_loops.h"
#include "common.hpp"

#include <cstring> // for memcpy
#include <limits>
#include <cstdio>

using namespace np::simd;
#if NPY_HWY
namespace hn = np::simd::hn;
#endif

// Helper function to set float status
inline void set_float_status(bool overflow, bool divbyzero) {
    if (overflow) {
        npy_set_floatstatus_overflow();
    }
    if (divbyzero) {
        npy_set_floatstatus_divbyzero();
    }
}
#if NPY_HWY

// Signed integer  DIVIDE  by scalar

template <typename T>
void simd_divide_by_scalar_contig_signed(T* src, T scalar, T* dst, npy_intp len) {
    using D = _Tag<T>;
    const D d;
    const size_t N = Lanes(T{});

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
        const auto vec_min_val = Set(d, static_cast<T>(std::numeric_limits<T>::min()));
        size_t i = 0;
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = LoadU(src + i);
            const auto is_min_val = Eq(vec_src, vec_min_val);
            const auto vec_res = hn::IfThenElse(is_min_val, vec_min_val, hn::Neg(vec_src));
            StoreU(vec_res, dst + i);
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
        const auto vec_scalar = Set(d, scalar);
        const auto one = Set(d, static_cast<T>(1));
        const auto vec_zero = Xor(one, one);
        size_t i = 0;
        
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = LoadU(src + i);
            auto vec_div = Div(vec_src, vec_scalar);
            const auto vec_mul = Mul(vec_div, vec_scalar);
            const auto eq_mask = Eq(vec_src, vec_mul);
            const auto diff_signs = Lt(Xor(vec_src, vec_scalar), vec_zero);
            const auto adjust = AndNot(eq_mask, diff_signs);
            
            vec_div = hn::MaskedSubOr(vec_div, adjust, vec_div, one);
            StoreU(vec_div, dst + i);
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

// Unsigned integer  DIVIDE  by scalar

template <typename T>
void simd_divide_by_scalar_contig_unsigned(T* src, T scalar, T* dst, npy_intp len) {

    const size_t N = Lanes(T{});
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
        const auto vec_scalar = Set(scalar);
        size_t i = 0;
        for (; i + N <= static_cast<size_t>(len); i += N) {
            const auto vec_src = LoadU(src + i);
            const auto vec_res = Div(vec_src, vec_scalar);
            StoreU(vec_res, dst + i);
        }
        // Handle remaining elements
        for (; i < static_cast<size_t>(len); i++) {
            dst[i] = src[i] / scalar;
        }
    }

    set_float_status(false, raise_divbyzero);
}

// Signed integer  DIVIDE  array / array

template <typename T>
void simd_divide_contig_signed(T* src1, T* src2, T* dst, npy_intp len) {
    using D = _Tag<T>;
    const D d;
    const size_t N = Lanes(T{});

    bool raise_overflow = false;
    bool raise_divbyzero = false;
    const auto vec_one = Set(d, static_cast<T>(1));
    const auto vec_zero = Xor(vec_one, vec_one);
    const auto vec_min_val = Set(d, static_cast<T>(std::numeric_limits<T>::min()));
    const auto vec_neg_one = Set(d, static_cast<T>(-1));

    size_t i = 0;
    for (; i + N <= static_cast<size_t>(len); i += N) {
        const auto vec_a = LoadU(src1 + i);
        const auto vec_b = LoadU(src2 + i);
        
        const auto b_is_zero = Eq(vec_b, vec_zero);
        const auto a_is_min = Eq(vec_a, vec_min_val);
        const auto b_is_neg_one = Eq(vec_b, vec_neg_one);
        const auto overflow_cond = And(a_is_min, b_is_neg_one);
        
        const auto safe_div_mask = hn::Not(Or(b_is_zero, overflow_cond));
        const auto safe_b = hn::IfThenElse(Or(b_is_zero, overflow_cond), vec_one, vec_b);
        
        auto vec_div = Div(vec_a, safe_b);
        
        if (!hn::AllFalse(d, safe_div_mask)) {
            const auto vec_mul = Mul(vec_div, safe_b);
            const auto has_remainder = hn::Ne(vec_a, vec_mul);
            const auto a_sign = Lt(vec_a, vec_zero);
            const auto b_sign = Lt(vec_b, vec_zero);
            const auto different_signs = Xor(a_sign, b_sign);
            const auto needs_adjustment = And(safe_div_mask,
                                                And(has_remainder, different_signs));
            
            vec_div = hn::MaskedSubOr(vec_div, needs_adjustment, vec_div, vec_one);
        }
        
        vec_div = hn::IfThenElse(b_is_zero, vec_zero, vec_div);
        vec_div = hn::IfThenElse(overflow_cond, vec_min_val, vec_div);
        
        StoreU(vec_div, dst + i);
        
        if (!raise_divbyzero && !hn::AllFalse(d, b_is_zero)) {
            raise_divbyzero = true;
        }
        if (!raise_overflow && !hn::AllFalse(d, overflow_cond)) {
            raise_overflow = true;
        }
    }
    
    // Handle remaining elements
    for (; i < static_cast<size_t>(len); i++) {
        T a = src1[i];
        T b = src2[i];

        if (b == 0) {
            dst[i] = 0;
            raise_divbyzero = true;
        } 
        else if (a == std::numeric_limits<T>::min() && b == -1) {
            dst[i] = std::numeric_limits<T>::min();
            raise_overflow = true;
        } 
        else {
            T r = a / b;
            if (((a > 0) != (b > 0)) && ((r * b) != a)) {
                --r;
            }
            dst[i] = r;
        }
    }
    
    npy_clear_floatstatus();
    set_float_status(raise_overflow, raise_divbyzero);
}

// Unsigned integer  DIVIDE  array / array

template <typename T>
void simd_divide_contig_unsigned(T* src1, T* src2, T* dst, npy_intp len) {
    using D = _Tag<T>;
    const D d;
    const size_t N = Lanes(T{});

    bool raise_divbyzero = false;
    const auto vec_one = Set(d, static_cast<T>(1));
    const auto vec_zero = Xor(vec_one, vec_one);

    size_t i = 0;
    for (; i + N <= static_cast<size_t>(len); i += N) {
        const auto vec_a = LoadU(src1 + i);
        const auto vec_b = LoadU(src2 + i);

        const auto b_is_zero = Eq(vec_b, vec_zero);
        
        const auto safe_b = hn::IfThenElse(b_is_zero, vec_one, vec_b);
        
        auto vec_div = Div(vec_a, safe_b);
        
        vec_div = hn::IfThenElse(b_is_zero, vec_zero, vec_div);
        
        StoreU(vec_div, dst + i);
        
        if (!raise_divbyzero && !hn::AllFalse(d, b_is_zero)) {
            raise_divbyzero = true;
        }
    }
    
    // Handle remaining elements
    for (; i < static_cast<size_t>(len); i++) {
        T a = src1[i];
        T b = src2[i];

        if (b == 0) {
            dst[i] = 0;
            raise_divbyzero = true;
        } else {
            dst[i] = a / b;
        }
    }
    
    npy_clear_floatstatus();
    set_float_status(false, raise_divbyzero);
}

#endif // NPY_HWY

// Floor division for signed integers
template <typename T>
T floor_div(T n, T d) {
    if (NPY_UNLIKELY(d == 0 || (n == std::numeric_limits<T>::min() && d == -1))) {
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
    npy_clear_floatstatus();
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(T) {
            const T divisor = *reinterpret_cast<T*>(ip2);
            if (NPY_UNLIKELY(divisor == 0)) {
                npy_set_floatstatus_divbyzero();
                io1 = 0;
            } else if (NPY_UNLIKELY(io1 == std::numeric_limits<T>::min() && divisor == -1)) {
                npy_set_floatstatus_overflow();
                io1 = std::numeric_limits<T>::min();
            } else {
                io1 = floor_div(io1, divisor);
            }
        }
        *reinterpret_cast<T*>(iop1) = io1;
        return;
    }

#if NPY_HWY
    // Handle array-array case
    if (IS_BLOCKABLE_BINARY(sizeof(T), kMaxLanes<uint8_t>)) 
    {
        bool no_overlap = nomemoverlap(args[2], steps[2], args[0], steps[0], dimensions[0]) &&
                         nomemoverlap(args[2], steps[2], args[1], steps[1], dimensions[0]);
        // Check if we can use SIMD for contiguous arrays - all steps must equal to sizeof(T)
        if (steps[0] == sizeof(T) && steps[1] == sizeof(T) && steps[2] == sizeof(T) && no_overlap) {
            T* src1 = (T*)args[0];
            T* src2 = (T*)args[1];
            T* dst = (T*)args[2];
            simd_divide_contig_signed(src1, src2, dst, dimensions[0]);
            return;
        }
    }
    else if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(T), kMaxLanes<uint8_t>) &&
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
#endif // NPY_HWY

    // Scalar fallback
    // Fallback for non-blockable, in-place, or zero divisor cases
    BINARY_LOOP {
        const T dividend = *reinterpret_cast<T*>(ip1);
        const T divisor = *reinterpret_cast<T*>(ip2);
        T* result = reinterpret_cast<T*>(op1);

        if (NPY_UNLIKELY(divisor == 0)) {
            npy_set_floatstatus_divbyzero();
            *result = 0;
        } else if (NPY_UNLIKELY(dividend == std::numeric_limits<T>::min() && divisor == -1)) {
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
    npy_clear_floatstatus();
    if (IS_BINARY_REDUCE) {
        BINARY_REDUCE_LOOP(T) {
            const T d = *reinterpret_cast<T*>(ip2);
            if (NPY_UNLIKELY(d == 0)) {
                npy_set_floatstatus_divbyzero();
                io1 = 0;
            } else {
                io1 = io1 / d;
            }
        }
        *reinterpret_cast<T*>(iop1) = io1;
        return;
    }
#if NPY_HWY
    // Handle array-array case
    if (IS_BLOCKABLE_BINARY(sizeof(T), kMaxLanes<uint8_t>)) {
        bool no_overlap = nomemoverlap(args[2], steps[2], args[0], steps[0], dimensions[0]) &&
                         nomemoverlap(args[2], steps[2], args[1], steps[1], dimensions[0]);
        // Check if we can use SIMD for contiguous arrays - all steps must equal to sizeof(T)
        if (steps[0] == sizeof(T) && steps[1] == sizeof(T) && steps[2] == sizeof(T) && no_overlap) {
            T* src1 = (T*)args[0];
            T* src2 = (T*)args[1];
            T* dst = (T*)args[2];
            simd_divide_contig_unsigned(src1, src2, dst, dimensions[0]);
            return;
        }
    }
    else if (IS_BLOCKABLE_BINARY_SCALAR2(sizeof(T), kMaxLanes<uint8_t>) &&
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
#endif // NPY_HWY

    // Fallback for non-blockable, in-place, or zero divisor cases
    BINARY_LOOP {
        const T in1 = *reinterpret_cast<T*>(ip1);
        const T in2 = *reinterpret_cast<T*>(ip2);
        if (NPY_UNLIKELY(in2 == 0)) {
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

        if (NPY_UNLIKELY(divisor == 0)) {
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
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_divide)(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func) { \
            TYPE_divide<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
        NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_divide_indexed)(PyArrayMethod_Context *context, char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *func) { \
            return TYPE_divide_indexed<SCALAR_TYPE>(context, args, dimensions, steps, func); \
        }

// Macro to define the dispatch functions for unsigned types
#define DEFINE_DIVIDE_FUNCTION_UNSIGNED(TYPE, SCALAR_TYPE) \
        NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(TYPE##_divide)(char **args, npy_intp const *dimensions, npy_intp const *steps, void *func) { \
            TYPE_divide_unsigned<SCALAR_TYPE>(args, dimensions, steps, func); \
        } \
        NPY_NO_EXPORT int NPY_CPU_DISPATCH_CURFX(TYPE##_divide_indexed)(PyArrayMethod_Context *context, char * const*args, npy_intp const *dimensions, npy_intp const *steps, NpyAuxData *func) { \
            return TYPE_divide_unsigned_indexed<SCALAR_TYPE>(context, args, dimensions, steps, func); \
        }


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
