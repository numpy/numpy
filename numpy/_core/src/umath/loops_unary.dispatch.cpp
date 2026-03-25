#include "numpy/npy_math.h"
#include "simd/simd.hpp"
#include "loops.h"
#include "loops_utils.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"
#include <hwy/highway.h>

#include <type_traits>

namespace {
using namespace np::simd;

#if NPY_HWY
template <typename T>
HWY_INLINE HWY_ATTR Vec<T> simd_neg(Vec<T> v)
{
    if constexpr (std::is_unsigned_v<T>) {
        return hn::Sub(Zero<T>(), v);  
    } else {
        return hn::Neg(v);             
    }
}

// contiguous in, contiguous out
template <typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_negative_cc(const T* ip, T* op, npy_intp len)
{
    constexpr int UNROLL = 4;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    const int wstep = vstep * UNROLL;
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for (int i = 0; i < UNROLL; ++i)
            StoreU(simd_neg<T>(LoadU(ip + i * vstep)), op + i * vstep);
    }
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep)
        StoreU(simd_neg<T>(LoadU(ip)), op);
    for (; len > 0; --len, ++ip, ++op) { *op = -(*ip); }
}

// contiguous in, strided out  (32/64-bit types only)
template <typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_negative_cn(
        const T* ip, T* op, npy_intp ostride, npy_intp len)
{
    using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int64_t>;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    // Shift base pointer down to prevent negative Gather/Scatter indices
    npy_intp o_base_offset = (ostride < 0) ? (vstep - 1) * ostride : 0;
    auto idx = hn::Mul(hn::Iota(_Tag<IdxT>(), 0),
                       hn::Set(_Tag<IdxT>(), static_cast<IdxT>(ostride)));
    if (ostride < 0) {
        idx = hn::Sub(idx, hn::Set(_Tag<IdxT>(), static_cast<IdxT>(o_base_offset)));
    }

    constexpr int UNROLL = 4;
    const int wstep = vstep * UNROLL;
    for (; len >= wstep; len -= wstep, ip += static_cast<npy_intp>(wstep), op += ostride * wstep) {
        for (int i = 0; i < UNROLL; ++i)
            hn::ScatterIndex(simd_neg<T>(LoadU(ip + static_cast<npy_intp>(i) * vstep)),
                             _Tag<T>(), op + static_cast<npy_intp>(i) * vstep * ostride + o_base_offset, idx);
    }
    for (; len >= vstep; len -= vstep, ip += static_cast<npy_intp>(vstep), op += ostride * vstep)
        hn::ScatterIndex(simd_neg<T>(LoadU(ip)), _Tag<T>(), op + o_base_offset, idx);
    for (; len > 0; --len, ++ip, op += ostride) { *op = -(*ip); }
}

// strided in, contiguous out  (32/64-bit types only)
template <typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_negative_nc(
        const T* ip, npy_intp istride, T* op, npy_intp len)
{
    using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int64_t>;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();

    // Shift base pointer down to prevent negative Gather/Scatter indices
    npy_intp i_base_offset = (istride < 0) ? (vstep - 1) * istride : 0;
    auto idx = hn::Mul(hn::Iota(_Tag<IdxT>(), 0),
                       hn::Set(_Tag<IdxT>(), static_cast<IdxT>(istride)));
    if (istride < 0) {
        idx = hn::Sub(idx, hn::Set(_Tag<IdxT>(), static_cast<IdxT>(i_base_offset)));
    }

    constexpr int UNROLL = 4;
    const int wstep = vstep * UNROLL;
    for (; len >= wstep; len -= wstep, ip += istride * static_cast<npy_intp>(wstep), op += static_cast<npy_intp>(wstep)) {
        for (int i = 0; i < UNROLL; ++i)
            StoreU(simd_neg<T>(
                       hn::GatherIndex(_Tag<T>(), ip + static_cast<npy_intp>(i) * vstep * istride + i_base_offset, idx)),
                   op + static_cast<npy_intp>(i) * vstep);
    }
    for (; len >= vstep; len -= vstep, ip += istride * static_cast<npy_intp>(vstep), op += static_cast<npy_intp>(vstep))
        StoreU(simd_neg<T>(hn::GatherIndex(_Tag<T>(), ip + i_base_offset, idx)), op);
    for (; len > 0; --len, ip += istride, ++op) { *op = -(*ip); }
}

// strided in, strided out  (32/64-bit types only, non-SSE2)
#ifndef NPY_HAVE_SSE2
template <typename T>
HWY_ATTR SIMD_MSVC_NOINLINE
static void simd_unary_negative_nn(
        const T* ip, npy_intp istride, T* op, npy_intp ostride, npy_intp len)
{
    using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int64_t>;
    HWY_LANES_CONSTEXPR int vstep = Lanes<T>();
    npy_intp i_base_offset = (istride < 0) ? (vstep - 1) * istride : 0;
    auto iidx = hn::Mul(hn::Iota(_Tag<IdxT>(), 0), 
                        hn::Set(_Tag<IdxT>(), static_cast<IdxT>(istride)));
    if (istride < 0) {
        iidx = hn::Sub(iidx, hn::Set(_Tag<IdxT>(), static_cast<IdxT>(i_base_offset)));
    }
    npy_intp o_base_offset = (ostride < 0) ? (vstep - 1) * ostride : 0;
    auto oidx = hn::Mul(hn::Iota(_Tag<IdxT>(), 0), 
                        hn::Set(_Tag<IdxT>(), static_cast<IdxT>(ostride)));
    if (ostride < 0) {
        oidx = hn::Sub(oidx, hn::Set(_Tag<IdxT>(), static_cast<IdxT>(o_base_offset)));
    }

    constexpr int UNROLL = 2;
    const int wstep = vstep * UNROLL;
    for (; len >= wstep; len -= wstep, ip += istride * static_cast<npy_intp>(wstep), op += ostride * static_cast<npy_intp>(wstep)) {
        for (int i = 0; i < UNROLL; ++i)
            hn::ScatterIndex(
                simd_neg<T>(hn::GatherIndex(_Tag<T>(), ip + static_cast<npy_intp>(i) * vstep * istride + i_base_offset, iidx)),
                _Tag<T>(), op + static_cast<npy_intp>(i) * vstep * ostride + o_base_offset, oidx);
    }
    for (; len >= vstep; len -= vstep, ip += istride * static_cast<npy_intp>(vstep), op += ostride * static_cast<npy_intp>(vstep))
        hn::ScatterIndex(
            simd_neg<T>(hn::GatherIndex(_Tag<T>(), ip + i_base_offset, iidx)), _Tag<T>(), op + o_base_offset, oidx);
    for (; len > 0; --len, ip += istride, op += ostride) { *op = -(*ip); }
}
#endif // !NPY_HAVE_SSE2

template <typename T>
static NPY_INLINE int
run_simd_negative(char** args, npy_intp const* dimensions, npy_intp const* steps)
{
    npy_intp istep = steps[0], ostep = steps[1];
    npy_intp len   = dimensions[0];
    npy_intp istride_mut = istep / sizeof(T);
    npy_intp ostride_mut = ostep / sizeof(T);

    const T* ip = reinterpret_cast<const T*>(args[0]);
    T* op = reinterpret_cast<T*>(args[1]);

    // If both strides are negative, shift views and flip signs
    if (istride_mut < 0 && ostride_mut < 0) {
        ip += (len - 1) * istride_mut;
        op += (len - 1) * ostride_mut;
        istride_mut = -istride_mut;
        ostride_mut = -ostride_mut;
    }

    const npy_intp istride = istride_mut;
    const npy_intp ostride = ostride_mut;

    if (!is_mem_overlap(args[0], istep, args[1], ostep, len)) {
        if (istride == 1 && ostride == 1) {
            simd_unary_negative_cc(ip, op, len);
            return 1;
        }    
        if constexpr (sizeof(T) >= 4) {
            if (istride == 1 && ostride != 1) {
                simd_unary_negative_cn(ip, op, ostride, len);
                return 1;
            }
            if (istride != 1 && ostride == 1) {
                simd_unary_negative_nc(ip, istride, op, len);
                return 1;
            }
    #ifndef NPY_HAVE_SSE2
            if (istride != 1 && ostride != 1) {
                simd_unary_negative_nn(ip, istride, op, ostride, len);
                return 1;
            }
    #endif
        }
    } 
    return 0;
}

#endif // NPY_HWY

// Since, Highway only knows fixed-width lane types, we map platform-dependent
// integer types to fixed-width equivalents.
template <typename T>
struct SimdLaneType { using type = T; };
template <> struct SimdLaneType<long> {
    using type = std::conditional_t<sizeof(long) == 4, int32_t, int64_t>;
};
template <> struct SimdLaneType<unsigned long> {
    using type = std::conditional_t<sizeof(unsigned long) == 4, uint32_t, uint64_t>;
};
template <> struct SimdLaneType<long double> {
    using type = void;  // No HWY support for 80-bit extended precision
};
template <> struct SimdLaneType<long long> {
    using type = int64_t;
};
template <> struct SimdLaneType<unsigned long long> {
    using type = uint64_t;
};

template <typename T>
using SimdLaneType_t = typename SimdLaneType<T>::type;

// Dispatcher
template <typename T>
static NPY_INLINE void
unary_ufunc_loop(char** args, npy_intp const* dimensions, npy_intp const* steps)
{
#if NPY_HWY
    using ST = SimdLaneType_t<T>;
    if constexpr (!std::is_void_v<ST> && kSupportLane<ST>) {
        if (run_simd_negative<ST>(args, dimensions, steps)) { return; }
    }
#endif
    char *ip = args[0], *op = args[1];
    npy_intp istep = steps[0], ostep = steps[1], len = dimensions[0];
    constexpr int UNROLL = 8;
    for (; len >= UNROLL; len -= UNROLL, ip += istep*UNROLL, op += ostep*UNROLL) {
        for (int i = 0; i < UNROLL; ++i)
            *reinterpret_cast<T*>(op + i*ostep) = -(*reinterpret_cast<const T*>(ip + i*istep));
    }
    for (; len > 0; --len, ip += istep, op += ostep)
        *reinterpret_cast<T*>(op) = -(*reinterpret_cast<const T*>(ip));
}
}

// C API
extern "C" {

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(UBYTE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<uint8_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(USHORT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<uint16_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(UINT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<uint32_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(ULONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<unsigned long>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(ULONGLONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<unsigned long long>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BYTE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<int8_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(SHORT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<int16_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(INT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<int32_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<long>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONGLONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<long long>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<float>(args, dimensions, steps);
#if NPY_HWY
    npy_clear_floatstatus_barrier((char*)dimensions);
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<double>(args, dimensions, steps);
#if NPY_HWY_F64
    npy_clear_floatstatus_barrier((char*)dimensions);
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONGDOUBLE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<long double>(args, dimensions, steps);
    npy_clear_floatstatus_barrier((char*)dimensions);
}
} // extern "C"
