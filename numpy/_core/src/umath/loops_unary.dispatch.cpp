#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops.h"
#include "loops_utils.h"
#include "lowlevel_strided_loops.h"
#include "fast_loop_macros.h"

#include <type_traits>

#if NPY_SIMD
// 8-bit signed
static NPY_INLINE npyv_s8
npyv_negative_s8(npyv_s8 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 8 < 64)
    return npyv_reinterpret_s8_s8(vnegq_s8(npyv_reinterpret_s8_s8(v)));
#else
    const npyv_s8 m1 = npyv_setall_s8((npyv_lanetype_s8)-1);
    return npyv_sub_s8(npyv_xor_s8(v, m1), m1);
#endif
}

// 8-bit unsigned
static NPY_INLINE npyv_u8
npyv_negative_u8(npyv_u8 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 8 < 64)
    return npyv_reinterpret_u8_s8(vnegq_s8(npyv_reinterpret_s8_u8(v)));
#else
    const npyv_u8 m1 = npyv_setall_u8((npyv_lanetype_u8)-1);
    return npyv_sub_u8(npyv_xor_u8(v, m1), m1);
#endif
}

// 16-bit signed
static NPY_INLINE npyv_s16
npyv_negative_s16(npyv_s16 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 16 < 64)
    return npyv_reinterpret_s16_s16(vnegq_s16(npyv_reinterpret_s16_s16(v)));
#else
    const npyv_s16 m1 = npyv_setall_s16((npyv_lanetype_s16)-1);
    return npyv_sub_s16(npyv_xor_s16(v, m1), m1);
#endif
}

// 16-bit unsigned
static NPY_INLINE npyv_u16
npyv_negative_u16(npyv_u16 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 16 < 64)
    return npyv_reinterpret_u16_s16(vnegq_s16(npyv_reinterpret_s16_u16(v)));
#else
    const npyv_u16 m1 = npyv_setall_u16((npyv_lanetype_u16)-1);
    return npyv_sub_u16(npyv_xor_u16(v, m1), m1);
#endif
}

// 32-bit signed
static NPY_INLINE npyv_s32
npyv_negative_s32(npyv_s32 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 32 < 64)
    return npyv_reinterpret_s32_s32(vnegq_s32(npyv_reinterpret_s32_s32(v)));
#else
    const npyv_s32 m1 = npyv_setall_s32((npyv_lanetype_s32)-1);
    return npyv_sub_s32(npyv_xor_s32(v, m1), m1);
#endif
}

// 32-bit unsigned
static NPY_INLINE npyv_u32
npyv_negative_u32(npyv_u32 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 32 < 64)
    return npyv_reinterpret_u32_s32(vnegq_s32(npyv_reinterpret_s32_u32(v)));
#else
    const npyv_u32 m1 = npyv_setall_u32((npyv_lanetype_u32)-1);
    return npyv_sub_u32(npyv_xor_u32(v, m1), m1);
#endif
}

// 64-bit signed
static NPY_INLINE npyv_s64
npyv_negative_s64(npyv_s64 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 64 < 64)
    return npyv_reinterpret_s64_s64(vnegq_s64(npyv_reinterpret_s64_s64(v)));
#else
    const npyv_s64 m1 = npyv_setall_s64((npyv_lanetype_s64)-1);
    return npyv_sub_s64(npyv_xor_s64(v, m1), m1);
#endif
}

// 64-bit unsigned
static NPY_INLINE npyv_u64
npyv_negative_u64(npyv_u64 v)
{
#if defined(NPY_HAVE_NEON) && (defined(__aarch64__) || 64 < 64)
    return npyv_reinterpret_u64_s64(vnegq_s64(npyv_reinterpret_s64_u64(v)));
#else
    const npyv_u64 m1 = npyv_setall_u64((npyv_lanetype_u64)-1);
    return npyv_sub_u64(npyv_xor_u64(v, m1), m1);
#endif
}

// 32-bit float
#if NPY_SIMD_F32
static NPY_INLINE npyv_f32
npyv_negative_f32(npyv_f32 v)
{
#if defined(NPY_HAVE_NEON)
    return vnegq_f32(v);
#else
    const npyv_f32 signmask = npyv_setall_f32(-0.f);
    return npyv_xor_f32(v, signmask);
#endif
}
#endif // NPY_SIMD_F32

// 64-bit float
#if NPY_SIMD_F64
static NPY_INLINE npyv_f64
npyv_negative_f64(npyv_f64 v)
{
#if defined(NPY_HAVE_NEON)
    return vnegq_f64(v);
#else
    const npyv_f64 signmask = npyv_setall_f64(-0.);
    return npyv_xor_f64(v, signmask);
#endif
}
#endif // NPY_SIMD_F64
#endif // NPY_SIMD

struct negative_t {};

// SIMD Type Traits
template<typename T> struct SIMDTypeTraits;

template<> struct SIMDTypeTraits<signed char> {
#if NPY_SIMD
    using simd_type = npyv_s8;
    using lane_type = npyv_lanetype_s8;
#else
    using simd_type = void;
    using lane_type = npy_int8;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = false;
};
template<> struct SIMDTypeTraits<unsigned char> {
#if NPY_SIMD
    using simd_type = npyv_u8;
    using lane_type = npyv_lanetype_u8;
#else
    using simd_type = void;
    using lane_type = npy_uint8;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = false;
};

template<> struct SIMDTypeTraits<short> {
#if NPY_SIMD
    using simd_type = npyv_s16;
    using lane_type = npyv_lanetype_s16;
#else
    using simd_type = void;
    using lane_type = npy_int16;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = false;
};
template<> struct SIMDTypeTraits<unsigned short> {
#if NPY_SIMD
    using simd_type = npyv_u16;
    using lane_type = npyv_lanetype_u16;
#else
    using simd_type = void;
    using lane_type = npy_uint16;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = false;
};

template<> struct SIMDTypeTraits<int> {
#if NPY_SIMD
    using simd_type = npyv_s32;
    using lane_type = npyv_lanetype_s32;
#else
    using simd_type = void;
    using lane_type = npy_int32;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = true;
};
template<> struct SIMDTypeTraits<unsigned int> {
#if NPY_SIMD
    using simd_type = npyv_u32;
    using lane_type = npyv_lanetype_u32;
#else
    using simd_type = void;
    using lane_type = npy_uint32;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = true;
};

template<> struct SIMDTypeTraits<long> {
#if NPY_SIMD
    #if NPY_SIZEOF_LONG == 4
        using simd_type = npyv_s32;
        using lane_type = npyv_lanetype_s32;
    #else
        using simd_type = npyv_s64;
        using lane_type = npyv_lanetype_s64;
    #endif
#else
    using simd_type = void;
    using lane_type = long;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = true;
};
template<> struct SIMDTypeTraits<unsigned long> {
#if NPY_SIMD
    #if NPY_SIZEOF_LONG == 4
        using simd_type = npyv_u32;
        using lane_type = npyv_lanetype_u32;
    #else
        using simd_type = npyv_u64;
        using lane_type = npyv_lanetype_u64;
    #endif
#else
    using simd_type = void;
    using lane_type = unsigned long;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = true;
};

template<> struct SIMDTypeTraits<long long> {
#if NPY_SIMD
    using simd_type = npyv_s64;
    using lane_type = npyv_lanetype_s64;
#else
    using simd_type = void;
    using lane_type = long long;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = true;
};
template<> struct SIMDTypeTraits<unsigned long long> {
#if NPY_SIMD
    using simd_type = npyv_u64;
    using lane_type = npyv_lanetype_u64;
#else
    using simd_type = void;
    using lane_type = unsigned long long;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD);
    static constexpr bool supports_ncontig = true;
};

template<> struct SIMDTypeTraits<float> {
#if NPY_SIMD_F32
    using simd_type = npyv_f32;
    using lane_type = npyv_lanetype_f32;
#else
    using simd_type = void;
    using lane_type = float;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD_F32);
    static constexpr bool supports_ncontig = true;
};
template<> struct SIMDTypeTraits<double> {
#if NPY_SIMD_F64
    using simd_type = npyv_f64;
    using lane_type = npyv_lanetype_f64;
#else
    using simd_type = void;
    using lane_type = double;
#endif
    static constexpr bool has_simd = static_cast<bool>(NPY_SIMD_F64);
    static constexpr bool supports_ncontig = true;
};

template<> struct SIMDTypeTraits<long double> {
    using simd_type = void;
    using lane_type = long double;
    static constexpr bool has_simd = false;
    static constexpr bool supports_ncontig = false;
};

/* Scalar Operations */
template <typename T>
static constexpr T scalar_negative(T x) noexcept { return -x; }

// Operation Traits
template <typename Op>
struct UnaryOpTraits;

template <>
struct UnaryOpTraits<negative_t> {
    template <typename T>
    static constexpr T scalar_op(T x) noexcept { 
        return scalar_negative(x); 
    }

#if NPY_SIMD
    // Dispatch based on scalar type T
    template <typename T, typename SIMD>
    static NPY_INLINE SIMD simd_op(SIMD v) {
        if constexpr (std::is_same_v<T, int8_t>) {
            return npyv_negative_s8(v);
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return npyv_negative_u8(v);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return npyv_negative_s16(v);
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return npyv_negative_u16(v);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return npyv_negative_s32(v);
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return npyv_negative_u32(v);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return npyv_negative_s64(v);
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            return npyv_negative_u64(v);
        } else if constexpr (std::is_same_v<T, long>) {
#if SIZEOF_LONG == 4
            return npyv_negative_s32(v);
#else
            return npyv_negative_s64(v);
#endif
        } else if constexpr (std::is_same_v<T, unsigned long>) {
#if SIZEOF_LONG == 4
            return npyv_negative_u32(v);
#else
            return npyv_negative_u64(v);
#endif
        } else if constexpr (std::is_same_v<T, long long>) {
            return npyv_negative_s64(v);
        } else if constexpr (std::is_same_v<T, unsigned long long>) {
            return npyv_negative_u64(v);
        }
#if NPY_SIMD_F32
        else if constexpr (std::is_same_v<T, float>) {
            return npyv_negative_f32(v);
        }
#endif
#if NPY_SIMD_F64
        else if constexpr (std::is_same_v<T, double>) {
            return npyv_negative_f64(v);
        }
#endif
    }
#endif // NPY_SIMD
};

/* SIMD helper utilities. */
#if NPY_SIMD
template<typename T>
static constexpr int simd_nlanes() {
    // Dispatch on scalar type T
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        return npyv_nlanes_u8;
    } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
        return npyv_nlanes_u16;
    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
        return npyv_nlanes_u32;
    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        return npyv_nlanes_u64;
    } else if constexpr (std::is_same_v<T, long>) {
#if SIZEOF_LONG == 4
        return npyv_nlanes_u32;
#else
        return npyv_nlanes_u64;
#endif
    } else if constexpr (std::is_same_v<T, unsigned long>) {
#if SIZEOF_LONG == 4
        return npyv_nlanes_u32;
#else
        return npyv_nlanes_u64;
#endif
    } else if constexpr (std::is_same_v<T, float>) {
        return npyv_nlanes_f32;
    } else if constexpr (std::is_same_v<T, double>) {
        return npyv_nlanes_f64;
    }
    return 0; // Fallback
}

template <typename T>
static NPY_INLINE auto simd_load(const T* ptr) {
    // Dispatch on scalar type T
    if constexpr (std::is_same_v<T, int8_t>) {
        return npyv_load_s8(reinterpret_cast<const npyv_lanetype_s8*>(ptr));
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return npyv_load_u8(reinterpret_cast<const npyv_lanetype_u8*>(ptr));
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return npyv_load_s16(reinterpret_cast<const npyv_lanetype_s16*>(ptr));
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return npyv_load_u16(reinterpret_cast<const npyv_lanetype_u16*>(ptr));
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return npyv_load_s32(reinterpret_cast<const npyv_lanetype_s32*>(ptr));
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return npyv_load_u32(reinterpret_cast<const npyv_lanetype_u32*>(ptr));
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return npyv_load_s64(reinterpret_cast<const npyv_lanetype_s64*>(ptr));
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return npyv_load_u64(reinterpret_cast<const npyv_lanetype_u64*>(ptr));
    } else if constexpr (std::is_same_v<T, long>) {
#if SIZEOF_LONG == 4
        return npyv_load_s32(reinterpret_cast<const npyv_lanetype_s32*>(ptr));
#else
        return npyv_load_s64(reinterpret_cast<const npyv_lanetype_s64*>(ptr));
#endif
    } else if constexpr (std::is_same_v<T, unsigned long>) {
#if SIZEOF_LONG == 4
        return npyv_load_u32(reinterpret_cast<const npyv_lanetype_u32*>(ptr));
#else
        return npyv_load_u64(reinterpret_cast<const npyv_lanetype_u64*>(ptr));
#endif
    } else if constexpr (std::is_same_v<T, float>) {
        return npyv_load_f32(reinterpret_cast<const npyv_lanetype_f32*>(ptr));
    } else if constexpr (std::is_same_v<T, double>) {
        return npyv_load_f64(reinterpret_cast<const npyv_lanetype_f64*>(ptr));
    }
    using SIMD = typename SIMDTypeTraits<T>::simd_type;
    return SIMD{}; // Fallback
}

template<typename T>
static NPY_INLINE void simd_store(T* ptr, typename SIMDTypeTraits<T>::simd_type v) {
    // Dispatch on scalar type T
    if constexpr (std::is_same_v<T, int8_t>) {
        npyv_store_s8(reinterpret_cast<npyv_lanetype_s8*>(ptr), v);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        npyv_store_u8(reinterpret_cast<npyv_lanetype_u8*>(ptr), v);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        npyv_store_s16(reinterpret_cast<npyv_lanetype_s16*>(ptr), v);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        npyv_store_u16(reinterpret_cast<npyv_lanetype_u16*>(ptr), v);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        npyv_store_s32(reinterpret_cast<npyv_lanetype_s32*>(ptr), v);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        npyv_store_u32(reinterpret_cast<npyv_lanetype_u32*>(ptr), v);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        npyv_store_s64(reinterpret_cast<npyv_lanetype_s64*>(ptr), v);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        npyv_store_u64(reinterpret_cast<npyv_lanetype_u64*>(ptr), v);
    } else if constexpr (std::is_same_v<T, long>) {
#if SIZEOF_LONG == 4
        npyv_store_s32(reinterpret_cast<npyv_lanetype_s32*>(ptr), v);
#else
        npyv_store_s64(reinterpret_cast<npyv_lanetype_s64*>(ptr), v);
#endif
    } else if constexpr (std::is_same_v<T, unsigned long>) {
#if SIZEOF_LONG == 4
        npyv_store_u32(reinterpret_cast<npyv_lanetype_u32*>(ptr), v);
#else
        npyv_store_u64(reinterpret_cast<npyv_lanetype_u64*>(ptr), v);
#endif
    } else if constexpr (std::is_same_v<T, float>) {
        npyv_store_f32(reinterpret_cast<npyv_lanetype_f32*>(ptr), v);
    } else if constexpr (std::is_same_v<T, double>) {
        npyv_store_f64(reinterpret_cast<npyv_lanetype_f64*>(ptr), v);
    }
}

// Load SIMD vector (strided/non-contiguous)
template<typename T>
static NPY_INLINE auto simd_loadn(const T* ptr, npy_intp stride) {
    // Dispatch on scalar type T
    if constexpr (std::is_same_v<T, int32_t>) {
        return npyv_loadn_s32(reinterpret_cast<const npyv_lanetype_s32*>(ptr), stride);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return npyv_loadn_u32(reinterpret_cast<const npyv_lanetype_u32*>(ptr), stride);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return npyv_loadn_s64(reinterpret_cast<const npyv_lanetype_s64*>(ptr), stride);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return npyv_loadn_u64(reinterpret_cast<const npyv_lanetype_u64*>(ptr), stride);
    } else if constexpr (std::is_same_v<T, float>) {
        return npyv_loadn_f32(reinterpret_cast<const npyv_lanetype_f32*>(ptr), stride);
    } else if constexpr (std::is_same_v<T, double>) {
        return npyv_loadn_f64(reinterpret_cast<const npyv_lanetype_f64*>(ptr), stride);
    } else if constexpr (std::is_same_v<T, long>) {
#if SIZEOF_LONG == 4
        return npyv_loadn_s32(reinterpret_cast<const npyv_lanetype_s32*>(ptr), stride);
#else
        return npyv_loadn_s64(reinterpret_cast<const npyv_lanetype_s64*>(ptr), stride);
#endif
    } else if constexpr (std::is_same_v<T, unsigned long>) {
#if SIZEOF_LONG == 4
        return npyv_loadn_u32(reinterpret_cast<const npyv_lanetype_u32*>(ptr), stride);
#else
        return npyv_loadn_u64(reinterpret_cast<const npyv_lanetype_u64*>(ptr), stride);
#endif
    } else if constexpr (std::is_same_v<T, long long>) {
        return npyv_loadn_s64(reinterpret_cast<const npyv_lanetype_s64*>(ptr), stride);
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        return npyv_loadn_u64(reinterpret_cast<const npyv_lanetype_u64*>(ptr), stride);
    }
    using SIMD = typename SIMDTypeTraits<T>::simd_type;
    return SIMD{};
}

// Store SIMD vector (strided/non-contiguous)
template<typename T>
static NPY_INLINE void simd_storen(T* ptr, npy_intp stride, typename SIMDTypeTraits<T>::simd_type v) {
    // Dispatch on scalar type T
    if constexpr (std::is_same_v<T, int32_t>) {
        npyv_storen_s32(reinterpret_cast<npyv_lanetype_s32*>(ptr), stride, v);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        npyv_storen_u32(reinterpret_cast<npyv_lanetype_u32*>(ptr), stride, v);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        npyv_storen_s64(reinterpret_cast<npyv_lanetype_s64*>(ptr), stride, v);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        npyv_storen_u64(reinterpret_cast<npyv_lanetype_u64*>(ptr), stride, v);
    } else if constexpr (std::is_same_v<T, float>) {
        npyv_storen_f32(reinterpret_cast<npyv_lanetype_f32*>(ptr), stride, v);
    } else if constexpr (std::is_same_v<T, double>) {
        npyv_storen_f64(reinterpret_cast<npyv_lanetype_f64*>(ptr), stride, v);
    } else if constexpr (std::is_same_v<T, long>) {
#if SIZEOF_LONG == 4
        npyv_storen_s32(reinterpret_cast<npyv_lanetype_s32*>(ptr), stride, v);
#else
        npyv_storen_s64(reinterpret_cast<npyv_lanetype_s64*>(ptr), stride, v);
#endif
    } else if constexpr (std::is_same_v<T, unsigned long>) {
#if SIZEOF_LONG == 4
        npyv_storen_u32(reinterpret_cast<npyv_lanetype_u32*>(ptr), stride, v);
#else
        npyv_storen_u64(reinterpret_cast<npyv_lanetype_u64*>(ptr), stride, v);
#endif
    } else if constexpr (std::is_same_v<T, long long>) {
        npyv_storen_s64(reinterpret_cast<npyv_lanetype_s64*>(ptr), stride, v);
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        npyv_storen_u64(reinterpret_cast<npyv_lanetype_u64*>(ptr), stride, v);
    }
}
#endif

/*
Following are to be ported:

// 1. contiguous-contiguous
simd_unary_cc_@intrin@_@sfx@(...)

// 2. contiguous-noncontiguous (only if @supports_ncontig@)
simd_unary_cn_@intrin@_@sfx@(...)

// 3. noncontiguous-contiguous (only if @supports_ncontig@)
simd_unary_nc_@intrin@_@sfx@(...)

// 4. noncontiguous-noncontiguous (only if @supports_ncontig@ and not SSE2)
simd_unary_nn_@intrin@_@sfx@(...)
*/
#if NPY_SIMD
template <typename T, typename Op>
static NPY_INLINE void simd_unary_cc(const T* ip, T* op, npy_intp len) {
    using SIMD = typename SIMDTypeTraits<T>::simd_type;
    using Traits = UnaryOpTraits<Op>;

    constexpr int UNROLL = 4;

    const int vstep = simd_nlanes<T>();
    const int wstep = vstep * UNROLL;

    // Unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += wstep) {
        for (int i = 0; i < UNROLL; i++) {
            SIMD v = simd_load<T>(ip + i * vstep);
            v = Traits::template simd_op<T>(v);
            simd_store<T>(op + i * vstep, v);
        }
    }

    // Single vector loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += vstep) {
        SIMD v = simd_load<T>(ip);
        SIMD r = Traits::template simd_op<T>(v);
        simd_store<T>(op, r);
    }

    // Scalar finish up any remaining iterations
    for (; len > 0; --len, ++ip, ++op) {
        *op = Traits::scalar_op(*ip);
    }
}

// Contiguous input, non-contiguous output
template <typename T, typename Op>
static NPY_INLINE void
simd_unary_cn(const T* ip, T* op, npy_intp ostride, npy_intp len)
{
    using SIMD = typename SIMDTypeTraits<T>::simd_type;
    using Traits = UnaryOpTraits<Op>;
    
    constexpr int UNROLL = 4;
    const int vstep = simd_nlanes<T>();
    const int wstep = vstep * UNROLL;
    
    // Unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += wstep, op += ostride * wstep) {
        for (int i = 0; i < UNROLL; i++) {
            SIMD v = simd_load<T>(ip + i * vstep);
            SIMD r = Traits::template simd_op<T>(v);
            simd_storen<T>(op + i * vstep * ostride, ostride, r);
        }
    }
    
    // Single vector loop
    for (; len >= vstep; len -= vstep, ip += vstep, op += ostride * vstep) {
        SIMD v = simd_load<T>(ip);
        SIMD r = Traits::template simd_op<T>(v);
        simd_storen<T>(op, ostride, r);
    }
    
    // Scalar finish up
    for (; len > 0; --len, ++ip, op += ostride) {
        *op = Traits::scalar_op(*ip);
    }
}

// Non-contiguous input, contiguous output
template <typename T, typename Op>
static NPY_INLINE void
simd_unary_nc(const T* ip, npy_intp istride, T* op, npy_intp len)
{
    using SIMD = typename SIMDTypeTraits<T>::simd_type;
    using Traits = UnaryOpTraits<Op>;
    
    constexpr int UNROLL = 4;
    const int vstep = simd_nlanes<T>();
    const int wstep = vstep * UNROLL;
    
    // Unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += istride * wstep, op += wstep) {
        for (int i = 0; i < UNROLL; i++) {
            SIMD v = simd_loadn<T>(ip + i * vstep * istride, istride);
            SIMD r = Traits::template simd_op<T>(v);
            simd_store<T>(op + i * vstep, r);
        }
    }
    
    // Single vector loop
    for (; len >= vstep; len -= vstep, ip += istride * vstep, op += vstep) {
        SIMD v = simd_loadn<T>(ip, istride);
        SIMD r = Traits::template simd_op<T>(v);
        simd_store<T>(op, r);
    }
    
    // Scalar finish up
    for (; len > 0; --len, ip += istride, ++op) {
        *op = Traits::scalar_op(*ip);
    }
}

// Non-contiguous input and output
#ifndef NPY_HAVE_SSE2
template <typename T, typename Op>
static NPY_INLINE void
simd_unary_nn(const T* ip, npy_intp istride, T* op, npy_intp ostride, npy_intp len)
{
    using SIMD = typename SIMDTypeTraits<T>::simd_type;
    using Traits = UnaryOpTraits<Op>;
    
    constexpr int UNROLL = 2;
    const int vstep = simd_nlanes<T>();
    const int wstep = vstep * UNROLL;
    
    // Unrolled vector loop
    for (; len >= wstep; len -= wstep, ip += istride * wstep, op += ostride * wstep) {
        for (int i = 0; i < UNROLL; i++) {
            SIMD v = simd_loadn<T>(ip + i * vstep * istride, istride);
            SIMD r = Traits::template simd_op<T>(v);
            simd_storen<T>(op + i * vstep * ostride, ostride, r);
        }
    }
    
    // Single vector loop
    for (; len >= vstep; len -= vstep, ip += istride * vstep, op += ostride * vstep) {
        SIMD v = simd_loadn<T>(ip, istride);
        SIMD r = Traits::template simd_op<T>(v);
        simd_storen<T>(op, ostride, r);
    }
    
    // Scalar finish up
    for (; len > 0; --len, ip += istride, op += ostride) {
        *op = Traits::scalar_op(*ip);
    }
}
#endif // !NPY_HAVE_SSE2
#endif // NPY_SIMD

/* Dispatcher */
template <typename T, typename Op>
static NPY_INLINE void
unary_ufunc_loop(char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    char *ip = args[0], *op = args[1];
    npy_intp istep = steps[0], ostep = steps[1];
    npy_intp len = dimensions[0];
    
    using Traits = SIMDTypeTraits<T>;
    using OpTraits = UnaryOpTraits<Op>;
    
    constexpr int UNROLL = 8;
    
#if NPY_SIMD
    if constexpr (Traits::has_simd) {
        if (!is_mem_overlap(ip, istep, op, ostep, len)) {
            if (IS_UNARY_CONT(T, T)) {
                // Both contiguous
                simd_unary_cc<T, Op>(
                    reinterpret_cast<const T*>(ip),
                    reinterpret_cast<T*>(op),
                        len
                );
                npyv_cleanup();
                return;
            }
            
            if constexpr (Traits::supports_ncontig) {
                const npy_intp istride = istep / sizeof(T);
                const npy_intp ostride = ostep / sizeof(T);
                
                if (istride == 1 && ostride != 1) {
                    // Contiguous input, non-contiguous output
                    simd_unary_cn<T, Op>(
                        reinterpret_cast<const T*>(ip),
                        reinterpret_cast<T*>(op),
                        ostride,
                        len
                    );
                    npyv_cleanup();
                    return;
                }
                else if (istride != 1 && ostride == 1) {
                    // Non-contiguous input, contiguous output
                    simd_unary_nc<T, Op>(
                        reinterpret_cast<const T*>(ip),
                        istride,
                        reinterpret_cast<T*>(op),
                        len
                    );
                    npyv_cleanup();
                    return;
                }
#ifndef NPY_HAVE_SSE2
                else if (istride != 1 && ostride != 1) {
                    // Both non-contiguous
                    simd_unary_nn<T, Op>(
                        reinterpret_cast<const T*>(ip),
                        istride,
                        reinterpret_cast<T*>(op),
                        ostride,
                        len
                    );
                    npyv_cleanup();
                    return;
                }
#endif
            }
        }
    }
#endif // NPY_SIMD

// Scalar fallback with unrolling
#ifndef NPY_DISABLE_OPTIMIZATION
    for (; len >= UNROLL; len -= UNROLL, ip += istep * UNROLL, op += ostep * UNROLL) {
        for (int i = 0; i < UNROLL; i++) {
            const T in_val = *reinterpret_cast<const T*>(ip + i * istep);
            *reinterpret_cast<T*>(op + i * ostep) = OpTraits::scalar_op(in_val);
        }
    }
#endif

    // Scalar remainder
    for (; len > 0; --len, ip += istep, op += ostep) {
        *reinterpret_cast<T*>(op) = OpTraits::scalar_op(*reinterpret_cast<const T*>(ip));
    }

#if NPY_SIMD
    if constexpr (Traits::has_simd) {
        npyv_cleanup();
    }
    return;
#endif
}

// C API
extern "C" {

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(UBYTE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<uint8_t, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(USHORT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<uint16_t, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(UINT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<uint32_t, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(ULONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<unsigned long, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(ULONGLONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<unsigned long long, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(BYTE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<int8_t, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(SHORT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<int16_t, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(INT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<int32_t, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<long, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONGLONG_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<long long, negative_t>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<float, negative_t>(args, dimensions, steps);
#if NPY_SIMD_F32
    npy_clear_floatstatus_barrier((char*)dimensions);
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<double, negative_t>(args, dimensions, steps);
#if NPY_SIMD_F64
    npy_clear_floatstatus_barrier((char*)dimensions);
#endif
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(LONGDOUBLE_negative)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    unary_ufunc_loop<long double, negative_t>(args, dimensions, steps);
    npy_clear_floatstatus_barrier((char*)dimensions);
}
} // extern "C"
