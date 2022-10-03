#ifndef NPY_SIMD
#error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SVE_MISC_H
#define _NPY_SIMD_SVE_MISC_H

// vector with zero lanes
#define npyv_zero_u8()  svdup_n_u8(0)
#define npyv_zero_u16() svdup_n_u16(0)
#define npyv_zero_u32() svdup_n_u32(0)
#define npyv_zero_u64() svdup_n_u64(0)
#define npyv_zero_s8()  svdup_n_s8(0)
#define npyv_zero_s16() svdup_n_s16(0)
#define npyv_zero_s32() svdup_n_s32(0)
#define npyv_zero_s64() svdup_n_s64(0)
#define npyv_zero_f32() svdup_n_f32(0.0)
#define npyv_zero_f64() svdup_n_f64(0.0)

// vector with a specific value set to all lanes
#define npyv_setall_u8 svdup_n_u8
#define npyv_setall_s8 svdup_n_s8
#define npyv_setall_u16 svdup_n_u16
#define npyv_setall_s16 svdup_n_s16
#define npyv_setall_u32 svdup_n_u32
#define npyv_setall_s32 svdup_n_s32
#define npyv_setall_u64 svdup_n_u64
#define npyv_setall_s64 svdup_n_s64
#define npyv_setall_f32 svdup_n_f32
#define npyv_setall_f64 svdup_n_f64

#define npyv__set_8(T, S)                                                 \
    NPY_FINLINE npyv_##S##8 npyv__set_##S##8(                             \
            T i0,  T i1,  T i2,  T i3,  T i4,  T i5,  T i6,  T i7,        \
            T i8,  T i9,  T i10, T i11, T i12, T i13, T i14, T i15,       \
            T i16, T i17, T i18, T i19, T i20, T i21, T i22, T i23,       \
            T i24, T i25, T i26, T i27, T i28, T i29, T i30, T i31,       \
            T i32, T i33, T i34, T i35, T i36, T i37, T i38, T i39,       \
            T i40, T i41, T i42, T i43, T i44, T i45, T i46, T i47,       \
            T i48, T i49, T i50, T i51, T i52, T i53, T i54, T i55,       \
            T i56, T i57, T i58, T i59, T i60, T i61, T i62, T i63)       \
    {                                                                     \
        const T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) data[npyv_nlanes_u8] = { \
                i0,  i1,  i2,  i3,  i4,  i5,  i6,  i7,                    \
                i8,  i9,  i10, i11, i12, i13, i14, i15,                   \
                i16, i17, i18, i19, i20, i21, i22, i23,                   \
                i24, i25, i26, i27, i28, i29, i30, i31,                   \
                i32, i33, i34, i35, i36, i37, i38, i39,                   \
                i40, i41, i42, i43, i44, i45, i46, i47,                   \
                i48, i49, i50, i51, i52, i53, i54, i55,                   \
                i56, i57, i58, i59, i60, i61, i62, i63};                  \
        return svld1_##S##8(svptrue_b8(), (const void *)data);            \
    }

#define npyv__set_16(T, S)                                                 \
    NPY_FINLINE npyv_##S##16 npyv__set_##S##16(                            \
            T i0,  T i1,  T i2,  T i3,  T i4,  T i5,  T i6,  T i7,         \
            T i8,  T i9,  T i10, T i11, T i12, T i13, T i14, T i15,        \
            T i16, T i17, T i18, T i19, T i20, T i21, T i22, T i23,        \
            T i24, T i25, T i26, T i27, T i28, T i29, T i30, T i31)        \
    {                                                                      \
        const T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) data[npyv_nlanes_u16] = { \
                i0,  i1,  i2,  i3,  i4,  i5,  i6,  i7,                     \
                i8,  i9,  i10, i11, i12, i13, i14, i15,                    \
                i16, i17, i18, i19, i20, i21, i22, i23,                    \
                i24, i25, i26, i27, i28, i29, i30, i31};                   \
        return svld1_##S##16(svptrue_b8(), (const void *)data);            \
    }

#define npyv__set_32(T, S)                                                    \
    NPY_FINLINE npyv_##S##32 npyv__set_##S##32(                               \
            T i0, T i1, T i2,  T i3,  T i4,  T i5,  T i6,  T i7,              \
            T i8, T i9, T i10, T i11, T i12, T i13, T i14, T i15)             \
    {                                                                         \
        const T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH)                              \
                data[npyv_nlanes_u32] = {i0, i1, i2,  i3,  i4,  i5,  i6,  i7, \
                        i8, i9, i10, i11, i12, i13, i14, i15};                \
        return svld1_##S##32(svptrue_b8(), (const void *)data);               \
    }

#define npyv__set_64(T, S)                                                   \
    NPY_FINLINE npyv_##S##64 npyv__set_##S##64(T i0, T i1, T i2, T i3, T i4, \
                                               T i5, T i6, T i7)             \
    {                                                                        \
        const T NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) data[npyv_nlanes_u64] =     \
                {i0, i1, i2, i3, i4, i5, i6, i7};                            \
        return svld1_##S##64(svptrue_b8(), (const void *)data);              \
    }

npyv__set_8(uint8_t, u)
npyv__set_8(int8_t, s)
npyv__set_16(uint16_t, u)
npyv__set_16(int16_t, s)
npyv__set_32(uint32_t, u)
npyv__set_32(int32_t, s)
npyv__set_64(uint64_t, u)
npyv__set_64(int64_t, s)
npyv__set_32(float32_t, f)
npyv__set_64(float64_t, f)

#define npyv_setf_u8(FILL, ...) \
    npyv__set_u8(NPYV__SET_FILL_64(char, FILL, __VA_ARGS__))
#define npyv_setf_s8(FILL, ...) \
    npyv__set_s8(NPYV__SET_FILL_64(char, FILL, __VA_ARGS__))
#define npyv_setf_u16(FILL, ...) \
    npyv__set_u16(NPYV__SET_FILL_32(short, FILL, __VA_ARGS__))
#define npyv_setf_s16(FILL, ...) \
    npyv__set_s16(NPYV__SET_FILL_32(short, FILL, __VA_ARGS__))
#define npyv_setf_u32(FILL, ...) \
    npyv__set_u32(NPYV__SET_FILL_16(int, FILL, __VA_ARGS__))
#define npyv_setf_s32(FILL, ...) \
    npyv__set_s32(NPYV__SET_FILL_16(int, FILL, __VA_ARGS__))
#define npyv_setf_u64(FILL, ...) \
    npyv__set_u64(NPYV__SET_FILL_8(npy_int64, FILL, __VA_ARGS__))
#define npyv_setf_s64(FILL, ...) \
    npyv__set_s64(NPYV__SET_FILL_8(npy_int64, FILL, __VA_ARGS__))
#define npyv_setf_f32(FILL, ...) \
    npyv__set_f32(NPYV__SET_FILL_16(float, FILL, __VA_ARGS__))
#define npyv_setf_f64(FILL, ...) \
    npyv__set_f64(NPYV__SET_FILL_8(double, FILL, __VA_ARGS__))

// vector with specific values set to each lane and
// set zero to all remained lanes
#define npyv_set_u8(...) npyv_setf_u8(0, __VA_ARGS__)
#define npyv_set_s8(...) npyv_setf_s8(0, __VA_ARGS__)
#define npyv_set_u16(...) npyv_setf_u16(0, __VA_ARGS__)
#define npyv_set_s16(...) npyv_setf_s16(0, __VA_ARGS__)
#define npyv_set_u32(...) npyv_setf_u32(0, __VA_ARGS__)
#define npyv_set_s32(...) npyv_setf_s32(0, __VA_ARGS__)
#define npyv_set_u64(...) npyv_setf_u64(0, __VA_ARGS__)
#define npyv_set_s64(...) npyv_setf_s64(0, __VA_ARGS__)
#define npyv_set_f32(...) npyv_setf_f32(0, __VA_ARGS__)
#define npyv_set_f64(...) npyv_setf_f64(0, __VA_ARGS__)

// Set mask for lane #0 - #(a-1)
#define NPYV_IMPL_SVE_SET_B(BITS)                    \
    NPY_FINLINE npyv_b##BITS npyv_set_b##BITS(int a) \
    {                                                \
        return svwhilelt_b##BITS##_s32(0, a);        \
    }

NPYV_IMPL_SVE_SET_B(8) NPYV_IMPL_SVE_SET_B(16)
NPYV_IMPL_SVE_SET_B(32)
NPYV_IMPL_SVE_SET_B(64)

// Per lane select
#define npyv_select_u8 svsel_u8
#define npyv_select_s8 svsel_s8
#define npyv_select_u16 svsel_u16
#define npyv_select_s16 svsel_s16
#define npyv_select_u32 svsel_u32
#define npyv_select_s32 svsel_s32
#define npyv_select_u64 svsel_u64
#define npyv_select_s64 svsel_s64
#define npyv_select_f32 svsel_f32
#define npyv_select_f64 svsel_f64

// extract the first vector's lane
#define npyv_extract0_u8(A) svlastb_u8(svptrue_pat_b8(SV_VL1), A)
#define npyv_extract0_u16(A) svlastb_u16(svptrue_pat_b16(SV_VL1), A)
#define npyv_extract0_u32(A) svlastb_u32(svptrue_pat_b32(SV_VL1), A)
#define npyv_extract0_u64(A) svlastb_u64(svptrue_pat_b64(SV_VL1), A)
#define npyv_extract0_s8(A) svlastb_s8(svptrue_pat_b8(SV_VL1), A)
#define npyv_extract0_s16(A) svlastb_s16(svptrue_pat_b16(SV_VL1), A)
#define npyv_extract0_s32(A) svlastb_s32(svptrue_pat_b32(SV_VL1), A)
#define npyv_extract0_s64(A) svlastb_s64(svptrue_pat_b64(SV_VL1), A)
#define npyv_extract0_f32(A) svlastb_f32(svptrue_pat_b32(SV_VL1), A)
#define npyv_extract0_f64(A) svlastb_f64(svptrue_pat_b64(SV_VL1), A)

// Reinterpret
#define npyv_reinterpret_u8_u8(X) X
#define npyv_reinterpret_u8_s8 svreinterpret_u8_s8
#define npyv_reinterpret_u8_u16 svreinterpret_u8_u16
#define npyv_reinterpret_u8_s16 svreinterpret_u8_s16
#define npyv_reinterpret_u8_u32 svreinterpret_u8_u32
#define npyv_reinterpret_u8_s32 svreinterpret_u8_s32
#define npyv_reinterpret_u8_u64 svreinterpret_u8_u64
#define npyv_reinterpret_u8_s64 svreinterpret_u8_s64
#define npyv_reinterpret_u8_f32 svreinterpret_u8_f32
#define npyv_reinterpret_u8_f64 svreinterpret_u8_f64

#define npyv_reinterpret_s8_s8(X) X
#define npyv_reinterpret_s8_u8 svreinterpret_s8_u8
#define npyv_reinterpret_s8_u16 svreinterpret_s8_u16
#define npyv_reinterpret_s8_s16 svreinterpret_s8_s16
#define npyv_reinterpret_s8_u32 svreinterpret_s8_u32
#define npyv_reinterpret_s8_s32 svreinterpret_s8_s32
#define npyv_reinterpret_s8_u64 svreinterpret_s8_u64
#define npyv_reinterpret_s8_s64 svreinterpret_s8_s64
#define npyv_reinterpret_s8_f32 svreinterpret_s8_f32
#define npyv_reinterpret_s8_f64 svreinterpret_s8_f64

#define npyv_reinterpret_u16_u16(X) X
#define npyv_reinterpret_u16_u8 svreinterpret_u16_u8
#define npyv_reinterpret_u16_s8 svreinterpret_u16_s8
#define npyv_reinterpret_u16_s16 svreinterpret_u16_s16
#define npyv_reinterpret_u16_u32 svreinterpret_u16_u32
#define npyv_reinterpret_u16_s32 svreinterpret_u16_s32
#define npyv_reinterpret_u16_u64 svreinterpret_u16_u64
#define npyv_reinterpret_u16_s64 svreinterpret_u16_s64
#define npyv_reinterpret_u16_f32 svreinterpret_u16_f32
#define npyv_reinterpret_u16_f64 svreinterpret_u16_f64

#define npyv_reinterpret_s16_s16(X) X
#define npyv_reinterpret_s16_u8 svreinterpret_s16_u8
#define npyv_reinterpret_s16_s8 svreinterpret_s16_s8
#define npyv_reinterpret_s16_u16 svreinterpret_s16_u16
#define npyv_reinterpret_s16_u32 svreinterpret_s16_u32
#define npyv_reinterpret_s16_s32 svreinterpret_s16_s32
#define npyv_reinterpret_s16_u64 svreinterpret_s16_u64
#define npyv_reinterpret_s16_s64 svreinterpret_s16_s64
#define npyv_reinterpret_s16_f32 svreinterpret_s16_f32
#define npyv_reinterpret_s16_f64 svreinterpret_s16_f64

#define npyv_reinterpret_u32_u32(X) X
#define npyv_reinterpret_u32_u8 svreinterpret_u32_u8
#define npyv_reinterpret_u32_s8 svreinterpret_u32_s8
#define npyv_reinterpret_u32_u16 svreinterpret_u32_u16
#define npyv_reinterpret_u32_s16 svreinterpret_u32_s16
#define npyv_reinterpret_u32_s32 svreinterpret_u32_s32
#define npyv_reinterpret_u32_u64 svreinterpret_u32_u64
#define npyv_reinterpret_u32_s64 svreinterpret_u32_s64
#define npyv_reinterpret_u32_f32 svreinterpret_u32_f32
#define npyv_reinterpret_u32_f64 svreinterpret_u32_f64

#define npyv_reinterpret_s32_s32(X) X
#define npyv_reinterpret_s32_u8 svreinterpret_s32_u8
#define npyv_reinterpret_s32_s8 svreinterpret_s32_s8
#define npyv_reinterpret_s32_u16 svreinterpret_s32_u16
#define npyv_reinterpret_s32_s16 svreinterpret_s32_s16
#define npyv_reinterpret_s32_u32 svreinterpret_s32_u32
#define npyv_reinterpret_s32_u64 svreinterpret_s32_u64
#define npyv_reinterpret_s32_s64 svreinterpret_s32_s64
#define npyv_reinterpret_s32_f32 svreinterpret_s32_f32
#define npyv_reinterpret_s32_f64 svreinterpret_s32_f64

#define npyv_reinterpret_u64_u64(X) X
#define npyv_reinterpret_u64_u8 svreinterpret_u64_u8
#define npyv_reinterpret_u64_s8 svreinterpret_u64_s8
#define npyv_reinterpret_u64_u16 svreinterpret_u64_u16
#define npyv_reinterpret_u64_s16 svreinterpret_u64_s16
#define npyv_reinterpret_u64_u32 svreinterpret_u64_u32
#define npyv_reinterpret_u64_s32 svreinterpret_u64_s32
#define npyv_reinterpret_u64_s64 svreinterpret_u64_s64
#define npyv_reinterpret_u64_f32 svreinterpret_u64_f32
#define npyv_reinterpret_u64_f64 svreinterpret_u64_f64

#define npyv_reinterpret_s64_s64(X) X
#define npyv_reinterpret_s64_u8 svreinterpret_s64_u8
#define npyv_reinterpret_s64_s8 svreinterpret_s64_s8
#define npyv_reinterpret_s64_u16 svreinterpret_s64_u16
#define npyv_reinterpret_s64_s16 svreinterpret_s64_s16
#define npyv_reinterpret_s64_u32 svreinterpret_s64_u32
#define npyv_reinterpret_s64_s32 svreinterpret_s64_s32
#define npyv_reinterpret_s64_u64 svreinterpret_s64_u64
#define npyv_reinterpret_s64_f32 svreinterpret_s64_f32
#define npyv_reinterpret_s64_f64 svreinterpret_s64_f64

#define npyv_reinterpret_f32_f32(X) X
#define npyv_reinterpret_f32_u8 svreinterpret_f32_u8
#define npyv_reinterpret_f32_s8 svreinterpret_f32_s8
#define npyv_reinterpret_f32_u16 svreinterpret_f32_u16
#define npyv_reinterpret_f32_s16 svreinterpret_f32_s16
#define npyv_reinterpret_f32_u32 svreinterpret_f32_u32
#define npyv_reinterpret_f32_s32 svreinterpret_f32_s32
#define npyv_reinterpret_f32_u64 svreinterpret_f32_u64
#define npyv_reinterpret_f32_s64 svreinterpret_f32_s64
#define npyv_reinterpret_f32_f64 svreinterpret_f32_f64

#define npyv_reinterpret_f64_f64(X) X
#define npyv_reinterpret_f64_u8 svreinterpret_f64_u8
#define npyv_reinterpret_f64_s8 svreinterpret_f64_s8
#define npyv_reinterpret_f64_u16 svreinterpret_f64_u16
#define npyv_reinterpret_f64_s16 svreinterpret_f64_s16
#define npyv_reinterpret_f64_u32 svreinterpret_f64_u32
#define npyv_reinterpret_f64_s32 svreinterpret_f64_s32
#define npyv_reinterpret_f64_u64 svreinterpret_f64_u64
#define npyv_reinterpret_f64_s64 svreinterpret_f64_s64
#define npyv_reinterpret_f64_f32 svreinterpret_f64_f32

// broadcast simd lane#0 to others

#define NPYV_IMPL_SVE_BROADCAST_LANE0(W, S)              \
    NPY_FINLINE npyv_##S##W npyv_broadcast_lane0_##S##W( \
            npyv_##S##W a)                               \
    {                                                    \
        return svdup_lane_##S##W(a, 0);                  \
    }

NPYV_IMPL_SVE_BROADCAST_LANE0(8, u)
NPYV_IMPL_SVE_BROADCAST_LANE0(16, u)
NPYV_IMPL_SVE_BROADCAST_LANE0(32, u)
NPYV_IMPL_SVE_BROADCAST_LANE0(64, u)
NPYV_IMPL_SVE_BROADCAST_LANE0(8, s)
NPYV_IMPL_SVE_BROADCAST_LANE0(16, s)
NPYV_IMPL_SVE_BROADCAST_LANE0(32, s)
NPYV_IMPL_SVE_BROADCAST_LANE0(64, s)
NPYV_IMPL_SVE_BROADCAST_LANE0(32, f)
NPYV_IMPL_SVE_BROADCAST_LANE0(64, f)

/* Following npyv_popcnt_b(8/16/32/64) function requried
   to declare #define NPY_SIMD_POPCNT 1. */
#define npyv_popcnt_b8(a)  svcntp_b8(svptrue_b8(), a)
#define npyv_popcnt_b16(a) svcntp_b16(svptrue_b16(), a)
#define npyv_popcnt_b32(a) svcntp_b32(svptrue_b32(), a)
#define npyv_popcnt_b64(a) svcntp_b64(svptrue_b64(), a)


// Only required by AVX2/AVX512
#define npyv_cleanup() ((void)0)

#endif  // _NPY_SIMD_NEON_MISC_H
