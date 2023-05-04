#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_MISC_H
#define _NPY_SIMD_VEC_MISC_H

// vector with zero lanes
#define npyv_zero_u8()  ((npyv_u8)   npyv_setall_s32(0))
#define npyv_zero_s8()  ((npyv_s8)   npyv_setall_s32(0))
#define npyv_zero_u16() ((npyv_u16)  npyv_setall_s32(0))
#define npyv_zero_s16() ((npyv_s16)  npyv_setall_s32(0))
#define npyv_zero_u32() npyv_setall_u32(0)
#define npyv_zero_s32() npyv_setall_s32(0)
#define npyv_zero_u64() ((npyv_u64) npyv_setall_s32(0))
#define npyv_zero_s64() ((npyv_s64) npyv_setall_s32(0))
#if NPY_SIMD_F32
    #define npyv_zero_f32() npyv_setall_f32(0.0f)
#endif
#define npyv_zero_f64() npyv_setall_f64(0.0)

// vector with a specific value set to all lanes
// the safest way to generate vsplti* and vsplt* instructions
#define NPYV_IMPL_VEC_SPLTB(T_VEC, V) ((T_VEC){V, V, V, V, V, V, V, V, V, V, V, V, V, V, V, V})
#define NPYV_IMPL_VEC_SPLTH(T_VEC, V) ((T_VEC){V, V, V, V, V, V, V, V})
#define NPYV_IMPL_VEC_SPLTW(T_VEC, V) ((T_VEC){V, V, V, V})
#define NPYV_IMPL_VEC_SPLTD(T_VEC, V) ((T_VEC){V, V})

#define npyv_setall_u8(VAL)  NPYV_IMPL_VEC_SPLTB(npyv_u8,  (unsigned char)(VAL))
#define npyv_setall_s8(VAL)  NPYV_IMPL_VEC_SPLTB(npyv_s8,  (signed char)(VAL))
#define npyv_setall_u16(VAL) NPYV_IMPL_VEC_SPLTH(npyv_u16, (unsigned short)(VAL))
#define npyv_setall_s16(VAL) NPYV_IMPL_VEC_SPLTH(npyv_s16, (short)(VAL))
#define npyv_setall_u32(VAL) NPYV_IMPL_VEC_SPLTW(npyv_u32, (unsigned int)(VAL))
#define npyv_setall_s32(VAL) NPYV_IMPL_VEC_SPLTW(npyv_s32, (int)(VAL))
#if NPY_SIMD_F32
    #define npyv_setall_f32(VAL) NPYV_IMPL_VEC_SPLTW(npyv_f32, (VAL))
#endif
#define npyv_setall_u64(VAL) NPYV_IMPL_VEC_SPLTD(npyv_u64, (npy_uint64)(VAL))
#define npyv_setall_s64(VAL) NPYV_IMPL_VEC_SPLTD(npyv_s64, (npy_int64)(VAL))
#define npyv_setall_f64(VAL) NPYV_IMPL_VEC_SPLTD(npyv_f64, VAL)

// vector with specific values set to each lane and
// set a specific value to all remained lanes
#define npyv_setf_u8(FILL, ...)  ((npyv_u8){NPYV__SET_FILL_16(unsigned char, FILL, __VA_ARGS__)})
#define npyv_setf_s8(FILL, ...)  ((npyv_s8){NPYV__SET_FILL_16(signed char, FILL, __VA_ARGS__)})
#define npyv_setf_u16(FILL, ...) ((npyv_u16){NPYV__SET_FILL_8(unsigned short, FILL, __VA_ARGS__)})
#define npyv_setf_s16(FILL, ...) ((npyv_s16){NPYV__SET_FILL_8(short, FILL, __VA_ARGS__)})
#define npyv_setf_u32(FILL, ...) ((npyv_u32){NPYV__SET_FILL_4(unsigned int, FILL, __VA_ARGS__)})
#define npyv_setf_s32(FILL, ...) ((npyv_s32){NPYV__SET_FILL_4(int, FILL, __VA_ARGS__)})
#define npyv_setf_u64(FILL, ...) ((npyv_u64){NPYV__SET_FILL_2(npy_uint64, FILL, __VA_ARGS__)})
#define npyv_setf_s64(FILL, ...) ((npyv_s64){NPYV__SET_FILL_2(npy_int64, FILL, __VA_ARGS__)})
#if NPY_SIMD_F32
    #define npyv_setf_f32(FILL, ...) ((npyv_f32){NPYV__SET_FILL_4(float, FILL, __VA_ARGS__)})
#endif
#define npyv_setf_f64(FILL, ...) ((npyv_f64){NPYV__SET_FILL_2(double, FILL, __VA_ARGS__)})

// vector with specific values set to each lane and
// set zero to all remained lanes
#define npyv_set_u8(...)  npyv_setf_u8(0,  __VA_ARGS__)
#define npyv_set_s8(...)  npyv_setf_s8(0,  __VA_ARGS__)
#define npyv_set_u16(...) npyv_setf_u16(0, __VA_ARGS__)
#define npyv_set_s16(...) npyv_setf_s16(0, __VA_ARGS__)
#define npyv_set_u32(...) npyv_setf_u32(0, __VA_ARGS__)
#define npyv_set_s32(...) npyv_setf_s32(0, __VA_ARGS__)
#define npyv_set_u64(...) npyv_setf_u64(0, __VA_ARGS__)
#define npyv_set_s64(...) npyv_setf_s64(0, __VA_ARGS__)
#if NPY_SIMD_F32
    #define npyv_set_f32(...) npyv_setf_f32(0, __VA_ARGS__)
#endif
#define npyv_set_f64(...) npyv_setf_f64(0, __VA_ARGS__)

// Per lane select
#define npyv_select_u8(MASK, A, B) vec_sel(B, A, MASK)
#define npyv_select_s8  npyv_select_u8
#define npyv_select_u16 npyv_select_u8
#define npyv_select_s16 npyv_select_u8
#define npyv_select_u32 npyv_select_u8
#define npyv_select_s32 npyv_select_u8
#define npyv_select_u64 npyv_select_u8
#define npyv_select_s64 npyv_select_u8
#if NPY_SIMD_F32
    #define npyv_select_f32 npyv_select_u8
#endif
#define npyv_select_f64 npyv_select_u8

// extract the first vector's lane
#define npyv_extract0_u8(A) ((npy_uint8)vec_extract(A, 0))
#define npyv_extract0_s8(A) ((npy_int8)vec_extract(A, 0))
#define npyv_extract0_u16(A) ((npy_uint16)vec_extract(A, 0))
#define npyv_extract0_s16(A) ((npy_int16)vec_extract(A, 0))
#define npyv_extract0_u32(A) ((npy_uint32)vec_extract(A, 0))
#define npyv_extract0_s32(A) ((npy_int32)vec_extract(A, 0))
#define npyv_extract0_u64(A) ((npy_uint64)vec_extract(A, 0))
#define npyv_extract0_s64(A) ((npy_int64)vec_extract(A, 0))
#if NPY_SIMD_F32
    #define npyv_extract0_f32(A) vec_extract(A, 0)
#endif
#define npyv_extract0_f64(A) vec_extract(A, 0)

// Reinterpret
#define npyv_reinterpret_u8_u8(X) X
#define npyv_reinterpret_u8_s8(X) ((npyv_u8)X)
#define npyv_reinterpret_u8_u16 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_s16 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_u32 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_s32 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_u64 npyv_reinterpret_u8_s8
#define npyv_reinterpret_u8_s64 npyv_reinterpret_u8_s8
#if NPY_SIMD_F32
    #define npyv_reinterpret_u8_f32 npyv_reinterpret_u8_s8
#endif
#define npyv_reinterpret_u8_f64 npyv_reinterpret_u8_s8

#define npyv_reinterpret_s8_s8(X) X
#define npyv_reinterpret_s8_u8(X) ((npyv_s8)X)
#define npyv_reinterpret_s8_u16 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_s16 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_u32 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_s32 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_u64 npyv_reinterpret_s8_u8
#define npyv_reinterpret_s8_s64 npyv_reinterpret_s8_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_s8_f32 npyv_reinterpret_s8_u8
#endif
#define npyv_reinterpret_s8_f64 npyv_reinterpret_s8_u8

#define npyv_reinterpret_u16_u16(X) X
#define npyv_reinterpret_u16_u8(X) ((npyv_u16)X)
#define npyv_reinterpret_u16_s8  npyv_reinterpret_u16_u8
#define npyv_reinterpret_u16_s16 npyv_reinterpret_u16_u8
#define npyv_reinterpret_u16_u32 npyv_reinterpret_u16_u8
#define npyv_reinterpret_u16_s32 npyv_reinterpret_u16_u8
#define npyv_reinterpret_u16_u64 npyv_reinterpret_u16_u8
#define npyv_reinterpret_u16_s64 npyv_reinterpret_u16_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_u16_f32 npyv_reinterpret_u16_u8
#endif
#define npyv_reinterpret_u16_f64 npyv_reinterpret_u16_u8

#define npyv_reinterpret_s16_s16(X) X
#define npyv_reinterpret_s16_u8(X) ((npyv_s16)X)
#define npyv_reinterpret_s16_s8  npyv_reinterpret_s16_u8
#define npyv_reinterpret_s16_u16 npyv_reinterpret_s16_u8
#define npyv_reinterpret_s16_u32 npyv_reinterpret_s16_u8
#define npyv_reinterpret_s16_s32 npyv_reinterpret_s16_u8
#define npyv_reinterpret_s16_u64 npyv_reinterpret_s16_u8
#define npyv_reinterpret_s16_s64 npyv_reinterpret_s16_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_s16_f32 npyv_reinterpret_s16_u8
#endif
#define npyv_reinterpret_s16_f64 npyv_reinterpret_s16_u8

#define npyv_reinterpret_u32_u32(X) X
#define npyv_reinterpret_u32_u8(X) ((npyv_u32)X)
#define npyv_reinterpret_u32_s8  npyv_reinterpret_u32_u8
#define npyv_reinterpret_u32_u16 npyv_reinterpret_u32_u8
#define npyv_reinterpret_u32_s16 npyv_reinterpret_u32_u8
#define npyv_reinterpret_u32_s32 npyv_reinterpret_u32_u8
#define npyv_reinterpret_u32_u64 npyv_reinterpret_u32_u8
#define npyv_reinterpret_u32_s64 npyv_reinterpret_u32_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_u32_f32 npyv_reinterpret_u32_u8
#endif
#define npyv_reinterpret_u32_f64 npyv_reinterpret_u32_u8

#define npyv_reinterpret_s32_s32(X) X
#define npyv_reinterpret_s32_u8(X) ((npyv_s32)X)
#define npyv_reinterpret_s32_s8  npyv_reinterpret_s32_u8
#define npyv_reinterpret_s32_u16 npyv_reinterpret_s32_u8
#define npyv_reinterpret_s32_s16 npyv_reinterpret_s32_u8
#define npyv_reinterpret_s32_u32 npyv_reinterpret_s32_u8
#define npyv_reinterpret_s32_u64 npyv_reinterpret_s32_u8
#define npyv_reinterpret_s32_s64 npyv_reinterpret_s32_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_s32_f32 npyv_reinterpret_s32_u8
#endif
#define npyv_reinterpret_s32_f64 npyv_reinterpret_s32_u8

#define npyv_reinterpret_u64_u64(X) X
#define npyv_reinterpret_u64_u8(X) ((npyv_u64)X)
#define npyv_reinterpret_u64_s8  npyv_reinterpret_u64_u8
#define npyv_reinterpret_u64_u16 npyv_reinterpret_u64_u8
#define npyv_reinterpret_u64_s16 npyv_reinterpret_u64_u8
#define npyv_reinterpret_u64_u32 npyv_reinterpret_u64_u8
#define npyv_reinterpret_u64_s32 npyv_reinterpret_u64_u8
#define npyv_reinterpret_u64_s64 npyv_reinterpret_u64_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_u64_f32 npyv_reinterpret_u64_u8
#endif
#define npyv_reinterpret_u64_f64 npyv_reinterpret_u64_u8

#define npyv_reinterpret_s64_s64(X) X
#define npyv_reinterpret_s64_u8(X) ((npyv_s64)X)
#define npyv_reinterpret_s64_s8  npyv_reinterpret_s64_u8
#define npyv_reinterpret_s64_u16 npyv_reinterpret_s64_u8
#define npyv_reinterpret_s64_s16 npyv_reinterpret_s64_u8
#define npyv_reinterpret_s64_u32 npyv_reinterpret_s64_u8
#define npyv_reinterpret_s64_s32 npyv_reinterpret_s64_u8
#define npyv_reinterpret_s64_u64 npyv_reinterpret_s64_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_s64_f32 npyv_reinterpret_s64_u8
#endif
#define npyv_reinterpret_s64_f64 npyv_reinterpret_s64_u8

#if NPY_SIMD_F32
    #define npyv_reinterpret_f32_f32(X) X
    #define npyv_reinterpret_f32_u8(X) ((npyv_f32)X)
    #define npyv_reinterpret_f32_s8  npyv_reinterpret_f32_u8
    #define npyv_reinterpret_f32_u16 npyv_reinterpret_f32_u8
    #define npyv_reinterpret_f32_s16 npyv_reinterpret_f32_u8
    #define npyv_reinterpret_f32_u32 npyv_reinterpret_f32_u8
    #define npyv_reinterpret_f32_s32 npyv_reinterpret_f32_u8
    #define npyv_reinterpret_f32_u64 npyv_reinterpret_f32_u8
    #define npyv_reinterpret_f32_s64 npyv_reinterpret_f32_u8
    #define npyv_reinterpret_f32_f64 npyv_reinterpret_f32_u8
#endif

#define npyv_reinterpret_f64_f64(X) X
#define npyv_reinterpret_f64_u8(X) ((npyv_f64)X)
#define npyv_reinterpret_f64_s8  npyv_reinterpret_f64_u8
#define npyv_reinterpret_f64_u16 npyv_reinterpret_f64_u8
#define npyv_reinterpret_f64_s16 npyv_reinterpret_f64_u8
#define npyv_reinterpret_f64_u32 npyv_reinterpret_f64_u8
#define npyv_reinterpret_f64_s32 npyv_reinterpret_f64_u8
#define npyv_reinterpret_f64_u64 npyv_reinterpret_f64_u8
#define npyv_reinterpret_f64_s64 npyv_reinterpret_f64_u8
#if NPY_SIMD_F32
    #define npyv_reinterpret_f64_f32 npyv_reinterpret_f64_u8
#endif
// Only required by AVX2/AVX512
#define npyv_cleanup() ((void)0)

#endif // _NPY_SIMD_VEC_MISC_H
