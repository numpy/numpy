#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_OPERATORS_H
#define _NPY_SIMD_NEON_OPERATORS_H

/***************************
 * Shifting
 ***************************/

// left
#define npyv_shl_u16(A, C) vshlq_u16(A, npyv_setall_s16(C))
#define npyv_shl_s16(A, C) vshlq_s16(A, npyv_setall_s16(C))
#define npyv_shl_u32(A, C) vshlq_u32(A, npyv_setall_s32(C))
#define npyv_shl_s32(A, C) vshlq_s32(A, npyv_setall_s32(C))
#define npyv_shl_u64(A, C) vshlq_u64(A, npyv_setall_s64(C))
#define npyv_shl_s64(A, C) vshlq_s64(A, npyv_setall_s64(C))

// left by an immediate constant
#define npyv_shli_u16 vshlq_n_u16
#define npyv_shli_s16 vshlq_n_s16
#define npyv_shli_u32 vshlq_n_u32
#define npyv_shli_s32 vshlq_n_s32
#define npyv_shli_u64 vshlq_n_u64
#define npyv_shli_s64 vshlq_n_s64

// right
#define npyv_shr_u16(A, C) vshlq_u16(A, npyv_setall_s16(-(C)))
#define npyv_shr_s16(A, C) vshlq_s16(A, npyv_setall_s16(-(C)))
#define npyv_shr_u32(A, C) vshlq_u32(A, npyv_setall_s32(-(C)))
#define npyv_shr_s32(A, C) vshlq_s32(A, npyv_setall_s32(-(C)))
#define npyv_shr_u64(A, C) vshlq_u64(A, npyv_setall_s64(-(C)))
#define npyv_shr_s64(A, C) vshlq_s64(A, npyv_setall_s64(-(C)))

// right by an immediate constant
#define npyv_shri_u16 vshrq_n_u16
#define npyv_shri_s16 vshrq_n_s16
#define npyv_shri_u32 vshrq_n_u32
#define npyv_shri_s32 vshrq_n_s32
#define npyv_shri_u64 vshrq_n_u64
#define npyv_shri_s64 vshrq_n_s64

/***************************
 * Logical
 ***************************/

// AND
#define npyv_and_u8  vandq_u8
#define npyv_and_s8  vandq_s8
#define npyv_and_u16 vandq_u16
#define npyv_and_s16 vandq_s16
#define npyv_and_u32 vandq_u32
#define npyv_and_s32 vandq_s32
#define npyv_and_u64 vandq_u64
#define npyv_and_s64 vandq_s64
#define npyv_and_f32(A, B) \
    vreinterpretq_f32_u8(vandq_u8(vreinterpretq_u8_f32(A), vreinterpretq_u8_f32(B)))
#define npyv_and_f64(A, B) \
    vreinterpretq_f64_u8(vandq_u8(vreinterpretq_u8_f64(A), vreinterpretq_u8_f64(B)))
#define npyv_and_b8   vandq_u8
#define npyv_and_b16  vandq_u16
#define npyv_and_b32  vandq_u32
#define npyv_and_b64  vandq_u64

// OR
#define npyv_or_u8  vorrq_u8
#define npyv_or_s8  vorrq_s8
#define npyv_or_u16 vorrq_u16
#define npyv_or_s16 vorrq_s16
#define npyv_or_u32 vorrq_u32
#define npyv_or_s32 vorrq_s32
#define npyv_or_u64 vorrq_u64
#define npyv_or_s64 vorrq_s64
#define npyv_or_f32(A, B) \
    vreinterpretq_f32_u8(vorrq_u8(vreinterpretq_u8_f32(A), vreinterpretq_u8_f32(B)))
#define npyv_or_f64(A, B) \
    vreinterpretq_f64_u8(vorrq_u8(vreinterpretq_u8_f64(A), vreinterpretq_u8_f64(B)))
#define npyv_or_b8   vorrq_u8
#define npyv_or_b16  vorrq_u16
#define npyv_or_b32  vorrq_u32
#define npyv_or_b64  vorrq_u64


// XOR
#define npyv_xor_u8  veorq_u8
#define npyv_xor_s8  veorq_s8
#define npyv_xor_u16 veorq_u16
#define npyv_xor_s16 veorq_s16
#define npyv_xor_u32 veorq_u32
#define npyv_xor_s32 veorq_s32
#define npyv_xor_u64 veorq_u64
#define npyv_xor_s64 veorq_s64
#define npyv_xor_f32(A, B) \
    vreinterpretq_f32_u8(veorq_u8(vreinterpretq_u8_f32(A), vreinterpretq_u8_f32(B)))
#define npyv_xor_f64(A, B) \
    vreinterpretq_f64_u8(veorq_u8(vreinterpretq_u8_f64(A), vreinterpretq_u8_f64(B)))
#define npyv_xor_b8   veorq_u8
#define npyv_xor_b16  veorq_u16
#define npyv_xor_b32  veorq_u32
#define npyv_xor_b64  veorq_u64

// NOT
#define npyv_not_u8  vmvnq_u8
#define npyv_not_s8  vmvnq_s8
#define npyv_not_u16 vmvnq_u16
#define npyv_not_s16 vmvnq_s16
#define npyv_not_u32 vmvnq_u32
#define npyv_not_s32 vmvnq_s32
#define npyv_not_u64(A) vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(A)))
#define npyv_not_s64(A) vreinterpretq_s64_u8(vmvnq_u8(vreinterpretq_u8_s64(A)))
#define npyv_not_f32(A) vreinterpretq_f32_u8(vmvnq_u8(vreinterpretq_u8_f32(A)))
#define npyv_not_f64(A) vreinterpretq_f64_u8(vmvnq_u8(vreinterpretq_u8_f64(A)))
#define npyv_not_b8   vmvnq_u8
#define npyv_not_b16  vmvnq_u16
#define npyv_not_b32  vmvnq_u32
#define npyv_not_b64  npyv_not_u64

/***************************
 * Comparison
 ***************************/

// equal
#define npyv_cmpeq_u8  vceqq_u8
#define npyv_cmpeq_s8  vceqq_s8
#define npyv_cmpeq_u16 vceqq_u16
#define npyv_cmpeq_s16 vceqq_s16
#define npyv_cmpeq_u32 vceqq_u32
#define npyv_cmpeq_s32 vceqq_s32
#define npyv_cmpeq_f32 vceqq_f32
#define npyv_cmpeq_f64 vceqq_f64

#ifdef __aarch64__
    #define npyv_cmpeq_u64 vceqq_u64
    #define npyv_cmpeq_s64 vceqq_s64
#else
    NPY_FINLINE uint64x2_t npyv_cmpeq_u64(uint64x2_t a, uint64x2_t b)
    {
        uint64x2_t cmpeq = vreinterpretq_u64_u32(vceqq_u32(
            vreinterpretq_u32_u64(a), vreinterpretq_u32_u64(b)
        ));
        uint64x2_t cmpeq_h = vshlq_n_u64(cmpeq, 32);
        uint64x2_t test = vandq_u64(cmpeq, cmpeq_h);
        return vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_u64(test), 32));
    }
    #define npyv_cmpeq_s64(A, B) \
        npyv_cmpeq_u64(vreinterpretq_u64_s64(A), vreinterpretq_u64_s64(B))
#endif

// not Equal
#define npyv_cmpneq_u8(A, B)  vmvnq_u8(vceqq_u8(A, B))
#define npyv_cmpneq_s8(A, B)  vmvnq_u8(vceqq_s8(A, B))
#define npyv_cmpneq_u16(A, B) vmvnq_u16(vceqq_u16(A, B))
#define npyv_cmpneq_s16(A, B) vmvnq_u16(vceqq_s16(A, B))
#define npyv_cmpneq_u32(A, B) vmvnq_u32(vceqq_u32(A, B))
#define npyv_cmpneq_s32(A, B) vmvnq_u32(vceqq_s32(A, B))
#define npyv_cmpneq_u64(A, B) npyv_not_u64(npyv_cmpeq_u64(A, B))
#define npyv_cmpneq_s64(A, B) npyv_not_u64(npyv_cmpeq_s64(A, B))
#define npyv_cmpneq_f32(A, B) vmvnq_u32(vceqq_f32(A, B))
#define npyv_cmpneq_f64(A, B) npyv_not_u64(vceqq_f64(A, B))

// greater than
#define npyv_cmpgt_u8  vcgtq_u8
#define npyv_cmpgt_s8  vcgtq_s8
#define npyv_cmpgt_u16 vcgtq_u16
#define npyv_cmpgt_s16 vcgtq_s16
#define npyv_cmpgt_u32 vcgtq_u32
#define npyv_cmpgt_s32 vcgtq_s32
#define npyv_cmpgt_f32 vcgtq_f32
#define npyv_cmpgt_f64 vcgtq_f64

#ifdef __aarch64__
    #define npyv_cmpgt_u64 vcgtq_u64
    #define npyv_cmpgt_s64 vcgtq_s64
#else
    NPY_FINLINE uint64x2_t npyv_cmpgt_s64(int64x2_t a, int64x2_t b)
    {
        int64x2_t sub = vsubq_s64(b, a);
        uint64x2_t nsame_sbit = vreinterpretq_u64_s64(veorq_s64(a, b));
        int64x2_t test = vbslq_s64(nsame_sbit, b, sub);
        int64x2_t extend_sbit = vshrq_n_s64(test, 63);
        return  vreinterpretq_u64_s64(extend_sbit);
    }
    NPY_FINLINE uint64x2_t npyv_cmpgt_u64(uint64x2_t a, uint64x2_t b)
    {
        const uint64x2_t sbit = npyv_setall_u64(0x8000000000000000);
        a = npyv_xor_u64(a, sbit);
        b = npyv_xor_u64(b, sbit);
        return npyv_cmpgt_s64(vreinterpretq_s64_u64(a), vreinterpretq_s64_u64(b));
    }
#endif

// greater than or equal
#define npyv_cmpge_u8  vcgeq_u8
#define npyv_cmpge_s8  vcgeq_s8
#define npyv_cmpge_u16 vcgeq_u16
#define npyv_cmpge_s16 vcgeq_s16
#define npyv_cmpge_u32 vcgeq_u32
#define npyv_cmpge_s32 vcgeq_s32
#define npyv_cmpge_f32 vcgeq_f32
#define npyv_cmpge_f64 vcgeq_f64

#ifdef __aarch64__
    #define npyv_cmpge_u64 vcgeq_u64
    #define npyv_cmpge_s64 vcgeq_s64
#else
    #define npyv_cmpge_u64(A, B) npyv_not_u64(npyv_cmpgt_u64(B, A))
    #define npyv_cmpge_s64(A, B) npyv_not_u64(npyv_cmpgt_s64(B, A))
#endif

// less than
#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)
#define npyv_cmplt_f32(A, B) npyv_cmpgt_f32(B, A)
#define npyv_cmplt_f64(A, B) npyv_cmpgt_f64(B, A)

// less than or equal
#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)
#define npyv_cmple_f32(A, B) npyv_cmpge_f32(B, A)
#define npyv_cmple_f64(A, B) npyv_cmpge_f64(B, A)

// check special cases
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{ return vceqq_f32(a, a); }
#if NPY_SIMD_F64
    NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
    { return vceqq_f64(a, a); }
#endif

#endif // _NPY_SIMD_NEON_OPERATORS_H
