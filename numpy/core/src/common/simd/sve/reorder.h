#ifndef NPY_SIMD
#error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_SVE_REORDER_H
#define _NPY_SIMD_SVE_REORDER_H

// combine lower part of two vectors
#define NPYV_IMPL_SVE_COMBINEL(T, S, W)                                  \
    NPY_FINLINE npyv_##S##W npyv_combinel_##S##W(                        \
            npyv_##S##W a, npyv_##S##W b)                                \
    {                                                                    \
        const svbool_t mask = svwhilelt_b##W##_u32(0, NPY_SIMD / W / 2); \
        sv##T t = svext_##S##W(b, b, NPY_SIMD / W / 2);                  \
        return svsel_##S##W(mask, a, t);                                 \
    }

NPYV_IMPL_SVE_COMBINEL(uint8_t, u, 8)
NPYV_IMPL_SVE_COMBINEL(uint16_t, u, 16)
NPYV_IMPL_SVE_COMBINEL(uint32_t, u, 32)
NPYV_IMPL_SVE_COMBINEL(uint64_t, u, 64)
NPYV_IMPL_SVE_COMBINEL(int8_t, s, 8)
NPYV_IMPL_SVE_COMBINEL(int16_t, s, 16)
NPYV_IMPL_SVE_COMBINEL(int32_t, s, 32)
NPYV_IMPL_SVE_COMBINEL(int64_t, s, 64)
NPYV_IMPL_SVE_COMBINEL(float32_t, f, 32)
NPYV_IMPL_SVE_COMBINEL(float64_t, f, 64)

// combine higher part of two vectors
#define NPYV_IMPL_SVE_COMBINEH(T, S, W)                                  \
    NPY_FINLINE npyv_##S##W npyv_combineh_##S##W(                        \
            npyv_##S##W a, npyv_##S##W b)                                \
    {                                                                    \
        const svbool_t mask = svwhilelt_b##W##_u32(0, NPY_SIMD / W / 2); \
        sv##T t = svext_##S##W(a, a, NPY_SIMD / W / 2);                  \
        return svsel_##S##W(mask, t, b);                                 \
    }

NPYV_IMPL_SVE_COMBINEH(uint8_t, u, 8)
NPYV_IMPL_SVE_COMBINEH(uint16_t, u, 16)
NPYV_IMPL_SVE_COMBINEH(uint32_t, u, 32)
NPYV_IMPL_SVE_COMBINEH(uint64_t, u, 64)
NPYV_IMPL_SVE_COMBINEH(int8_t, s, 8)
NPYV_IMPL_SVE_COMBINEH(int16_t, s, 16)
NPYV_IMPL_SVE_COMBINEH(int32_t, s, 32)
NPYV_IMPL_SVE_COMBINEH(int64_t, s, 64)
NPYV_IMPL_SVE_COMBINEH(float32_t, f, 32)
NPYV_IMPL_SVE_COMBINEH(float64_t, f, 64)

// combine two vectors from lower and higher parts of two other vectors
#define NPYV_IMPL_SVE_COMBINE(S, W)                  \
    NPY_FINLINE npyv_##S##W##x2 npyv_combine_##S##W( \
            npyv_##S##W a, npyv_##S##W b)            \
    {                                                \
        npyv_##S##W##x2 r;                           \
                                                     \
        r.val[0] = npyv_combinel_##S##W(a, b);       \
        r.val[1] = npyv_combineh_##S##W(a, b);       \
        return r;                                    \
    }

NPYV_IMPL_SVE_COMBINE(u, 8)
NPYV_IMPL_SVE_COMBINE(u, 16)
NPYV_IMPL_SVE_COMBINE(u, 32)
NPYV_IMPL_SVE_COMBINE(u, 64)
NPYV_IMPL_SVE_COMBINE(s, 8)
NPYV_IMPL_SVE_COMBINE(s, 16)
NPYV_IMPL_SVE_COMBINE(s, 32)
NPYV_IMPL_SVE_COMBINE(s, 64)
NPYV_IMPL_SVE_COMBINE(f, 32)
NPYV_IMPL_SVE_COMBINE(f, 64)

// interleave two vectors
#define NPYV_IMPL_SVE_ZIP(T, S, W)                       \
    NPY_FINLINE npyv_##S##W##x2 npyv_zip_##S##W(         \
            npyv_##S##W a, npyv_##S##W b)                \
    {                                                    \
        npyv_##S##W##x2 r;                               \
                                                         \
        r.val[0] = svzip1_##S##W(a, b);                  \
        sv##T t0 = svext_##S##W(a, a, NPY_SIMD / W / 2); \
        sv##T t1 = svext_##S##W(b, b, NPY_SIMD / W / 2); \
        r.val[1] = svzip1_##S##W(t0, t1);                \
        return r;                                        \
    }

NPYV_IMPL_SVE_ZIP(uint8_t, u, 8)
NPYV_IMPL_SVE_ZIP(uint16_t, u, 16)
NPYV_IMPL_SVE_ZIP(uint32_t, u, 32)
NPYV_IMPL_SVE_ZIP(uint64_t, u, 64)
NPYV_IMPL_SVE_ZIP(int8_t, s, 8)
NPYV_IMPL_SVE_ZIP(int16_t, s, 16)
NPYV_IMPL_SVE_ZIP(int32_t, s, 32)
NPYV_IMPL_SVE_ZIP(int64_t, s, 64)
NPYV_IMPL_SVE_ZIP(float32_t, f, 32)
NPYV_IMPL_SVE_ZIP(float64_t, f, 64)

// Reverse elements of each 64-bit lane
#define npyv_rev64_u8(a) \
    svrev_u8(svreinterpret_u8_u64(svrev_u64(svreinterpret_u64_u8(a))))
#define npyv_rev64_u16(a) \
    svrev_u16(svreinterpret_u16_u64(svrev_u64(svreinterpret_u64_u16(a))))
#define npyv_rev64_u32(a) \
    svrev_u32(svreinterpret_u32_u64(svrev_u64(svreinterpret_u64_u32(a))))
#define npyv_rev64_u64(a) svrev_u64(a)
#define npyv_rev64_s8(a) \
    svrev_s8(svreinterpret_s8_u64(svrev_u64(svreinterpret_u64_s8(a))))
#define npyv_rev64_s16(a) \
    svrev_s16(svreinterpret_s16_u64(svrev_u64(svreinterpret_u64_s16(a))))
#define npyv_rev64_s32(a) \
    svrev_s32(svreinterpret_s32_u64(svrev_u64(svreinterpret_u64_s32(a))))
#define npyv_rev64_s64(a) svrev_s64(a)
#define npyv_rev64_f32(a) \
    svrev_f32(svreinterpret_f32_u64(svrev_u64(svreinterpret_u64_f32(a))))
#define npyv_rev64_f64(a) svrev_f64(a)

#endif  // _NPY_SIMD_SVE_REORDER_H
