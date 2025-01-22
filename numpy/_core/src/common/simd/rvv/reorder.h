#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_REORDER_H
#define _NPY_SIMD_RVV_REORDER_H

// combine lower part of two vectors
NPY_FINLINE npyv_u8 npyv_combinel_u8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vslideup_vx_u8m1(a, b, 8, 16);
}

NPY_FINLINE npyv_s8 npyv_combinel_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vslideup_vx_i8m1(a, b, 8, 16);
}

NPY_FINLINE npyv_u16 npyv_combinel_u16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vslideup_vx_u16m1(a, b, 4, 8);
}

NPY_FINLINE npyv_s16 npyv_combinel_s16(npyv_s16 a, npyv_s16 b)
{
    return __riscv_vslideup_vx_i16m1(a, b, 4, 8);
}

NPY_FINLINE npyv_u32 npyv_combinel_u32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vslideup_vx_u32m1(a, b, 2, 4);
}

NPY_FINLINE npyv_s32 npyv_combinel_s32(npyv_s32 a, npyv_s32 b)
{
    return __riscv_vslideup_vx_i32m1(a, b, 2, 4);
}

NPY_FINLINE npyv_u64 npyv_combinel_u64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vslideup_vx_u64m1(a, b, 1, 2);
}

NPY_FINLINE npyv_s64 npyv_combinel_s64(npyv_s64 a, npyv_s64 b)
{
    return __riscv_vslideup_vx_i64m1(a, b, 1, 2);
}

NPY_FINLINE npyv_f32 npyv_combinel_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vslideup_vx_f32m1(a, b, 2, 4);
}

NPY_FINLINE npyv_f64 npyv_combinel_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vslideup_vx_f64m1(a, b, 1, 2);
}

// combine higher part of two vectors
NPY_FINLINE npyv_u8 npyv_combineh_u8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vslideup_vx_u8m1(
        __riscv_vslidedown_vx_u8m1(a, 8, 16),
        __riscv_vslidedown_vx_u8m1(b, 8, 16),
        8,
        16
    );
}

NPY_FINLINE npyv_u16 npyv_combineh_u16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vslideup_vx_u16m1(
        __riscv_vslidedown_vx_u16m1(a, 4, 8),
        __riscv_vslidedown_vx_u16m1(b, 4, 8),
        4,
        8
    );
}

NPY_FINLINE npyv_u32 npyv_combineh_u32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vslideup_vx_u32m1(
        __riscv_vslidedown_vx_u32m1(a, 2, 4),
        __riscv_vslidedown_vx_u32m1(b, 2, 4),
        2,
        4
    );
}

NPY_FINLINE npyv_u64 npyv_combineh_u64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vslideup_vx_u64m1(
        __riscv_vslidedown_vx_u64m1(a, 1, 2),
        __riscv_vslidedown_vx_u64m1(b, 1, 2),
        1,
        2
    );
}

NPY_FINLINE npyv_s8 npyv_combineh_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vslideup_vx_i8m1(
        __riscv_vslidedown_vx_i8m1(a, 8, 16),
        __riscv_vslidedown_vx_i8m1(b, 8, 16),
        8,
        16
    );
}

NPY_FINLINE npyv_s16 npyv_combineh_s16(npyv_s16 a, npyv_s16 b)
{
    return __riscv_vslideup_vx_i16m1(
        __riscv_vslidedown_vx_i16m1(a, 4, 8),
        __riscv_vslidedown_vx_i16m1(b, 4, 8),
        4,
        8
    );
}

NPY_FINLINE npyv_s32 npyv_combineh_s32(npyv_s32 a, npyv_s32 b)
{
    return __riscv_vslideup_vx_i32m1(
        __riscv_vslidedown_vx_i32m1(a, 2, 4),
        __riscv_vslidedown_vx_i32m1(b, 2, 4),
        2,
        4
    );
}

NPY_FINLINE npyv_s64 npyv_combineh_s64(npyv_s64 a, npyv_s64 b)
{
    return __riscv_vslideup_vx_i64m1(
        __riscv_vslidedown_vx_i64m1(a, 1, 2),
        __riscv_vslidedown_vx_i64m1(b, 1, 2),
        1,
        2
    );
}

NPY_FINLINE npyv_f32 npyv_combineh_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vslideup_vx_f32m1(
        __riscv_vslidedown_vx_f32m1(a, 2, 4),
        __riscv_vslidedown_vx_f32m1(b, 2, 4),
        2,
        4
    );
}

NPY_FINLINE npyv_f64 npyv_combineh_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vslideup_vx_f64m1(
        __riscv_vslidedown_vx_f64m1(a, 1, 2),
        __riscv_vslidedown_vx_f64m1(b, 1, 2),
        1,
        2
    );
}

// combine two vectors from lower and higher parts of two other vectors
#define NPYV_IMPL_RVV_COMBINE(T_VEC, SFX)                     \
    NPY_FINLINE T_VEC##x2 npyv_combine_##SFX(T_VEC a, T_VEC b) \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = NPY_CAT(npyv_combinel_, SFX)(a, b);         \
        r.val[1] = NPY_CAT(npyv_combineh_, SFX)(a, b);         \
        return r;                                              \
    }

NPYV_IMPL_RVV_COMBINE(npyv_u8,  u8)
NPYV_IMPL_RVV_COMBINE(npyv_s8,  s8)
NPYV_IMPL_RVV_COMBINE(npyv_u16, u16)
NPYV_IMPL_RVV_COMBINE(npyv_s16, s16)
NPYV_IMPL_RVV_COMBINE(npyv_u32, u32)
NPYV_IMPL_RVV_COMBINE(npyv_s32, s32)
NPYV_IMPL_RVV_COMBINE(npyv_u64, u64)
NPYV_IMPL_RVV_COMBINE(npyv_s64, s64)
NPYV_IMPL_RVV_COMBINE(npyv_f32, f32)
NPYV_IMPL_RVV_COMBINE(npyv_f64, f64)

// interleave & deinterleave two vectors
NPY_FINLINE npyv_u8 vzip1q_u8(npyv_u8 a, npyv_u8 b)
{
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);
    
    npyv_u8 interleave_idx = __riscv_vsrl_vx_u8m1(index, 1, vl);
    
    npyv_u8 a_gathered = __riscv_vrgather_vv_u8m1(a, interleave_idx, vl);
    npyv_u8 b_gathered = __riscv_vrgather_vv_u8m1(b, interleave_idx, vl);
    
    vbool8_t mask = __riscv_vmsne_vx_u8m1_b8(
        __riscv_vand_vx_u8m1(index, 1, vl),
        0,
        vl
    );

    return __riscv_vmerge_vvm_u8m1(a_gathered, b_gathered, mask, vl);
}

NPY_FINLINE npyv_u8 vzip2q_u8(npyv_u8 a, npyv_u8 b)
{
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);

    npyv_u8 interleave_idx = __riscv_vadd_vx_u8m1(
        __riscv_vsrl_vx_u8m1(index, 1, vl),
        8,
        vl
    );

    npyv_u8 a_gathered = __riscv_vrgather_vv_u8m1(a, interleave_idx, vl);
    npyv_u8 b_gathered = __riscv_vrgather_vv_u8m1(b, interleave_idx, vl);
    
    vbool8_t mask = __riscv_vmsne_vx_u8m1_b8(
        __riscv_vand_vx_u8m1(index, 1, vl),
        0,
        vl
    );

    return __riscv_vmerge_vvm_u8m1(a_gathered, b_gathered, mask, vl);
}

NPY_FINLINE npyv_u8 vuzp1q_u8(npyv_u8 a, npyv_u8 b) {
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);

    npyv_u8 gather_idx_a = __riscv_vmul_vx_u8m1(index, 2, vl);
    npyv_u8 gather_idx_b = gather_idx_a;

    vbool8_t high_mask = __riscv_vmsgtu_vx_u8m1_b8(index, 7, vl);

    gather_idx_b = __riscv_vsub_vx_u8m1(gather_idx_b, 16, vl);

    npyv_u8 result_a = __riscv_vrgather_vv_u8m1(a, gather_idx_a, vl);
    npyv_u8 result_b = __riscv_vrgather_vv_u8m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_u8m1(result_a, result_b, high_mask, vl);
}

NPY_FINLINE npyv_u8 vuzp2q_u8(npyv_u8 a, npyv_u8 b) 
{
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);

    npyv_u8 gather_idx_a = __riscv_vadd_vx_u8m1(
        __riscv_vmul_vx_u8m1(index, 2, vl),
        1,
        vl
    );

    npyv_u8 gather_idx_b = gather_idx_a;

    vbool8_t high_mask = __riscv_vmsgtu_vx_u8m1_b8(index, 7, vl);

    gather_idx_b = __riscv_vsub_vx_u8m1(gather_idx_b, 16, vl);

    npyv_u8 result_a = __riscv_vrgather_vv_u8m1(a, gather_idx_a, vl);
    npyv_u8 result_b = __riscv_vrgather_vv_u8m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_u8m1(result_a, result_b, high_mask, vl);
}

NPY_FINLINE npyv_s8 vzip1q_s8(npyv_s8 a, npyv_s8 b)
{
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);

    npyv_u8 interleave_idx = __riscv_vsrl_vx_u8m1(index, 1, vl);

    npyv_s8 a_gathered = __riscv_vrgather_vv_i8m1(a, interleave_idx, vl);
    npyv_s8 b_gathered = __riscv_vrgather_vv_i8m1(b, interleave_idx, vl);

    vbool8_t mask = __riscv_vmsne_vx_u8m1_b8(
        __riscv_vand_vx_u8m1(index, 1, vl),
        0,
        vl
    );

    return __riscv_vmerge_vvm_i8m1(a_gathered, b_gathered, mask, vl);
}

NPY_FINLINE npyv_s8 vzip2q_s8(npyv_s8 a, npyv_s8 b)
{
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);

    npyv_u8 interleave_idx = __riscv_vadd_vx_u8m1(
        __riscv_vsrl_vx_u8m1(index, 1, vl),
        8,
        vl
    );

    npyv_s8 a_gathered = __riscv_vrgather_vv_i8m1(a, interleave_idx, vl);
    npyv_s8 b_gathered = __riscv_vrgather_vv_i8m1(b, interleave_idx, vl);

    vbool8_t mask = __riscv_vmsne_vx_u8m1_b8(
        __riscv_vand_vx_u8m1(index, 1, vl),
        0,
        vl
    );

    return __riscv_vmerge_vvm_i8m1(a_gathered, b_gathered, mask, vl);
}

NPY_FINLINE npyv_s8 vuzp1q_s8(npyv_s8 a, npyv_s8 b)
{
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);

    npyv_u8 gather_idx_a = __riscv_vmul_vx_u8m1(index, 2, vl);
    npyv_u8 gather_idx_b = gather_idx_a;

    vbool8_t high_mask = __riscv_vmsgtu_vx_u8m1_b8(index, 7, vl);

    gather_idx_b = __riscv_vsub_vx_u8m1(gather_idx_b, 16, vl);

    npyv_s8 a_gathered = __riscv_vrgather_vv_i8m1(a, gather_idx_a, vl);
    npyv_s8 b_gathered = __riscv_vrgather_vv_i8m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_i8m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_s8 vuzp2q_s8(npyv_s8 a, npyv_s8 b)
{
    size_t vl = __riscv_vsetvlmax_e8m1();
    npyv_u8 index = __riscv_vid_v_u8m1(vl);

    npyv_u8 gather_idx_a = __riscv_vadd_vx_u8m1(
        __riscv_vmul_vx_u8m1(index, 2, vl),
        1,
        vl
    );

    npyv_u8 gather_idx_b = gather_idx_a;

    vbool8_t high_mask = __riscv_vmsgtu_vx_u8m1_b8(index, 7, vl);

    gather_idx_b = __riscv_vsub_vx_u8m1(gather_idx_b, 16, vl);

    npyv_s8 a_gathered = __riscv_vrgather_vv_i8m1(a, gather_idx_a, vl);
    npyv_s8 b_gathered = __riscv_vrgather_vv_i8m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_i8m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_u16 vzip1q_u16(npyv_u16 a, npyv_u16 b)
{
    size_t vl = __riscv_vsetvlmax_e16m1();
    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 interleave_idx = __riscv_vsrl_vx_u16m1(index, 1, vl);

    npyv_u16 a_gathered = __riscv_vrgather_vv_u16m1(a, interleave_idx, vl);
    npyv_u16 b_gathered = __riscv_vrgather_vv_u16m1(b, interleave_idx, vl);

    vbool16_t mask = __riscv_vmsne_vx_u16m1_b16(
        __riscv_vand_vx_u16m1(index, 1, vl),
        0,
        vl
    );

    return __riscv_vmerge_vvm_u16m1(a_gathered, b_gathered, mask, vl);
}

NPY_FINLINE npyv_u16 vzip2q_u16(npyv_u16 a, npyv_u16 b)
{
    size_t vl = __riscv_vsetvlmax_e16m1();
    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 interleave_idx = __riscv_vadd_vx_u16m1(
        __riscv_vsrl_vx_u16m1(index, 1, vl),
        4,
        vl
    );

    npyv_u16 a_gathered = __riscv_vrgather_vv_u16m1(a, interleave_idx, vl);
    npyv_u16 b_gathered = __riscv_vrgather_vv_u16m1(b, interleave_idx, vl);

    vbool16_t mask = __riscv_vmsne_vx_u16m1_b16(
        __riscv_vand_vx_u16m1(index, 1, vl),
        0,
        vl
    );

    return __riscv_vmerge_vvm_u16m1(a_gathered, b_gathered, mask, vl);
}

NPY_FINLINE npyv_u16 vuzp1q_u16(npyv_u16 a, npyv_u16 b)
{
    size_t vl = __riscv_vsetvlmax_e16m1();
    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 gather_idx_a = __riscv_vmul_vx_u16m1(index, 2, vl);

    npyv_u16 gather_idx_b = __riscv_vmul_vx_u16m1(
        __riscv_vsub_vx_u16m1(index, 4, vl),
        2,
        vl
    );

    npyv_s16 signed_index = __riscv_vreinterpret_v_u16m1_i16m1(index);
    vbool16_t high_mask = __riscv_vmsgt_vx_i16m1_b16(signed_index, 3, vl);

    npyv_u16 a_gathered = __riscv_vrgather_vv_u16m1(a, gather_idx_a, vl);
    npyv_u16 b_gathered = __riscv_vrgather_vv_u16m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_u16m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_u16 vuzp2q_u16(npyv_u16 a, npyv_u16 b)
{
    size_t vl = __riscv_vsetvlmax_e16m1();
    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 gather_idx_a = __riscv_vadd_vx_u16m1(
        __riscv_vmul_vx_u16m1(index, 2, vl),
        1,
        vl
    );

    npyv_u16 gather_idx_b = __riscv_vadd_vx_u16m1(
        __riscv_vmul_vx_u16m1(
            __riscv_vsub_vx_u16m1(index, 4, vl),
            2,
            vl
        ),
        1,
        vl
    );

    npyv_s16 signed_index = __riscv_vreinterpret_v_u16m1_i16m1(index);
    vbool16_t high_mask = __riscv_vmsgt_vx_i16m1_b16(signed_index, 3, vl);

    npyv_u16 a_gathered = __riscv_vrgather_vv_u16m1(a, gather_idx_a, vl);
    npyv_u16 b_gathered = __riscv_vrgather_vv_u16m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_u16m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_s16 vzip1q_s16(npyv_s16 a, npyv_s16 b)
{
    size_t vl = __riscv_vsetvlmax_e16m1();

    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 gather_idx = __riscv_vsrl_vx_u16m1(index, 1, vl);

    vbool16_t sel_mask = __riscv_vmsne_vx_u16m1_b16(
        __riscv_vand_vx_u16m1(index, 1, vl),
        0,
        vl
    );

    npyv_s16 a_gathered = __riscv_vrgather_vv_i16m1(a, gather_idx, vl);
    npyv_s16 b_gathered = __riscv_vrgather_vv_i16m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_i16m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_s16 vzip2q_s16(npyv_s16 a, npyv_s16 b)
{
    size_t vl = __riscv_vsetvlmax_e16m1();

    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 gather_idx = __riscv_vadd_vx_u16m1(
     __riscv_vsrl_vx_u16m1(index, 1, vl),
     4,
     vl
    );

    vbool16_t sel_mask = __riscv_vmsne_vx_u16m1_b16(
     __riscv_vand_vx_u16m1(index, 1, vl),
     0,
     vl
    );

    npyv_s16 a_gathered = __riscv_vrgather_vv_i16m1(a, gather_idx, vl);
    npyv_s16 b_gathered = __riscv_vrgather_vv_i16m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_i16m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_s16 vuzp1q_s16(npyv_s16 a, npyv_s16 b)
{
    size_t vl = __riscv_vsetvlmax_e16m1();

    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 gather_idx_a = __riscv_vmul_vx_u16m1(index, 2, vl);

    npyv_u16 gather_idx_b = __riscv_vmul_vx_u16m1(
        __riscv_vsub_vx_u16m1(index, 4, vl),
        2,
        vl
    );

    npyv_s16 signed_index = __riscv_vreinterpret_v_u16m1_i16m1(index);
    vbool16_t high_mask = __riscv_vmsgt_vx_i16m1_b16(signed_index, 3, vl);

    npyv_s16 a_gathered = __riscv_vrgather_vv_i16m1(a, gather_idx_a, vl);
    npyv_s16 b_gathered = __riscv_vrgather_vv_i16m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_i16m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_s16 vuzp2q_s16(npyv_s16 a, npyv_s16 b) 
{
    size_t vl = __riscv_vsetvlmax_e16m1();

    npyv_u16 index = __riscv_vid_v_u16m1(vl);

    npyv_u16 gather_idx_a = __riscv_vadd_vx_u16m1(
        __riscv_vmul_vx_u16m1(index, 2, vl),
        1,
        vl
    );

    npyv_u16 gather_idx_b = __riscv_vadd_vx_u16m1(
        __riscv_vmul_vx_u16m1(
            __riscv_vsub_vx_u16m1(index, 4, vl),
            2,
            vl
        ),
        1,
        vl
    );

    npyv_s16 signed_index = __riscv_vreinterpret_v_u16m1_i16m1(index);
    vbool16_t high_mask = __riscv_vmsgt_vx_i16m1_b16(signed_index, 3, vl);

    npyv_s16 a_gathered = __riscv_vrgather_vv_i16m1(a, gather_idx_a, vl);
    npyv_s16 b_gathered = __riscv_vrgather_vv_i16m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_i16m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_u32 vzip1q_u32(npyv_u32 a, npyv_u32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx = __riscv_vsrl_vx_u32m1(index, 1, vl);

    vbool32_t sel_mask = __riscv_vmsne_vx_u32m1_b32(
        __riscv_vand_vx_u32m1(index, 1, vl),
        0,
        vl
    );

    npyv_u32 a_gathered = __riscv_vrgather_vv_u32m1(a, gather_idx, vl);
    npyv_u32 b_gathered = __riscv_vrgather_vv_u32m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_u32m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_u32 vzip2q_u32(npyv_u32 a, npyv_u32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx = __riscv_vadd_vx_u32m1(
        __riscv_vsrl_vx_u32m1(index, 1, vl),
        2,
        vl
    );

    vbool32_t sel_mask = __riscv_vmsne_vx_u32m1_b32(
        __riscv_vand_vx_u32m1(index, 1, vl),
        0,
        vl
    );

    npyv_u32 a_gathered = __riscv_vrgather_vv_u32m1(a, gather_idx, vl);
    npyv_u32 b_gathered = __riscv_vrgather_vv_u32m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_u32m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_u32 vuzp1q_u32(npyv_u32 a, npyv_u32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx_a = __riscv_vmul_vx_u32m1(index, 2, vl);

    npyv_u32 gather_idx_b = __riscv_vmul_vx_u32m1(
        __riscv_vsub_vx_u32m1(index, 2, vl),
        2,
        vl
    );

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t high_mask = __riscv_vmsgt_vx_i32m1_b32(signed_index, 1, vl);

    npyv_u32 a_gathered = __riscv_vrgather_vv_u32m1(a, gather_idx_a, vl);
    npyv_u32 b_gathered = __riscv_vrgather_vv_u32m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_u32m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_u32 vuzp2q_u32(npyv_u32 a, npyv_u32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx_a = __riscv_vadd_vx_u32m1(
        __riscv_vmul_vx_u32m1(index, 2, vl),
        1,
        vl
    );

    npyv_u32 gather_idx_b = __riscv_vadd_vx_u32m1(
        __riscv_vmul_vx_u32m1(
            __riscv_vsub_vx_u32m1(index, 2, vl),
            2,
            vl
        ),
        1,
        vl
    );

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t high_mask = __riscv_vmsgt_vx_i32m1_b32(signed_index, 1, vl);

    npyv_u32 a_gathered = __riscv_vrgather_vv_u32m1(a, gather_idx_a, vl);
    npyv_u32 b_gathered = __riscv_vrgather_vv_u32m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_u32m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_s32 vzip1q_s32(npyv_s32 a, npyv_s32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx = __riscv_vsrl_vx_u32m1(index, 1, vl);

    vbool32_t sel_mask = __riscv_vmsne_vx_u32m1_b32(
        __riscv_vand_vx_u32m1(index, 1, vl),
        0,
        vl
    );

    npyv_s32 a_gathered = __riscv_vrgather_vv_i32m1(a, gather_idx, vl);
    npyv_s32 b_gathered = __riscv_vrgather_vv_i32m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_i32m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_s32 vzip2q_s32(npyv_s32 a, npyv_s32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx = __riscv_vadd_vx_u32m1(
        __riscv_vsrl_vx_u32m1(index, 1, vl),
        2,
        vl
    );

    vbool32_t sel_mask = __riscv_vmsne_vx_u32m1_b32(
        __riscv_vand_vx_u32m1(index, 1, vl),
        0,
        vl
    );

    npyv_s32 a_gathered = __riscv_vrgather_vv_i32m1(a, gather_idx, vl);
    npyv_s32 b_gathered = __riscv_vrgather_vv_i32m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_i32m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_s32 vuzp1q_s32(npyv_s32 a, npyv_s32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx_a = __riscv_vmul_vx_u32m1(index, 2, vl);

    npyv_u32 gather_idx_b = __riscv_vmul_vx_u32m1(
        __riscv_vsub_vx_u32m1(index, 2, vl),
        2,
        vl
    );

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t high_mask = __riscv_vmsgt_vx_i32m1_b32(signed_index, 1, vl);

    npyv_s32 a_gathered = __riscv_vrgather_vv_i32m1(a, gather_idx_a, vl);
    npyv_s32 b_gathered = __riscv_vrgather_vv_i32m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_i32m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_s32 vuzp2q_s32(npyv_s32 a, npyv_s32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx_a = __riscv_vadd_vx_u32m1(
        __riscv_vmul_vx_u32m1(index, 2, vl),
        1,
        vl
    );

    npyv_u32 gather_idx_b = __riscv_vadd_vx_u32m1(
        __riscv_vmul_vx_u32m1(
            __riscv_vsub_vx_u32m1(index, 2, vl),
            2,
            vl
        ),
        1,
        vl
    );

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t high_mask = __riscv_vmsgt_vx_i32m1_b32(signed_index, 1, vl);

    npyv_s32 a_gathered = __riscv_vrgather_vv_i32m1(a, gather_idx_a, vl);
    npyv_s32 b_gathered = __riscv_vrgather_vv_i32m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_i32m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_f32 vzip1q_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx = __riscv_vsrl_vx_u32m1(index, 1, vl);

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t sel_mask = __riscv_vmsgt_vx_i32m1_b32(
        __riscv_vand_vx_i32m1(signed_index, 1, vl),
        0,
        vl
    );

    npyv_f32 a_gathered = __riscv_vrgather_vv_f32m1(a, gather_idx, vl);
    npyv_f32 b_gathered = __riscv_vrgather_vv_f32m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_f32m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_f32 vzip2q_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx = __riscv_vadd_vx_u32m1(
        __riscv_vsrl_vx_u32m1(index, 1, vl),
        2,  
        vl
    );

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t sel_mask = __riscv_vmsgt_vx_i32m1_b32(
        __riscv_vand_vx_i32m1(signed_index, 1, vl),
        0,
        vl
    );

    npyv_f32 a_gathered = __riscv_vrgather_vv_f32m1(a, gather_idx, vl);
    npyv_f32 b_gathered = __riscv_vrgather_vv_f32m1(b, gather_idx, vl);

    return __riscv_vmerge_vvm_f32m1(a_gathered, b_gathered, sel_mask, vl);
}

NPY_FINLINE npyv_f32 vuzp1q_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx_a = __riscv_vmul_vx_u32m1(index, 2, vl);

    npyv_u32 gather_idx_b = __riscv_vmul_vx_u32m1(
        __riscv_vsub_vx_u32m1(index, 2, vl),
        2,
        vl
    );

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t high_mask = __riscv_vmsgt_vx_i32m1_b32(signed_index, 1, vl);

    npyv_f32 a_gathered = __riscv_vrgather_vv_f32m1(a, gather_idx_a, vl);
    npyv_f32 b_gathered = __riscv_vrgather_vv_f32m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_f32m1(a_gathered, b_gathered, high_mask, vl);
}

NPY_FINLINE npyv_f32 vuzp2q_f32(npyv_f32 a, npyv_f32 b)
{
    size_t vl = __riscv_vsetvlmax_e32m1();

    npyv_u32 index = __riscv_vid_v_u32m1(vl);

    npyv_u32 gather_idx_a = __riscv_vadd_vx_u32m1(
        __riscv_vmul_vx_u32m1(index, 2, vl),
        1,
        vl
    );

    npyv_u32 gather_idx_b = __riscv_vadd_vx_u32m1(
        __riscv_vmul_vx_u32m1(
            __riscv_vsub_vx_u32m1(index, 2, vl),
            2,
            vl
        ),
        1,
        vl
    );

    npyv_s32 signed_index = __riscv_vreinterpret_v_u32m1_i32m1(index);
    vbool32_t high_mask = __riscv_vmsgt_vx_i32m1_b32(signed_index, 1, vl);

    npyv_f32 a_gathered = __riscv_vrgather_vv_f32m1(a, gather_idx_a, vl);
    npyv_f32 b_gathered = __riscv_vrgather_vv_f32m1(b, gather_idx_b, vl);

    return __riscv_vmerge_vvm_f32m1(a_gathered, b_gathered, high_mask, vl);
}

// interleave & deinterleave two vectors
#define NPYV_IMPL_RVV_ZIP(T_VEC, SFX)                       \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)   \
    {                                                        \
        T_VEC##x2 r;                                         \
        r.val[0] = vzip1q_##SFX(a, b);                       \
        r.val[1] = vzip2q_##SFX(a, b);                       \
        return r;                                            \
    }                                                        \
    NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b) \
    {                                                        \
        T_VEC##x2 r;                                         \
        r.val[0] = vuzp1q_##SFX(a, b);                       \
        r.val[1] = vuzp2q_##SFX(a, b);                       \
        return r;                                            \
    }

NPYV_IMPL_RVV_ZIP(npyv_u8,  u8)
NPYV_IMPL_RVV_ZIP(npyv_s8,  s8)
NPYV_IMPL_RVV_ZIP(npyv_u16, u16)
NPYV_IMPL_RVV_ZIP(npyv_s16, s16)
NPYV_IMPL_RVV_ZIP(npyv_u32, u32)
NPYV_IMPL_RVV_ZIP(npyv_s32, s32)
NPYV_IMPL_RVV_ZIP(npyv_f32, f32)

#define NPYV_IMPL_RVV_ZIP2(SFX)                                              \
    NPY_FINLINE npyv_##SFX##x2 npyv_zip_##SFX(npyv_##SFX a, npyv_##SFX b)    \
    {                                                                        \
        npyv_##SFX##x2 r;                                                    \
        r.val[0] = npyv_combinel_##SFX(a, b);                                \
        r.val[1] = npyv_combineh_##SFX(a, b);                                \
        return r;                                                            \
    }                                                                        \
                                                                             \
    NPY_FINLINE npyv_##SFX##x2 npyv_unzip_##SFX(npyv_##SFX a, npyv_##SFX b)  \
    {                                                                        \
        npyv_##SFX##x2 r;                                                    \
        r.val[0] = npyv_combinel_##SFX(a, b);                                \
        r.val[1] = npyv_combineh_##SFX(a, b);                                \
        return r;                                                            \
    }

NPYV_IMPL_RVV_ZIP2(u64)
NPYV_IMPL_RVV_ZIP2(s64)
NPYV_IMPL_RVV_ZIP2(f64)

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
    npyv_u8 vid = __riscv_vid_v_u8m1(8);
    npyv_u8 vid_slideup = __riscv_vslideup_vx_u8m1(vid, vid, 8, 16);
    npyv_u8 sub = __riscv_vslideup_vx_u8m1(__riscv_vmv_v_x_u8m1(7, 16), __riscv_vmv_v_x_u8m1(7 + 8, 16), 8, 16);
    npyv_u8 idxs = __riscv_vsub_vv_u8m1(sub, vid_slideup, 16);
    return __riscv_vrgather_vv_u8m1(a, idxs, 16);
}

NPY_FINLINE npyv_s8 npyv_rev64_s8(npyv_s8 a) {
    npyv_u8 vid = __riscv_vid_v_u8m1(8);
    npyv_u8 vid_slideup = __riscv_vslideup_vx_u8m1(vid, vid, 8, 16);
    npyv_u8 sub = __riscv_vslideup_vx_u8m1(__riscv_vmv_v_x_u8m1(7,16), __riscv_vmv_v_x_u8m1(7 + 8, 16), 8, 16);
    npyv_u8 idxs = __riscv_vsub_vv_u8m1(sub, vid_slideup, 16);
    return __riscv_vrgather_vv_i8m1(a, idxs, 16);
}

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a) {
    npyv_u16 vid = __riscv_vid_v_u16m1(4);
    npyv_u16 vid_slideup = __riscv_vslideup_vx_u16m1(vid, vid, 4, 8);
    npyv_u16 sub = __riscv_vslideup_vx_u16m1(__riscv_vmv_v_x_u16m1(3, 8), __riscv_vmv_v_x_u16m1(3 + 4, 8), 4, 8);
    npyv_u16 idxs = __riscv_vsub_vv_u16m1(sub, vid_slideup, 8);
    return __riscv_vrgather_vv_u16m1(a, idxs, 8);
}

NPY_FINLINE npyv_s16 npyv_rev64_s16(npyv_s16 a) {
    npyv_u16 vid = __riscv_vid_v_u16m1(4);
    npyv_u16 vid_slideup = __riscv_vslideup_vx_u16m1(vid, vid, 4, 8);
    npyv_u16 sub = __riscv_vslideup_vx_u16m1(__riscv_vmv_v_x_u16m1(3, 8), __riscv_vmv_v_x_u16m1(3 + 4, 8), 4, 8);
    npyv_u16 idxs = __riscv_vsub_vv_u16m1(sub, vid_slideup, 8);
    return __riscv_vrgather_vv_i16m1(a, idxs, 8);
}

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a) {
    npyv_u32 vid = __riscv_vid_v_u32m1(2);
    npyv_u32 vid_slideup = __riscv_vslideup_vx_u32m1(vid, vid, 2, 4);
    npyv_u32 sub = __riscv_vslideup_vx_u32m1(__riscv_vmv_v_x_u32m1(1, 4), __riscv_vmv_v_x_u32m1(1 + 2, 4), 2, 4);
    npyv_u32 idxs = __riscv_vsub_vv_u32m1(sub, vid_slideup, 4);
    return __riscv_vrgather_vv_u32m1(a, idxs, 4);
}

NPY_FINLINE npyv_s32 npyv_rev64_s32(npyv_s32 a) {
    npyv_u32 vid = __riscv_vid_v_u32m1(2);
    npyv_u32 vid_slideup = __riscv_vslideup_vx_u32m1(vid, vid, 2, 4);
    npyv_u32 sub = __riscv_vslideup_vx_u32m1(__riscv_vmv_v_x_u32m1(1, 4), __riscv_vmv_v_x_u32m1(1 + 2, 4), 2, 4);
    npyv_u32 idxs = __riscv_vsub_vv_u32m1(sub, vid_slideup, 4);
    return __riscv_vrgather_vv_i32m1(a, idxs, 4);
}

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a) {
    npyv_u32 vid = __riscv_vid_v_u32m1(2);
    npyv_u32 vid_slideup = __riscv_vslideup_vx_u32m1(vid, vid, 2, 4);
    npyv_u32 sub = __riscv_vslideup_vx_u32m1(__riscv_vmv_v_x_u32m1(1, 4), __riscv_vmv_v_x_u32m1(1 + 2, 4), 2, 4);
    npyv_u32 idxs = __riscv_vsub_vv_u32m1(sub, vid_slideup, 4);
    return __riscv_vrgather_vv_f32m1(a, idxs, 4);
}

// Permuting the elements of each 128-bit lane by immediate index for
// each element.
#define npyv_permi128_u32(A, E0, E1, E2, E3)          \
    npyv_set_u32(                                     \
        __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(A, E0, 4)), __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(A, E1, 4)), \
        __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(A, E2, 4)), __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(A, E3, 4))  \
    )
#define npyv_permi128_s32(A, E0, E1, E2, E3)          \
    npyv_set_s32(                                     \
        __riscv_vmv_x_s_i32m1_i32(__riscv_vslidedown_vx_i32m1(A, E0, 4)), __riscv_vmv_x_s_i32m1_i32(__riscv_vslidedown_vx_i32m1(A, E1, 4)), \
        __riscv_vmv_x_s_i32m1_i32(__riscv_vslidedown_vx_i32m1(A, E2, 4)), __riscv_vmv_x_s_i32m1_i32(__riscv_vslidedown_vx_i32m1(A, E3, 4))  \
    )
#define npyv_permi128_f32(A, E0, E1, E2, E3)          \
    npyv_set_f32(                                     \
        __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(A, E0, 4)), __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(A, E1, 4)), \
        __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(A, E2, 4)), __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(A, E3, 4))  \
    )

#define npyv_permi128_u64(A, E0, E1)                  \
    npyv_set_u64(                                     \
        __riscv_vmv_x_s_u64m1_u64(__riscv_vslidedown_vx_u64m1(A, E0, 2)), __riscv_vmv_x_s_u64m1_u64(__riscv_vslidedown_vx_u64m1(A, E1, 2))  \
    )
#define npyv_permi128_s64(A, E0, E1)                  \
    npyv_set_s64(                                     \
        __riscv_vmv_x_s_i64m1_i64(__riscv_vslidedown_vx_i64m1(A, E0, 2)), __riscv_vmv_x_s_i64m1_i64(__riscv_vslidedown_vx_i64m1(A, E1, 2))  \
    )
#define npyv_permi128_f64(A, E0, E1)                  \
    npyv_set_f64(                                     \
        __riscv_vfmv_f_s_f64m1_f64(__riscv_vslidedown_vx_f64m1(A, E0, 2)), __riscv_vfmv_f_s_f64m1_f64(__riscv_vslidedown_vx_f64m1(A, E1, 2))  \
    )

#endif // _NPY_SIMD_RVV_REORDER_H
