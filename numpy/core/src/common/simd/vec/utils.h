#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_UTILS_H
#define _NPY_SIMD_VEC_UTILS_H

// the following intrinsics may not some|all by zvector API on gcc/clang
#ifdef NPY_HAVE_VX
    #ifndef vec_neg
        #define vec_neg(a) (-(a)) // Vector Negate
    #endif
    #ifndef vec_add
        #define vec_add(a, b) ((a) + (b)) // Vector Add
    #endif
    #ifndef vec_sub
        #define vec_sub(a, b) ((a) - (b)) // Vector Subtract
    #endif
    #ifndef vec_mul
        #define vec_mul(a, b) ((a) * (b)) // Vector Multiply
    #endif
    #ifndef vec_div
        #define vec_div(a, b) ((a) / (b)) // Vector Divide
    #endif
    #ifndef vec_neg
        #define vec_neg(a) (-(a))
    #endif
    #ifndef vec_and
        #define vec_and(a, b) ((a) & (b)) // Vector AND
    #endif
    #ifndef vec_or
        #define vec_or(a, b) ((a) | (b)) // Vector OR
    #endif
    #ifndef vec_xor
        #define vec_xor(a, b) ((a) ^ (b)) // Vector XOR
    #endif
    #ifndef vec_sl
        #define vec_sl(a, b) ((a) << (b)) // Vector Shift Left
    #endif
    #ifndef vec_sra
        #define vec_sra(a, b) ((a) >> (b)) // Vector Shift Right
    #endif
    #ifndef vec_sr
        #define vec_sr(a, b) ((a) >> (b)) // Vector Shift Right Algebraic
    #endif
    #ifndef vec_slo
        #define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
    #endif
    #ifndef vec_sro
        #define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet
    #endif
    // vec_doublee maps to wrong intrin "vfll".
    // see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=100871
    #if defined(__GNUC__) && !defined(__clang__)
        #define npyv_doublee __builtin_s390_vflls
    #else
        #define npyv_doublee vec_doublee
    #endif
    // compatibility with vsx
    #ifndef vec_vbpermq
        #define vec_vbpermq vec_bperm_u128
    #endif
    // zvector requires second operand to signed while vsx api expected to be
    // unsigned, the following macros are set to remove this conflict
    #define vec_sl_s8(a, b)   vec_sl(a, (npyv_s8)(b))
    #define vec_sl_s16(a, b)  vec_sl(a, (npyv_s16)(b))
    #define vec_sl_s32(a, b)  vec_sl(a, (npyv_s32)(b))
    #define vec_sl_s64(a, b)  vec_sl(a, (npyv_s64)(b))
    #define vec_sra_s8(a, b)  vec_sra(a, (npyv_s8)(b))
    #define vec_sra_s16(a, b) vec_sra(a, (npyv_s16)(b))
    #define vec_sra_s32(a, b) vec_sra(a, (npyv_s32)(b))
    #define vec_sra_s64(a, b) vec_sra(a, (npyv_s64)(b))
#else
    #define vec_sl_s8 vec_sl
    #define vec_sl_s16 vec_sl
    #define vec_sl_s32 vec_sl
    #define vec_sl_s64 vec_sl
    #define vec_sra_s8 vec_sra
    #define vec_sra_s16 vec_sra
    #define vec_sra_s32 vec_sra
    #define vec_sra_s64 vec_sra
#endif

#endif // _NPY_SIMD_VEC_UTILS_H
