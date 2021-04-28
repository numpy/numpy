/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *  After computing exp(x) in high-low parts, an accurate computation is performed to obtain exp(x)-1
 *  *  Typical exp() implementation, except that:
 *  *   - tables are small (32 elements), allowing for fast gathers
 *  *   - all arguments processed in the main path
 *  *       - final VSCALEF assists branch-free design (correct overflow/underflow and special case responses)
 *  *       - a VAND is used to ensure the reduced argument |R|<2, even for large inputs
 *  *       - RZ mode used to avoid oveflow to +/-Inf for x*log2(e); helps with special case handling
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_expm1f16_z0_0:

	.align    16,0x90
	.globl __svml_expm1f16

__svml_expm1f16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_sexpm1_data_internal_avx512(%rip), %zmm5
        vmovups   320+__svml_sexpm1_data_internal_avx512(%rip), %zmm3
        vmovups   512+__svml_sexpm1_data_internal_avx512(%rip), %zmm8
        vmovups   576+__svml_sexpm1_data_internal_avx512(%rip), %zmm4
        vmovups   __svml_sexpm1_data_internal_avx512(%rip), %zmm6

/* polynomial */
        vmovups   704+__svml_sexpm1_data_internal_avx512(%rip), %zmm9
        vmovups   768+__svml_sexpm1_data_internal_avx512(%rip), %zmm12
        vmovups   128+__svml_sexpm1_data_internal_avx512(%rip), %zmm11
        vmovups   384+__svml_sexpm1_data_internal_avx512(%rip), %zmm2

/* Th - 1 */
        vmovups   832+__svml_sexpm1_data_internal_avx512(%rip), %zmm14
        vmovaps   %zmm0, %zmm1

/* 2^(52-5)*1.5 + x * log2(e) */
        vfmadd213ps {rn-sae}, %zmm3, %zmm1, %zmm5
        vcmpps    $29, {sae}, %zmm2, %zmm1, %k0

/* Z0 ~ x*log2(e), rounded to 5 fractional bits */
        vsubps    {rn-sae}, %zmm3, %zmm5, %zmm7
        vpermt2ps 64+__svml_sexpm1_data_internal_avx512(%rip), %zmm5, %zmm6
        vpermt2ps 192+__svml_sexpm1_data_internal_avx512(%rip), %zmm5, %zmm11
        vandps    448+__svml_sexpm1_data_internal_avx512(%rip), %zmm1, %zmm0

/* R = x - Z0*log(2) */
        vfnmadd213ps {rn-sae}, %zmm1, %zmm7, %zmm8

/* scale Th */
        vscalefps {rn-sae}, %zmm7, %zmm6, %zmm2
        vfnmadd231ps {rn-sae}, %zmm7, %zmm4, %zmm8
        kmovw     %k0, %edx

/* ensure |R|<2 even for special cases */
        vandps    640+__svml_sexpm1_data_internal_avx512(%rip), %zmm8, %zmm13
        vsubps    {rn-sae}, %zmm14, %zmm2, %zmm8
        vmulps    {rn-sae}, %zmm13, %zmm13, %zmm10
        vfmadd231ps {rn-sae}, %zmm13, %zmm9, %zmm12

/* Tlr + R+ R2*Poly */
        vfmadd213ps {rn-sae}, %zmm11, %zmm10, %zmm12
        vaddps    {rn-sae}, %zmm13, %zmm12, %zmm15

/* (Th-1)+Th*(Tlr + R+ R*Poly) */
        vfmadd213ps {rn-sae}, %zmm8, %zmm15, %zmm2
        vorps     %zmm0, %zmm2, %zmm0
        testl     %edx, %edx
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm1, 64(%rsp)
        vmovups   %zmm0, 128(%rsp)
        je        .LBL_1_2


        xorl      %eax, %eax


        vzeroupper
        kmovw     %k4, 24(%rsp)
        kmovw     %k5, 16(%rsp)
        kmovw     %k6, 8(%rsp)
        kmovw     %k7, (%rsp)
        movq      %rsi, 40(%rsp)
        movq      %rdi, 32(%rsp)
        movq      %r12, 56(%rsp)
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x68, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x78, 0xff, 0xff, 0xff, 0x22
        movl      %eax, %r12d
        movq      %r13, 48(%rsp)
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x70, 0xff, 0xff, 0xff, 0x22
        movl      %edx, %r13d
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xff, 0xff, 0xff, 0x22

.LBL_1_7:

        btl       %r12d, %r13d
        jc        .LBL_1_10

.LBL_1_8:

        incl      %r12d
        cmpl      $16, %r12d
        jl        .LBL_1_7


        kmovw     24(%rsp), %k4
	.cfi_restore 122
        kmovw     16(%rsp), %k5
	.cfi_restore 123
        kmovw     8(%rsp), %k6
	.cfi_restore 124
        kmovw     (%rsp), %k7
	.cfi_restore 125
        vmovups   128(%rsp), %zmm0
        movq      40(%rsp), %rsi
	.cfi_restore 4
        movq      32(%rsp), %rdi
	.cfi_restore 5
        movq      56(%rsp), %r12
	.cfi_restore 12
        movq      48(%rsp), %r13
	.cfi_restore 13
        jmp       .LBL_1_2
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x68, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x78, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x70, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xff, 0xff, 0xff, 0x22

.LBL_1_10:

        lea       64(%rsp,%r12,4), %rdi
        lea       128(%rsp,%r12,4), %rsi

        call      __svml_sexpm1_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_expm1f16,@function
	.size	__svml_expm1f16,.-__svml_expm1f16
..LN__svml_expm1f16.0:

.L_2__routine_start___svml_sexpm1_cout_rare_internal_1:

	.align    16,0x90

__svml_sexpm1_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movss     (%rdi), %xmm8
        pxor      %xmm0, %xmm0
        comiss    %xmm8, %xmm0
        ja        .LBL_2_8


        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_9


        movss     .L_2il0floatpacket.45(%rip), %xmm0
        comiss    %xmm8, %xmm0
        jb        .LBL_2_6


        comiss    .L_2il0floatpacket.46(%rip), %xmm8
        jb        .LBL_2_7


        movss     .L_2il0floatpacket.48(%rip), %xmm0
        mulss     %xmm8, %xmm0
        movss     %xmm0, -24(%rsp)
        movss     -24(%rsp), %xmm1
        movss     .L_2il0floatpacket.49(%rip), %xmm2
        movss     .L_2il0floatpacket.56(%rip), %xmm6
        addss     %xmm2, %xmm1
        movss     %xmm1, -20(%rsp)
        movss     -20(%rsp), %xmm3
        movss     .L_2il0floatpacket.52(%rip), %xmm7
        subss     %xmm2, %xmm3
        movss     %xmm3, -24(%rsp)
        movss     -24(%rsp), %xmm4
        mulss     .L_2il0floatpacket.50(%rip), %xmm4
        movss     -24(%rsp), %xmm5
        subss     %xmm4, %xmm8
        mulss     .L_2il0floatpacket.51(%rip), %xmm5
        movl      -20(%rsp), %edx
        subss     %xmm5, %xmm8
        mulss     %xmm8, %xmm6
        shll      $23, %edx
        addss     .L_2il0floatpacket.55(%rip), %xmm6
        mulss     %xmm8, %xmm6
        addss     .L_2il0floatpacket.54(%rip), %xmm6
        mulss     %xmm8, %xmm6
        addss     .L_2il0floatpacket.53(%rip), %xmm6
        mulss     %xmm8, %xmm6
        addss     %xmm7, %xmm6
        mulss     %xmm8, %xmm6
        addss     %xmm6, %xmm7
        movss     %xmm7, -16(%rsp)
        addl      -16(%rsp), %edx
        movl      %edx, (%rsi)
        ret

.LBL_2_6:

        movss     .L_2il0floatpacket.47(%rip), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, -16(%rsp)
        movl      -16(%rsp), %eax
        movl      %eax, (%rsi)
        movl      $3, %eax

.LBL_2_7:

        ret

.LBL_2_8:

        movl      $-1082130432, (%rsi)
        ret

.LBL_2_9:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_11

.LBL_2_10:

        mulss     %xmm8, %xmm8
        movss     %xmm8, (%rsi)
        ret

.LBL_2_11:

        testl     $8388607, (%rdi)
        jne       .LBL_2_10


        movss     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sexpm1_cout_rare_internal,@function
	.size	__svml_sexpm1_cout_rare_internal,.-__svml_sexpm1_cout_rare_internal
..LN__svml_sexpm1_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_sexpm1_data_internal_avx512:
	.long	1065353216
	.long	1065536903
	.long	1065724611
	.long	1065916431
	.long	1066112450
	.long	1066312762
	.long	1066517459
	.long	1066726640
	.long	1066940400
	.long	1067158842
	.long	1067382066
	.long	1067610179
	.long	1067843287
	.long	1068081499
	.long	1068324927
	.long	1068573686
	.long	1068827891
	.long	1069087663
	.long	1069353124
	.long	1069624397
	.long	1069901610
	.long	1070184894
	.long	1070474380
	.long	1070770206
	.long	1071072509
	.long	1071381432
	.long	1071697119
	.long	1072019719
	.long	1072349383
	.long	1072686266
	.long	1073030525
	.long	1073382323
	.long	0
	.long	3007986186
	.long	860277610
	.long	3010384254
	.long	2991457809
	.long	3008462297
	.long	860562562
	.long	3004532446
	.long	856238081
	.long	3001480295
	.long	857441778
	.long	815380209
	.long	3003456168
	.long	3001196762
	.long	2986372182
	.long	3006683458
	.long	848495278
	.long	851809756
	.long	3003311522
	.long	2995654817
	.long	833868005
	.long	3004843819
	.long	835836658
	.long	3003498340
	.long	2994528642
	.long	3002229827
	.long	2981408986
	.long	2983889551
	.long	2983366846
	.long	3000350873
	.long	833659207
	.long	2987748092
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	1118652779
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	2969756424
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	3221225471
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1042983923
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1056964854
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.type	__svml_sexpm1_data_internal_avx512,@object
	.size	__svml_sexpm1_data_internal_avx512,896
	.align 4
.L_2il0floatpacket.44:
	.long	0xbf800000
	.type	.L_2il0floatpacket.44,@object
	.size	.L_2il0floatpacket.44,4
	.align 4
.L_2il0floatpacket.45:
	.long	0x42b17217
	.type	.L_2il0floatpacket.45,@object
	.size	.L_2il0floatpacket.45,4
	.align 4
.L_2il0floatpacket.46:
	.long	0xc2cff1b4
	.type	.L_2il0floatpacket.46,@object
	.size	.L_2il0floatpacket.46,4
	.align 4
.L_2il0floatpacket.47:
	.long	0x7f7fffff
	.type	.L_2il0floatpacket.47,@object
	.size	.L_2il0floatpacket.47,4
	.align 4
.L_2il0floatpacket.48:
	.long	0x3fb8aa3b
	.type	.L_2il0floatpacket.48,@object
	.size	.L_2il0floatpacket.48,4
	.align 4
.L_2il0floatpacket.49:
	.long	0x4b400000
	.type	.L_2il0floatpacket.49,@object
	.size	.L_2il0floatpacket.49,4
	.align 4
.L_2il0floatpacket.50:
	.long	0x3f317200
	.type	.L_2il0floatpacket.50,@object
	.size	.L_2il0floatpacket.50,4
	.align 4
.L_2il0floatpacket.51:
	.long	0x35bfbe8e
	.type	.L_2il0floatpacket.51,@object
	.size	.L_2il0floatpacket.51,4
	.align 4
.L_2il0floatpacket.52:
	.long	0x3f800001
	.type	.L_2il0floatpacket.52,@object
	.size	.L_2il0floatpacket.52,4
	.align 4
.L_2il0floatpacket.53:
	.long	0x3efffe85
	.type	.L_2il0floatpacket.53,@object
	.size	.L_2il0floatpacket.53,4
	.align 4
.L_2il0floatpacket.54:
	.long	0x3e2aa9c6
	.type	.L_2il0floatpacket.54,@object
	.size	.L_2il0floatpacket.54,4
	.align 4
.L_2il0floatpacket.55:
	.long	0x3d2bb1b6
	.type	.L_2il0floatpacket.55,@object
	.size	.L_2il0floatpacket.55,4
	.align 4
.L_2il0floatpacket.56:
	.long	0x3c0950ef
	.type	.L_2il0floatpacket.56,@object
	.size	.L_2il0floatpacket.56,4
