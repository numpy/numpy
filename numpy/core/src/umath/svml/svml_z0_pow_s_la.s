/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *   Typical computation sequences for log2() and exp2(),
 *  *     with smaller tables (32- and 16-element tables)
 *  *    The log2() part uses VGETEXP/VGETMANT (which treat denormals correctly),
 *  *         similar to DP ln() algorithm
 *  *    Branches are not needed for overflow/underflow:
 *  *     - RZ mode used to prevent overflow to +/-Inf in intermediate computations
 *  *     - final VSCALEF properly handles overflow and underflow cases
 *  *    Callout is still used for Inf/NaNs or x<=0
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_powf16_z0_0:

	.align    16,0x90
	.globl __svml_powf16

__svml_powf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $256, %rsp

/* GetMant(x), normalized to [0.5,1) for x>=0, NaN for x<0 */
        vgetmantps $10, {sae}, %zmm0, %zmm4
        vmovups   384+__svml_spow_data_internal_avx512(%rip), %zmm9
        vmovups   448+__svml_spow_data_internal_avx512(%rip), %zmm11
        vmovups   512+__svml_spow_data_internal_avx512(%rip), %zmm12
        vmovups   320+__svml_spow_data_internal_avx512(%rip), %zmm8
        vmovups   576+__svml_spow_data_internal_avx512(%rip), %zmm15

/* GetExp(x) */
        vgetexpps {sae}, %zmm0, %zmm14

/* Table lookup */
        vmovups   __svml_spow_data_internal_avx512(%rip), %zmm13
        vmovups   128+__svml_spow_data_internal_avx512(%rip), %zmm3

/* SglRcp ~ 1/Mantissa */
        vrcp14ps  %zmm4, %zmm6

/* x<=0 or Inf/NaN? */
        vfpclassps $223, %zmm0, %k0

/* round SglRcp to 5 fractional bits (RN mode, no Precision exception) */
        vrndscaleps $88, {sae}, %zmm6, %zmm7
        vmovups   704+__svml_spow_data_internal_avx512(%rip), %zmm6
        kmovw     %k0, %edx

/* Reduced argument: R = (SglRcp*Mantissa - 1) */
        vfmsub213ps {rn-sae}, %zmm9, %zmm7, %zmm4

/* Prepare exponent correction: SglRcp<1.5? */
        vcmpps    $17, {sae}, %zmm8, %zmm7, %k1

/* start polynomial evaluation */
        vfmadd231ps {rn-sae}, %zmm4, %zmm11, %zmm12

/* Prepare table index */
        vpsrld    $18, %zmm7, %zmm10
        vmovups   768+__svml_spow_data_internal_avx512(%rip), %zmm7
        vfmadd231ps {rn-sae}, %zmm4, %zmm12, %zmm15

/* add 1 to Expon if SglRcp<1.5 */
        vaddps    {rn-sae}, %zmm9, %zmm14, %zmm14{%k1}
        vpermt2ps 64+__svml_spow_data_internal_avx512(%rip), %zmm10, %zmm13
        vpermt2ps 192+__svml_spow_data_internal_avx512(%rip), %zmm10, %zmm3

/* Th+Expon */
        vaddps    {rn-sae}, %zmm14, %zmm13, %zmm2
        vmovaps   %zmm1, %zmm5
        vmovups   640+__svml_spow_data_internal_avx512(%rip), %zmm1

/* y Inf/NaN? */
        vfpclassps $153, %zmm5, %k2
        vfmadd231ps {rn-sae}, %zmm4, %zmm15, %zmm1
        kmovw     %k2, %eax

/* Poly_low */
        vfmadd231ps {rn-sae}, %zmm4, %zmm1, %zmm6

/* Th+Expon+R*c1h */
        vmovaps   %zmm2, %zmm9
        orl       %eax, %edx
        vfmadd231ps {rn-sae}, %zmm4, %zmm7, %zmm9

/* Tl + R*Poly_low */
        vfmadd231ps {rn-sae}, %zmm4, %zmm6, %zmm3
        vmovups   960+__svml_spow_data_internal_avx512(%rip), %zmm6

/* (R*c1h)_high */
        vsubps    {rn-sae}, %zmm2, %zmm9, %zmm8

/* High1 + Tl */
        vaddps    {rn-sae}, %zmm3, %zmm9, %zmm11

/* (R*c1h)_low */
        vfmsub213ps {rn-sae}, %zmm8, %zmm7, %zmm4
        vmovups   1088+__svml_spow_data_internal_avx512(%rip), %zmm7

/* y*High */
        vmulps    {rz-sae}, %zmm5, %zmm11, %zmm12

/* Tlh */
        vsubps    {rn-sae}, %zmm9, %zmm11, %zmm10

/* (y*High)_low */
        vfmsub213ps {rz-sae}, %zmm12, %zmm5, %zmm11

/* Tll */
        vsubps    {rn-sae}, %zmm10, %zmm3, %zmm3

/* Tll + (R*c1h)_low */
        vaddps    {rn-sae}, %zmm4, %zmm3, %zmm13
        vmovups   832+__svml_spow_data_internal_avx512(%rip), %zmm4

/* Zl = y*Tll + Zl */
        vfmadd213ps {rz-sae}, %zmm11, %zmm5, %zmm13

/*
 * scaled result
 * Filter very large |y*log2(x)| and scale final result for LRB2
 */
        vmovups   1408+__svml_spow_data_internal_avx512(%rip), %zmm11
        vaddps    {rz-sae}, %zmm13, %zmm12, %zmm2
        vsubps    {rn-sae}, %zmm12, %zmm2, %zmm14
        vaddps    {rd-sae}, %zmm4, %zmm2, %zmm1

/*
 * /
 * exp2 computation starts here
 */
        vreduceps $65, {sae}, %zmm2, %zmm15
        vmovups   1024+__svml_spow_data_internal_avx512(%rip), %zmm12
        vsubps    {rn-sae}, %zmm14, %zmm13, %zmm3

/* Table lookup: The, Tle/The */
        vpermps   256+__svml_spow_data_internal_avx512(%rip), %zmm1, %zmm10
        vandps    1344+__svml_spow_data_internal_avx512(%rip), %zmm2, %zmm2
        vaddps    {rn-sae}, %zmm3, %zmm15, %zmm4
        vpslld    $19, %zmm1, %zmm1
        vcmpps    $22, {sae}, %zmm11, %zmm2, %k3

/* ensure |R|<2 even for special cases */
        vandps    896+__svml_spow_data_internal_avx512(%rip), %zmm4, %zmm8
        vandps    1472+__svml_spow_data_internal_avx512(%rip), %zmm1, %zmm13
        kmovw     %k3, %ecx

/* R*The */
        vmulps    {rn-sae}, %zmm8, %zmm10, %zmm9

/* polynomial */
        vfmadd231ps {rn-sae}, %zmm8, %zmm6, %zmm12
        vfmadd213ps {rn-sae}, %zmm7, %zmm8, %zmm12
        orl       %ecx, %edx

/* The + The*R*poly */
        vfmadd213ps {rn-sae}, %zmm10, %zmm9, %zmm12
        vmulps    {rn-sae}, %zmm13, %zmm12, %zmm1
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        vmovaps   %zmm1, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm5, 128(%rsp)
        vmovups   %zmm1, 192(%rsp)
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
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x28, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x20, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x38, 0xff, 0xff, 0xff, 0x22
        movl      %eax, %r12d
        movq      %r13, 48(%rsp)
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x30, 0xff, 0xff, 0xff, 0x22
        movl      %edx, %r13d
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x18, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22

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
        vmovups   192(%rsp), %zmm1
        movq      40(%rsp), %rsi
	.cfi_restore 4
        movq      32(%rsp), %rdi
	.cfi_restore 5
        movq      56(%rsp), %r12
	.cfi_restore 12
        movq      48(%rsp), %r13
	.cfi_restore 13
        jmp       .LBL_1_2
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x28, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x20, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x38, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x30, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x18, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22

.LBL_1_10:

        lea       64(%rsp,%r12,4), %rdi
        lea       128(%rsp,%r12,4), %rsi
        lea       192(%rsp,%r12,4), %rdx

        call      __svml_spow_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_powf16,@function
	.size	__svml_powf16,.-__svml_powf16
..LN__svml_powf16.0:

.L_2__routine_start___spow_la_TestIntFunc_1:

	.align    16,0x90

__spow_la_TestIntFunc:


	.cfi_startproc
..L53:

        movd      %xmm0, %edx
        andl      $2147483647, %edx
        lea       -1065353216(%rdx), %eax
        cmpl      $1073741824, %eax
        jae       .LBL_2_5


        cmpl      $1266679808, %edx
        jge       .LBL_2_7


        movl      %edx, %ecx
        andl      $-8388608, %ecx
        addl      $8388608, %ecx
        shrl      $23, %ecx
        shll      %cl, %edx
        testl     $8388607, %edx
        jne       .LBL_2_5


        andl      $16777215, %edx
        xorl      %eax, %eax
        cmpl      $8388608, %edx
        setne     %al
        incl      %eax
        ret

.LBL_2_5:

        xorl      %eax, %eax
        ret

.LBL_2_7:

        movl      $2, %eax
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__spow_la_TestIntFunc,@function
	.size	__spow_la_TestIntFunc,.-__spow_la_TestIntFunc
..LN__spow_la_TestIntFunc.1:

.L_2__routine_start___svml_spow_cout_rare_internal_2:

	.align    16,0x90

__svml_spow_cout_rare_internal:


	.cfi_startproc
..L56:

        pushq     %r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rbx
	.cfi_def_cfa_offset 48
	.cfi_offset 3, -48
        pushq     %rbp
	.cfi_def_cfa_offset 56
	.cfi_offset 6, -56
        subq      $88, %rsp
	.cfi_def_cfa_offset 144
        movq      %rdx, %r15
        movss     (%rdi), %xmm4
        pxor      %xmm1, %xmm1
        movss     (%rsi), %xmm3
        movaps    %xmm4, %xmm2
        movl      $0, 64(%rsp)
        movaps    %xmm3, %xmm0
        ucomiss   %xmm1, %xmm4
        jp        .LBL_3_2
        je        .LBL_3_3

.LBL_3_2:

        movss     %xmm4, 8(%rsp)
        jmp       .LBL_3_4

.LBL_3_3:

        movaps    %xmm4, %xmm2
        addss     %xmm4, %xmm2
        movss     %xmm2, 8(%rsp)

.LBL_3_4:

        pxor      %xmm1, %xmm1
        ucomiss   %xmm1, %xmm3
        jp        .LBL_3_5
        je        .LBL_3_6

.LBL_3_5:

        movss     %xmm3, 28(%rsp)
        jmp       .LBL_3_7

.LBL_3_6:

        movaps    %xmm3, %xmm0
        addss     %xmm3, %xmm0
        movss     %xmm0, 28(%rsp)

.LBL_3_7:

        movb      31(%rsp), %al
        xorl      %ebx, %ebx
        andb      $-128, %al
        xorl      %ebp, %ebp
        shrb      $7, %al
        testl     $8388607, 8(%rsp)
        movzwl    30(%rsp), %r13d
        sete      %bl
        andl      $32640, %r13d
        testl     $8388607, 28(%rsp)
        movzwl    10(%rsp), %r14d
        movzbl    11(%rsp), %r12d
        sete      %bpl
        andl      $32640, %r14d
        andl      $128, %r12d
        shrl      $7, %r13d
        shrl      $7, %r14d
        shrl      $7, %r12d
        movb      %al, 72(%rsp)
        cmpl      $255, %r13d
        je        .LBL_3_59


        movl      $1, 8(%rsp)
        movss     %xmm2, 32(%rsp)
        movss     %xmm0, 40(%rsp)
        movss     %xmm3, 48(%rsp)
        movss     %xmm4, 56(%rsp)

        call      __spow_la_TestIntFunc


        movss     56(%rsp), %xmm4
        movl      %eax, %edx
        movss     48(%rsp), %xmm3
        movss     40(%rsp), %xmm0
        movss     32(%rsp), %xmm2
        testl     %r12d, %r12d
        jne       .LBL_3_10


        cmpl      $127, %r14d
        je        .LBL_3_56

.LBL_3_10:

        testl     %r13d, %r13d
        jne       .LBL_3_12

.LBL_3_11:

        testl     %ebp, %ebp
        jne       .LBL_3_38

.LBL_3_12:

        cmpl      $255, %r14d
        je        .LBL_3_14

.LBL_3_13:

        movl      $1, %eax
        jmp       .LBL_3_15

.LBL_3_14:

        xorl      %eax, %eax

.LBL_3_15:

        orl       %eax, %ebx
        je        .LBL_3_37


        orl       8(%rsp), %ebp
        je        .LBL_3_37


        pxor      %xmm1, %xmm1
        ucomiss   %xmm1, %xmm2
        jp        .LBL_3_18
        je        .LBL_3_53

.LBL_3_18:

        ucomiss   .L_2il0floatpacket.121(%rip), %xmm2
        jp        .LBL_3_19
        je        .LBL_3_50

.LBL_3_19:

        testl     %eax, %eax
        je        .LBL_3_30


        cmpl      $0, 8(%rsp)
        je        .LBL_3_30


        pxor      %xmm0, %xmm0
        comiss    %xmm0, %xmm2
        ja        .LBL_3_23


        testl     %edx, %edx
        je        .LBL_3_29

.LBL_3_23:

        lea       1484+__spow_la_CoutTab(%rip), %rax
        andl      %r12d, %edx
        movss     %xmm2, 12(%rsp)
        xorl      %ecx, %ecx
        andb      $127, 15(%rsp)
        movss     (%rax,%rdx,4), %xmm12
        testl     %r14d, %r14d
        jne       .LBL_3_25


        movss     12(%rsp), %xmm0
        movl      $-64, %ecx
        mulss     .L_2il0floatpacket.122(%rip), %xmm0
        movss     %xmm0, 12(%rsp)
        jmp       .LBL_3_26

.LBL_3_25:

        movss     12(%rsp), %xmm0

.LBL_3_26:

        movzwl    14(%rsp), %edi
        lea       __spow_la_CoutTab(%rip), %rsi
        andl      $32640, %edi
        pxor      %xmm1, %xmm1
        shrl      $7, %edi
        movl      12(%rsp), %edx
        shll      $23, %edi
        andl      $8388607, %edx
        movss     %xmm0, 16(%rsp)
        orl       %edx, %edi
        movzwl    18(%rsp), %eax
        addl      $-1060634624, %edi
        andl      $-32641, %eax
        addl      $16256, %eax
        movw      %ax, 18(%rsp)
        sarl      $23, %edi
        addl      %ecx, %edi
        movl      16(%rsp), %ecx
        andl      $7864320, %ecx
        addl      $524288, %ecx
        cvtsi2ss  %edi, %xmm1
        shrl      $20, %ecx
        movss     (%rsi,%rcx,4), %xmm4
        addl      %ecx, %ecx
        movss     36(%rsi,%rcx,4), %xmm13
        movaps    %xmm13, %xmm7
        movss     16(%rsp), %xmm0
        addss     %xmm1, %xmm7
        movaps    %xmm0, %xmm6
        mulss     %xmm4, %xmm6
        movss     %xmm7, 20(%rsp)
        movss     20(%rsp), %xmm3
        movd      %xmm6, %ebx
        subss     %xmm3, %xmm1
        andl      $1966080, %ebx
        addss     %xmm1, %xmm13
        addl      $131072, %ebx
        shrl      $18, %ebx
        movss     108(%rsi,%rbx,4), %xmm11
        addl      %ebx, %ebx
        movss     144(%rsi,%rbx,4), %xmm5
        movss     %xmm13, 24(%rsp)
        movaps    %xmm5, %xmm14
        movss     20(%rsp), %xmm1
        addss     %xmm1, %xmm14
        mulss     %xmm11, %xmm6
        mulss     %xmm11, %xmm4
        movd      %xmm6, %ebp
        movss     24(%rsp), %xmm7
        movss     %xmm14, 20(%rsp)
        movss     20(%rsp), %xmm3
        andl      $507904, %ebp
        addl      $16384, %ebp
        subss     %xmm3, %xmm1
        shrl      $15, %ebp
        addss     %xmm1, %xmm5
        movss     216(%rsi,%rbp,4), %xmm15
        addl      %ebp, %ebp
        movss     284(%rsi,%rbp,4), %xmm2
        movss     %xmm5, 24(%rsp)
        movaps    %xmm2, %xmm13
        movss     20(%rsp), %xmm1
        movss     24(%rsp), %xmm5
        addss     %xmm1, %xmm13
        mulss     %xmm15, %xmm4
        movss     %xmm13, 20(%rsp)
        movss     20(%rsp), %xmm3
        movss     .L_2il0floatpacket.124(%rip), %xmm11
        subss     %xmm3, %xmm1
        addss     %xmm1, %xmm2
        mulss     %xmm15, %xmm6
        movaps    %xmm11, %xmm15
        movaps    %xmm6, %xmm3
        mulss     %xmm0, %xmm15
        subss     .L_2il0floatpacket.123(%rip), %xmm3
        movss     %xmm2, 24(%rsp)
        movss     20(%rsp), %xmm1
        movss     24(%rsp), %xmm2
        movss     %xmm15, 20(%rsp)
        movss     20(%rsp), %xmm13
        movss     40(%rsi,%rcx,4), %xmm9
        movss     148(%rsi,%rbx,4), %xmm8
        movss     288(%rsi,%rbp,4), %xmm10
        subss     16(%rsp), %xmm13
        movss     %xmm13, 24(%rsp)
        movss     20(%rsp), %xmm13
        movss     24(%rsp), %xmm14
        subss     %xmm14, %xmm13
        movss     %xmm13, 20(%rsp)
        movss     20(%rsp), %xmm15
        subss     %xmm15, %xmm0
        movss     %xmm0, 24(%rsp)
        movaps    %xmm4, %xmm0
        mulss     %xmm11, %xmm0
        movss     20(%rsp), %xmm13
        movss     24(%rsp), %xmm14
        movss     %xmm0, 20(%rsp)
        movss     20(%rsp), %xmm15
        subss     %xmm4, %xmm15
        movss     %xmm15, 24(%rsp)
        movss     20(%rsp), %xmm15
        movss     24(%rsp), %xmm0
        subss     %xmm0, %xmm15
        movss     %xmm15, 20(%rsp)
        movss     20(%rsp), %xmm0
        subss     %xmm0, %xmm4
        movaps    %xmm13, %xmm0
        movss     %xmm4, 24(%rsp)
        movss     20(%rsp), %xmm4
        mulss     %xmm4, %xmm0
        mulss     %xmm14, %xmm4
        subss     %xmm6, %xmm0
        movaps    %xmm3, %xmm6
        addss     %xmm4, %xmm0
        addss     %xmm1, %xmm6
        movss     24(%rsp), %xmm15
        movss     %xmm6, 20(%rsp)
        movss     20(%rsp), %xmm4
        mulss     %xmm15, %xmm13
        subss     %xmm4, %xmm1
        mulss     %xmm15, %xmm14
        addss     %xmm13, %xmm0
        addss     %xmm3, %xmm1
        addss     %xmm14, %xmm0
        movss     %xmm1, 24(%rsp)
        movss     20(%rsp), %xmm6
        movss     %xmm6, 8(%rsp)
        movzwl    10(%rsp), %eax
        andl      $32640, %eax
        shrl      $7, %eax
        addl      %r13d, %eax
        movss     24(%rsp), %xmm4
        cmpl      $265, %eax
        jge       .LBL_3_49


        cmpl      $192, %eax
        jg        .LBL_3_40


        movl      $1065353216, 20(%rsp)
        movss     20(%rsp), %xmm0
        addss     .L_2il0floatpacket.133(%rip), %xmm0
        movss     %xmm0, 20(%rsp)
        movss     20(%rsp), %xmm1
        mulss     %xmm12, %xmm1
        movss     %xmm1, (%r15)
        jmp       .LBL_3_39

.LBL_3_29:

        movl      $1, 64(%rsp)
        pxor      %xmm0, %xmm0
        movss     %xmm0, 20(%rsp)
        movss     20(%rsp), %xmm2
        movss     20(%rsp), %xmm1
        divss     %xmm1, %xmm2
        movss     %xmm2, 20(%rsp)
        movl      20(%rsp), %eax
        movl      %eax, (%r15)
        jmp       .LBL_3_39

.LBL_3_30:

        cmpl      $127, %r14d
        jge       .LBL_3_34


        movb      72(%rsp), %al
        testb     %al, %al
        je        .LBL_3_33


        mulss     %xmm0, %xmm0
        movss     %xmm0, (%r15)
        jmp       .LBL_3_39

.LBL_3_33:

        pxor      %xmm0, %xmm0
        movss     %xmm0, (%r15)
        jmp       .LBL_3_39

.LBL_3_34:

        movb      72(%rsp), %al
        testb     %al, %al
        je        .LBL_3_36


        lea       1512+__spow_la_CoutTab(%rip), %rax
        andl      %r12d, %edx
        movl      (%rax,%rdx,4), %ecx
        movl      %ecx, 12(%rsp)
        movl      %ecx, (%r15)
        jmp       .LBL_3_39

.LBL_3_36:

        mulss     %xmm2, %xmm2
        lea       1484+__spow_la_CoutTab(%rip), %rax
        mulss     %xmm0, %xmm2
        andl      %r12d, %edx
        mulss     (%rax,%rdx,4), %xmm2
        movss     %xmm2, (%r15)
        jmp       .LBL_3_39

.LBL_3_37:

        addss     %xmm3, %xmm4
        movss     %xmm4, (%r15)
        jmp       .LBL_3_39

.LBL_3_38:

        addss     %xmm0, %xmm2
        movss     %xmm2, 20(%rsp)
        movl      $1065353216, 24(%rsp)
        movb      23(%rsp), %al
        movb      27(%rsp), %dl
        andb      $-128, %al
        andb      $127, %dl
        orb       %al, %dl
        movb      %dl, 27(%rsp)
        movss     24(%rsp), %xmm1
        movss     24(%rsp), %xmm0
        mulss     %xmm0, %xmm1
        movss     %xmm1, (%r15)

.LBL_3_39:

        movl      64(%rsp), %eax
        addq      $88, %rsp
	.cfi_def_cfa_offset 56
	.cfi_restore 6
        popq      %rbp
	.cfi_def_cfa_offset 48
	.cfi_restore 3
        popq      %rbx
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12
	.cfi_def_cfa_offset 8
        ret
	.cfi_def_cfa_offset 144
	.cfi_offset 3, -48
	.cfi_offset 6, -56
	.cfi_offset 12, -16
	.cfi_offset 13, -24
	.cfi_offset 14, -32
	.cfi_offset 15, -40

.LBL_3_40:

        movss     .L_2il0floatpacket.128(%rip), %xmm1
        lea       424+__spow_la_CoutTab(%rip), %rdx
        addss     %xmm0, %xmm3
        addss     %xmm5, %xmm7
        addss     %xmm4, %xmm0
        mulss     %xmm3, %xmm1
        addss     %xmm7, %xmm2
        addss     .L_2il0floatpacket.127(%rip), %xmm1
        addss     %xmm2, %xmm9
        mulss     %xmm3, %xmm1
        addss     %xmm9, %xmm8
        addss     .L_2il0floatpacket.126(%rip), %xmm1
        addss     %xmm8, %xmm10
        mulss     %xmm3, %xmm1
        addss     %xmm0, %xmm10
        addss     .L_2il0floatpacket.125(%rip), %xmm1
        mulss     %xmm3, %xmm1
        movaps    %xmm10, %xmm5
        movss     8(%rsp), %xmm4
        movaps    %xmm11, %xmm8
        addss     %xmm1, %xmm6
        lea       20(%rsp), %rax
        movss     %xmm6, (%rax)
        movss     (%rax), %xmm0
        subss     %xmm0, %xmm4
        movaps    %xmm11, %xmm0
        addss     %xmm1, %xmm4
        lea       24(%rsp), %rcx
        movss     %xmm4, (%rcx)
        movss     (%rax), %xmm7
        movss     (%rcx), %xmm3
        addss     %xmm7, %xmm5
        movss     %xmm5, (%rax)
        movss     (%rax), %xmm6
        subss     %xmm6, %xmm7
        addss     %xmm10, %xmm7
        movss     %xmm7, (%rcx)
        movss     (%rax), %xmm10
        mulss     %xmm10, %xmm8
        movss     (%rcx), %xmm2
        movss     %xmm8, (%rax)
        addss     %xmm3, %xmm2
        movss     (%rax), %xmm9
        lea       28(%rsp), %rbx
        movss     (%rbx), %xmm7
        subss     %xmm10, %xmm9
        mulss     %xmm7, %xmm0
        mulss     (%rbx), %xmm2
        movss     %xmm9, (%rcx)
        movss     (%rax), %xmm14
        movss     (%rcx), %xmm13
        movss     .L_2il0floatpacket.129(%rip), %xmm9
        subss     %xmm13, %xmm14
        movss     %xmm14, (%rax)
        movss     (%rax), %xmm15
        subss     %xmm15, %xmm10
        movss     %xmm10, (%rcx)
        movss     (%rax), %xmm8
        movss     (%rcx), %xmm10
        movss     %xmm0, (%rax)
        movss     (%rax), %xmm1
        subss     (%rbx), %xmm1
        movss     %xmm1, (%rcx)
        movss     (%rax), %xmm5
        movss     (%rcx), %xmm4
        subss     %xmm4, %xmm5
        movss     %xmm5, (%rax)
        movss     (%rax), %xmm6
        subss     %xmm6, %xmm7
        movss     %xmm7, (%rcx)
        movss     (%rax), %xmm1
        movss     (%rcx), %xmm15
        movaps    %xmm1, %xmm0
        mulss     %xmm8, %xmm0
        mulss     %xmm10, %xmm1
        mulss     %xmm15, %xmm8
        mulss     %xmm15, %xmm10
        addss     %xmm8, %xmm1
        movaps    %xmm0, %xmm8
        addss     %xmm1, %xmm8
        movaps    %xmm8, %xmm3
        subss     %xmm8, %xmm0
        addss     %xmm9, %xmm3
        addss     %xmm0, %xmm1
        movss     %xmm3, (%rax)
        movaps    %xmm8, %xmm3
        movss     (%rax), %xmm13
        addss     %xmm1, %xmm10
        subss     %xmm9, %xmm13
        addss     %xmm2, %xmm10
        movss     %xmm13, (%rsp)
        movss     (%rsp), %xmm14
        movss     .L_2il0floatpacket.132(%rip), %xmm2
        subss     %xmm14, %xmm3
        movss     %xmm3, 4(%rsp)
        movss     4(%rsp), %xmm4
        movl      (%rax), %eax
        movl      %eax, %ebx
        andl      $127, %eax
        addss     %xmm10, %xmm4
        mulss     %xmm4, %xmm2
        addl      %eax, %eax
        movss     -4(%rdx,%rax,4), %xmm1
        shll      $10, %ebx
        addss     .L_2il0floatpacket.131(%rip), %xmm2
        mulss     %xmm4, %xmm2
        sarl      $17, %ebx
        addss     .L_2il0floatpacket.130(%rip), %xmm2
        mulss     %xmm4, %xmm2
        mulss     %xmm1, %xmm2
        addss     (%rdx,%rax,4), %xmm2
        movaps    %xmm2, %xmm5
        addss     %xmm1, %xmm5
        movss     %xmm5, 12(%rsp)
        movzwl    14(%rsp), %edx
        movl      %edx, %ecx
        andl      $32640, %ecx
        shrl      $7, %ecx
        lea       -127(%rbx,%rcx), %eax
        cmpl      $128, %eax
        jge       .LBL_3_48


        cmpl      $-126, %eax
        jl        .LBL_3_43


        andl      $-32641, %edx
        lea       127(%rax), %eax
        movzbl    %al, %eax
        shll      $7, %eax
        orl       %eax, %edx
        movw      %dx, 14(%rsp)
        movss     12(%rsp), %xmm0
        mulss     %xmm12, %xmm0
        movss     %xmm0, 12(%rsp)
        movss     %xmm0, (%r15)
        jmp       .LBL_3_39

.LBL_3_43:

        cmpl      $-136, %eax
        jl        .LBL_3_45


        lea       20(%rsp), %rdx
        movss     %xmm5, (%rdx)
        movl      $1065353216, %eax
        movss     (%rdx), %xmm0
        addl      $191, %ebx
        movl      %eax, 8(%rsp)
        subss     %xmm0, %xmm1
        shrl      $16, %eax
        addss     %xmm1, %xmm2
        movss     %xmm2, 24(%rsp)
        movss     (%rdx), %xmm5
        mulss     %xmm5, %xmm11
        movss     24(%rsp), %xmm6
        movss     %xmm11, (%rdx)
        movss     (%rdx), %xmm1
        movzwl    %ax, %edx
        subss     %xmm5, %xmm1
        lea       24(%rsp), %rax
        movss     %xmm1, (%rax)
        andl      $-32641, %edx
        lea       20(%rsp), %rcx
        movss     (%rcx), %xmm3
        movss     (%rax), %xmm2
        movzbl    %bl, %ebx
        subss     %xmm2, %xmm3
        movss     %xmm3, (%rcx)
        movss     (%rcx), %xmm4
        shll      $7, %ebx
        subss     %xmm4, %xmm5
        movss     %xmm5, (%rax)
        orl       %ebx, %edx
        movss     (%rcx), %xmm8
        movss     (%rax), %xmm14
        movw      %dx, 10(%rsp)
        addss     %xmm6, %xmm14
        movss     8(%rsp), %xmm7
        mulss     %xmm7, %xmm14
        mulss     %xmm7, %xmm8
        lea       20(%rsp), %rdx
        movl      $8388608, (%rdx)
        addss     %xmm8, %xmm14
        movss     (%rdx), %xmm10
        movss     (%rdx), %xmm9
        mulss     %xmm9, %xmm10
        mulss     .L_2il0floatpacket.135(%rip), %xmm14
        movss     %xmm10, (%rdx)
        movss     (%rdx), %xmm13
        addss     %xmm13, %xmm14
        mulss     %xmm14, %xmm12
        movss     %xmm14, 12(%rsp)
        movss     %xmm12, (%r15)
        jmp       .LBL_3_39

.LBL_3_45:

        cmpl      $-159, %eax
        jl        .LBL_3_47


        movl      $1065353216, %eax
        addl      $191, %ebx
        movl      %eax, 8(%rsp)
        shrl      $16, %eax
        movzwl    %ax, %edx
        movzbl    %bl, %ebx
        andl      $-32641, %edx
        shll      $7, %ebx
        orl       %ebx, %edx
        movw      %dx, 10(%rsp)
        movss     8(%rsp), %xmm0
        movss     .L_2il0floatpacket.135(%rip), %xmm1
        mulss     %xmm0, %xmm5
        mulss     %xmm1, %xmm12
        lea       20(%rsp), %rdx
        movl      $8388608, (%rdx)
        movss     (%rdx), %xmm3
        movss     (%rdx), %xmm2
        mulss     %xmm2, %xmm3
        mulss     %xmm12, %xmm5
        movss     %xmm3, (%rdx)
        movss     (%rdx), %xmm4
        subss     %xmm4, %xmm5
        movss     %xmm5, 12(%rsp)
        movss     %xmm5, (%r15)
        jmp       .LBL_3_39

.LBL_3_47:

        lea       20(%rsp), %rax
        movl      $8388608, (%rax)
        movss     (%rax), %xmm1
        movss     (%rax), %xmm0
        mulss     %xmm0, %xmm1
        movss     %xmm1, (%rax)
        movss     (%rax), %xmm2
        mulss     %xmm2, %xmm12
        movss     %xmm12, 12(%rsp)
        movss     %xmm12, (%r15)
        jmp       .LBL_3_39

.LBL_3_48:

        lea       20(%rsp), %rax
        movl      $2130706432, (%rax)
        movss     (%rax), %xmm1
        movss     (%rax), %xmm0
        mulss     %xmm0, %xmm1
        movss     %xmm1, (%rax)
        movss     (%rax), %xmm2
        mulss     %xmm2, %xmm12
        movss     %xmm12, 12(%rsp)
        movss     %xmm12, (%r15)
        jmp       .LBL_3_39

.LBL_3_49:

        movb      11(%rsp), %al
        lea       1472+__spow_la_CoutTab(%rip), %rcx
        andb      $-128, %al
        movb      72(%rsp), %dl
        shrb      $7, %al
        xorb      %al, %dl
        movzbl    %dl, %ebx
        movss     (%rcx,%rbx,4), %xmm0
        mulss     %xmm0, %xmm0
        mulss     %xmm12, %xmm0
        movss     %xmm0, (%r15)
        jmp       .LBL_3_39

.LBL_3_50:

        testl     %edx, %edx
        jne       .LBL_3_52


        cmpl      $0, 8(%rsp)
        jne       .LBL_3_19

.LBL_3_52:

        lea       1484+__spow_la_CoutTab(%rip), %rax
        andl      $1, %edx
        movl      (%rax,%rdx,4), %ecx
        movl      %ecx, (%r15)
        jmp       .LBL_3_39

.LBL_3_53:

        movb      72(%rsp), %al
        mulss     %xmm2, %xmm2
        testb     %al, %al
        je        .LBL_3_55


        lea       1484+__spow_la_CoutTab(%rip), %rax
        andl      %r12d, %edx
        movl      $1, 64(%rsp)
        movss     (%rax,%rdx,4), %xmm0
        divss     %xmm2, %xmm0
        movss     %xmm0, (%r15)
        jmp       .LBL_3_39

.LBL_3_55:

        lea       1484+__spow_la_CoutTab(%rip), %rax
        andl      %r12d, %edx
        movss     (%rax,%rdx,4), %xmm0
        mulss     %xmm2, %xmm0
        movss     %xmm0, (%r15)
        jmp       .LBL_3_39

.LBL_3_56:

        testl     %ebx, %ebx
        jne       .LBL_3_38


        testl     %r13d, %r13d
        jne       .LBL_3_13
        jmp       .LBL_3_11

.LBL_3_59:

        movl      $0, 8(%rsp)
        movss     %xmm2, 32(%rsp)
        movss     %xmm0, 40(%rsp)
        movss     %xmm3, 48(%rsp)
        movss     %xmm4, 56(%rsp)

        call      __spow_la_TestIntFunc


        movss     56(%rsp), %xmm4
        movl      %eax, %edx
        movss     48(%rsp), %xmm3
        movss     40(%rsp), %xmm0
        movss     32(%rsp), %xmm2
        testl     %r12d, %r12d
        jne       .LBL_3_12


        cmpl      $127, %r14d
        jne       .LBL_3_12


        testl     %ebx, %ebx
        je        .LBL_3_13
        jmp       .LBL_3_38
	.align    16,0x90

	.cfi_endproc

	.type	__svml_spow_cout_rare_internal,@function
	.size	__svml_spow_cout_rare_internal,.-__svml_spow_cout_rare_internal
..LN__svml_spow_cout_rare_internal.2:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_spow_data_internal_avx512:
	.long	0
	.long	3174420480
	.long	3182632960
	.long	3187958784
	.long	3190685696
	.long	3193338880
	.long	3195920384
	.long	3197247488
	.long	3198473216
	.long	3199668736
	.long	3200835072
	.long	3201974272
	.long	3203086848
	.long	3204174848
	.long	3204843520
	.long	3205364224
	.long	1054113792
	.long	1053115392
	.long	1052137472
	.long	1051179008
	.long	1050238976
	.long	1049316864
	.long	1048248320
	.long	1046471680
	.long	1044726784
	.long	1043013632
	.long	1041329152
	.long	1039161344
	.long	1035907072
	.long	1032706048
	.long	1027317760
	.long	1018830848
	.long	0
	.long	3067311503
	.long	890262383
	.long	916311190
	.long	3058814943
	.long	914835756
	.long	3056977939
	.long	3052757441
	.long	905348701
	.long	921801496
	.long	900652061
	.long	916473404
	.long	3063873943
	.long	3048020321
	.long	3055557319
	.long	921573027
	.long	3050426335
	.long	918574590
	.long	913737309
	.long	3045697063
	.long	3029223305
	.long	866568163
	.long	3063765991
	.long	3057827840
	.long	910185982
	.long	3062847489
	.long	917965485
	.long	903301016
	.long	882039287
	.long	910858241
	.long	3059117133
	.long	3029061382
	.long	1065353216
	.long	1065724611
	.long	1066112450
	.long	1066517459
	.long	1066940400
	.long	1067382066
	.long	1067843287
	.long	1068324927
	.long	1068827891
	.long	1069353124
	.long	1069901610
	.long	1070474380
	.long	1071072509
	.long	1071697119
	.long	1072349383
	.long	1073030525
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
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
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	1049872133
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	3199775725
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	1056323663
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	3208161851
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
	.long	849703116
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
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
	.long	1228933104
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
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1030247627
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1047916908
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	1060205090
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	60
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	1123745792
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.type	__svml_spow_data_internal_avx512,@object
	.size	__svml_spow_data_internal_avx512,1536
	.align 32
__spow_la_CoutTab:
	.long	1065353216
	.long	1063518208
	.long	1061945344
	.long	1060765696
	.long	1059717120
	.long	1058930688
	.long	1058144256
	.long	1057488896
	.long	1056964608
	.long	0
	.long	0
	.long	1043013120
	.long	912347133
	.long	1051178752
	.long	920491638
	.long	1055650560
	.long	910207939
	.long	3201407744
	.long	3065009147
	.long	3197864192
	.long	3046757530
	.long	3192020992
	.long	3064938684
	.long	3182631936
	.long	3069048210
	.long	0
	.long	0
	.long	1065353216
	.long	1064828928
	.long	1064304640
	.long	1066008576
	.long	1065877504
	.long	1065746432
	.long	1065615360
	.long	1065484288
	.long	1065353216
	.long	0
	.long	0
	.long	1027315712
	.long	915720665
	.long	1035907072
	.long	882039287
	.long	3185459200
	.long	3062137179
	.long	3182631936
	.long	3069048210
	.long	3179763712
	.long	3059908234
	.long	3174420480
	.long	3067311503
	.long	3166121984
	.long	3066774241
	.long	0
	.long	0
	.long	1069056000
	.long	1069006848
	.long	1068957696
	.long	1068908544
	.long	1068875776
	.long	1069318144
	.long	1069301760
	.long	1069268992
	.long	1069252608
	.long	1069219840
	.long	1069203456
	.long	1069170688
	.long	1069154304
	.long	1069121536
	.long	1069105152
	.long	1069072384
	.long	1069056000
	.long	0
	.long	0
	.long	1002471424
	.long	906080490
	.long	1010884608
	.long	916991201
	.long	1016135680
	.long	905081895
	.long	1018261504
	.long	918286540
	.long	3170725888
	.long	3067774196
	.long	3169697792
	.long	3068476631
	.long	3167637504
	.long	3069858259
	.long	3166609408
	.long	3020376912
	.long	3164540928
	.long	3039629966
	.long	3163504640
	.long	3043319364
	.long	3160350720
	.long	3046704089
	.long	3158269952
	.long	3047249178
	.long	3154083840
	.long	3046609959
	.long	3149905920
	.long	3045301896
	.long	3137339392
	.long	3034784887
	.long	0
	.long	0
	.long	1065353216
	.long	0
	.long	1065398766
	.long	3014665830
	.long	1065444562
	.long	2982428624
	.long	1065490608
	.long	3015478276
	.long	1065536902
	.long	865643564
	.long	1065583450
	.long	3014725705
	.long	1065630248
	.long	868461790
	.long	1065677302
	.long	858550072
	.long	1065724612
	.long	3013096376
	.long	1065772178
	.long	3013897664
	.long	1065820002
	.long	3008545510
	.long	1065868086
	.long	3011512679
	.long	1065916430
	.long	864064219
	.long	1065965038
	.long	819487640
	.long	1066013910
	.long	3012212369
	.long	1066063046
	.long	856316133
	.long	1066112450
	.long	2992679841
	.long	1066162122
	.long	3001970243
	.long	1066212064
	.long	3013902756
	.long	1066262276
	.long	847285146
	.long	1066312762
	.long	3010032741
	.long	1066363522
	.long	3018332471
	.long	1066414556
	.long	856041677
	.long	1066465868
	.long	864808677
	.long	1066517460
	.long	3012318446
	.long	1066569330
	.long	863709796
	.long	1066621484
	.long	3016813593
	.long	1066673920
	.long	3018872036
	.long	1066726640
	.long	3006136850
	.long	1066779646
	.long	864474828
	.long	1066832942
	.long	3016286184
	.long	1066886526
	.long	3015052933
	.long	1066940400
	.long	857938801
	.long	1066994568
	.long	2993474036
	.long	1067049030
	.long	3009003152
	.long	1067103786
	.long	872191232
	.long	1067158842
	.long	3003929955
	.long	1067214196
	.long	3013071165
	.long	1067269850
	.long	3006375425
	.long	1067325806
	.long	843377209
	.long	1067382066
	.long	859906882
	.long	1067438632
	.long	848662531
	.long	1067495506
	.long	3018868367
	.long	1067552686
	.long	868910405
	.long	1067610180
	.long	3019699127
	.long	1067667984
	.long	3013023741
	.long	1067726102
	.long	3005475891
	.long	1067784536
	.long	3010626242
	.long	1067843286
	.long	866758993
	.long	1067902356
	.long	869265128
	.long	1067961748
	.long	3004575030
	.long	1068021462
	.long	3018425550
	.long	1068081498
	.long	867494524
	.long	1068141862
	.long	858118433
	.long	1068202554
	.long	3004476802
	.long	1068263574
	.long	866434624
	.long	1068324926
	.long	870990497
	.long	1068386612
	.long	858100843
	.long	1068448632
	.long	867002634
	.long	1068510990
	.long	3000050815
	.long	1068573686
	.long	3011271336
	.long	1068636722
	.long	3006477262
	.long	1068700100
	.long	840255625
	.long	1068763822
	.long	866280780
	.long	1068827892
	.long	3016492578
	.long	1068892308
	.long	3006218836
	.long	1068957074
	.long	2993076596
	.long	1069022192
	.long	3000356208
	.long	1069087664
	.long	3015220484
	.long	1069153490
	.long	856315927
	.long	1069219674
	.long	867308350
	.long	1069286218
	.long	863888852
	.long	1069353124
	.long	3007401960
	.long	1069420392
	.long	832069785
	.long	1069488026
	.long	3004369690
	.long	1069556026
	.long	866250961
	.long	1069624396
	.long	868902513
	.long	1069693138
	.long	851736822
	.long	1069762252
	.long	869934231
	.long	1069831742
	.long	869028661
	.long	1069901610
	.long	839559223
	.long	1069971856
	.long	867543588
	.long	1070042484
	.long	868789178
	.long	1070113496
	.long	859381756
	.long	1070184894
	.long	3010667426
	.long	1070256678
	.long	859604257
	.long	1070328852
	.long	872346226
	.long	1070401420
	.long	3010682756
	.long	1070474380
	.long	841546788
	.long	1070547736
	.long	869210393
	.long	1070621492
	.long	2996061011
	.long	1070695648
	.long	3013455510
	.long	1070770206
	.long	3009158570
	.long	1070845168
	.long	865699227
	.long	1070920538
	.long	866897902
	.long	1070996318
	.long	2955948569
	.long	1071072508
	.long	868931229
	.long	1071149114
	.long	3014890061
	.long	1071226134
	.long	3002473793
	.long	1071303572
	.long	861820308
	.long	1071381432
	.long	3008383516
	.long	1071459714
	.long	3010850715
	.long	1071538420
	.long	864181775
	.long	1071617554
	.long	870234352
	.long	1071697118
	.long	871115413
	.long	1071777114
	.long	872414852
	.long	1071857546
	.long	3012378998
	.long	1071938412
	.long	866137918
	.long	1072019718
	.long	870808707
	.long	1072101466
	.long	866840096
	.long	1072183658
	.long	857766040
	.long	1072266296
	.long	855693471
	.long	1072349382
	.long	870833444
	.long	1072432920
	.long	867585053
	.long	1072516912
	.long	846646433
	.long	1072601360
	.long	3008357562
	.long	1072686266
	.long	3007858250
	.long	1072771632
	.long	866626825
	.long	1072857464
	.long	3015943680
	.long	1072943760
	.long	2995197552
	.long	1073030526
	.long	3018513273
	.long	1073117762
	.long	3012791488
	.long	1073205472
	.long	3012359471
	.long	1073293658
	.long	3003728983
	.long	1073382322
	.long	870019626
	.long	1073471470
	.long	3012762127
	.long	1073561100
	.long	835668076
	.long	1073651218
	.long	3013837936
	.long	980050793
	.long	3199320925
	.long	1042575209
	.long	3182108321
	.long	1060205080
	.long	1047920112
	.long	1029920839
	.long	2130706432
	.long	8388608
	.long	0
	.long	1065353216
	.long	3212836864
	.long	1203765248
	.long	1069056000
	.long	1166018560
	.long	1602224128
	.long	528482304
	.long	0
	.long	2147483648
	.type	__spow_la_CoutTab,@object
	.size	__spow_la_CoutTab,1520
	.align 4
.L_2il0floatpacket.121:
	.long	0xbf800000
	.type	.L_2il0floatpacket.121,@object
	.size	.L_2il0floatpacket.121,4
	.align 4
.L_2il0floatpacket.122:
	.long	0x5f800000
	.type	.L_2il0floatpacket.122,@object
	.size	.L_2il0floatpacket.122,4
	.align 4
.L_2il0floatpacket.123:
	.long	0x3fb88000
	.type	.L_2il0floatpacket.123,@object
	.size	.L_2il0floatpacket.123,4
	.align 4
.L_2il0floatpacket.124:
	.long	0x45800800
	.type	.L_2il0floatpacket.124,@object
	.size	.L_2il0floatpacket.124,4
	.align 4
.L_2il0floatpacket.125:
	.long	0x3a6a6369
	.type	.L_2il0floatpacket.125,@object
	.size	.L_2il0floatpacket.125,4
	.align 4
.L_2il0floatpacket.126:
	.long	0xbeb1c35d
	.type	.L_2il0floatpacket.126,@object
	.size	.L_2il0floatpacket.126,4
	.align 4
.L_2il0floatpacket.127:
	.long	0x3e246f69
	.type	.L_2il0floatpacket.127,@object
	.size	.L_2il0floatpacket.127,4
	.align 4
.L_2il0floatpacket.128:
	.long	0xbdab1ea1
	.type	.L_2il0floatpacket.128,@object
	.size	.L_2il0floatpacket.128,4
	.align 4
.L_2il0floatpacket.129:
	.long	0x47c00000
	.type	.L_2il0floatpacket.129,@object
	.size	.L_2il0floatpacket.129,4
	.align 4
.L_2il0floatpacket.130:
	.long	0x3f317218
	.type	.L_2il0floatpacket.130,@object
	.size	.L_2il0floatpacket.130,4
	.align 4
.L_2il0floatpacket.131:
	.long	0x3e75fdf0
	.type	.L_2il0floatpacket.131,@object
	.size	.L_2il0floatpacket.131,4
	.align 4
.L_2il0floatpacket.132:
	.long	0x3d635847
	.type	.L_2il0floatpacket.132,@object
	.size	.L_2il0floatpacket.132,4
	.align 4
.L_2il0floatpacket.133:
	.long	0x00800000
	.type	.L_2il0floatpacket.133,@object
	.size	.L_2il0floatpacket.133,4
	.align 4
.L_2il0floatpacket.134:
	.long	0x7f000000
	.type	.L_2il0floatpacket.134,@object
	.size	.L_2il0floatpacket.134,4
	.align 4
.L_2il0floatpacket.135:
	.long	0x1f800000
	.type	.L_2il0floatpacket.135,@object
	.size	.L_2il0floatpacket.135,4
	.align 4
.L_2il0floatpacket.136:
	.long	0x3f800000
	.type	.L_2il0floatpacket.136,@object
	.size	.L_2il0floatpacket.136,4
