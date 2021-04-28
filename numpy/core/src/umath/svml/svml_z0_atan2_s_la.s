/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *      For    0.0    <= x <=  7.0/16.0: atan(x) = atan(0.0) + atan(s), where s=(x-0.0)/(1.0+0.0*x)
 *      For  7.0/16.0 <= x <= 11.0/16.0: atan(x) = atan(0.5) + atan(s), where s=(x-0.5)/(1.0+0.5*x)
 *      For 11.0/16.0 <= x <= 19.0/16.0: atan(x) = atan(1.0) + atan(s), where s=(x-1.0)/(1.0+1.0*x)
 *      For 19.0/16.0 <= x <= 39.0/16.0: atan(x) = atan(1.5) + atan(s), where s=(x-1.5)/(1.0+1.5*x)
 *      For 39.0/16.0 <= x <=    inf   : atan(x) = atan(inf) + atan(s), where s=-1.0/x
 *      Where atan(s) ~= s+s^3*Poly11(s^2) on interval |s|<7.0/0.16.
 * --
 * 
 */


	.text
.L_2__routine_start___svml_atan2f16_z0_0:

	.align    16,0x90
	.globl __svml_atan2f16

__svml_atan2f16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $256, %rsp
        xorl      %edx, %edx


        vmovups   256+__svml_satan2_data_internal(%rip), %zmm6
        vmovups   64+__svml_satan2_data_internal(%rip), %zmm3

/* Testing on working interval. */
        vmovups   1024+__svml_satan2_data_internal(%rip), %zmm9
        vmovups   1088+__svml_satan2_data_internal(%rip), %zmm14

/*
 * 1) If y<x then a= y, b=x, PIO2=0
 * 2) If y>x then a=-x, b=y, PIO2=Pi/2
 */
        vmovups   320+__svml_satan2_data_internal(%rip), %zmm4
        vpternlogd $255, %zmm13, %zmm13, %zmm13
        vmovaps   %zmm1, %zmm8
        vandps    %zmm6, %zmm8, %zmm2
        vandps    %zmm6, %zmm0, %zmm1
        vorps     192+__svml_satan2_data_internal(%rip), %zmm2, %zmm5
        vpsubd    %zmm9, %zmm2, %zmm10
        vpsubd    %zmm9, %zmm1, %zmm12
        vxorps    %zmm2, %zmm8, %zmm7
        vxorps    %zmm1, %zmm0, %zmm6
        vcmpps    $17, {sae}, %zmm2, %zmm1, %k1
        vpcmpgtd  %zmm10, %zmm14, %k2
        vpcmpgtd  %zmm12, %zmm14, %k3
        vmovups   576+__svml_satan2_data_internal(%rip), %zmm14
        vblendmps %zmm1, %zmm5, %zmm11{%k1}
        vblendmps %zmm2, %zmm1, %zmm5{%k1}
        vxorps    %zmm4, %zmm4, %zmm4{%k1}

/*
 * Division a/b.
 * Enabled when FMA is available and
 * performance is better with NR iteration
 */
        vrcp14ps  %zmm5, %zmm15
        vfnmadd231ps {rn-sae}, %zmm5, %zmm15, %zmm3
        vfmadd213ps {rn-sae}, %zmm15, %zmm3, %zmm15
        vmulps    {rn-sae}, %zmm15, %zmm11, %zmm3
        vfnmadd231ps {rn-sae}, %zmm5, %zmm3, %zmm11
        vfmadd213ps {rn-sae}, %zmm3, %zmm11, %zmm15
        vmovups   448+__svml_satan2_data_internal(%rip), %zmm11
        vpternlogd $255, %zmm3, %zmm3, %zmm3

/* Polynomial. */
        vmulps    {rn-sae}, %zmm15, %zmm15, %zmm9
        vpandnd   %zmm10, %zmm10, %zmm13{%k2}
        vmulps    {rn-sae}, %zmm9, %zmm9, %zmm10
        vfmadd231ps {rn-sae}, %zmm10, %zmm11, %zmm14
        vmovups   640+__svml_satan2_data_internal(%rip), %zmm11
        vpandnd   %zmm12, %zmm12, %zmm3{%k3}
        vpord     %zmm3, %zmm13, %zmm3
        vmovups   704+__svml_satan2_data_internal(%rip), %zmm13
        vmovups   512+__svml_satan2_data_internal(%rip), %zmm12
        vptestmd  %zmm3, %zmm3, %k0
        vfmadd213ps {rn-sae}, %zmm13, %zmm10, %zmm14
        vfmadd231ps {rn-sae}, %zmm10, %zmm12, %zmm11
        vmovups   768+__svml_satan2_data_internal(%rip), %zmm12
        vmovups   832+__svml_satan2_data_internal(%rip), %zmm13

/* =========== Special branch for fast (vector) processing of zero arguments ================ */
        kortestw  %k0, %k0
        vfmadd213ps {rn-sae}, %zmm12, %zmm10, %zmm11
        vmovups   896+__svml_satan2_data_internal(%rip), %zmm12
        vfmadd213ps {rn-sae}, %zmm13, %zmm10, %zmm14
        vmovups   960+__svml_satan2_data_internal(%rip), %zmm13
        vfmadd213ps {rn-sae}, %zmm12, %zmm10, %zmm11
        vfmadd213ps {rn-sae}, %zmm13, %zmm10, %zmm14
        vfmadd213ps {rn-sae}, %zmm14, %zmm9, %zmm11

/* Reconstruction. */
        vfmadd213ps {rn-sae}, %zmm4, %zmm15, %zmm11

/* if x<0, sPI = Pi, else sPI =0 */
        vmovups   __svml_satan2_data_internal(%rip), %zmm15
        vorps     %zmm7, %zmm11, %zmm9
        vcmpps    $18, {sae}, %zmm15, %zmm8, %k1
        vmovups   384+__svml_satan2_data_internal(%rip), %zmm11
        vaddps    {rn-sae}, %zmm11, %zmm9, %zmm9{%k1}
        vorps     %zmm6, %zmm9, %zmm10
        jne       .LBL_1_12

.LBL_1_2:


/*
 * =========== Special branch for fast (vector) processing of zero arguments ================
 * -------------- The end of implementation ----------------
 */
        testl     %edx, %edx
        jne       .LBL_1_4

.LBL_1_3:


/* no invcbrt in libm, so taking it out here */
        vmovaps   %zmm10, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_4:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm8, 128(%rsp)
        vmovups   %zmm10, 192(%rsp)
        je        .LBL_1_3


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

.LBL_1_8:

        btl       %r12d, %r13d
        jc        .LBL_1_11

.LBL_1_9:

        incl      %r12d
        cmpl      $16, %r12d
        jl        .LBL_1_8


        kmovw     24(%rsp), %k4
	.cfi_restore 122
        kmovw     16(%rsp), %k5
	.cfi_restore 123
        kmovw     8(%rsp), %k6
	.cfi_restore 124
        kmovw     (%rsp), %k7
	.cfi_restore 125
        vmovups   192(%rsp), %zmm10
        movq      40(%rsp), %rsi
	.cfi_restore 4
        movq      32(%rsp), %rdi
	.cfi_restore 5
        movq      56(%rsp), %r12
	.cfi_restore 12
        movq      48(%rsp), %r13
	.cfi_restore 13
        jmp       .LBL_1_3
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x28, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x20, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x38, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x30, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x18, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22

.LBL_1_11:

        lea       64(%rsp,%r12,4), %rdi
        lea       128(%rsp,%r12,4), %rsi
        lea       192(%rsp,%r12,4), %rdx

        call      __svml_satan2_cout_rare_internal
        jmp       .LBL_1_9
	.cfi_restore 4
	.cfi_restore 5
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 122
	.cfi_restore 123
	.cfi_restore 124
	.cfi_restore 125

.LBL_1_12:


/* Check if at least on of Y or Y is zero: iAXAYZERO */
        vmovups   __svml_satan2_data_internal(%rip), %zmm9

/* Check if both X & Y are not NaNs:  iXYnotNAN */
        vcmpps    $3, {sae}, %zmm8, %zmm8, %k1
        vcmpps    $3, {sae}, %zmm0, %zmm0, %k2
        vpcmpd    $4, %zmm9, %zmm2, %k3
        vpternlogd $255, %zmm12, %zmm12, %zmm12
        vpternlogd $255, %zmm13, %zmm13, %zmm13
        vpternlogd $255, %zmm14, %zmm14, %zmm14
        vpandnd   %zmm8, %zmm8, %zmm12{%k1}
        vpcmpd    $4, %zmm9, %zmm1, %k1
        vpandnd   %zmm0, %zmm0, %zmm13{%k2}

/*
 * -------- Path for zero arguments (at least one of both) --------------
 * Check if both args are zeros (den. is zero)
 */
        vcmpps    $4, {sae}, %zmm9, %zmm5, %k2
        vandps    %zmm13, %zmm12, %zmm12
        vpandnd   %zmm2, %zmm2, %zmm14{%k3}
        vpternlogd $255, %zmm2, %zmm2, %zmm2

/* Res = sign(Y)*(X<0)?(PIO2+PI):PIO2 */
        vpcmpgtd  %zmm8, %zmm9, %k3
        vpandnd   %zmm1, %zmm1, %zmm2{%k1}
        vpord     %zmm2, %zmm14, %zmm15
        vpternlogd $255, %zmm2, %zmm2, %zmm2
        vpandnd   %zmm5, %zmm5, %zmm2{%k2}

/* Set sPIO2 to zero if den. is zero */
        vpandnd   %zmm4, %zmm2, %zmm4
        vpandd    %zmm2, %zmm9, %zmm5
        vpord     %zmm5, %zmm4, %zmm2
        vorps     %zmm7, %zmm2, %zmm7
        vaddps    {rn-sae}, %zmm11, %zmm7, %zmm7{%k3}
        vorps     %zmm6, %zmm7, %zmm6

/* Check if at least on of Y or Y is zero and not NaN: iAXAYZEROnotNAN */
        vpandd    %zmm12, %zmm15, %zmm1

/* Exclude from previous callout mask zero (and not NaN) arguments */
        vpandnd   %zmm3, %zmm1, %zmm3

/* Go to callout */
        vptestmd  %zmm3, %zmm3, %k0
        kmovw     %k0, %edx

/* Merge results from main and spec path */
        vpandnd   %zmm10, %zmm1, %zmm10
        vpandd    %zmm1, %zmm6, %zmm11
        vpord     %zmm11, %zmm10, %zmm10
        jmp       .LBL_1_2
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atan2f16,@function
	.size	__svml_atan2f16,.-__svml_atan2f16
..LN__svml_atan2f16.0:

.L_2__routine_start___svml_satan2_cout_rare_internal_1:

	.align    16,0x90

__svml_satan2_cout_rare_internal:


	.cfi_startproc
..L61:

        pxor      %xmm0, %xmm0
        movss     (%rdi), %xmm3
        pxor      %xmm1, %xmm1
        movss     (%rsi), %xmm2
        movq      %rdx, %r8
        cvtss2sd  %xmm3, %xmm0
        cvtss2sd  %xmm2, %xmm1
        movss     %xmm3, -32(%rsp)
        movss     %xmm2, -28(%rsp)
        movsd     %xmm0, -48(%rsp)
        movsd     %xmm1, -40(%rsp)
        movzwl    -30(%rsp), %edi
        andl      $32640, %edi
        movb      -25(%rsp), %dl
        movzwl    -42(%rsp), %eax
        andb      $-128, %dl
        movzwl    -34(%rsp), %r9d
        andl      $32752, %eax
        andl      $32752, %r9d
        shrl      $7, %edi
        movb      -29(%rsp), %cl
        shrb      $7, %cl
        shrb      $7, %dl
        shrl      $4, %eax
        shrl      $4, %r9d
        cmpl      $255, %edi
        je        .LBL_2_35


        movzwl    -26(%rsp), %esi
        andl      $32640, %esi
        cmpl      $32640, %esi
        je        .LBL_2_35


        testl     %eax, %eax
        jne       .LBL_2_5


        testl     $8388607, -32(%rsp)
        je        .LBL_2_30

.LBL_2_5:

        testl     %r9d, %r9d
        jne       .LBL_2_7


        testl     $8388607, -28(%rsp)
        je        .LBL_2_27

.LBL_2_7:

        negl      %r9d
        movsd     %xmm0, -48(%rsp)
        addl      %eax, %r9d
        movsd     %xmm1, -40(%rsp)
        movb      -41(%rsp), %dil
        movb      -33(%rsp), %sil
        andb      $127, %dil
        andb      $127, %sil
        cmpl      $-54, %r9d
        jle       .LBL_2_22


        cmpl      $54, %r9d
        jge       .LBL_2_19


        movb      %sil, -33(%rsp)
        movb      %dil, -41(%rsp)
        testb     %dl, %dl
        jne       .LBL_2_11


        movsd     1976+__satan2_la_CoutTab(%rip), %xmm1
        movaps    %xmm1, %xmm0
        jmp       .LBL_2_12

.LBL_2_11:

        movsd     1936+__satan2_la_CoutTab(%rip), %xmm1
        movsd     1944+__satan2_la_CoutTab(%rip), %xmm0

.LBL_2_12:

        movsd     -48(%rsp), %xmm4
        movsd     -40(%rsp), %xmm2
        movaps    %xmm4, %xmm5
        divsd     %xmm2, %xmm5
        movzwl    -42(%rsp), %esi
        movsd     %xmm5, -16(%rsp)
        testl     %eax, %eax
        jle       .LBL_2_34


        cmpl      $2046, %eax
        jge       .LBL_2_15


        andl      $-32753, %esi
        addl      $-1023, %eax
        movsd     %xmm4, -48(%rsp)
        addl      $16368, %esi
        movw      %si, -42(%rsp)
        jmp       .LBL_2_16

.LBL_2_15:

        movsd     1992+__satan2_la_CoutTab(%rip), %xmm3
        movl      $1022, %eax
        mulsd     %xmm3, %xmm4
        movsd     %xmm4, -48(%rsp)

.LBL_2_16:

        negl      %eax
        movq      1888+__satan2_la_CoutTab(%rip), %rsi
        addl      $1023, %eax
        movq      %rsi, -40(%rsp)
        andl      $2047, %eax
        shrq      $48, %rsi
        shll      $4, %eax
        andl      $-32753, %esi
        orl       %eax, %esi
        movw      %si, -34(%rsp)
        movsd     -40(%rsp), %xmm3
        mulsd     %xmm3, %xmm2
        comisd    1880+__satan2_la_CoutTab(%rip), %xmm5
        jb        .LBL_2_18


        movsd     2000+__satan2_la_CoutTab(%rip), %xmm12
        movaps    %xmm2, %xmm3
        mulsd     %xmm2, %xmm12
        movsd     %xmm12, -72(%rsp)
        movsd     -72(%rsp), %xmm13
        movsd     %xmm5, -24(%rsp)
        subsd     %xmm2, %xmm13
        movsd     %xmm13, -64(%rsp)
        movsd     -72(%rsp), %xmm15
        movsd     -64(%rsp), %xmm14
        movl      -20(%rsp), %edi
        movl      %edi, %r9d
        andl      $-524288, %edi
        andl      $-1048576, %r9d
        addl      $262144, %edi
        subsd     %xmm14, %xmm15
        movsd     %xmm15, -72(%rsp)
        andl      $1048575, %edi
        movsd     -72(%rsp), %xmm4
        orl       %edi, %r9d
        movl      $0, -24(%rsp)
        subsd     %xmm4, %xmm3
        movl      %r9d, -20(%rsp)
        movsd     %xmm3, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -24(%rsp), %xmm11
        movsd     -64(%rsp), %xmm9
        mulsd     %xmm11, %xmm5
        mulsd     %xmm11, %xmm9
        movsd     1968+__satan2_la_CoutTab(%rip), %xmm8
        mulsd     %xmm8, %xmm5
        mulsd     %xmm8, %xmm9
        movaps    %xmm5, %xmm7
        movzwl    -10(%rsp), %esi
        addsd     %xmm9, %xmm7
        movsd     %xmm7, -72(%rsp)
        andl      $32752, %esi
        movsd     -72(%rsp), %xmm6
        shrl      $4, %esi
        subsd     %xmm6, %xmm5
        movl      -12(%rsp), %eax
        addsd     %xmm5, %xmm9
        movsd     %xmm9, -64(%rsp)
        andl      $1048575, %eax
        movsd     -48(%rsp), %xmm9
        movsd     -72(%rsp), %xmm3
        movaps    %xmm9, %xmm12
        movsd     -64(%rsp), %xmm10
        movaps    %xmm9, %xmm14
        movaps    %xmm9, %xmm6
        addsd     %xmm3, %xmm12
        movsd     %xmm12, -72(%rsp)
        movsd     -72(%rsp), %xmm13
        shll      $20, %esi
        subsd     %xmm13, %xmm14
        movsd     %xmm14, -64(%rsp)
        orl       %eax, %esi
        movsd     -72(%rsp), %xmm4
        addl      $-1069547520, %esi
        movsd     -64(%rsp), %xmm15
        movl      $113, %eax
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm13
        addsd     %xmm15, %xmm4
        movsd     %xmm4, -56(%rsp)
        movsd     -64(%rsp), %xmm8
        sarl      $19, %esi
        addsd     %xmm3, %xmm8
        movsd     %xmm8, -64(%rsp)
        cmpl      $113, %esi
        movsd     -56(%rsp), %xmm7
        cmovl     %esi, %eax
        subsd     %xmm7, %xmm6
        movsd     %xmm6, -56(%rsp)
        addl      %eax, %eax
        movsd     -64(%rsp), %xmm12
        lea       __satan2_la_CoutTab(%rip), %rsi
        movsd     -56(%rsp), %xmm5
        movslq    %eax, %rax
        addsd     %xmm5, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -72(%rsp), %xmm7
        mulsd     %xmm7, %xmm13
        movsd     -56(%rsp), %xmm8
        movsd     %xmm13, -72(%rsp)
        addsd     %xmm10, %xmm8
        movsd     -72(%rsp), %xmm4
        movaps    %xmm9, %xmm10
        mulsd     2000+__satan2_la_CoutTab(%rip), %xmm10
        subsd     %xmm7, %xmm4
        movsd     %xmm4, -64(%rsp)
        movsd     -72(%rsp), %xmm3
        movsd     -64(%rsp), %xmm14
        subsd     %xmm14, %xmm3
        movsd     %xmm3, -72(%rsp)
        movsd     -72(%rsp), %xmm15
        subsd     %xmm15, %xmm7
        movsd     %xmm7, -64(%rsp)
        movsd     -72(%rsp), %xmm7
        movsd     -64(%rsp), %xmm4
        movsd     %xmm10, -72(%rsp)
        movaps    %xmm2, %xmm10
        addsd     %xmm4, %xmm8
        movsd     -72(%rsp), %xmm4
        subsd     -48(%rsp), %xmm4
        movsd     %xmm4, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm3
        subsd     %xmm3, %xmm6
        movaps    %xmm2, %xmm3
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        subsd     %xmm5, %xmm9
        movsd     %xmm9, -64(%rsp)
        movsd     -72(%rsp), %xmm12
        movsd     -64(%rsp), %xmm9
        mulsd     %xmm11, %xmm12
        mulsd     %xmm11, %xmm9
        movaps    %xmm12, %xmm11
        addsd     %xmm9, %xmm11
        movsd     %xmm11, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        subsd     %xmm4, %xmm12
        addsd     %xmm9, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -72(%rsp), %xmm15
        movsd     -64(%rsp), %xmm6
        addsd     %xmm15, %xmm3
        movsd     %xmm3, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm3
        subsd     %xmm5, %xmm10
        movsd     %xmm10, -64(%rsp)
        movsd     -72(%rsp), %xmm13
        movsd     -64(%rsp), %xmm11
        addsd     %xmm11, %xmm13
        movsd     %xmm13, -56(%rsp)
        movsd     -64(%rsp), %xmm14
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm13
        addsd     %xmm14, %xmm15
        movsd     %xmm15, -64(%rsp)
        movsd     -56(%rsp), %xmm4
        movsd     1888+__satan2_la_CoutTab(%rip), %xmm14
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -56(%rsp)
        movsd     -64(%rsp), %xmm4
        movsd     -56(%rsp), %xmm2
        addsd     %xmm2, %xmm4
        movsd     %xmm4, -56(%rsp)
        movsd     -72(%rsp), %xmm12
        mulsd     %xmm12, %xmm3
        movsd     -56(%rsp), %xmm5
        movsd     %xmm3, -72(%rsp)
        addsd     %xmm6, %xmm5
        movsd     -72(%rsp), %xmm9
        subsd     %xmm12, %xmm9
        movsd     %xmm9, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm2
        subsd     %xmm2, %xmm10
        movsd     %xmm10, -72(%rsp)
        movsd     -72(%rsp), %xmm11
        subsd     %xmm11, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -72(%rsp), %xmm9
        divsd     %xmm9, %xmm14
        mulsd     %xmm14, %xmm13
        movsd     -64(%rsp), %xmm10
        movsd     %xmm13, -64(%rsp)
        addsd     %xmm10, %xmm5
        movsd     -64(%rsp), %xmm15
        movsd     1888+__satan2_la_CoutTab(%rip), %xmm12
        subsd     %xmm14, %xmm15
        movsd     %xmm15, -56(%rsp)
        movsd     -64(%rsp), %xmm2
        movsd     -56(%rsp), %xmm4
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm13
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -56(%rsp)
        movsd     -56(%rsp), %xmm3
        mulsd     %xmm3, %xmm9
        movsd     -56(%rsp), %xmm11
        subsd     %xmm9, %xmm12
        mulsd     %xmm11, %xmm5
        movsd     %xmm5, -64(%rsp)
        movsd     -64(%rsp), %xmm5
        subsd     %xmm5, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -64(%rsp), %xmm2
        movq      -56(%rsp), %r10
        movsd     -64(%rsp), %xmm6
        movsd     -56(%rsp), %xmm4
        movq      %r10, -40(%rsp)
        movsd     -40(%rsp), %xmm3
        movaps    %xmm3, %xmm5
        addsd     1888+__satan2_la_CoutTab(%rip), %xmm2
        mulsd     %xmm7, %xmm5
        mulsd     %xmm6, %xmm2
        mulsd     %xmm4, %xmm2
        mulsd     %xmm2, %xmm7
        mulsd     %xmm8, %xmm2
        mulsd     %xmm3, %xmm8
        addsd     %xmm2, %xmm7
        movsd     1872+__satan2_la_CoutTab(%rip), %xmm3
        addsd     %xmm8, %xmm7
        movsd     %xmm7, -72(%rsp)
        movaps    %xmm5, %xmm7
        movsd     -72(%rsp), %xmm4
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm6
        addsd     %xmm4, %xmm7
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm8
        subsd     %xmm8, %xmm5
        addsd     %xmm4, %xmm5
        movsd     %xmm5, -64(%rsp)
        movsd     -72(%rsp), %xmm11
        movaps    %xmm11, %xmm2
        mulsd     %xmm11, %xmm2
        mulsd     %xmm11, %xmm6
        mulsd     %xmm2, %xmm3
        movsd     -64(%rsp), %xmm4
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm7
        addsd     1864+__satan2_la_CoutTab(%rip), %xmm3
        subsd     %xmm11, %xmm7
        mulsd     %xmm2, %xmm3
        movsd     %xmm7, -64(%rsp)
        movsd     -72(%rsp), %xmm9
        movsd     -64(%rsp), %xmm8
        addsd     1856+__satan2_la_CoutTab(%rip), %xmm3
        subsd     %xmm8, %xmm9
        mulsd     %xmm2, %xmm3
        movsd     %xmm9, -72(%rsp)
        movsd     -72(%rsp), %xmm10
        addsd     1848+__satan2_la_CoutTab(%rip), %xmm3
        subsd     %xmm10, %xmm11
        mulsd     %xmm2, %xmm3
        movsd     %xmm11, -64(%rsp)
        addsd     1840+__satan2_la_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        addsd     1832+__satan2_la_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        addsd     1824+__satan2_la_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        mulsd     %xmm3, %xmm13
        movsd     -72(%rsp), %xmm2
        movsd     -64(%rsp), %xmm12
        movsd     %xmm13, -72(%rsp)
        addsd     %xmm12, %xmm4
        movsd     -72(%rsp), %xmm14
        subsd     %xmm3, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm15
        subsd     %xmm15, %xmm5
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm6
        subsd     %xmm6, %xmm3
        movsd     %xmm3, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm5
        movaps    %xmm6, %xmm12
        movaps    %xmm5, %xmm3
        mulsd     %xmm4, %xmm6
        mulsd     %xmm4, %xmm3
        mulsd     %xmm2, %xmm5
        mulsd     %xmm2, %xmm12
        addsd     %xmm3, %xmm6
        movaps    %xmm12, %xmm7
        movaps    %xmm12, %xmm8
        addsd     %xmm5, %xmm6
        addsd     %xmm2, %xmm7
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm3
        subsd     %xmm3, %xmm8
        movsd     %xmm8, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movsd     -64(%rsp), %xmm11
        addsd     %xmm11, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -56(%rsp), %xmm2
        subsd     %xmm2, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -64(%rsp), %xmm14
        movsd     -56(%rsp), %xmm13
        addsd     %xmm13, %xmm14
        movsd     %xmm14, -56(%rsp)
        movq      -72(%rsp), %r11
        movsd     -56(%rsp), %xmm15
        movq      %r11, -40(%rsp)
        addsd     %xmm15, %xmm4
        movsd     -40(%rsp), %xmm8
        addsd     %xmm5, %xmm4
        movsd     %xmm4, -32(%rsp)
        movaps    %xmm8, %xmm4
        movaps    %xmm8, %xmm2
        addsd     (%rsi,%rax,8), %xmm4
        movsd     %xmm4, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm3
        addsd     %xmm3, %xmm5
        movsd     %xmm5, -56(%rsp)
        movsd     -64(%rsp), %xmm6
        addsd     (%rsi,%rax,8), %xmm6
        movsd     %xmm6, -64(%rsp)
        movsd     -56(%rsp), %xmm7
        subsd     %xmm7, %xmm8
        movsd     %xmm8, -56(%rsp)
        movsd     -64(%rsp), %xmm10
        movsd     -56(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movq      -72(%rsp), %rdi
        movq      %rdi, -40(%rsp)


        movsd     -56(%rsp), %xmm2
        movaps    %xmm1, %xmm3
        shrq      $56, %rdi
        addsd     -32(%rsp), %xmm2
        shlb      $7, %cl
        addsd     8(%rsi,%rax,8), %xmm2
        movb      %dl, %al
        andb      $127, %dil
        shlb      $7, %al
        movsd     %xmm2, -32(%rsp)
        orb       %al, %dil
        movb      %dil, -33(%rsp)
        movsd     -40(%rsp), %xmm9
        movaps    %xmm9, %xmm5
        addsd     %xmm9, %xmm3
        movsd     %xmm3, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        movb      -25(%rsp), %sil
        movb      %sil, %r9b
        shrb      $7, %sil
        subsd     %xmm4, %xmm5
        movsd     %xmm5, -64(%rsp)
        movsd     -72(%rsp), %xmm7
        movsd     -64(%rsp), %xmm6
        xorb      %sil, %dl
        andb      $127, %r9b
        shlb      $7, %dl
        addsd     %xmm6, %xmm7
        movsd     %xmm7, -56(%rsp)
        movsd     -64(%rsp), %xmm8
        addsd     %xmm8, %xmm1
        movsd     %xmm1, -64(%rsp)
        orb       %dl, %r9b
        movsd     -56(%rsp), %xmm1
        movb      %r9b, -25(%rsp)
        subsd     %xmm1, %xmm9
        movsd     %xmm9, -56(%rsp)
        movsd     -64(%rsp), %xmm11
        movsd     -56(%rsp), %xmm10
        addsd     %xmm10, %xmm11
        movsd     %xmm11, -56(%rsp)
        movq      -72(%rsp), %rdx
        movsd     -56(%rsp), %xmm12
        movq      %rdx, -40(%rsp)
        addsd     %xmm12, %xmm0
        movsd     -40(%rsp), %xmm13
        addsd     -32(%rsp), %xmm0
        movsd     %xmm0, -32(%rsp)
        addsd     %xmm0, %xmm13
        movsd     %xmm13, -24(%rsp)
        movb      -17(%rsp), %r10b
        andb      $127, %r10b
        orb       %cl, %r10b
        movb      %r10b, -17(%rsp)
        movsd     -24(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)
        jmp       .LBL_2_33

.LBL_2_18:

        movsd     -48(%rsp), %xmm12
        movb      %dl, %dil
        movaps    %xmm12, %xmm7
        mulsd     2000+__satan2_la_CoutTab(%rip), %xmm7
        shlb      $7, %dil
        shlb      $7, %cl
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm8
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm13
        movsd     1888+__satan2_la_CoutTab(%rip), %xmm7
        mulsd     %xmm2, %xmm13
        subsd     -48(%rsp), %xmm8
        movsd     %xmm8, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm9
        subsd     %xmm9, %xmm10
        movsd     %xmm10, -72(%rsp)
        movsd     -72(%rsp), %xmm11
        subsd     %xmm11, %xmm12
        movsd     %xmm12, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm5
        movsd     %xmm13, -72(%rsp)
        movsd     -72(%rsp), %xmm14
        subsd     %xmm2, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -72(%rsp), %xmm4
        movsd     -64(%rsp), %xmm15
        subsd     %xmm15, %xmm4
        movsd     %xmm4, -72(%rsp)
        movsd     -72(%rsp), %xmm3
        movsd     1888+__satan2_la_CoutTab(%rip), %xmm4
        subsd     %xmm3, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -72(%rsp), %xmm12
        divsd     %xmm12, %xmm7
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm2
        mulsd     %xmm7, %xmm2
        movsd     -64(%rsp), %xmm14
        movsd     %xmm2, -64(%rsp)
        movsd     -64(%rsp), %xmm8
        subsd     %xmm7, %xmm8
        movsd     %xmm8, -56(%rsp)
        movsd     -64(%rsp), %xmm10
        movsd     -56(%rsp), %xmm9
        subsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movsd     -56(%rsp), %xmm11
        mulsd     %xmm11, %xmm12
        movsd     -56(%rsp), %xmm13
        subsd     %xmm12, %xmm4
        mulsd     %xmm13, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -64(%rsp), %xmm15
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm13
        subsd     %xmm15, %xmm4
        movsd     %xmm4, -64(%rsp)
        movsd     -64(%rsp), %xmm7
        movq      -56(%rsp), %rax
        movsd     -64(%rsp), %xmm2
        movsd     -56(%rsp), %xmm3
        movq      %rax, -40(%rsp)
        movsd     -40(%rsp), %xmm8
        movaps    %xmm8, %xmm9
        addsd     1888+__satan2_la_CoutTab(%rip), %xmm7
        mulsd     %xmm6, %xmm9
        mulsd     %xmm5, %xmm8
        mulsd     %xmm2, %xmm7
        movsd     -16(%rsp), %xmm2
        mulsd     %xmm2, %xmm2
        mulsd     %xmm3, %xmm7
        movsd     1872+__satan2_la_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        mulsd     %xmm7, %xmm6
        mulsd     %xmm5, %xmm7
        addsd     1864+__satan2_la_CoutTab(%rip), %xmm3
        addsd     %xmm7, %xmm6
        mulsd     %xmm2, %xmm3
        addsd     %xmm8, %xmm6
        addsd     1856+__satan2_la_CoutTab(%rip), %xmm3
        mulsd     %xmm2, %xmm3
        movaps    %xmm9, %xmm5
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        addsd     1848+__satan2_la_CoutTab(%rip), %xmm3
        addsd     %xmm4, %xmm5
        mulsd     %xmm2, %xmm3
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     2000+__satan2_la_CoutTab(%rip), %xmm5
        subsd     %xmm6, %xmm9
        addsd     1840+__satan2_la_CoutTab(%rip), %xmm3
        addsd     %xmm4, %xmm9
        mulsd     %xmm2, %xmm3
        movsd     %xmm9, -64(%rsp)
        movsd     -72(%rsp), %xmm11
        mulsd     %xmm11, %xmm5
        addsd     1832+__satan2_la_CoutTab(%rip), %xmm3
        movsd     -64(%rsp), %xmm4
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm7
        mulsd     %xmm2, %xmm3
        subsd     %xmm11, %xmm7
        movsd     %xmm7, -64(%rsp)
        movsd     -72(%rsp), %xmm8
        movsd     -64(%rsp), %xmm6
        addsd     1824+__satan2_la_CoutTab(%rip), %xmm3
        subsd     %xmm6, %xmm8
        mulsd     %xmm2, %xmm3
        movsd     %xmm8, -72(%rsp)
        movsd     -72(%rsp), %xmm10
        mulsd     %xmm3, %xmm13
        subsd     %xmm10, %xmm11
        movsd     %xmm11, -64(%rsp)
        movsd     -72(%rsp), %xmm2
        movsd     -64(%rsp), %xmm12
        movsd     %xmm13, -72(%rsp)
        addsd     %xmm12, %xmm4
        movsd     -72(%rsp), %xmm14
        subsd     %xmm3, %xmm14
        movsd     %xmm14, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm15
        subsd     %xmm15, %xmm5
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm6
        subsd     %xmm6, %xmm3
        movsd     %xmm3, -64(%rsp)
        movsd     -72(%rsp), %xmm6
        movsd     -64(%rsp), %xmm5
        movaps    %xmm6, %xmm12
        movaps    %xmm5, %xmm3
        mulsd     %xmm4, %xmm6
        mulsd     %xmm4, %xmm3
        mulsd     %xmm2, %xmm5
        mulsd     %xmm2, %xmm12
        addsd     %xmm3, %xmm6
        movaps    %xmm12, %xmm7
        movaps    %xmm12, %xmm8
        addsd     %xmm5, %xmm6
        addsd     %xmm2, %xmm7
        movsd     %xmm6, -72(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     %xmm7, -72(%rsp)
        movsd     -72(%rsp), %xmm3
        subsd     %xmm3, %xmm8
        movsd     %xmm8, -64(%rsp)
        movsd     -72(%rsp), %xmm10
        movsd     -64(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -56(%rsp)
        movsd     -64(%rsp), %xmm11
        addsd     %xmm11, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -56(%rsp), %xmm2
        subsd     %xmm2, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -64(%rsp), %xmm14
        movsd     -56(%rsp), %xmm13
        addsd     %xmm13, %xmm14
        movsd     %xmm14, -56(%rsp)
        movq      -72(%rsp), %rsi
        movsd     -56(%rsp), %xmm15
        movq      %rsi, -40(%rsp)
        addsd     %xmm15, %xmm4
        shrq      $56, %rsi
        addsd     %xmm5, %xmm4
        andb      $127, %sil
        orb       %dil, %sil
        movb      %sil, -33(%rsp)
        movsd     %xmm4, -32(%rsp)
        movaps    %xmm1, %xmm4
        movsd     -40(%rsp), %xmm7
        movaps    %xmm7, %xmm2
        addsd     %xmm7, %xmm4
        movsd     %xmm4, -72(%rsp)
        movsd     -72(%rsp), %xmm4
        movb      -25(%rsp), %r9b
        movb      %r9b, %r10b
        shrb      $7, %r9b
        subsd     %xmm4, %xmm2
        movsd     %xmm2, -64(%rsp)
        movsd     -72(%rsp), %xmm5
        movsd     -64(%rsp), %xmm3
        xorb      %r9b, %dl
        andb      $127, %r10b
        shlb      $7, %dl
        addsd     %xmm3, %xmm5
        movsd     %xmm5, -56(%rsp)
        movsd     -64(%rsp), %xmm6
        addsd     %xmm6, %xmm1
        movsd     %xmm1, -64(%rsp)
        orb       %dl, %r10b
        movsd     -56(%rsp), %xmm1
        movb      %r10b, -25(%rsp)
        subsd     %xmm1, %xmm7
        movsd     %xmm7, -56(%rsp)
        movsd     -64(%rsp), %xmm2
        movsd     -56(%rsp), %xmm1
        addsd     %xmm1, %xmm2
        movsd     %xmm2, -56(%rsp)
        movq      -72(%rsp), %rdx
        movsd     -56(%rsp), %xmm3
        movq      %rdx, -40(%rsp)
        addsd     %xmm3, %xmm0
        movsd     -40(%rsp), %xmm4
        addsd     -32(%rsp), %xmm0
        movsd     %xmm0, -32(%rsp)
        addsd     %xmm0, %xmm4
        movsd     %xmm4, -24(%rsp)
        movb      -17(%rsp), %r11b
        andb      $127, %r11b
        orb       %cl, %r11b
        movb      %r11b, -17(%rsp)
        movsd     -24(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)
        jmp       .LBL_2_33

.LBL_2_19:

        cmpl      $74, %r9d
        jge       .LBL_2_21


        movb      %dil, -41(%rsp)
        divsd     -48(%rsp), %xmm1
        movsd     1928+__satan2_la_CoutTab(%rip), %xmm0
        shlb      $7, %cl
        subsd     %xmm1, %xmm0
        addsd     1920+__satan2_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%r8)
        jmp       .LBL_2_33

.LBL_2_21:

        movsd     1920+__satan2_la_CoutTab(%rip), %xmm0
        shlb      $7, %cl
        addsd     1928+__satan2_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%r8)
        jmp       .LBL_2_33

.LBL_2_22:

        testb     %dl, %dl
        jne       .LBL_2_32


        movb      %dil, -41(%rsp)
        pxor      %xmm0, %xmm0
        movb      %sil, -33(%rsp)
        movsd     -48(%rsp), %xmm2
        divsd     -40(%rsp), %xmm2
        cvtsd2ss  %xmm2, %xmm0
        movss     %xmm0, -8(%rsp)
        movzwl    -6(%rsp), %eax
        movsd     %xmm2, -24(%rsp)
        testl     $32640, %eax
        je        .LBL_2_25


        movsd     1888+__satan2_la_CoutTab(%rip), %xmm0
        shlb      $7, %cl
        addsd     %xmm2, %xmm0
        movsd     %xmm0, -72(%rsp)
        movsd     -72(%rsp), %xmm1
        mulsd     %xmm1, %xmm2
        movsd     %xmm2, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm2
        cvtsd2ss  %xmm2, %xmm2
        movss     %xmm2, (%r8)
        jmp       .LBL_2_33

.LBL_2_25:

        movsd     -24(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        shlb      $7, %cl
        movss     %xmm0, -8(%rsp)
        movss     -8(%rsp), %xmm2
        movss     -8(%rsp), %xmm1
        mulss     %xmm1, %xmm2
        movss     %xmm2, -8(%rsp)
        movss     -8(%rsp), %xmm3
        cvtss2sd  %xmm3, %xmm3
        addsd     -24(%rsp), %xmm3
        movsd     %xmm3, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm4
        cvtsd2ss  %xmm4, %xmm4
        movss     %xmm4, (%r8)
        jmp       .LBL_2_33

.LBL_2_27:

        testl     %eax, %eax
        jne       .LBL_2_21


        testl     $8388607, -32(%rsp)
        jne       .LBL_2_21

.LBL_2_30:

        testb     %dl, %dl
        jne       .LBL_2_32

.LBL_2_31:

        shlb      $7, %cl
        movq      1976+__satan2_la_CoutTab(%rip), %rax
        movq      %rax, -24(%rsp)
        shrq      $56, %rax
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)
        jmp       .LBL_2_33

.LBL_2_32:

        movsd     1936+__satan2_la_CoutTab(%rip), %xmm0
        shlb      $7, %cl
        addsd     1944+__satan2_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%r8)

.LBL_2_33:

        xorl      %eax, %eax
        ret

.LBL_2_34:

        movsd     1984+__satan2_la_CoutTab(%rip), %xmm3
        movl      $-1022, %eax
        mulsd     %xmm3, %xmm4
        movsd     %xmm4, -48(%rsp)
        jmp       .LBL_2_16

.LBL_2_35:

        cmpl      $2047, %eax
        je        .LBL_2_48

.LBL_2_36:

        cmpl      $2047, %r9d
        je        .LBL_2_46

.LBL_2_37:

        movzwl    -26(%rsp), %eax
        andl      $32640, %eax
        cmpl      $32640, %eax
        jne       .LBL_2_21


        cmpl      $255, %edi
        je        .LBL_2_43


        testb     %dl, %dl
        je        .LBL_2_31
        jmp       .LBL_2_32

.LBL_2_43:

        testb     %dl, %dl
        jne       .LBL_2_45


        movsd     1904+__satan2_la_CoutTab(%rip), %xmm0
        shlb      $7, %cl
        addsd     1912+__satan2_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%r8)
        jmp       .LBL_2_33

.LBL_2_45:

        movsd     1952+__satan2_la_CoutTab(%rip), %xmm0
        shlb      $7, %cl
        addsd     1960+__satan2_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %al
        andb      $127, %al
        orb       %cl, %al
        movb      %al, -17(%rsp)
        movsd     -24(%rsp), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%r8)
        jmp       .LBL_2_33

.LBL_2_46:

        testl     $8388607, -28(%rsp)
        je        .LBL_2_37

.LBL_2_47:

        addss     %xmm2, %xmm3
        movss     %xmm3, (%r8)
        jmp       .LBL_2_33

.LBL_2_48:

        testl     $8388607, -32(%rsp)
        jne       .LBL_2_47
        jmp       .LBL_2_36
	.align    16,0x90

	.cfi_endproc

	.type	__svml_satan2_cout_rare_internal,@function
	.size	__svml_satan2_cout_rare_internal,.-__svml_satan2_cout_rare_internal
..LN__svml_satan2_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_satan2_data_internal:
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
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
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
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	993144000
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	3162449457
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	1026278276
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	3180885545
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	1037657204
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	3188810232
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	1045215135
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
	.long	3198855753
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
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	2164260864
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.long	4227858432
	.type	__svml_satan2_data_internal,@object
	.size	__svml_satan2_data_internal,1152
	.align 32
__satan2_la_CoutTab:
	.long	3892314112
	.long	1069799150
	.long	2332892550
	.long	1039715405
	.long	1342177280
	.long	1070305495
	.long	270726690
	.long	1041535749
	.long	939524096
	.long	1070817911
	.long	2253973841
	.long	3188654726
	.long	3221225472
	.long	1071277294
	.long	3853927037
	.long	1043226911
	.long	2818572288
	.long	1071767563
	.long	2677759107
	.long	1044314101
	.long	3355443200
	.long	1072103591
	.long	1636578514
	.long	3191094734
	.long	1476395008
	.long	1072475260
	.long	1864703685
	.long	3188646936
	.long	805306368
	.long	1072747407
	.long	192551812
	.long	3192726267
	.long	2013265920
	.long	1072892781
	.long	2240369452
	.long	1043768538
	.long	0
	.long	1072999953
	.long	3665168337
	.long	3192705970
	.long	402653184
	.long	1073084787
	.long	1227953434
	.long	3192313277
	.long	2013265920
	.long	1073142981
	.long	3853283127
	.long	1045277487
	.long	805306368
	.long	1073187261
	.long	1676192264
	.long	3192868861
	.long	134217728
	.long	1073217000
	.long	4290763938
	.long	1042034855
	.long	671088640
	.long	1073239386
	.long	994303084
	.long	3189643768
	.long	402653184
	.long	1073254338
	.long	1878067156
	.long	1042652475
	.long	1610612736
	.long	1073265562
	.long	670314820
	.long	1045138554
	.long	3221225472
	.long	1073273048
	.long	691126919
	.long	3189987794
	.long	3489660928
	.long	1073278664
	.long	1618990832
	.long	3188194509
	.long	1207959552
	.long	1073282409
	.long	2198872939
	.long	1044806069
	.long	3489660928
	.long	1073285217
	.long	2633982383
	.long	1042307894
	.long	939524096
	.long	1073287090
	.long	1059367786
	.long	3189114230
	.long	2281701376
	.long	1073288494
	.long	3158525533
	.long	1044484961
	.long	3221225472
	.long	1073289430
	.long	286581777
	.long	1044893263
	.long	4026531840
	.long	1073290132
	.long	2000245215
	.long	3191647611
	.long	134217728
	.long	1073290601
	.long	4205071590
	.long	1045035927
	.long	536870912
	.long	1073290952
	.long	2334392229
	.long	1043447393
	.long	805306368
	.long	1073291186
	.long	2281458177
	.long	3188885569
	.long	3087007744
	.long	1073291361
	.long	691611507
	.long	1044733832
	.long	3221225472
	.long	1073291478
	.long	1816229550
	.long	1044363390
	.long	2281701376
	.long	1073291566
	.long	1993843750
	.long	3189837440
	.long	134217728
	.long	1073291625
	.long	3654754496
	.long	1044970837
	.long	4026531840
	.long	1073291668
	.long	3224300229
	.long	3191935390
	.long	805306368
	.long	1073291698
	.long	2988777976
	.long	3188950659
	.long	536870912
	.long	1073291720
	.long	1030371341
	.long	1043402665
	.long	3221225472
	.long	1073291734
	.long	1524463765
	.long	1044361356
	.long	3087007744
	.long	1073291745
	.long	2754295320
	.long	1044731036
	.long	134217728
	.long	1073291753
	.long	3099629057
	.long	1044970710
	.long	2281701376
	.long	1073291758
	.long	962914160
	.long	3189838838
	.long	805306368
	.long	1073291762
	.long	3543908206
	.long	3188950786
	.long	4026531840
	.long	1073291764
	.long	1849909620
	.long	3191935434
	.long	3221225472
	.long	1073291766
	.long	1641333636
	.long	1044361352
	.long	536870912
	.long	1073291768
	.long	1373968792
	.long	1043402654
	.long	134217728
	.long	1073291769
	.long	2033191599
	.long	1044970710
	.long	3087007744
	.long	1073291769
	.long	4117947437
	.long	1044731035
	.long	805306368
	.long	1073291770
	.long	315378368
	.long	3188950787
	.long	2281701376
	.long	1073291770
	.long	2428571750
	.long	3189838838
	.long	3221225472
	.long	1073291770
	.long	1608007466
	.long	1044361352
	.long	4026531840
	.long	1073291770
	.long	1895711420
	.long	3191935434
	.long	134217728
	.long	1073291771
	.long	2031108713
	.long	1044970710
	.long	536870912
	.long	1073291771
	.long	1362518342
	.long	1043402654
	.long	805306368
	.long	1073291771
	.long	317461253
	.long	3188950787
	.long	939524096
	.long	1073291771
	.long	4117231784
	.long	1044731035
	.long	1073741824
	.long	1073291771
	.long	1607942376
	.long	1044361352
	.long	1207959552
	.long	1073291771
	.long	2428929577
	.long	3189838838
	.long	1207959552
	.long	1073291771
	.long	2031104645
	.long	1044970710
	.long	1342177280
	.long	1073291771
	.long	1895722602
	.long	3191935434
	.long	1342177280
	.long	1073291771
	.long	317465322
	.long	3188950787
	.long	1342177280
	.long	1073291771
	.long	1362515546
	.long	1043402654
	.long	1342177280
	.long	1073291771
	.long	1607942248
	.long	1044361352
	.long	1342177280
	.long	1073291771
	.long	4117231610
	.long	1044731035
	.long	1342177280
	.long	1073291771
	.long	2031104637
	.long	1044970710
	.long	1342177280
	.long	1073291771
	.long	1540251232
	.long	1045150466
	.long	1342177280
	.long	1073291771
	.long	2644671394
	.long	1045270303
	.long	1342177280
	.long	1073291771
	.long	2399244691
	.long	1045360181
	.long	1342177280
	.long	1073291771
	.long	803971124
	.long	1045420100
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192879152
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192849193
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192826724
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192811744
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192800509
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192793019
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192787402
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192783657
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192780848
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192778976
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192777572
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192776635
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192775933
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192775465
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192775114
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774880
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774704
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774587
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774500
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774441
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774397
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774368
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774346
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774331
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774320
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774313
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774308
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774304
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774301
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774299
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774298
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774297
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1466225875
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1343512524
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1251477510
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1190120835
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1144103328
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1113424990
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1090416237
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1075077068
	.long	3192774295
	.long	1431655765
	.long	3218429269
	.long	2576978363
	.long	1070176665
	.long	2453154343
	.long	3217180964
	.long	4189149139
	.long	1069314502
	.long	1775019125
	.long	3216459198
	.long	273199057
	.long	1068739452
	.long	874748308
	.long	3215993277
	.long	0
	.long	1069547520
	.long	0
	.long	1072693248
	.long	0
	.long	1073741824
	.long	1413754136
	.long	1072243195
	.long	856972295
	.long	1015129638
	.long	1413754136
	.long	1073291771
	.long	856972295
	.long	1016178214
	.long	1413754136
	.long	1074340347
	.long	856972295
	.long	1017226790
	.long	2134057426
	.long	1073928572
	.long	1285458442
	.long	1016756537
	.long	0
	.long	3220176896
	.long	0
	.long	0
	.long	0
	.long	2144337920
	.long	0
	.long	1048576
	.long	33554432
	.long	1101004800
	.type	__satan2_la_CoutTab,@object
	.size	__satan2_la_CoutTab,2008
