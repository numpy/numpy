/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/


	.text
.L_2__routine_start___svml_sinhf16_z0_0:

	.align    16,0x90
	.globl __svml_sinhf16

__svml_sinhf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovaps   %zmm0, %zmm5

/*
 * ----------------------------------- Implementation  ---------------------
 * ............... Abs argument ............................
 */
        vandps    1152+__svml_ssinh_data_internal(%rip), %zmm5, %zmm4

/*
 * ...............Check for overflow\underflow .............
 */
        vpternlogd $255, %zmm6, %zmm6, %zmm6
        vmovups   1280+__svml_ssinh_data_internal(%rip), %zmm7

/*
 * ............... Load argument ............................
 * dM = x/log(2) + RShifter
 */
        vmovups   960+__svml_ssinh_data_internal(%rip), %zmm11
        vmovups   1024+__svml_ssinh_data_internal(%rip), %zmm8
        vmovups   1088+__svml_ssinh_data_internal(%rip), %zmm10
        vmovups   1856+__svml_ssinh_data_internal(%rip), %zmm12
        vmovups   1728+__svml_ssinh_data_internal(%rip), %zmm0
        vmovups   1792+__svml_ssinh_data_internal(%rip), %zmm3

/* x^2 */
        vmovups   1536+__svml_ssinh_data_internal(%rip), %zmm2
        vxorps    %zmm5, %zmm4, %zmm1
        vfmadd213ps {rn-sae}, %zmm7, %zmm1, %zmm11
        vpcmpd    $2, 1408+__svml_ssinh_data_internal(%rip), %zmm1, %k1

/*
 * ............... G1,G2 2^N,2^(-N) ...........
 * iM now is an EXP(2^N)
 */
        vpslld    $23, %zmm11, %zmm13

/*
 * ................... R ...................................
 * sN = sM - RShifter
 */
        vsubps    {rn-sae}, %zmm7, %zmm11, %zmm9
        vpaddd    %zmm13, %zmm12, %zmm14
        vpsubd    %zmm13, %zmm12, %zmm15

/* sG1 = 2^(N-1)+2^(-N-1) */
        vaddps    {rn-sae}, %zmm15, %zmm14, %zmm7
        vpandnd   %zmm1, %zmm1, %zmm6{%k1}

/* sR = sX - sN*Log2_hi */
        vfnmadd231ps {rn-sae}, %zmm8, %zmm9, %zmm1
        vptestmd  %zmm6, %zmm6, %k0

/* sG2 = 2^(N-1)-2^(-N-1) */
        vsubps    {rn-sae}, %zmm15, %zmm14, %zmm8

/* sR = (sX - sN*Log2_hi) - sN*Log2_lo */
        vfnmadd231ps {rn-sae}, %zmm10, %zmm9, %zmm1

/*
 * ....sinh(r) = r*((a1=1)+r^2*(a3+r^2*(a5+{v1 r^2*a7})))) = r + r*(r^2*(a3+r^2*(a5+r^2*a7))) ....
 * sSinh_r = (a3+r^2*a5)
 */
        vmovups   1600+__svml_ssinh_data_internal(%rip), %zmm14
        kmovw     %k0, %edx

/* sR2 = sR^2 */
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm6
        vfmadd231ps {rn-sae}, %zmm6, %zmm0, %zmm14

/* sSinh_r = r^2*(a3+r^2*a5) */
        vmulps    {rn-sae}, %zmm6, %zmm14, %zmm0

/* sSinh_r = r + r*(r^2*(a3+r^2*a5)) */
        vfmadd213ps {rn-sae}, %zmm1, %zmm1, %zmm0

/*
 * sinh(X) = sG2 + sG1*sinh(dR) + sG2*sR2*(a2+sR2*(a4+a6*sR2)
 * sOut = (a4 +a6*sR2)
 */
        vmovups   1664+__svml_ssinh_data_internal(%rip), %zmm1
        vfmadd231ps {rn-sae}, %zmm6, %zmm3, %zmm1

/* sOut = a2+sR2*(a4+a6*sR2) */
        vfmadd213ps {rn-sae}, %zmm2, %zmm6, %zmm1

/* sOut = sR2*(a2+sR2*(a4+a6*sR2) */
        vmulps    {rn-sae}, %zmm6, %zmm1, %zmm2

/* sOut = sG2*sR2*(a2+sR2*(a4+a6*sR2) */
        vmulps    {rn-sae}, %zmm8, %zmm2, %zmm3

/* sOut = sG1*sinh(dR)+sG2*sR2*(a2+sR2*(a4+a6*sR2) */
        vfmadd213ps {rn-sae}, %zmm3, %zmm0, %zmm7

/* sOut = sG2 + sG1*sinh(dR) + sG2*sR2*(a2+sR2*(a4+a6*sR2) */
        vaddps    {rn-sae}, %zmm8, %zmm7, %zmm9

/* ................... Ret H ...................... */
        vorps     %zmm9, %zmm4, %zmm0
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

        vmovups   %zmm5, 64(%rsp)
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

        call      __svml_ssinh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sinhf16,@function
	.size	__svml_sinhf16,.-__svml_sinhf16
..LN__svml_sinhf16.0:

.L_2__routine_start___svml_ssinh_cout_rare_internal_1:

	.align    16,0x90

__svml_ssinh_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      %rsi, %r9
        movzwl    2(%rdi), %edx
        xorl      %eax, %eax
        andl      $32640, %edx
        movss     (%rdi), %xmm2
        cmpl      $32640, %edx
        je        .LBL_2_17


        cvtss2sd  %xmm2, %xmm2
        movsd     %xmm2, -8(%rsp)
        movzwl    -2(%rsp), %edx
        andl      $32752, %edx
        movsd     %xmm2, -32(%rsp)
        shrl      $4, %edx
        andb      $127, -25(%rsp)
        testl     %edx, %edx
        jle       .LBL_2_16


        cmpl      $969, %edx
        jle       .LBL_2_14


        movsd     -32(%rsp), %xmm0
        movsd     1136+__ssinh_la_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_13


        movsd     1184+__ssinh_la_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_9


        comisd    1176+__ssinh_la_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movsd     1112+__ssinh_la_CoutTab(%rip), %xmm3
        lea       __ssinh_la_CoutTab(%rip), %rcx
        mulsd     %xmm0, %xmm3
        movsd     1144+__ssinh_la_CoutTab(%rip), %xmm10
        movq      8+__ssinh_la_CoutTab(%rip), %r10
        movq      %r10, %rsi
        shrq      $48, %rsi
        addsd     1120+__ssinh_la_CoutTab(%rip), %xmm3
        movsd     %xmm3, -40(%rsp)
        andl      $-32753, %esi
        movsd     -40(%rsp), %xmm13
        movl      -40(%rsp), %r8d
        movl      %r8d, %r11d
        shrl      $6, %r11d
        andl      $63, %r8d
        movq      %r10, -16(%rsp)
        subsd     1120+__ssinh_la_CoutTab(%rip), %xmm13
        mulsd     %xmm13, %xmm10
        lea       1023(%r11), %edi
        xorps     .L_2il0floatpacket.98(%rip), %xmm13
        addl      $1022, %r11d
        mulsd     1152+__ssinh_la_CoutTab(%rip), %xmm13
        subsd     %xmm10, %xmm0
        movaps    %xmm0, %xmm5
        movaps    %xmm0, %xmm11
        andl      $2047, %r11d
        lea       (%r8,%r8), %edx
        negl      %edi
        lea       1(%r8,%r8), %r8d
        movsd     (%rcx,%rdx,8), %xmm8
        negl      %edx
        shll      $4, %r11d
        addl      $-4, %edi
        orl       %r11d, %esi
        andl      $2047, %edi
        movw      %si, -10(%rsp)
        andl      $-32753, %esi
        shll      $4, %edi
        addsd     %xmm13, %xmm5
        movsd     %xmm5, -24(%rsp)
        orl       %edi, %esi
        movsd     -24(%rsp), %xmm7
        movsd     1128+__ssinh_la_CoutTab(%rip), %xmm5
        subsd     %xmm7, %xmm11
        movsd     %xmm11, -56(%rsp)
        movsd     -24(%rsp), %xmm4
        movsd     -56(%rsp), %xmm12
        movsd     (%rcx,%r8,8), %xmm6
        addsd     %xmm12, %xmm4
        movsd     %xmm4, -48(%rsp)
        movsd     -56(%rsp), %xmm9
        movsd     -16(%rsp), %xmm4
        addsd     %xmm9, %xmm13
        mulsd     %xmm4, %xmm8
        mulsd     %xmm4, %xmm6
        movsd     %xmm13, -56(%rsp)
        movaps    %xmm8, %xmm9
        movsd     -48(%rsp), %xmm15
        movw      %si, -10(%rsp)
        lea       128(%rdx), %esi
        movsd     -16(%rsp), %xmm14
        addl      $129, %edx
        subsd     %xmm15, %xmm0
        movaps    %xmm8, %xmm15
        movsd     %xmm0, -48(%rsp)
        movsd     -56(%rsp), %xmm3
        movsd     -48(%rsp), %xmm0
        addsd     %xmm0, %xmm3
        movsd     %xmm3, -48(%rsp)
        movsd     -24(%rsp), %xmm10
        mulsd     %xmm10, %xmm5
        movaps    %xmm10, %xmm2
        mulsd     %xmm10, %xmm2
        movsd     -48(%rsp), %xmm3
        movaps    %xmm10, %xmm1
        movsd     %xmm5, -24(%rsp)
        movsd     -24(%rsp), %xmm7
        subsd     %xmm10, %xmm7
        movsd     %xmm7, -56(%rsp)
        movsd     -24(%rsp), %xmm12
        movsd     -56(%rsp), %xmm11
        subsd     %xmm11, %xmm12
        movsd     1064+__ssinh_la_CoutTab(%rip), %xmm11
        mulsd     %xmm2, %xmm11
        movsd     %xmm12, -24(%rsp)
        movsd     1072+__ssinh_la_CoutTab(%rip), %xmm12
        mulsd     %xmm2, %xmm12
        addsd     1048+__ssinh_la_CoutTab(%rip), %xmm11
        mulsd     %xmm2, %xmm11
        addsd     1056+__ssinh_la_CoutTab(%rip), %xmm12
        mulsd     %xmm2, %xmm12
        mulsd     %xmm10, %xmm11
        addsd     1040+__ssinh_la_CoutTab(%rip), %xmm12
        addsd     %xmm11, %xmm10
        mulsd     %xmm2, %xmm12
        movsd     (%rcx,%rsi,8), %xmm2
        mulsd     %xmm14, %xmm2
        movsd     -24(%rsp), %xmm0
        subsd     %xmm2, %xmm9
        subsd     %xmm0, %xmm1
        movsd     %xmm1, -56(%rsp)
        movsd     -24(%rsp), %xmm7
        movsd     -56(%rsp), %xmm5
        movsd     %xmm9, -24(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     (%rcx,%rdx,8), %xmm1
        subsd     %xmm13, %xmm15
        mulsd     %xmm14, %xmm1
        subsd     %xmm2, %xmm15
        movsd     %xmm15, -56(%rsp)
        movaps    %xmm8, %xmm13
        movsd     -24(%rsp), %xmm14
        addsd     %xmm2, %xmm13
        movsd     -56(%rsp), %xmm9
        movaps    %xmm14, %xmm0
        movb      -1(%rsp), %cl
        addsd     %xmm6, %xmm9
        addsd     %xmm1, %xmm6
        subsd     %xmm1, %xmm9
        andb      $-128, %cl
        addsd     %xmm9, %xmm0
        movsd     %xmm0, -24(%rsp)
        movsd     -24(%rsp), %xmm4
        subsd     %xmm4, %xmm14
        addsd     %xmm14, %xmm9
        movsd     %xmm9, -56(%rsp)
        movsd     -24(%rsp), %xmm9
        movsd     -56(%rsp), %xmm0
        movsd     %xmm13, -24(%rsp)
        movsd     -24(%rsp), %xmm15
        subsd     %xmm15, %xmm8
        addsd     %xmm8, %xmm2
        movsd     %xmm2, -56(%rsp)
        movsd     -24(%rsp), %xmm2
        movsd     -56(%rsp), %xmm4
        addsd     %xmm6, %xmm4
        movaps    %xmm2, %xmm6
        addsd     %xmm4, %xmm6
        movsd     %xmm6, -24(%rsp)
        movsd     -24(%rsp), %xmm8
        movsd     1128+__ssinh_la_CoutTab(%rip), %xmm6
        subsd     %xmm8, %xmm2
        addsd     %xmm2, %xmm4
        movsd     %xmm4, -56(%rsp)
        movsd     -24(%rsp), %xmm1
        mulsd     %xmm1, %xmm6
        movsd     -56(%rsp), %xmm2
        movsd     %xmm6, -24(%rsp)
        movaps    %xmm1, %xmm6
        movsd     -24(%rsp), %xmm14
        mulsd     %xmm2, %xmm10
        subsd     %xmm1, %xmm14
        movsd     %xmm14, -56(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm8
        subsd     %xmm8, %xmm13
        movsd     %xmm13, -24(%rsp)
        movaps    %xmm11, %xmm13
        movsd     -24(%rsp), %xmm15
        mulsd     %xmm1, %xmm13
        subsd     %xmm15, %xmm6
        mulsd     %xmm3, %xmm1
        mulsd     %xmm2, %xmm3
        movaps    %xmm12, %xmm15
        movaps    %xmm13, %xmm4
        mulsd     %xmm9, %xmm15
        mulsd     %xmm0, %xmm12
        addsd     %xmm15, %xmm4
        addsd     %xmm0, %xmm12
        movsd     %xmm6, -56(%rsp)
        addsd     %xmm1, %xmm12
        movsd     -24(%rsp), %xmm8
        addsd     %xmm3, %xmm12
        movsd     -56(%rsp), %xmm6
        movsd     %xmm4, -24(%rsp)
        movsd     -24(%rsp), %xmm14
        subsd     %xmm14, %xmm13
        addsd     %xmm13, %xmm15
        movsd     %xmm15, -56(%rsp)
        movaps    %xmm7, %xmm15
        mulsd     %xmm8, %xmm15
        mulsd     %xmm5, %xmm8
        mulsd     %xmm6, %xmm5
        mulsd     %xmm6, %xmm7
        movsd     -24(%rsp), %xmm14
        movaps    %xmm14, %xmm13
        movsd     -56(%rsp), %xmm4
        addsd     %xmm15, %xmm13
        addsd     %xmm8, %xmm4
        movsd     %xmm13, -24(%rsp)
        addsd     %xmm5, %xmm4
        movsd     -24(%rsp), %xmm13
        addsd     %xmm7, %xmm4
        subsd     %xmm13, %xmm15
        addsd     %xmm4, %xmm12
        addsd     %xmm15, %xmm14
        movsd     %xmm14, -56(%rsp)
        movaps    %xmm9, %xmm15
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm14
        addsd     %xmm13, %xmm15
        addsd     %xmm14, %xmm12
        movsd     %xmm15, -24(%rsp)
        movsd     -24(%rsp), %xmm15
        subsd     %xmm15, %xmm9
        addsd     %xmm9, %xmm13
        movsd     %xmm13, -56(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm9
        addsd     %xmm9, %xmm12
        addsd     %xmm12, %xmm13
        addsd     %xmm13, %xmm10
        movsd     %xmm10, -32(%rsp)
        movb      -25(%rsp), %dil
        andb      $127, %dil
        orb       %cl, %dil
        movb      %dil, -25(%rsp)
        movsd     -32(%rsp), %xmm10
        cvtsd2ss  %xmm10, %xmm10
        movss     %xmm10, (%r9)
        ret

.LBL_2_8:

        movaps    %xmm0, %xmm2
        mulsd     %xmm0, %xmm2
        movsd     1104+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        movb      -1(%rsp), %dl
        andb      $-128, %dl
        addsd     1096+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        addsd     1088+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        addsd     1080+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm1, %xmm2
        mulsd     %xmm0, %xmm2
        addsd     %xmm2, %xmm0
        movsd     %xmm0, -32(%rsp)
        movb      -25(%rsp), %cl
        andb      $127, %cl
        orb       %dl, %cl
        movb      %cl, -25(%rsp)
        movsd     -32(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)
        ret

.LBL_2_9:

        movsd     1112+__ssinh_la_CoutTab(%rip), %xmm1
        lea       __ssinh_la_CoutTab(%rip), %r8
        mulsd     %xmm0, %xmm1
        movsd     1144+__ssinh_la_CoutTab(%rip), %xmm2
        movsd     1152+__ssinh_la_CoutTab(%rip), %xmm3
        movq      8+__ssinh_la_CoutTab(%rip), %rdx
        movq      %rdx, -16(%rsp)
        addsd     1120+__ssinh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm4
        movsd     1072+__ssinh_la_CoutTab(%rip), %xmm1
        movl      -40(%rsp), %edx
        movl      %edx, %esi
        andl      $63, %esi
        subsd     1120+__ssinh_la_CoutTab(%rip), %xmm4
        mulsd     %xmm4, %xmm2
        lea       (%rsi,%rsi), %ecx
        mulsd     %xmm3, %xmm4
        subsd     %xmm2, %xmm0
        movsd     (%r8,%rcx,8), %xmm5
        lea       1(%rsi,%rsi), %edi
        shrl      $6, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm0, %xmm1
        addl      $1022, %edx
        andl      $2047, %edx
        addsd     1064+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1056+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1048+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1040+__ssinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        addsd     %xmm0, %xmm1
        mulsd     %xmm5, %xmm1
        addsd     (%r8,%rdi,8), %xmm1
        addsd     %xmm5, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_11


        movq      8+__ssinh_la_CoutTab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -10(%rsp)
        movsd     -16(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        movsd     %xmm0, -32(%rsp)
        jmp       .LBL_2_12

.LBL_2_11:

        decl      %edx
        andl      $2047, %edx
        movzwl    -10(%rsp), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -10(%rsp)
        movsd     -16(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        mulsd     1024+__ssinh_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -32(%rsp)

.LBL_2_12:

        movb      -25(%rsp), %cl
        movb      -1(%rsp), %dl
        andb      $127, %cl
        andb      $-128, %dl
        orb       %dl, %cl
        movb      %cl, -25(%rsp)
        movsd     -32(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)
        ret

.LBL_2_13:

        movsd     1168+__ssinh_la_CoutTab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm2, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)
        ret

.LBL_2_14:

        movsd     __ssinh_la_CoutTab(%rip), %xmm0
        addsd     1160+__ssinh_la_CoutTab(%rip), %xmm0
        mulsd     %xmm2, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r9)


        ret

.LBL_2_16:

        movsd     1160+__ssinh_la_CoutTab(%rip), %xmm0
        mulsd     %xmm0, %xmm2
        movsd     %xmm2, -24(%rsp)
        pxor      %xmm2, %xmm2
        cvtss2sd  (%rdi), %xmm2
        movsd     -24(%rsp), %xmm1
        movq      8+__ssinh_la_CoutTab(%rip), %rdx
        addsd     %xmm1, %xmm2
        cvtsd2ss  %xmm2, %xmm2
        movq      %rdx, -16(%rsp)
        movss     %xmm2, (%r9)
        ret

.LBL_2_17:

        addss     %xmm2, %xmm2
        movss     %xmm2, (%r9)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_ssinh_cout_rare_internal,@function
	.size	__svml_ssinh_cout_rare_internal,.-__svml_ssinh_cout_rare_internal
..LN__svml_ssinh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_ssinh_data_internal:
	.long	1056964608
	.long	1057148295
	.long	1057336003
	.long	1057527823
	.long	1057723842
	.long	1057924154
	.long	1058128851
	.long	1058338032
	.long	1058551792
	.long	1058770234
	.long	1058993458
	.long	1059221571
	.long	1059454679
	.long	1059692891
	.long	1059936319
	.long	1060185078
	.long	1060439283
	.long	1060699055
	.long	1060964516
	.long	1061235789
	.long	1061513002
	.long	1061796286
	.long	1062085772
	.long	1062381598
	.long	1062683901
	.long	1062992824
	.long	1063308511
	.long	1063631111
	.long	1063960775
	.long	1064297658
	.long	1064641917
	.long	1064993715
	.long	0
	.long	2999887785
	.long	852465809
	.long	3003046475
	.long	2984291233
	.long	3001644133
	.long	854021668
	.long	2997748242
	.long	849550193
	.long	2995541347
	.long	851518274
	.long	809701978
	.long	2997656926
	.long	2996185864
	.long	2980965110
	.long	3002882728
	.long	844097402
	.long	848217591
	.long	2999013352
	.long	2992006718
	.long	831170615
	.long	3002278818
	.long	833158180
	.long	3000769962
	.long	2991891850
	.long	2999994908
	.long	2979965785
	.long	2982419430
	.long	2982221534
	.long	2999469642
	.long	833168438
	.long	2987538264
	.long	1056964608
	.long	1056605107
	.long	1056253309
	.long	1055909050
	.long	1055572167
	.long	1055242503
	.long	1054919903
	.long	1054604216
	.long	1054295293
	.long	1053992990
	.long	1053697164
	.long	1053407678
	.long	1053124394
	.long	1052847181
	.long	1052575908
	.long	1052310447
	.long	1052050675
	.long	1051796470
	.long	1051547711
	.long	1051304283
	.long	1051066071
	.long	1050832963
	.long	1050604850
	.long	1050381626
	.long	1050163184
	.long	1049949424
	.long	1049740243
	.long	1049535546
	.long	1049335234
	.long	1049139215
	.long	1048947395
	.long	1048759687
	.long	0
	.long	2979149656
	.long	824779830
	.long	2991081034
	.long	2973832926
	.long	2974030822
	.long	2971577177
	.long	2991606300
	.long	2983503242
	.long	2992381354
	.long	824769572
	.long	2993890210
	.long	822782007
	.long	2983618110
	.long	2990624744
	.long	839828983
	.long	835708794
	.long	2994494120
	.long	2972576502
	.long	2987797256
	.long	2989268318
	.long	801313370
	.long	843129666
	.long	2987152739
	.long	841161585
	.long	2989359634
	.long	845633060
	.long	2993255525
	.long	2975902625
	.long	2994657867
	.long	844077201
	.long	2991499177
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
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	1220542465
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
	.long	1118743631
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
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1056964676
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
	.long	1042983605
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
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
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
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	1262485504
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
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
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.type	__svml_ssinh_data_internal,@object
	.size	__svml_ssinh_data_internal,1920
	.align 32
__ssinh_la_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	1048019041
	.long	1072704666
	.long	1398474845
	.long	3161559171
	.long	3541402996
	.long	1072716208
	.long	2759177317
	.long	1015903202
	.long	410360776
	.long	1072727877
	.long	1269990655
	.long	1013024446
	.long	1828292879
	.long	1072739672
	.long	1255956747
	.long	1016636974
	.long	852742562
	.long	1072751596
	.long	667253587
	.long	1010842135
	.long	3490863953
	.long	1072763649
	.long	960797498
	.long	3163997456
	.long	2930322912
	.long	1072775834
	.long	2599499422
	.long	3163762623
	.long	1014845819
	.long	1072788152
	.long	3117910646
	.long	3162607681
	.long	3949972341
	.long	1072800603
	.long	2068408548
	.long	1015962444
	.long	828946858
	.long	1072813191
	.long	10642492
	.long	1016988014
	.long	2288159958
	.long	1072825915
	.long	2169144469
	.long	1015924597
	.long	1853186616
	.long	1072838778
	.long	3066496371
	.long	1016705150
	.long	1709341917
	.long	1072851781
	.long	2571168217
	.long	1015201075
	.long	4112506593
	.long	1072864925
	.long	2947355221
	.long	1015419624
	.long	2799960843
	.long	1072878213
	.long	1423655381
	.long	1016070727
	.long	171030293
	.long	1072891646
	.long	3526460132
	.long	1015477354
	.long	2992903935
	.long	1072905224
	.long	2218154406
	.long	1016276769
	.long	926591435
	.long	1072918951
	.long	3208833762
	.long	3163962090
	.long	887463927
	.long	1072932827
	.long	3596744163
	.long	3161842742
	.long	1276261410
	.long	1072946854
	.long	300981948
	.long	1015732745
	.long	569847338
	.long	1072961034
	.long	472945272
	.long	3160339305
	.long	1617004845
	.long	1072975368
	.long	82804944
	.long	1011391354
	.long	3049340112
	.long	1072989858
	.long	3062915824
	.long	1014219171
	.long	3577096743
	.long	1073004506
	.long	2951496418
	.long	1014842263
	.long	1990012071
	.long	1073019314
	.long	3529070563
	.long	3163861769
	.long	1453150082
	.long	1073034283
	.long	498154669
	.long	3162536638
	.long	917841882
	.long	1073049415
	.long	18715565
	.long	1016707884
	.long	3712504873
	.long	1073064711
	.long	88491949
	.long	1016476236
	.long	363667784
	.long	1073080175
	.long	813753950
	.long	1016833785
	.long	2956612997
	.long	1073095806
	.long	2118169751
	.long	3163784129
	.long	2186617381
	.long	1073111608
	.long	2270764084
	.long	3164321289
	.long	1719614413
	.long	1073127582
	.long	330458198
	.long	3164331316
	.long	1013258799
	.long	1073143730
	.long	1748797611
	.long	3161177658
	.long	3907805044
	.long	1073160053
	.long	2257091225
	.long	3162598983
	.long	1447192521
	.long	1073176555
	.long	1462857171
	.long	3163563097
	.long	1944781191
	.long	1073193236
	.long	3993278767
	.long	3162772855
	.long	919555682
	.long	1073210099
	.long	3121969534
	.long	1013996802
	.long	2571947539
	.long	1073227145
	.long	3558159064
	.long	3164425245
	.long	2604962541
	.long	1073244377
	.long	2614425274
	.long	3164587768
	.long	1110089947
	.long	1073261797
	.long	1451641639
	.long	1016523249
	.long	2568320822
	.long	1073279406
	.long	2732824428
	.long	1015401491
	.long	2966275557
	.long	1073297207
	.long	2176155324
	.long	3160891335
	.long	2682146384
	.long	1073315202
	.long	2082178513
	.long	3164411995
	.long	2191782032
	.long	1073333393
	.long	2960257726
	.long	1014791238
	.long	2069751141
	.long	1073351782
	.long	1562170675
	.long	3163773257
	.long	2990417245
	.long	1073370371
	.long	3683467745
	.long	3164417902
	.long	1434058175
	.long	1073389163
	.long	251133233
	.long	1016134345
	.long	2572866477
	.long	1073408159
	.long	878562433
	.long	1016570317
	.long	3092190715
	.long	1073427362
	.long	814012168
	.long	3160571998
	.long	4076559943
	.long	1073446774
	.long	2119478331
	.long	3161806927
	.long	2420883922
	.long	1073466398
	.long	2049810052
	.long	1015168464
	.long	3716502172
	.long	1073486235
	.long	2303740125
	.long	1015091301
	.long	777507147
	.long	1073506289
	.long	4282924205
	.long	1016236109
	.long	3706687593
	.long	1073526560
	.long	3521726939
	.long	1014301643
	.long	1242007932
	.long	1073547053
	.long	1132034716
	.long	3164388407
	.long	3707479175
	.long	1073567768
	.long	3613079303
	.long	1015213314
	.long	64696965
	.long	1073588710
	.long	1768797490
	.long	1016865536
	.long	863738719
	.long	1073609879
	.long	1326992220
	.long	3163661773
	.long	3884662774
	.long	1073631278
	.long	2158611599
	.long	1015258761
	.long	2728693978
	.long	1073652911
	.long	396109971
	.long	3164511267
	.long	3999357479
	.long	1073674779
	.long	2258941616
	.long	1016973300
	.long	1533953344
	.long	1073696886
	.long	769171851
	.long	1016714209
	.long	2174652632
	.long	1073719233
	.long	4087714590
	.long	1015498835
	.long	0
	.long	1073741824
	.long	0
	.long	0
	.long	0
	.long	1071644672
	.long	1431652600
	.long	1069897045
	.long	1431670732
	.long	1067799893
	.long	984555731
	.long	1065423122
	.long	472530941
	.long	1062650218
	.long	1431655765
	.long	1069897045
	.long	286331153
	.long	1065423121
	.long	436314138
	.long	1059717536
	.long	2773927732
	.long	1053236707
	.long	1697350398
	.long	1079448903
	.long	0
	.long	1127743488
	.long	33554432
	.long	1101004800
	.long	2684354560
	.long	1079401119
	.long	4277796864
	.long	1065758274
	.long	3164486458
	.long	1025308570
	.long	1
	.long	1048576
	.long	4294967295
	.long	2146435071
	.long	3671843104
	.long	1067178892
	.long	3875694624
	.long	1077247184
	.type	__ssinh_la_CoutTab,@object
	.size	__ssinh_la_CoutTab,1192
	.space 8, 0x00 	
	.align 16
.L_2il0floatpacket.98:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.98,@object
	.size	.L_2il0floatpacket.98,16
