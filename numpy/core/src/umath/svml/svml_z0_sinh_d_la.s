/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/


	.text
.L_2__routine_start___svml_sinh8_z0_0:

	.align    16,0x90
	.globl __svml_sinh8

__svml_sinh8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        lea       1608+__svml_dsinh_data_internal(%rip), %rax
        vmovaps   %zmm0, %zmm8

/* ............... Abs argument ............................ */
        vandpd    1408+__svml_dsinh_data_internal(%rip), %zmm8, %zmm7
        vmovups   3648+__svml_dsinh_data_internal(%rip), %zmm13

/*
 * ............... Load argument ............................
 * dM = x*2^K/log(2) + RShifter
 */
        vmovups   1216+__svml_dsinh_data_internal(%rip), %zmm12
        vmovups   1280+__svml_dsinh_data_internal(%rip), %zmm14
        vmovups   4032+__svml_dsinh_data_internal(%rip), %zmm6

/* VLOAD_CONST( D, dPC[0],         TAB._dPC1 ); */
        vmovups   3968+__svml_dsinh_data_internal(%rip), %zmm4
        vxorpd    %zmm8, %zmm7, %zmm5
        kxnorw    %k0, %k0, %k1
        kxnorw    %k0, %k0, %k2
        vfmadd213pd {rn-sae}, %zmm13, %zmm5, %zmm12

/*
 * ...............Check for overflow\underflow .............
 * 
 */
        vpsrlq    $32, %zmm5, %zmm9

/*
 * ................... R ...................................
 * dN = dM - RShifter
 */
        vsubpd    {rn-sae}, %zmm13, %zmm12, %zmm2
        vpmovqd   %zmm9, %ymm10
        vmovups   1344+__svml_dsinh_data_internal(%rip), %zmm9

/* dR = dX - dN*Log2_hi/2^K */
        vfnmadd231pd {rn-sae}, %zmm14, %zmm2, %zmm5

/*
 * ....sinh(r) = r*((a1=1)+r^2*(a3+r^2*a5)) = r + r*(r^2*(a3+r^2*a5)) ....
 * dSinh_r = (a3+r^2*a5)
 */
        vmovups   3904+__svml_dsinh_data_internal(%rip), %zmm14

/* dR = (dX - dN*Log2_hi/2^K) - dN*Log2_lo/2^K */
        vfnmadd231pd {rn-sae}, %zmm9, %zmm2, %zmm5
        vpcmpgtd  3712+__svml_dsinh_data_internal(%rip), %ymm10, %ymm11
        vmovmskps %ymm11, %edx

/* dR2 = dR^2 */
        vmulpd    {rn-sae}, %zmm5, %zmm5, %zmm2
        vfmadd231pd {rn-sae}, %zmm2, %zmm6, %zmm14

/*
 * .............. Index and lookup .........................
 * j
 */
        vpandq    4224+__svml_dsinh_data_internal(%rip), %zmm12, %zmm15
        vpsllq    $4, %zmm15, %zmm1
        vpmovqd   %zmm1, %ymm0
        vpxord    %zmm11, %zmm11, %zmm11
        vpxord    %zmm10, %zmm10, %zmm10
        vgatherdpd (%rax,%ymm0), %zmm11{%k1}
        vgatherdpd -8(%rax,%ymm0), %zmm10{%k2}

/* split j and N */
        vpxorq    %zmm15, %zmm12, %zmm3

/*
 * ............... G1,G2,G3: dTdif,dTn * 2^N,2^(-N) ...........
 * lM now is an EXP(2^N)
 */
        vpsllq    $45, %zmm3, %zmm3
        vpaddq    %zmm3, %zmm10, %zmm1

/*  */
        vpaddq    %zmm3, %zmm11, %zmm12

/*  */
        vpsubq    %zmm3, %zmm11, %zmm13

/* dSinh_r = r^2*(a3+r^2*a5) */
        vmulpd    {rn-sae}, %zmm2, %zmm14, %zmm3

/* dG2 = dTn*2^N - dTn*2^-N */
        vsubpd    {rn-sae}, %zmm13, %zmm12, %zmm15

/* dG3 = dTn*2^N + dTn*2^-N */
        vaddpd    {rn-sae}, %zmm13, %zmm12, %zmm0

/* dSinh_r = r + r*(r^2*(a3+r^2*a5)) */
        vfmadd213pd {rn-sae}, %zmm5, %zmm5, %zmm3

/*
 * poly(r) = (dG2+dG1)+dG3*sinh(dR)+dG1*sinh(dR)+(dG1+dG2)*dR2*(a2 +a4*dR2)
 * dOut = (a2 +a4*dR2)
 */
        vmovups   3840+__svml_dsinh_data_internal(%rip), %zmm5

/* dG1 += dG3 */
        vaddpd    {rn-sae}, %zmm0, %zmm1, %zmm6
        vfmadd231pd {rn-sae}, %zmm2, %zmm4, %zmm5

/* dOut = dR2*(a2 +a4*dR2) */
        vmulpd    {rn-sae}, %zmm2, %zmm5, %zmm4

/* dG2 += dG1 */
        vaddpd    {rn-sae}, %zmm15, %zmm1, %zmm2

/* dOut = dG2*dR2*(a2 +a4*dR2) */
        vmulpd    {rn-sae}, %zmm2, %zmm4, %zmm4

/* dOut = dG1*sinh(dR)+dG2*dR2*(a2 +a4*dR2) */
        vfmadd213pd {rn-sae}, %zmm4, %zmm6, %zmm3

/* dOut = dG2 + dG1*sinh(dR)+dG2*dR2*(a2 +a4*dR2) */
        vaddpd    {rn-sae}, %zmm2, %zmm3, %zmm0

/* ................... Ret H ...................... */
        vorpd     %zmm0, %zmm7, %zmm0
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

        vmovups   %zmm8, 64(%rsp)
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
        cmpl      $8, %r12d
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

        lea       64(%rsp,%r12,8), %rdi
        lea       128(%rsp,%r12,8), %rsi

        call      __svml_dsinh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sinh8,@function
	.size	__svml_sinh8,.-__svml_sinh8
..LN__svml_sinh8.0:

.L_2__routine_start___svml_dsinh_cout_rare_internal_1:

	.align    16,0x90

__svml_dsinh_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      %rsi, %r9
        movzwl    6(%rdi), %ecx
        xorl      %eax, %eax
        andl      $32752, %ecx
        shrl      $4, %ecx
        movsd     (%rdi), %xmm2
        movb      7(%rdi), %dl
        movsd     %xmm2, -8(%rsp)
        cmpl      $2047, %ecx
        je        .LBL_2_17


        testl     %ecx, %ecx
        jle       .LBL_2_16


        andb      $127, %dl
        movsd     %xmm2, -32(%rsp)
        movb      %dl, -25(%rsp)
        cmpl      $969, %ecx
        jle       .LBL_2_14


        movsd     -32(%rsp), %xmm0
        movsd     1136+__dsinh_la_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_13


        movsd     1184+__dsinh_la_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_9


        comisd    1176+__dsinh_la_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movsd     1112+__dsinh_la_CoutTab(%rip), %xmm1
        lea       __dsinh_la_CoutTab(%rip), %rcx
        mulsd     %xmm0, %xmm1
        movsd     1144+__dsinh_la_CoutTab(%rip), %xmm4
        movq      8+__dsinh_la_CoutTab(%rip), %r10
        movq      %r10, %rsi
        shrq      $48, %rsi
        addsd     1120+__dsinh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        andl      $-32753, %esi
        movsd     -40(%rsp), %xmm10
        movl      -40(%rsp), %r8d
        movl      %r8d, %r11d
        shrl      $6, %r11d
        andl      $63, %r8d
        movq      %r10, -16(%rsp)
        subsd     1120+__dsinh_la_CoutTab(%rip), %xmm10
        mulsd     %xmm10, %xmm4
        lea       1023(%r11), %edi
        xorps     .L_2il0floatpacket.97(%rip), %xmm10
        addl      $1022, %r11d
        mulsd     1152+__dsinh_la_CoutTab(%rip), %xmm10
        subsd     %xmm4, %xmm0
        movaps    %xmm10, %xmm2
        movaps    %xmm0, %xmm8
        andl      $2047, %r11d
        lea       (%r8,%r8), %edx
        negl      %edi
        lea       1(%r8,%r8), %r8d
        movsd     (%rcx,%rdx,8), %xmm9
        negl      %edx
        shll      $4, %r11d
        addl      $-4, %edi
        orl       %r11d, %esi
        andl      $2047, %edi
        movw      %si, -10(%rsp)
        andl      $-32753, %esi
        shll      $4, %edi
        addsd     %xmm0, %xmm2
        movsd     %xmm2, -24(%rsp)
        orl       %edi, %esi
        movsd     -24(%rsp), %xmm6
        movsd     1128+__dsinh_la_CoutTab(%rip), %xmm2
        subsd     %xmm6, %xmm8
        movsd     %xmm8, -56(%rsp)
        movsd     -24(%rsp), %xmm12
        movsd     -56(%rsp), %xmm11
        movsd     (%rcx,%r8,8), %xmm7
        addsd     %xmm11, %xmm12
        movsd     %xmm12, -48(%rsp)
        movsd     -56(%rsp), %xmm5
        addsd     %xmm5, %xmm10
        movsd     -16(%rsp), %xmm5
        mulsd     %xmm5, %xmm9
        mulsd     %xmm5, %xmm7
        movsd     %xmm10, -56(%rsp)
        movaps    %xmm9, %xmm10
        movsd     -48(%rsp), %xmm13
        movw      %si, -10(%rsp)
        lea       128(%rdx), %esi
        movsd     -16(%rsp), %xmm14
        addl      $129, %edx
        subsd     %xmm13, %xmm0
        movsd     %xmm0, -48(%rsp)
        movsd     -56(%rsp), %xmm1
        movsd     -48(%rsp), %xmm15
        addsd     %xmm15, %xmm1
        movsd     %xmm1, -48(%rsp)
        movsd     -24(%rsp), %xmm4
        mulsd     %xmm4, %xmm2
        movaps    %xmm4, %xmm3
        mulsd     %xmm4, %xmm3
        movsd     -48(%rsp), %xmm1
        movaps    %xmm4, %xmm0
        movsd     %xmm2, -24(%rsp)
        movsd     -24(%rsp), %xmm6
        movsd     1064+__dsinh_la_CoutTab(%rip), %xmm2
        subsd     %xmm4, %xmm6
        mulsd     %xmm3, %xmm2
        movsd     %xmm6, -56(%rsp)
        movsd     -24(%rsp), %xmm11
        movsd     -56(%rsp), %xmm8
        subsd     %xmm8, %xmm11
        addsd     1048+__dsinh_la_CoutTab(%rip), %xmm2
        movsd     %xmm11, -24(%rsp)
        movsd     1072+__dsinh_la_CoutTab(%rip), %xmm11
        mulsd     %xmm3, %xmm11
        mulsd     %xmm3, %xmm2
        addsd     1056+__dsinh_la_CoutTab(%rip), %xmm11
        mulsd     %xmm4, %xmm2
        mulsd     %xmm3, %xmm11
        movsd     -24(%rsp), %xmm12
        addsd     1040+__dsinh_la_CoutTab(%rip), %xmm11
        subsd     %xmm12, %xmm0
        mulsd     %xmm3, %xmm11
        movsd     (%rcx,%rsi,8), %xmm3
        movaps    %xmm9, %xmm12
        mulsd     %xmm14, %xmm3
        movsd     %xmm0, -56(%rsp)
        subsd     %xmm3, %xmm10
        movsd     -24(%rsp), %xmm8
        movsd     -56(%rsp), %xmm6
        movsd     %xmm10, -24(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     (%rcx,%rdx,8), %xmm0
        subsd     %xmm13, %xmm12
        mulsd     %xmm14, %xmm0
        subsd     %xmm3, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -24(%rsp), %xmm14
        movsd     -56(%rsp), %xmm12
        movaps    %xmm14, %xmm5
        movb      -1(%rsp), %cl
        addsd     %xmm7, %xmm12
        addsd     %xmm0, %xmm7
        subsd     %xmm0, %xmm12
        andb      $-128, %cl
        addsd     %xmm12, %xmm5
        movsd     %xmm5, -24(%rsp)
        movaps    %xmm9, %xmm5
        movsd     -24(%rsp), %xmm15
        addsd     %xmm3, %xmm5
        subsd     %xmm15, %xmm14
        addsd     %xmm14, %xmm12
        movsd     %xmm12, -56(%rsp)
        movsd     -24(%rsp), %xmm10
        movsd     -56(%rsp), %xmm12
        movsd     %xmm5, -24(%rsp)
        movsd     -24(%rsp), %xmm13
        subsd     %xmm13, %xmm9
        addsd     %xmm9, %xmm3
        movsd     %xmm3, -56(%rsp)
        movsd     -24(%rsp), %xmm3
        movsd     -56(%rsp), %xmm5
        addsd     %xmm7, %xmm5
        movaps    %xmm3, %xmm7
        addsd     %xmm5, %xmm7
        movsd     %xmm7, -24(%rsp)
        movsd     -24(%rsp), %xmm9
        movsd     1128+__dsinh_la_CoutTab(%rip), %xmm7
        subsd     %xmm9, %xmm3
        addsd     %xmm3, %xmm5
        movsd     %xmm5, -56(%rsp)
        movsd     -24(%rsp), %xmm0
        mulsd     %xmm0, %xmm7
        movsd     -56(%rsp), %xmm3
        movsd     %xmm7, -24(%rsp)
        movaps    %xmm0, %xmm7
        movsd     -24(%rsp), %xmm14
        mulsd     %xmm3, %xmm4
        subsd     %xmm0, %xmm14
        movsd     %xmm14, -56(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm9
        subsd     %xmm9, %xmm13
        movsd     %xmm13, -24(%rsp)
        movaps    %xmm0, %xmm13
        movsd     -24(%rsp), %xmm15
        mulsd     %xmm2, %xmm13
        subsd     %xmm15, %xmm7
        mulsd     %xmm1, %xmm0
        mulsd     %xmm3, %xmm1
        mulsd     %xmm3, %xmm2
        movaps    %xmm10, %xmm15
        movaps    %xmm13, %xmm5
        mulsd     %xmm11, %xmm15
        mulsd     %xmm12, %xmm11
        addsd     %xmm15, %xmm5
        addsd     %xmm12, %xmm11
        movsd     %xmm7, -56(%rsp)
        addsd     %xmm0, %xmm11
        movsd     -24(%rsp), %xmm9
        addsd     %xmm1, %xmm11
        movsd     -56(%rsp), %xmm7
        addsd     %xmm2, %xmm11
        movsd     %xmm5, -24(%rsp)
        addsd     %xmm4, %xmm11
        movsd     -24(%rsp), %xmm14
        subsd     %xmm14, %xmm13
        addsd     %xmm13, %xmm15
        movsd     %xmm15, -56(%rsp)
        movaps    %xmm8, %xmm15
        mulsd     %xmm9, %xmm15
        mulsd     %xmm6, %xmm9
        mulsd     %xmm7, %xmm6
        mulsd     %xmm7, %xmm8
        movsd     -24(%rsp), %xmm14
        movaps    %xmm15, %xmm13
        movsd     -56(%rsp), %xmm5
        addsd     %xmm14, %xmm13
        addsd     %xmm9, %xmm5
        movsd     %xmm13, -24(%rsp)
        addsd     %xmm6, %xmm5
        movsd     -24(%rsp), %xmm13
        addsd     %xmm8, %xmm5
        subsd     %xmm13, %xmm15
        addsd     %xmm5, %xmm11
        addsd     %xmm15, %xmm14
        movsd     %xmm14, -56(%rsp)
        movaps    %xmm10, %xmm15
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm14
        addsd     %xmm13, %xmm15
        addsd     %xmm14, %xmm11
        movsd     %xmm15, -24(%rsp)
        movsd     -24(%rsp), %xmm15
        subsd     %xmm15, %xmm10
        addsd     %xmm10, %xmm13
        movsd     %xmm13, -56(%rsp)
        movsd     -24(%rsp), %xmm13
        movsd     -56(%rsp), %xmm10
        addsd     %xmm10, %xmm11
        addsd     %xmm11, %xmm13
        movsd     %xmm13, -32(%rsp)
        movb      -25(%rsp), %dil
        andb      $127, %dil
        orb       %cl, %dil
        movb      %dil, -25(%rsp)
        movq      -32(%rsp), %r10
        movq      %r10, (%r9)
        ret

.LBL_2_8:

        movaps    %xmm0, %xmm2
        mulsd     %xmm0, %xmm2
        movsd     1104+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        movb      -1(%rsp), %dl
        andb      $-128, %dl
        addsd     1096+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        addsd     1088+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm2, %xmm1
        addsd     1080+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm1, %xmm2
        mulsd     %xmm0, %xmm2
        addsd     %xmm2, %xmm0
        movsd     %xmm0, -32(%rsp)
        movb      -25(%rsp), %cl
        andb      $127, %cl
        orb       %dl, %cl
        movb      %cl, -25(%rsp)
        movq      -32(%rsp), %rsi
        movq      %rsi, (%r9)
        ret

.LBL_2_9:

        movsd     1112+__dsinh_la_CoutTab(%rip), %xmm1
        lea       __dsinh_la_CoutTab(%rip), %r8
        mulsd     %xmm0, %xmm1
        movsd     1144+__dsinh_la_CoutTab(%rip), %xmm2
        movsd     1152+__dsinh_la_CoutTab(%rip), %xmm3
        movq      8+__dsinh_la_CoutTab(%rip), %rdx
        movq      %rdx, -16(%rsp)
        addsd     1120+__dsinh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm4
        movsd     1072+__dsinh_la_CoutTab(%rip), %xmm1
        movl      -40(%rsp), %edx
        movl      %edx, %esi
        andl      $63, %esi
        subsd     1120+__dsinh_la_CoutTab(%rip), %xmm4
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
        addsd     1064+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1056+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1048+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1040+__dsinh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        addsd     %xmm0, %xmm1
        mulsd     %xmm5, %xmm1
        addsd     (%r8,%rdi,8), %xmm1
        addsd     %xmm5, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_11


        movq      8+__dsinh_la_CoutTab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -10(%rsp)
        movsd     -16(%rsp), %xmm0
        mulsd     %xmm0, %xmm1
        movsd     %xmm1, -32(%rsp)
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
        mulsd     %xmm0, %xmm1
        mulsd     1024+__dsinh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -32(%rsp)

.LBL_2_12:

        movb      -25(%rsp), %cl
        movb      -1(%rsp), %dl
        andb      $127, %cl
        andb      $-128, %dl
        orb       %dl, %cl
        movb      %cl, -25(%rsp)
        movq      -32(%rsp), %rsi
        movq      %rsi, (%r9)
        ret

.LBL_2_13:

        movsd     1168+__dsinh_la_CoutTab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm2, %xmm0
        movsd     %xmm0, (%r9)
        ret

.LBL_2_14:

        movsd     __dsinh_la_CoutTab(%rip), %xmm0
        addsd     1160+__dsinh_la_CoutTab(%rip), %xmm0
        mulsd     %xmm2, %xmm0
        movsd     %xmm0, (%r9)


        ret

.LBL_2_16:

        movsd     1160+__dsinh_la_CoutTab(%rip), %xmm0
        mulsd     %xmm0, %xmm2
        movsd     %xmm2, -24(%rsp)
        movsd     -24(%rsp), %xmm1
        movq      8+__dsinh_la_CoutTab(%rip), %rdx
        movq      %rdx, -16(%rsp)
        addsd     -8(%rsp), %xmm1
        movsd     %xmm1, (%r9)
        ret

.LBL_2_17:

        addsd     %xmm2, %xmm2
        movsd     %xmm2, (%r9)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dsinh_cout_rare_internal,@function
	.size	__svml_dsinh_cout_rare_internal,.-__svml_dsinh_cout_rare_internal
..LN__svml_dsinh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dsinh_data_internal:
	.long	0
	.long	1071644672
	.long	1828292879
	.long	1071691096
	.long	1014845819
	.long	1071739576
	.long	1853186616
	.long	1071790202
	.long	171030293
	.long	1071843070
	.long	1276261410
	.long	1071898278
	.long	3577096743
	.long	1071955930
	.long	3712504873
	.long	1072016135
	.long	1719614413
	.long	1072079006
	.long	1944781191
	.long	1072144660
	.long	1110089947
	.long	1072213221
	.long	2191782032
	.long	1072284817
	.long	2572866477
	.long	1072359583
	.long	3716502172
	.long	1072437659
	.long	3707479175
	.long	1072519192
	.long	2728693978
	.long	1072604335
	.long	0
	.long	0
	.long	1255956747
	.long	1015588398
	.long	3117910646
	.long	3161559105
	.long	3066496371
	.long	1015656574
	.long	3526460132
	.long	1014428778
	.long	300981948
	.long	1014684169
	.long	2951496418
	.long	1013793687
	.long	88491949
	.long	1015427660
	.long	330458198
	.long	3163282740
	.long	3993278767
	.long	3161724279
	.long	1451641639
	.long	1015474673
	.long	2960257726
	.long	1013742662
	.long	878562433
	.long	1015521741
	.long	2303740125
	.long	1014042725
	.long	3613079303
	.long	1014164738
	.long	396109971
	.long	3163462691
	.long	0
	.long	1071644672
	.long	2728693978
	.long	1071555759
	.long	3707479175
	.long	1071470616
	.long	3716502172
	.long	1071389083
	.long	2572866477
	.long	1071311007
	.long	2191782032
	.long	1071236241
	.long	1110089947
	.long	1071164645
	.long	1944781191
	.long	1071096084
	.long	1719614413
	.long	1071030430
	.long	3712504873
	.long	1070967559
	.long	3577096743
	.long	1070907354
	.long	1276261410
	.long	1070849702
	.long	171030293
	.long	1070794494
	.long	1853186616
	.long	1070741626
	.long	1014845819
	.long	1070691000
	.long	1828292879
	.long	1070642520
	.long	0
	.long	0
	.long	396109971
	.long	3162414115
	.long	3613079303
	.long	1013116162
	.long	2303740125
	.long	1012994149
	.long	878562433
	.long	1014473165
	.long	2960257726
	.long	1012694086
	.long	1451641639
	.long	1014426097
	.long	3993278767
	.long	3160675703
	.long	330458198
	.long	3162234164
	.long	88491949
	.long	1014379084
	.long	2951496418
	.long	1012745111
	.long	300981948
	.long	1013635593
	.long	3526460132
	.long	1013380202
	.long	3066496371
	.long	1014607998
	.long	3117910646
	.long	3160510529
	.long	1255956747
	.long	1014539822
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	0
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1
	.long	1123549184
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	1082453555
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	15
	.long	0
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	1431657638
	.long	1069897045
	.long	1431657638
	.long	1069897045
	.long	1431657638
	.long	1069897045
	.long	1431657638
	.long	1069897045
	.long	1431657638
	.long	1069897045
	.long	1431657638
	.long	1069897045
	.long	1431657638
	.long	1069897045
	.long	1431657638
	.long	1069897045
	.long	1431653196
	.long	1067799893
	.long	1431653196
	.long	1067799893
	.long	1431653196
	.long	1067799893
	.long	1431653196
	.long	1067799893
	.long	1431653196
	.long	1067799893
	.long	1431653196
	.long	1067799893
	.long	1431653196
	.long	1067799893
	.long	1431653196
	.long	1067799893
	.long	249327322
	.long	1065423121
	.long	249327322
	.long	1065423121
	.long	249327322
	.long	1065423121
	.long	249327322
	.long	1065423121
	.long	249327322
	.long	1065423121
	.long	249327322
	.long	1065423121
	.long	249327322
	.long	1065423121
	.long	249327322
	.long	1065423121
	.long	419584011
	.long	1062650220
	.long	419584011
	.long	1062650220
	.long	419584011
	.long	1062650220
	.long	419584011
	.long	1062650220
	.long	419584011
	.long	1062650220
	.long	419584011
	.long	1062650220
	.long	419584011
	.long	1062650220
	.long	419584011
	.long	1062650220
	.long	100753094
	.long	1059717741
	.long	100753094
	.long	1059717741
	.long	100753094
	.long	1059717741
	.long	100753094
	.long	1059717741
	.long	100753094
	.long	1059717741
	.long	100753094
	.long	1059717741
	.long	100753094
	.long	1059717741
	.long	100753094
	.long	1059717741
	.long	461398617
	.long	1056571820
	.long	461398617
	.long	1056571820
	.long	461398617
	.long	1056571820
	.long	461398617
	.long	1056571820
	.long	461398617
	.long	1056571820
	.long	461398617
	.long	1056571820
	.long	461398617
	.long	1056571820
	.long	461398617
	.long	1056571820
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	1697350398
	.long	1073157447
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	4277796864
	.long	1072049730
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	3164486458
	.long	1031600026
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
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
	.long	0
	.long	0
	.long	0
	.long	1071644672
	.long	431824500
	.long	1064709706
	.long	730821105
	.long	1071633346
	.long	1779301686
	.long	1065758303
	.long	2174652632
	.long	1071622081
	.long	872681311
	.long	1066443490
	.long	2912730644
	.long	1071610877
	.long	2882296449
	.long	1066806964
	.long	1533953344
	.long	1071599734
	.long	3305321028
	.long	1067170481
	.long	929806999
	.long	1071588651
	.long	340716357
	.long	1067492210
	.long	3999357479
	.long	1071577627
	.long	1073477808
	.long	1067674027
	.long	764307441
	.long	1071566664
	.long	3128166954
	.long	1067855881
	.long	2728693978
	.long	1071555759
	.long	3639221082
	.long	1068037778
	.long	4224142467
	.long	1071544913
	.long	4041415279
	.long	1068219723
	.long	3884662774
	.long	1071534126
	.long	1480599658
	.long	1068401722
	.long	351641897
	.long	1071523398
	.long	2997120266
	.long	1068541361
	.long	863738719
	.long	1071512727
	.long	928693471
	.long	1068632422
	.long	4076975200
	.long	1071502113
	.long	1708958952
	.long	1068723517
	.long	64696965
	.long	1071491558
	.long	3926287402
	.long	1068814649
	.long	382305176
	.long	1071481059
	.long	1878784442
	.long	1068905822
	.long	3707479175
	.long	1071470616
	.long	2754496392
	.long	1068997037
	.long	135105010
	.long	1071460231
	.long	861943228
	.long	1069088298
	.long	1242007932
	.long	1071449901
	.long	3400259254
	.long	1069179606
	.long	1432208378
	.long	1071439627
	.long	394759087
	.long	1069270966
	.long	3706687593
	.long	1071429408
	.long	3351980561
	.long	1069362378
	.long	2483480501
	.long	1071419245
	.long	2310349189
	.long	1069453847
	.long	777507147
	.long	1071409137
	.long	200254151
	.long	1069545375
	.long	1610600570
	.long	1071399083
	.long	4274709417
	.long	1069592241
	.long	3716502172
	.long	1071389083
	.long	2266782956
	.long	1069638068
	.long	1540824585
	.long	1071379138
	.long	1995599824
	.long	1069683928
	.long	2420883922
	.long	1071369246
	.long	647201135
	.long	1069729823
	.long	815859274
	.long	1071359408
	.long	4001939191
	.long	1069775753
	.long	4076559943
	.long	1071349622
	.long	664843213
	.long	1069821722
	.long	2380618042
	.long	1071339890
	.long	720494647
	.long	1069867729
	.long	3092190715
	.long	1071330210
	.long	1373458573
	.long	1069913776
	.long	697153126
	.long	1071320583
	.long	4128322810
	.long	1069959864
	.long	2572866477
	.long	1071311007
	.long	1904966097
	.long	1070005996
	.long	3218338682
	.long	1071301483
	.long	513564799
	.long	1070052172
	.long	1434058175
	.long	1071292011
	.long	1474894098
	.long	1070098393
	.long	321958744
	.long	1071282590
	.long	2020498546
	.long	1070144661
	.long	2990417245
	.long	1071273219
	.long	3682797359
	.long	1070190977
	.long	3964284211
	.long	1071263899
	.long	3705320722
	.long	1070237343
	.long	2069751141
	.long	1071254630
	.long	3632815436
	.long	1070283760
	.long	434316067
	.long	1071245411
	.long	721481577
	.long	1070330230
	.long	2191782032
	.long	1071236241
	.long	824045819
	.long	1070376753
	.long	1892288442
	.long	1071227121
	.long	1210063881
	.long	1070423331
	.long	2682146384
	.long	1071218050
	.long	3450994238
	.long	1070469965
	.long	3418903055
	.long	1071209028
	.long	535468266
	.long	1070516658
	.long	2966275557
	.long	1071200055
	.long	2639266259
	.long	1070563409
	.long	194117574
	.long	1071191131
	.long	3530359402
	.long	1070603158
	.long	2568320822
	.long	1071182254
	.long	3405407025
	.long	1070626595
	.long	380978316
	.long	1071173426
	.long	1749136243
	.long	1070650064
	.long	1110089947
	.long	1071164645
	.long	1517376385
	.long	1070673565
	.long	3649726105
	.long	1071155911
	.long	1375061601
	.long	1070697099
	.long	2604962541
	.long	1071147225
	.long	4286252247
	.long	1070720666
	.long	1176749997
	.long	1071138586
	.long	334352625
	.long	1070744269
	.long	2571947539
	.long	1071129993
	.long	1081936396
	.long	1070767906
	.long	1413356050
	.long	1071121447
	.long	916128083
	.long	1070791579
	.long	919555682
	.long	1071112947
	.long	2818494152
	.long	1070815288
	.long	19972402
	.long	1071104493
	.long	1185261260
	.long	1070839035
	.long	1944781191
	.long	1071096084
	.long	3302240303
	.long	1070862819
	.long	1339972927
	.long	1071087721
	.long	3575110344
	.long	1070886642
	.long	1447192521
	.long	1071079403
	.long	709375646
	.long	1070910505
	.long	1218806132
	.long	1071071130
	.long	2005420950
	.long	1070934407
	.long	3907805044
	.long	1071062901
	.long	1883763132
	.long	1070958350
	.long	4182873220
	.long	1071054717
	.long	3359975962
	.long	1070982334
	.long	1013258799
	.long	1071046578
	.long	864909351
	.long	1071006361
	.long	1963711167
	.long	1071038482
	.long	1719614413
	.long	1071030430
	.long	1719614413
	.long	1071030430
	.long	365628427
	.long	1071054543
	.long	3561793907
	.long	1071022421
	.long	4134867513
	.long	1071078699
	.long	2186617381
	.long	1071014456
	.long	3184944616
	.long	1071102901
	.long	885834528
	.long	1071006534
	.long	564029795
	.long	1071127149
	.long	2956612997
	.long	1070998654
	.long	3621005023
	.long	1071151442
	.long	3111574537
	.long	1070990817
	.long	2530717257
	.long	1071175783
	.long	363667784
	.long	1070983023
	.long	358839240
	.long	1071200172
	.long	2321106615
	.long	1070975270
	.long	177057508
	.long	1071224609
	.long	3712504873
	.long	1070967559
	.long	768195176
	.long	1071249095
	.long	3566716925
	.long	1070959890
	.long	921269482
	.long	1071273631
	.long	917841882
	.long	1070952263
	.long	3726549525
	.long	1071298217
	.long	3395129871
	.long	1070944676
	.long	3690744995
	.long	1071322855
	.long	1453150082
	.long	1070937131
	.long	3916966168
	.long	1071347545
	.long	2731501122
	.long	1070929626
	.long	3219913010
	.long	1071372288
	.long	1990012071
	.long	1070922162
	.long	420933669
	.long	1071397085
	.long	2583551245
	.long	1070914738
	.long	2938050448
	.long	1071421935
	.long	3577096743
	.long	1070907354
	.long	1016247609
	.long	1071446841
	.long	4040676318
	.long	1070900010
	.long	2087301532
	.long	1071471802
	.long	3049340112
	.long	1070892706
	.long	705101620
	.long	1071496820
	.long	3978100823
	.long	1070885441
	.long	20578973
	.long	1071521895
	.long	1617004845
	.long	1070878216
	.long	3191864199
	.long	1071547027
	.long	3645941911
	.long	1070871029
	.long	499478133
	.long	1071572219
	.long	569847338
	.long	1070863882
	.long	3706163032
	.long	1071597469
	.long	78413852
	.long	1070856773
	.long	3107302654
	.long	1071622780
	.long	1276261410
	.long	1070849702
	.long	945376945
	.long	1071646412
	.long	3272845541
	.long	1070842669
	.long	3773502825
	.long	1071659128
	.long	887463927
	.long	1070835675
	.long	3049734401
	.long	1071671876
	.long	1829099622
	.long	1070828718
	.long	379637879
	.long	1071684656
	.long	926591435
	.long	1070821799
	.long	1667720032
	.long	1071697467
	.long	1603444721
	.long	1070814917
	.long	4232573504
	.long	1071710310
	.long	2992903935
	.long	1070808072
	.long	1101891425
	.long	1071723187
	.long	4232894513
	.long	1070801264
	.long	2487351331
	.long	1071736096
	.long	171030293
	.long	1070794494
	.long	1424924347
	.long	1071749039
	.long	2839424854
	.long	1070787759
	.long	3839693941
	.long	1071762015
	.long	2799960843
	.long	1070781061
	.long	2776099954
	.long	1071775026
	.long	3504003472
	.long	1070774399
	.long	4167790294
	.long	1071788071
	.long	4112506593
	.long	1070767773
	.long	1067865225
	.long	1071801152
	.long	3790955393
	.long	1070761183
	.long	3713696612
	.long	1071814267
	.long	1709341917
	.long	1070754629
	.long	872270569
	.long	1071827419
	.long	1337108031
	.long	1070748110
	.long	2789908864
	.long	1071840606
	.long	1853186616
	.long	1070741626
	.long	2537611823
	.long	1071853830
	.long	2440944790
	.long	1070735177
	.long	1780910816
	.long	1071867091
	.long	2288159958
	.long	1070728763
	.long	2189982503
	.long	1071880389
	.long	586995997
	.long	1070722384
	.long	1144730516
	.long	1071893725
	.long	828946858
	.long	1070716039
	.long	324769168
	.long	1071907099
	.long	2214878420
	.long	1070709728
	.long	1414505409
	.long	1071920511
	.long	3949972341
	.long	1070703451
	.long	1808220925
	.long	1071933962
	.long	948735466
	.long	1070697209
	.long	3200056266
	.long	1071947452
	.long	1014845819
	.long	1070691000
	.long	2994125935
	.long	1071960982
	.long	3366293073
	.long	1070684824
	.long	2894502806
	.long	1071974552
	.long	2930322912
	.long	1070678682
	.long	315333498
	.long	1071988163
	.long	3228316108
	.long	1070672573
	.long	1265790390
	.long	1072001814
	.long	3490863953
	.long	1070666497
	.long	3170252699
	.long	1072015506
	.long	2952712987
	.long	1070660454
	.long	3458291493
	.long	1072029240
	.long	852742562
	.long	1070654444
	.long	3859687560
	.long	1072043016
	.long	728909815
	.long	1070648466
	.long	1814547538
	.long	1072056835
	.long	1828292879
	.long	1070642520
	.long	3358256687
	.long	1072070696
	.long	3402036099
	.long	1070636606
	.long	1646693443
	.long	1072084601
	.long	410360776
	.long	1070630725
	.long	2726084392
	.long	1072098549
	.long	702412510
	.long	1070624875
	.long	4058219142
	.long	1072112541
	.long	3541402996
	.long	1070619056
	.long	3110436433
	.long	1072126578
	.long	3899555717
	.long	1070613269
	.long	1650643112
	.long	1072140660
	.long	1048019041
	.long	1070607514
	.long	1452398678
	.long	1072154787
	.long	2851812149
	.long	1070601789
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	0
	.long	1120403456
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	1082531225
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	0
	.long	1072693248
	.long	4294966717
	.long	1071644671
	.long	4294966717
	.long	1071644671
	.long	4294966717
	.long	1071644671
	.long	4294966717
	.long	1071644671
	.long	4294966717
	.long	1071644671
	.long	4294966717
	.long	1071644671
	.long	4294966717
	.long	1071644671
	.long	4294966717
	.long	1071644671
	.long	1431655597
	.long	1069897045
	.long	1431655597
	.long	1069897045
	.long	1431655597
	.long	1069897045
	.long	1431655597
	.long	1069897045
	.long	1431655597
	.long	1069897045
	.long	1431655597
	.long	1069897045
	.long	1431655597
	.long	1069897045
	.long	1431655597
	.long	1069897045
	.long	3474379417
	.long	1067799893
	.long	3474379417
	.long	1067799893
	.long	3474379417
	.long	1067799893
	.long	3474379417
	.long	1067799893
	.long	3474379417
	.long	1067799893
	.long	3474379417
	.long	1067799893
	.long	3474379417
	.long	1067799893
	.long	3474379417
	.long	1067799893
	.long	1460859941
	.long	1065423121
	.long	1460859941
	.long	1065423121
	.long	1460859941
	.long	1065423121
	.long	1460859941
	.long	1065423121
	.long	1460859941
	.long	1065423121
	.long	1460859941
	.long	1065423121
	.long	1460859941
	.long	1065423121
	.long	1460859941
	.long	1065423121
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	65472
	.long	0
	.long	65472
	.long	0
	.long	65472
	.long	0
	.long	65472
	.long	0
	.long	65472
	.long	0
	.long	65472
	.long	0
	.long	65472
	.long	0
	.long	65472
	.long	0
	.long	127
	.long	0
	.long	127
	.long	0
	.long	127
	.long	0
	.long	127
	.long	0
	.long	127
	.long	0
	.long	127
	.long	0
	.long	127
	.long	0
	.long	127
	.long	0
	.type	__svml_dsinh_data_internal,@object
	.size	__svml_dsinh_data_internal,4288
	.space 320, 0x00 	
	.align 32
__dsinh_la_CoutTab:
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
	.long	2411329662
	.long	1082536910
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
	.type	__dsinh_la_CoutTab,@object
	.size	__dsinh_la_CoutTab,1192
	.space 8, 0x00 	
	.align 16
.L_2il0floatpacket.97:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.97,@object
	.size	.L_2il0floatpacket.97,16
