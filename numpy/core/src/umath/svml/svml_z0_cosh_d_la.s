/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/


	.text
.L_2__routine_start___svml_cosh8_z0_0:

	.align    16,0x90
	.globl __svml_cosh8

__svml_cosh8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   3648+__svml_dcosh_data_internal(%rip), %zmm11
        vmovups   384+__svml_dcosh_data_internal(%rip), %zmm15

/*
 * ............... Load argument ...........................
 * dM = x*2^K/log(2) + RShifter
 */
        vmovups   3008+__svml_dcosh_data_internal(%rip), %zmm4
        vmovups   3072+__svml_dcosh_data_internal(%rip), %zmm2
        vmovups   3136+__svml_dcosh_data_internal(%rip), %zmm3
        vmovups   832+__svml_dcosh_data_internal(%rip), %zmm8
        vmovups   768+__svml_dcosh_data_internal(%rip), %zmm9
        vmovups   512+__svml_dcosh_data_internal(%rip), %zmm7
        vmovups   576+__svml_dcosh_data_internal(%rip), %zmm6
        vmovaps   %zmm0, %zmm10

/* ............... Abs argument ............................ */
        vandnpd   %zmm10, %zmm11, %zmm5

/* .............. Index and lookup ......................... */
        vmovups   __svml_dcosh_data_internal(%rip), %zmm11
        vmovups   256+__svml_dcosh_data_internal(%rip), %zmm0
        vfmadd213pd {rn-sae}, %zmm15, %zmm5, %zmm4

/*
 * ...............Check for overflow\underflow .............
 * 
 */
        vpsrlq    $32, %zmm5, %zmm12

/* dN = dM - RShifter */
        vsubpd    {rn-sae}, %zmm15, %zmm4, %zmm1
        vpmovqd   %zmm12, %ymm13
        vpermt2pd 320+__svml_dcosh_data_internal(%rip), %zmm4, %zmm0
        vpermt2pd 64+__svml_dcosh_data_internal(%rip), %zmm4, %zmm11

/* dR = dX - dN*Log2_hi/2^K */
        vfnmadd231pd {rn-sae}, %zmm2, %zmm1, %zmm5

/*
 * poly(r) = Gmjp(1 + a2*r^2 + a4*r^4) + Gmjn*(r+ a3*r^3 +a5*r^5)       =
 * = Gmjp_h +Gmjp_l+ Gmjp*r^2*(a2 + a4*r^2) + Gmjn*(r+ r^3*(a3 +a5*r^2)
 */
        vmovups   704+__svml_dcosh_data_internal(%rip), %zmm12
        vpsllq    $48, %zmm4, %zmm2

/* dR = dX - dN*Log2_hi/2^K */
        vfnmadd231pd {rn-sae}, %zmm3, %zmm1, %zmm5
        vmulpd    {rn-sae}, %zmm5, %zmm5, %zmm1
        vfmadd231pd {rn-sae}, %zmm1, %zmm8, %zmm12
        vmovups   640+__svml_dcosh_data_internal(%rip), %zmm8
        vfmadd213pd {rn-sae}, %zmm6, %zmm1, %zmm12
        vfmadd231pd {rn-sae}, %zmm1, %zmm9, %zmm8
        vfmadd213pd {rn-sae}, %zmm7, %zmm1, %zmm8
        vpcmpgtd  3712+__svml_dcosh_data_internal(%rip), %ymm13, %ymm14
        vmovmskps %ymm14, %edx

/* dOut=r^2*(a2 + a4*r^2) */
        vmulpd    {rn-sae}, %zmm1, %zmm8, %zmm6

/* lM now is an EXP(2^N) */
        vpandq    3584+__svml_dcosh_data_internal(%rip), %zmm2, %zmm3
        vpaddq    %zmm3, %zmm11, %zmm4
        vpsubq    %zmm3, %zmm0, %zmm0
        vsubpd    {rn-sae}, %zmm0, %zmm4, %zmm14
        vaddpd    {rn-sae}, %zmm0, %zmm4, %zmm13

/* dM=r^2*(a3 +a5*r^2) */
        vmulpd    {rn-sae}, %zmm1, %zmm12, %zmm0
        vfmadd213pd {rn-sae}, %zmm13, %zmm13, %zmm6

/* dM= r + r^3*(a3 +a5*r^2) */
        vfmadd213pd {rn-sae}, %zmm5, %zmm5, %zmm0
        vfmadd213pd {rn-sae}, %zmm6, %zmm14, %zmm0
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

        vmovups   %zmm10, 64(%rsp)
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

        call      __svml_dcosh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_cosh8,@function
	.size	__svml_cosh8,.-__svml_cosh8
..LN__svml_cosh8.0:

.L_2__routine_start___svml_dcosh_cout_rare_internal_1:

	.align    16,0x90

__svml_dcosh_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      %rsi, %r8
        movzwl    6(%rdi), %edx
        xorl      %eax, %eax
        andl      $32752, %edx
        cmpl      $32752, %edx
        je        .LBL_2_12


        movq      (%rdi), %rdx
        movq      %rdx, -8(%rsp)
        shrq      $56, %rdx
        andl      $127, %edx
        movb      %dl, -1(%rsp)
        movzwl    -2(%rsp), %ecx
        andl      $32752, %ecx
        cmpl      $15504, %ecx
        jle       .LBL_2_10


        movsd     -8(%rsp), %xmm0
        movsd     1096+__dcosh_la_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_9


        movq      1128+__dcosh_la_CoutTab(%rip), %rdx
        movq      %rdx, -8(%rsp)
        comisd    1144+__dcosh_la_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movsd     1040+__dcosh_la_CoutTab(%rip), %xmm1
        lea       __dcosh_la_CoutTab(%rip), %r9
        mulsd     %xmm0, %xmm1
        addsd     1048+__dcosh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movsd     1088+__dcosh_la_CoutTab(%rip), %xmm1
        movl      -40(%rsp), %edx
        movl      %edx, %esi
        andl      $63, %esi
        subsd     1048+__dcosh_la_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       (%rsi,%rsi), %ecx
        movsd     -32(%rsp), %xmm3
        lea       1(%rsi,%rsi), %edi
        mulsd     1104+__dcosh_la_CoutTab(%rip), %xmm3
        movsd     -32(%rsp), %xmm4
        subsd     %xmm3, %xmm0
        mulsd     1112+__dcosh_la_CoutTab(%rip), %xmm4
        shrl      $6, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm0, %xmm1
        addl      $1022, %edx
        andl      $2047, %edx
        addsd     1080+__dcosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1072+__dcosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1064+__dcosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1056+__dcosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        addsd     %xmm0, %xmm1
        movsd     (%r9,%rcx,8), %xmm0
        mulsd     %xmm0, %xmm1
        addsd     (%r9,%rdi,8), %xmm1
        addsd     %xmm0, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_7


        movq      1128+__dcosh_la_CoutTab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        movsd     %xmm0, (%r8)
        ret

.LBL_2_7:

        decl      %edx
        andl      $2047, %edx
        movzwl    -2(%rsp), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm0, %xmm1
        mulsd     1024+__dcosh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, (%r8)
        ret

.LBL_2_8:

        movsd     1040+__dcosh_la_CoutTab(%rip), %xmm1
        lea       __dcosh_la_CoutTab(%rip), %rcx
        movzwl    -2(%rsp), %esi
        andl      $-32753, %esi
        movsd     1080+__dcosh_la_CoutTab(%rip), %xmm14
        mulsd     %xmm0, %xmm1
        addsd     1048+__dcosh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movl      -40(%rsp), %r10d
        movl      %r10d, %r9d
        shrl      $6, %r9d
        subsd     1048+__dcosh_la_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       1023(%r9), %edi
        andl      $63, %r10d
        addl      $1022, %r9d
        movsd     -32(%rsp), %xmm3
        andl      $2047, %r9d
        negl      %edi
        shll      $4, %r9d
        addl      $-4, %edi
        mulsd     1104+__dcosh_la_CoutTab(%rip), %xmm3
        lea       (%r10,%r10), %edx
        movsd     (%rcx,%rdx,8), %xmm15
        negl      %edx
        movsd     -32(%rsp), %xmm4
        orl       %r9d, %esi
        andl      $2047, %edi
        lea       1(%r10,%r10), %r11d
        mulsd     1112+__dcosh_la_CoutTab(%rip), %xmm4
        subsd     %xmm3, %xmm0
        movw      %si, -2(%rsp)
        andl      $-32753, %esi
        shll      $4, %edi
        subsd     %xmm4, %xmm0
        movsd     -8(%rsp), %xmm6
        orl       %edi, %esi
        movw      %si, -2(%rsp)
        lea       128(%rdx), %esi
        mulsd     %xmm6, %xmm15
        movaps    %xmm0, %xmm5
        mulsd     %xmm0, %xmm5
        movsd     -8(%rsp), %xmm7
        movaps    %xmm15, %xmm8
        movsd     (%rcx,%rsi,8), %xmm11
        addl      $129, %edx
        mulsd     %xmm7, %xmm11
        movaps    %xmm15, %xmm10
        mulsd     %xmm5, %xmm14
        addsd     %xmm11, %xmm8
        subsd     %xmm11, %xmm15
        addsd     1064+__dcosh_la_CoutTab(%rip), %xmm14
        movsd     %xmm8, -24(%rsp)
        movsd     (%rcx,%r11,8), %xmm12
        movsd     (%rcx,%rdx,8), %xmm13
        movsd     -24(%rsp), %xmm9
        mulsd     %xmm6, %xmm12
        subsd     %xmm9, %xmm10
        mulsd     %xmm7, %xmm13
        mulsd     %xmm5, %xmm14
        addsd     %xmm11, %xmm10
        mulsd     %xmm0, %xmm14
        movsd     1088+__dcosh_la_CoutTab(%rip), %xmm1
        movaps    %xmm12, %xmm11
        mulsd     %xmm5, %xmm1
        subsd     %xmm13, %xmm12
        mulsd     %xmm15, %xmm14
        mulsd     %xmm0, %xmm12
        addsd     1072+__dcosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm15, %xmm0
        mulsd     %xmm5, %xmm1
        addsd     %xmm12, %xmm11
        movsd     %xmm10, -16(%rsp)
        addsd     %xmm13, %xmm11
        addsd     1056+__dcosh_la_CoutTab(%rip), %xmm1
        addsd     %xmm14, %xmm11
        mulsd     %xmm5, %xmm1
        addsd     %xmm0, %xmm11
        movsd     -24(%rsp), %xmm3
        mulsd     %xmm3, %xmm1
        movsd     -16(%rsp), %xmm2
        addsd     %xmm1, %xmm11
        addsd     %xmm2, %xmm11
        movsd     %xmm11, -24(%rsp)
        movsd     -24(%rsp), %xmm0
        addsd     %xmm0, %xmm3
        movsd     %xmm3, (%r8)
        ret

.LBL_2_9:

        movsd     1120+__dcosh_la_CoutTab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm0, %xmm0
        movsd     %xmm0, (%r8)
        ret

.LBL_2_10:

        movsd     1136+__dcosh_la_CoutTab(%rip), %xmm0
        addsd     -8(%rsp), %xmm0
        movsd     %xmm0, (%r8)


        ret

.LBL_2_12:

        movsd     (%rdi), %xmm0
        mulsd     %xmm0, %xmm0
        movsd     %xmm0, (%r8)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dcosh_cout_rare_internal,@function
	.size	__svml_dcosh_cout_rare_internal,.-__svml_dcosh_cout_rare_internal
..LN__svml_dcosh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dcosh_data_internal:
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
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	4
	.long	1071644672
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1431655747
	.long	1069897045
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	1430802231
	.long	1067799893
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	287861260
	.long	1065423121
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	3658019094
	.long	1062650243
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	1993999322
	.long	1059717517
	.long	0
	.long	1071644672
	.long	4200250559
	.long	1071647514
	.long	2851812149
	.long	1071650365
	.long	339411585
	.long	1071653224
	.long	1048019041
	.long	1071656090
	.long	772914124
	.long	1071658964
	.long	3899555717
	.long	1071661845
	.long	1928746161
	.long	1071664735
	.long	3541402996
	.long	1071667632
	.long	238821257
	.long	1071670538
	.long	702412510
	.long	1071673451
	.long	728934454
	.long	1071676372
	.long	410360776
	.long	1071679301
	.long	4133881824
	.long	1071682237
	.long	3402036099
	.long	1071685182
	.long	2602514713
	.long	1071688135
	.long	1828292879
	.long	1071691096
	.long	1172597893
	.long	1071694065
	.long	728909815
	.long	1071697042
	.long	590962156
	.long	1071700027
	.long	852742562
	.long	1071703020
	.long	1608493509
	.long	1071706021
	.long	2952712987
	.long	1071709030
	.long	685187902
	.long	1071712048
	.long	3490863953
	.long	1071715073
	.long	2875075254
	.long	1071718107
	.long	3228316108
	.long	1071721149
	.long	351405227
	.long	1071724200
	.long	2930322912
	.long	1071727258
	.long	2471440686
	.long	1071730325
	.long	3366293073
	.long	1071733400
	.long	1416741826
	.long	1071736484
	.long	1014845819
	.long	1071739576
	.long	2257959872
	.long	1071742676
	.long	948735466
	.long	1071745785
	.long	1480023343
	.long	1071748902
	.long	3949972341
	.long	1071752027
	.long	4162030108
	.long	1071755161
	.long	2214878420
	.long	1071758304
	.long	2502433899
	.long	1071761455
	.long	828946858
	.long	1071764615
	.long	1588871207
	.long	1071767783
	.long	586995997
	.long	1071770960
	.long	2218315341
	.long	1071774145
	.long	2288159958
	.long	1071777339
	.long	897099801
	.long	1071780542
	.long	2440944790
	.long	1071783753
	.long	2725843665
	.long	1071786973
	.long	1853186616
	.long	1071790202
	.long	4219606026
	.long	1071793439
	.long	1337108031
	.long	1071796686
	.long	1897844341
	.long	1071799941
	.long	1709341917
	.long	1071803205
	.long	874372905
	.long	1071806478
	.long	3790955393
	.long	1071809759
	.long	1972484976
	.long	1071813050
	.long	4112506593
	.long	1071816349
	.long	1724976915
	.long	1071819658
	.long	3504003472
	.long	1071822975
	.long	964107055
	.long	1071826302
	.long	2799960843
	.long	1071829637
	.long	526652809
	.long	1071832982
	.long	2839424854
	.long	1071836335
	.long	1253935211
	.long	1071839698
	.long	171030293
	.long	1071843070
	.long	3991843581
	.long	1071846450
	.long	4232894513
	.long	1071849840
	.long	1000925746
	.long	1071853240
	.long	2992903935
	.long	1071856648
	.long	1726216749
	.long	1071860066
	.long	1603444721
	.long	1071863493
	.long	2732492859
	.long	1071866929
	.long	926591435
	.long	1071870375
	.long	589198666
	.long	1071873830
	.long	1829099622
	.long	1071877294
	.long	460407023
	.long	1071880768
	.long	887463927
	.long	1071884251
	.long	3219942644
	.long	1071887743
	.long	3272845541
	.long	1071891245
	.long	1156440435
	.long	1071894757
	.long	1276261410
	.long	1071898278
	.long	3743175029
	.long	1071901808
	.long	78413852
	.long	1071905349
	.long	3278348324
	.long	1071908898
	.long	569847338
	.long	1071912458
	.long	654919306
	.long	1071916027
	.long	3645941911
	.long	1071919605
	.long	1065662932
	.long	1071923194
	.long	1617004845
	.long	1071926792
	.long	1118294578
	.long	1071930400
	.long	3978100823
	.long	1071934017
	.long	1720398391
	.long	1071937645
	.long	3049340112
	.long	1071941282
	.long	3784486610
	.long	1071944929
	.long	4040676318
	.long	1071948586
	.long	3933059031
	.long	1071952253
	.long	3577096743
	.long	1071955930
	.long	3088564500
	.long	1071959617
	.long	2583551245
	.long	1071963314
	.long	2178460671
	.long	1071967021
	.long	1990012071
	.long	1071970738
	.long	2135241198
	.long	1071974465
	.long	2731501122
	.long	1071978202
	.long	3896463087
	.long	1071981949
	.long	1453150082
	.long	1071985707
	.long	4109806887
	.long	1071989474
	.long	3395129871
	.long	1071993252
	.long	3723038930
	.long	1071997040
	.long	917841882
	.long	1072000839
	.long	3689071823
	.long	1072004647
	.long	3566716925
	.long	1072008466
	.long	671025100
	.long	1072012296
	.long	3712504873
	.long	1072016135
	.long	4222122499
	.long	1072019985
	.long	2321106615
	.long	1072023846
	.long	2425981843
	.long	1072027717
	.long	363667784
	.long	1072031599
	.long	551349105
	.long	1072035491
	.long	3111574537
	.long	1072039393
	.long	3872257780
	.long	1072043306
	.long	2956612997
	.long	1072047230
	.long	488188413
	.long	1072051165
	.long	885834528
	.long	1072055110
	.long	4273770423
	.long	1072059065
	.long	2186617381
	.long	1072063032
	.long	3339203574
	.long	1072067009
	.long	3561793907
	.long	1072070997
	.long	2979960120
	.long	1072074996
	.long	1719614413
	.long	1072079006
	.long	4201977662
	.long	1072083026
	.long	1963711167
	.long	1072087058
	.long	3721688645
	.long	1072091100
	.long	1013258799
	.long	1072095154
	.long	2555984613
	.long	1072099218
	.long	4182873220
	.long	1072103293
	.long	1727278727
	.long	1072107380
	.long	3907805044
	.long	1072111477
	.long	2263535754
	.long	1072115586
	.long	1218806132
	.long	1072119706
	.long	903334909
	.long	1072123837
	.long	1447192521
	.long	1072127979
	.long	2980802057
	.long	1072132132
	.long	1339972927
	.long	1072136297
	.long	950803702
	.long	1072140473
	.long	1944781191
	.long	1072144660
	.long	158781403
	.long	1072148859
	.long	19972402
	.long	1072153069
	.long	1660913392
	.long	1072157290
	.long	919555682
	.long	1072161523
	.long	2224145553
	.long	1072165767
	.long	1413356050
	.long	1072170023
	.long	2916157145
	.long	1072174290
	.long	2571947539
	.long	1072178569
	.long	515457527
	.long	1072182860
	.long	1176749997
	.long	1072187162
	.long	396319521
	.long	1072191476
	.long	2604962541
	.long	1072195801
	.long	3643909174
	.long	1072200138
	.long	3649726105
	.long	1072204487
	.long	2759350287
	.long	1072208848
	.long	1110089947
	.long	1072213221
	.long	3134592888
	.long	1072217605
	.long	380978316
	.long	1072222002
	.long	1577608921
	.long	1072226410
	.long	2568320822
	.long	1072230830
	.long	3492293770
	.long	1072235262
	.long	194117574
	.long	1072239707
	.long	1403662306
	.long	1072244163
	.long	2966275557
	.long	1072248631
	.long	727685349
	.long	1072253112
	.long	3418903055
	.long	1072257604
	.long	2591453363
	.long	1072262109
	.long	2682146384
	.long	1072266626
	.long	3833209506
	.long	1072271155
	.long	1892288442
	.long	1072275697
	.long	1297350157
	.long	1072280251
	.long	2191782032
	.long	1072284817
	.long	424392917
	.long	1072289396
	.long	434316067
	.long	1072293987
	.long	2366108318
	.long	1072298590
	.long	2069751141
	.long	1072303206
	.long	3985553595
	.long	1072307834
	.long	3964284211
	.long	1072312475
	.long	2152073944
	.long	1072317129
	.long	2990417245
	.long	1072321795
	.long	2331271250
	.long	1072326474
	.long	321958744
	.long	1072331166
	.long	1405169241
	.long	1072335870
	.long	1434058175
	.long	1072340587
	.long	557149882
	.long	1072345317
	.long	3218338682
	.long	1072350059
	.long	977020788
	.long	1072354815
	.long	2572866477
	.long	1072359583
	.long	3861050111
	.long	1072364364
	.long	697153126
	.long	1072369159
	.long	1822067026
	.long	1072373966
	.long	3092190715
	.long	1072378786
	.long	364333489
	.long	1072383620
	.long	2380618042
	.long	1072388466
	.long	703710506
	.long	1072393326
	.long	4076559943
	.long	1072398198
	.long	4062661092
	.long	1072403084
	.long	815859274
	.long	1072407984
	.long	3080351519
	.long	1072412896
	.long	2420883922
	.long	1072417822
	.long	3287523847
	.long	1072422761
	.long	1540824585
	.long	1072427714
	.long	1631695677
	.long	1072432680
	.long	3716502172
	.long	1072437659
	.long	3657065772
	.long	1072442652
	.long	1610600570
	.long	1072447659
	.long	2029714210
	.long	1072452679
	.long	777507147
	.long	1072457713
	.long	2307442995
	.long	1072462760
	.long	2483480501
	.long	1072467821
	.long	1464976603
	.long	1072472896
	.long	3706687593
	.long	1072477984
	.long	778901109
	.long	1072483087
	.long	1432208378
	.long	1072488203
	.long	1532734324
	.long	1072493333
	.long	1242007932
	.long	1072498477
	.long	721996136
	.long	1072503635
	.long	135105010
	.long	1072508807
	.long	3939148246
	.long	1072513992
	.long	3707479175
	.long	1072519192
	.long	3898795731
	.long	1072524406
	.long	382305176
	.long	1072529635
	.long	1912561781
	.long	1072534877
	.long	64696965
	.long	1072540134
	.long	3594158869
	.long	1072545404
	.long	4076975200
	.long	1072550689
	.long	1679558232
	.long	1072555989
	.long	863738719
	.long	1072561303
	.long	1796832535
	.long	1072566631
	.long	351641897
	.long	1072571974
	.long	991358482
	.long	1072577331
	.long	3884662774
	.long	1072582702
	.long	610758006
	.long	1072588089
	.long	4224142467
	.long	1072593489
	.long	2009970496
	.long	1072598905
	.long	2728693978
	.long	1072604335
	.long	2256325230
	.long	1072609780
	.long	764307441
	.long	1072615240
	.long	2719515920
	.long	1072620714
	.long	3999357479
	.long	1072626203
	.long	481706282
	.long	1072631708
	.long	929806999
	.long	1072637227
	.long	1222472308
	.long	1072642761
	.long	1533953344
	.long	1072648310
	.long	2038973688
	.long	1072653874
	.long	2912730644
	.long	1072659453
	.long	35929225
	.long	1072665048
	.long	2174652632
	.long	1072670657
	.long	915592468
	.long	1072676282
	.long	730821105
	.long	1072681922
	.long	1797923801
	.long	1072687577
	.long	0
	.long	1072693248
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
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
	.long	4277927936
	.long	1072049730
	.long	4277927936
	.long	1072049730
	.long	4277927936
	.long	1072049730
	.long	4277927936
	.long	1072049730
	.long	4277927936
	.long	1072049730
	.long	4277927936
	.long	1072049730
	.long	4277927936
	.long	1072049730
	.long	4277927936
	.long	1072049730
	.long	2825664665
	.long	3182190860
	.long	2825664665
	.long	3182190860
	.long	2825664665
	.long	3182190860
	.long	2825664665
	.long	3182190860
	.long	2825664665
	.long	3182190860
	.long	2825664665
	.long	3182190860
	.long	2825664665
	.long	3182190860
	.long	2825664665
	.long	3182190860
	.long	0
	.long	1119354880
	.long	0
	.long	1119354880
	.long	0
	.long	1119354880
	.long	0
	.long	1119354880
	.long	0
	.long	1119354880
	.long	0
	.long	1119354880
	.long	0
	.long	1119354880
	.long	0
	.long	1119354880
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
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
	.long	1887518228
	.long	1069897045
	.long	1887518228
	.long	1069897045
	.long	1887518228
	.long	1069897045
	.long	1887518228
	.long	1069897045
	.long	1887518228
	.long	1069897045
	.long	1887518228
	.long	1069897045
	.long	1887518228
	.long	1069897045
	.long	1887518228
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
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
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
	.type	__svml_dcosh_data_internal,@object
	.size	__svml_dcosh_data_internal,3776
	.space 832, 0x00 	
	.align 32
__dcosh_la_CoutTab:
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
	.long	1697350398
	.long	1079448903
	.long	0
	.long	1127743488
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
	.long	2411329662
	.long	1082536910
	.long	4277796864
	.long	1065758274
	.long	3164486458
	.long	1025308570
	.long	4294967295
	.long	2146435071
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	3875694624
	.long	1077247184
	.type	__dcosh_la_CoutTab,@object
	.size	__dcosh_la_CoutTab,1152
