/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/


	.text
.L_2__routine_start___svml_exp2f16_z0_0:

	.align    16,0x90
	.globl __svml_exp2f16

__svml_exp2f16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_sexp2_data_internal_avx512(%rip), %zmm3

/*
 * Reduced argument
*/
        vreduceps $65, {sae}, %zmm0, %zmm6
        vmovups   192+__svml_sexp2_data_internal_avx512(%rip), %zmm5
        vmovups   128+__svml_sexp2_data_internal_avx512(%rip), %zmm10
        vmovups   384+__svml_sexp2_data_internal_avx512(%rip), %zmm2
        vmovups   64+__svml_sexp2_data_internal_avx512(%rip), %zmm7

/*
 * Integer form of K+0.b1b2b3b4 in lower bits - call K_plus_f0
 * Mantisssa of normalized single precision FP: 1.b1b2...b23
 */
        vaddps    {rd-sae}, %zmm3, %zmm0, %zmm4
        vandps    320+__svml_sexp2_data_internal_avx512(%rip), %zmm0, %zmm1

/* c3*r   + c2 */
        vfmadd231ps {rn-sae}, %zmm6, %zmm5, %zmm10
        vcmpps    $30, {sae}, %zmm2, %zmm1, %k0

/* c3*r^2 + c2*r + c1 */
        vfmadd213ps {rn-sae}, %zmm7, %zmm6, %zmm10

/* Table value: 2^(0.b1b2b3b4) */
        vpermps   __svml_sexp2_data_internal_avx512(%rip), %zmm4, %zmm9
        kmovw     %k0, %edx

/* T*r */
        vmulps    {rn-sae}, %zmm6, %zmm9, %zmm8

/* T + (T*r*(c3*r^2 + c2*r + c1) */
        vfmadd213ps {rn-sae}, %zmm9, %zmm8, %zmm10

/* Scaling placed at the end to avoid accuracy loss when T*r*scale underflows */
        vscalefps {rn-sae}, %zmm0, %zmm10, %zmm1
        testl     %edx, %edx
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
        vmovups   %zmm1, 128(%rsp)
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
        vmovups   128(%rsp), %zmm1
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

        call      __svml_sexp2_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_exp2f16,@function
	.size	__svml_exp2f16,.-__svml_exp2f16
..LN__svml_exp2f16.0:

.L_2__routine_start___svml_sexp2_cout_rare_internal_1:

	.align    16,0x90

__svml_sexp2_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_10


        movss     (%rdi), %xmm7
        movss     .L_2il0floatpacket.43(%rip), %xmm0
        movl      %eax, -12(%rsp)
        comiss    %xmm7, %xmm0
        jb        .LBL_2_8


        comiss    .L_2il0floatpacket.44(%rip), %xmm7
        jb        .LBL_2_7


        movaps    %xmm7, %xmm0
        movaps    %xmm7, %xmm5
        movss     %xmm0, -24(%rsp)
        movss     -24(%rsp), %xmm1
        movss     .L_2il0floatpacket.46(%rip), %xmm2
        movss     .L_2il0floatpacket.52(%rip), %xmm6
        addss     %xmm2, %xmm1
        movss     %xmm1, -20(%rsp)
        movss     -20(%rsp), %xmm3
        movswl    -20(%rsp), %edx
        subss     %xmm2, %xmm3
        movss     %xmm3, -24(%rsp)
        movss     -24(%rsp), %xmm4
        subss     %xmm4, %xmm5
        mulss     %xmm5, %xmm6
        addss     .L_2il0floatpacket.51(%rip), %xmm6
        mulss     %xmm5, %xmm6
        addss     .L_2il0floatpacket.50(%rip), %xmm6
        mulss     %xmm5, %xmm6
        addss     .L_2il0floatpacket.49(%rip), %xmm6
        mulss     %xmm5, %xmm6
        addss     .L_2il0floatpacket.48(%rip), %xmm6
        mulss     %xmm5, %xmm6
        addss     .L_2il0floatpacket.47(%rip), %xmm6
        movss     %xmm6, -16(%rsp)
        cmpl      $104, %edx
        jl        .LBL_2_6


        movzbl    %dl, %edx
        shll      $7, %edx
        movw      %dx, -10(%rsp)
        movss     -16(%rsp), %xmm0
        mulss     -12(%rsp), %xmm0
        movss     %xmm0, -16(%rsp)
        movss     -16(%rsp), %xmm1
        mulss     .L_2il0floatpacket.54(%rip), %xmm1
        movss     %xmm1, -16(%rsp)
        movl      -16(%rsp), %ecx
        movl      %ecx, (%rsi)
        ret

.LBL_2_6:

        addl      $-106, %edx
        cmpltss   .L_2il0floatpacket.56(%rip), %xmm7
        movzbl    %dl, %edx
        movzwl    -10(%rsp), %eax
        shll      $7, %edx
        andl      $-32641, %eax
        orl       %edx, %eax
        movss     -16(%rsp), %xmm0
        movw      %ax, -10(%rsp)
        mulss     -12(%rsp), %xmm0
        movd      %xmm7, %eax
        movss     %xmm0, -16(%rsp)
        movss     -16(%rsp), %xmm1
        andl      $4, %eax
        mulss     .L_2il0floatpacket.55(%rip), %xmm1
        movss     %xmm1, -16(%rsp)
        movl      -16(%rsp), %ecx
        movl      %ecx, (%rsi)
        ret

.LBL_2_7:

        movss     .L_2il0floatpacket.53(%rip), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, -16(%rsp)
        movl      -16(%rsp), %eax
        movl      %eax, (%rsi)
        movl      $4, %eax
        ret

.LBL_2_8:

        movss     .L_2il0floatpacket.45(%rip), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, -16(%rsp)
        movl      -16(%rsp), %eax
        movl      %eax, (%rsi)
        movl      $3, %eax


        ret

.LBL_2_10:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_12

.LBL_2_11:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_12:

        testl     $8388607, (%rdi)
        jne       .LBL_2_11


        movl      %eax, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sexp2_cout_rare_internal,@function
	.size	__svml_sexp2_cout_rare_internal,.-__svml_sexp2_cout_rare_internal
..LN__svml_sexp2_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_sexp2_data_internal_avx512:
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
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1047916907
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1030247626
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
	.long	1228931072
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
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.long	1123811328
	.type	__svml_sexp2_data_internal_avx512,@object
	.size	__svml_sexp2_data_internal_avx512,448
	.align 4
.L_2il0floatpacket.43:
	.long	0x43000000
	.type	.L_2il0floatpacket.43,@object
	.size	.L_2il0floatpacket.43,4
	.align 4
.L_2il0floatpacket.44:
	.long	0xc3160000
	.type	.L_2il0floatpacket.44,@object
	.size	.L_2il0floatpacket.44,4
	.align 4
.L_2il0floatpacket.45:
	.long	0x7f7fffff
	.type	.L_2il0floatpacket.45,@object
	.size	.L_2il0floatpacket.45,4
	.align 4
.L_2il0floatpacket.46:
	.long	0x4b400000
	.type	.L_2il0floatpacket.46,@object
	.size	.L_2il0floatpacket.46,4
	.align 4
.L_2il0floatpacket.47:
	.long	0x3f800001
	.type	.L_2il0floatpacket.47,@object
	.size	.L_2il0floatpacket.47,4
	.align 4
.L_2il0floatpacket.48:
	.long	0x3f317219
	.type	.L_2il0floatpacket.48,@object
	.size	.L_2il0floatpacket.48,4
	.align 4
.L_2il0floatpacket.49:
	.long	0x3e75fc83
	.type	.L_2il0floatpacket.49,@object
	.size	.L_2il0floatpacket.49,4
	.align 4
.L_2il0floatpacket.50:
	.long	0x3d635716
	.type	.L_2il0floatpacket.50,@object
	.size	.L_2il0floatpacket.50,4
	.align 4
.L_2il0floatpacket.51:
	.long	0x3c1e883d
	.type	.L_2il0floatpacket.51,@object
	.size	.L_2il0floatpacket.51,4
	.align 4
.L_2il0floatpacket.52:
	.long	0x3aafc483
	.type	.L_2il0floatpacket.52,@object
	.size	.L_2il0floatpacket.52,4
	.align 4
.L_2il0floatpacket.53:
	.long	0x00000001
	.type	.L_2il0floatpacket.53,@object
	.size	.L_2il0floatpacket.53,4
	.align 4
.L_2il0floatpacket.54:
	.long	0x7f000000
	.type	.L_2il0floatpacket.54,@object
	.size	.L_2il0floatpacket.54,4
	.align 4
.L_2il0floatpacket.55:
	.long	0x34000000
	.type	.L_2il0floatpacket.55,@object
	.size	.L_2il0floatpacket.55,4
	.align 4
.L_2il0floatpacket.56:
	.long	0xc2fc0000
	.type	.L_2il0floatpacket.56,@object
	.size	.L_2il0floatpacket.56,4
