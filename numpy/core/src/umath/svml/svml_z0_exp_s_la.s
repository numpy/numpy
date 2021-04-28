/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *   Argument representation:
 *   M = rint(X*2^k/ln2) = 2^k*N+j
 *   X = M*ln2/2^k + r = N*ln2 + ln2*(j/2^k) + r
 *   then -ln2/2^(k+1) < r < ln2/2^(k+1)
 *   Alternatively:
 *   M = trunc(X*2^k/ln2)
 *   then 0 < r < ln2/2^k
 * 
 *   Result calculation:
 *   exp(X) = exp(N*ln2 + ln2*(j/2^k) + r)
 *   = 2^N * 2^(j/2^k) * exp(r)
 *   2^N is calculated by bit manipulation
 *   2^(j/2^k) is computed from table lookup
 *   exp(r) is approximated by polynomial
 * 
 *   The table lookup is skipped if k = 0.
 *   For low accuracy approximation, exp(r) ~ 1 or 1+r.
 * 
 */


	.text
.L_2__routine_start___svml_expf16_z0_0:

	.align    16,0x90
	.globl __svml_expf16

__svml_expf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_sexp_data_internal_avx512(%rip), %zmm2
        vmovups   320+__svml_sexp_data_internal_avx512(%rip), %zmm1

/* x!=0? */
        vmovups   896+__svml_sexp_data_internal_avx512(%rip), %zmm7
        vmovups   384+__svml_sexp_data_internal_avx512(%rip), %zmm5

/* Table lookup: Tl, Tl = 2^(j/2^10), j = 0,...,2^5-1 */
        vmovups   __svml_sexp_data_internal_avx512(%rip), %zmm8

/* Table lookup: Th, Th = 2^(j/2^5), j = 0,...,2^5-1 */
        vmovups   128+__svml_sexp_data_internal_avx512(%rip), %zmm12

/* 2^(52-4)*1.5 + x * log2(e) in round-to-zero mode */
        vfmadd213ps {rz-sae}, %zmm1, %zmm0, %zmm2
        vmovups   448+__svml_sexp_data_internal_avx512(%rip), %zmm4
        vmovups   640+__svml_sexp_data_internal_avx512(%rip), %zmm10

/* ensure |R|<2 even for special cases */
        vmovups   512+__svml_sexp_data_internal_avx512(%rip), %zmm6
        vcmpps    $4, {sae}, %zmm7, %zmm0, %k1

/* Adjust index by right shift for 5 bits */
        vpsrld    $5, %zmm2, %zmm3

/* N ~ x*log2(e), round-to-zero to 10 fractional bits */
        vsubps    {rn-sae}, %zmm1, %zmm2, %zmm13
        vpermt2ps 64+__svml_sexp_data_internal_avx512(%rip), %zmm2, %zmm8
        vpermt2ps 192+__svml_sexp_data_internal_avx512(%rip), %zmm3, %zmm12

/* remove sign of x by "and" operation */
        vandps    576+__svml_sexp_data_internal_avx512(%rip), %zmm0, %zmm9

/* R = x - N*ln(2)_high */
        vfnmadd213ps {rn-sae}, %zmm0, %zmm13, %zmm5

/* Th*Tl ~ 2^(j/2^k) */
        vmulps    {rn-sae}, %zmm8, %zmm12, %zmm12{%k1}

/* compare against threshold */
        vcmpps    $29, {sae}, %zmm10, %zmm9, %k0

/* R = R - N*ln(2)_low = x - N*ln(2) */
        vfnmadd231ps {rn-sae}, %zmm13, %zmm4, %zmm5

/* set mask for overflow/underflow */
        kmovw     %k0, %edx
        vrangeps  $2, {sae}, %zmm6, %zmm5, %zmm11

/* 2^(j/2^k)*(r+1) */
        vfmadd213ps {rn-sae}, %zmm12, %zmm11, %zmm12

/* exp(x) = 2^N*2^(j/2^k)*(r+1) */
        vscalefps {rn-sae}, %zmm13, %zmm12, %zmm1

/*
 * Check general callout condition
 * Check VML specific mode related condition,
 * no check in case of other libraries
 * Above HA/LA/EP sequences produce
 * correct results even without going to callout.
 * Callout was only needed to raise flags
 * and set errno. If caller doesn't need that
 * then it is safe to proceed without callout
 */
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

        call      __svml_sexp_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_expf16,@function
	.size	__svml_expf16,.-__svml_expf16
..LN__svml_expf16.0:

.L_2__routine_start___svml_sexp_cout_rare_internal_1:

	.align    16,0x90

__svml_sexp_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_12


        movss     (%rdi), %xmm9
        movss     .L_2il0floatpacket.52(%rip), %xmm0
        movl      %eax, -12(%rsp)
        comiss    %xmm9, %xmm0
        jb        .LBL_2_10


        comiss    .L_2il0floatpacket.53(%rip), %xmm9
        jb        .LBL_2_9


        movss     .L_2il0floatpacket.55(%rip), %xmm0
        movaps    %xmm9, %xmm6
        mulss     %xmm9, %xmm0
        comiss    .L_2il0floatpacket.64(%rip), %xmm9
        movss     %xmm0, -24(%rsp)
        movss     -24(%rsp), %xmm1
        movss     .L_2il0floatpacket.56(%rip), %xmm2
        movss     .L_2il0floatpacket.63(%rip), %xmm7
        addss     %xmm2, %xmm1
        movss     %xmm1, -20(%rsp)
        movss     -20(%rsp), %xmm3
        movss     .L_2il0floatpacket.59(%rip), %xmm8
        subss     %xmm2, %xmm3
        movss     %xmm3, -24(%rsp)
        movss     -24(%rsp), %xmm4
        mulss     .L_2il0floatpacket.57(%rip), %xmm4
        movss     -24(%rsp), %xmm5
        subss     %xmm4, %xmm6
        mulss     .L_2il0floatpacket.58(%rip), %xmm5
        movswl    -20(%rsp), %ecx
        subss     %xmm5, %xmm6
        mulss     %xmm6, %xmm7
        addss     .L_2il0floatpacket.62(%rip), %xmm7
        mulss     %xmm6, %xmm7
        addss     .L_2il0floatpacket.61(%rip), %xmm7
        mulss     %xmm6, %xmm7
        addss     .L_2il0floatpacket.60(%rip), %xmm7
        mulss     %xmm6, %xmm7
        addss     %xmm8, %xmm7
        mulss     %xmm6, %xmm7
        addss     %xmm7, %xmm8
        movss     %xmm8, -16(%rsp)
        jb        .LBL_2_8


        lea       127(%rcx), %edx
        cmpl      $254, %edx
        ja        .LBL_2_7


        movzbl    %dl, %edx
        shll      $7, %edx
        movw      %dx, -10(%rsp)
        movss     -16(%rsp), %xmm0
        mulss     -12(%rsp), %xmm0
        movss     %xmm0, -16(%rsp)
        movl      -16(%rsp), %ecx
        movl      %ecx, (%rsi)
        ret

.LBL_2_7:

        addl      $126, %ecx
        movzbl    %cl, %ecx
        movzwl    -10(%rsp), %edx
        shll      $7, %ecx
        andl      $-32641, %edx
        orl       %ecx, %edx
        movss     -16(%rsp), %xmm0
        movw      %dx, -10(%rsp)
        mulss     -12(%rsp), %xmm0
        movss     %xmm0, -16(%rsp)
        movss     -16(%rsp), %xmm1
        mulss     .L_2il0floatpacket.67(%rip), %xmm1
        movss     %xmm1, -16(%rsp)
        movl      -16(%rsp), %edi
        movl      %edi, (%rsi)
        ret

.LBL_2_8:

        addl      $-69, %ecx
        movzbl    %cl, %ecx
        movzwl    -10(%rsp), %eax
        shll      $7, %ecx
        andl      $-32641, %eax
        orl       %ecx, %eax
        movss     -16(%rsp), %xmm0
        movw      %ax, -10(%rsp)
        movl      $4, %eax
        mulss     -12(%rsp), %xmm0
        movss     %xmm0, -16(%rsp)
        movss     -16(%rsp), %xmm1
        mulss     .L_2il0floatpacket.66(%rip), %xmm1
        movss     %xmm1, -16(%rsp)
        movl      -16(%rsp), %edx
        movl      %edx, (%rsi)
        ret

.LBL_2_9:

        movss     .L_2il0floatpacket.65(%rip), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, -16(%rsp)
        movl      -16(%rsp), %eax
        movl      %eax, (%rsi)
        movl      $4, %eax
        ret

.LBL_2_10:

        movss     .L_2il0floatpacket.54(%rip), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, -16(%rsp)
        movl      -16(%rsp), %eax
        movl      %eax, (%rsi)
        movl      $3, %eax


        ret

.LBL_2_12:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_14

.LBL_2_13:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_14:

        testl     $8388607, (%rdi)
        jne       .LBL_2_13


        movl      %eax, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sexp_cout_rare_internal,@function
	.size	__svml_sexp_cout_rare_internal,.-__svml_sexp_cout_rare_internal
..LN__svml_sexp_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_sexp_data_internal_avx512:
	.long	1065353217
	.long	1065358897
	.long	1065364581
	.long	1065370269
	.long	1065375961
	.long	1065381656
	.long	1065387356
	.long	1065393059
	.long	1065398766
	.long	1065404477
	.long	1065410192
	.long	1065415911
	.long	1065421634
	.long	1065427360
	.long	1065433091
	.long	1065438825
	.long	1065444563
	.long	1065450305
	.long	1065456051
	.long	1065461801
	.long	1065467554
	.long	1065473312
	.long	1065479074
	.long	1065484839
	.long	1065490608
	.long	1065496381
	.long	1065502159
	.long	1065507940
	.long	1065513725
	.long	1065519513
	.long	1065525306
	.long	1065531103
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
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
	.long	1178599424
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
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
	.long	796917760
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
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
	.long	3968
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
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
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
	.type	__svml_sexp_data_internal_avx512,@object
	.size	__svml_sexp_data_internal_avx512,1216
	.align 4
.L_2il0floatpacket.52:
	.long	0x42b17217
	.type	.L_2il0floatpacket.52,@object
	.size	.L_2il0floatpacket.52,4
	.align 4
.L_2il0floatpacket.53:
	.long	0xc2cff1b4
	.type	.L_2il0floatpacket.53,@object
	.size	.L_2il0floatpacket.53,4
	.align 4
.L_2il0floatpacket.54:
	.long	0x7f7fffff
	.type	.L_2il0floatpacket.54,@object
	.size	.L_2il0floatpacket.54,4
	.align 4
.L_2il0floatpacket.55:
	.long	0x3fb8aa3b
	.type	.L_2il0floatpacket.55,@object
	.size	.L_2il0floatpacket.55,4
	.align 4
.L_2il0floatpacket.56:
	.long	0x4b400000
	.type	.L_2il0floatpacket.56,@object
	.size	.L_2il0floatpacket.56,4
	.align 4
.L_2il0floatpacket.57:
	.long	0x3f317200
	.type	.L_2il0floatpacket.57,@object
	.size	.L_2il0floatpacket.57,4
	.align 4
.L_2il0floatpacket.58:
	.long	0x35bfbe8e
	.type	.L_2il0floatpacket.58,@object
	.size	.L_2il0floatpacket.58,4
	.align 4
.L_2il0floatpacket.59:
	.long	0x3f800001
	.type	.L_2il0floatpacket.59,@object
	.size	.L_2il0floatpacket.59,4
	.align 4
.L_2il0floatpacket.60:
	.long	0x3efffe85
	.type	.L_2il0floatpacket.60,@object
	.size	.L_2il0floatpacket.60,4
	.align 4
.L_2il0floatpacket.61:
	.long	0x3e2aa9c6
	.type	.L_2il0floatpacket.61,@object
	.size	.L_2il0floatpacket.61,4
	.align 4
.L_2il0floatpacket.62:
	.long	0x3d2bb1b6
	.type	.L_2il0floatpacket.62,@object
	.size	.L_2il0floatpacket.62,4
	.align 4
.L_2il0floatpacket.63:
	.long	0x3c0950ef
	.type	.L_2il0floatpacket.63,@object
	.size	.L_2il0floatpacket.63,4
	.align 4
.L_2il0floatpacket.64:
	.long	0xc2aeac4f
	.type	.L_2il0floatpacket.64,@object
	.size	.L_2il0floatpacket.64,4
	.align 4
.L_2il0floatpacket.65:
	.long	0x00000001
	.type	.L_2il0floatpacket.65,@object
	.size	.L_2il0floatpacket.65,4
	.align 4
.L_2il0floatpacket.66:
	.long	0x21800000
	.type	.L_2il0floatpacket.66,@object
	.size	.L_2il0floatpacket.66,4
	.align 4
.L_2il0floatpacket.67:
	.long	0x40000000
	.type	.L_2il0floatpacket.67,@object
	.size	.L_2il0floatpacket.67,4
