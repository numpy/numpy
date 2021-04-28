/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *   Argument representation:
 *   N = rint(X*2^k/ln2) = 2^k*M+j
 *   X = N*ln2/2^k + r = M*ln2 + ln2*(j/2^k) + r
 *   then -ln2/2^(k+1) < r < ln2/2^(k+1)
 *   Alternatively:
 *   N = trunc(X*2^k/ln2)
 *   then 0 < r < ln2/2^k
 * 
 *   Result calculation:
 *   exp(X) = exp(M*ln2 + ln2*(j/2^k) + r)
 *   = 2^M * 2^(j/2^k) * exp(r)
 *   2^M is calculated by bit manipulation
 *   2^(j/2^k) is stored in table
 *   exp(r) is approximated by polynomial
 * 
 *   The table lookup is skipped if k = 0.
 * 
 */


	.text
.L_2__routine_start___svml_exp8_z0_0:

	.align    16,0x90
	.globl __svml_exp8

__svml_exp8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_dexp_data_internal_avx512(%rip), %zmm3
        vmovups   320+__svml_dexp_data_internal_avx512(%rip), %zmm1
        vmovups   384+__svml_dexp_data_internal_avx512(%rip), %zmm4
        vmovups   448+__svml_dexp_data_internal_avx512(%rip), %zmm2
        vmovups   576+__svml_dexp_data_internal_avx512(%rip), %zmm5
        vmovups   640+__svml_dexp_data_internal_avx512(%rip), %zmm15

/* 2^(52-4)*1.5 + x * log2(e) in round-to-zero mode */
        vfmadd213pd {rz-sae}, %zmm1, %zmm0, %zmm3
        vmovups   704+__svml_dexp_data_internal_avx512(%rip), %zmm6

/* p34 = c4*r+c3 */
        vmovups   768+__svml_dexp_data_internal_avx512(%rip), %zmm9
        vmovups   832+__svml_dexp_data_internal_avx512(%rip), %zmm11
        vmovups   1024+__svml_dexp_data_internal_avx512(%rip), %zmm13

/* Table lookup: Th = 2^(j/2^k) */
        vmovups   __svml_dexp_data_internal_avx512(%rip), %zmm14

/* N ~ x*log2(e), round-to-zero to 4 fractional bits */
        vsubpd    {rn-sae}, %zmm1, %zmm3, %zmm1

/* remove sign of x by "and" operation */
        vandpd    960+__svml_dexp_data_internal_avx512(%rip), %zmm0, %zmm12
        vpermt2pd 64+__svml_dexp_data_internal_avx512(%rip), %zmm3, %zmm14

/* R = x - N*ln(2)_high */
        vfnmadd213pd {rn-sae}, %zmm0, %zmm1, %zmm4

/* compare against threshold */
        vcmppd    $29, {sae}, %zmm13, %zmm12, %k0

/* R = R - N*ln(2)_high = x - N*ln(2) */
        vfnmadd231pd {rn-sae}, %zmm1, %zmm2, %zmm4

/* set mask for overflow/underflow */
        kmovw     %k0, %edx

/* ensure |R|<2 even for special cases */
        vandpd    512+__svml_dexp_data_internal_avx512(%rip), %zmm4, %zmm7

/* r2 = r*r */
        vmulpd    {rn-sae}, %zmm7, %zmm7, %zmm8

/* p56 = c6*r+c5 */
        vfmadd231pd {rn-sae}, %zmm7, %zmm5, %zmm15
        vfmadd231pd {rn-sae}, %zmm7, %zmm6, %zmm9

/* r3 = r2*r = r^3 */
        vmulpd    {rn-sae}, %zmm7, %zmm8, %zmm10

/* p12 = c2*r2+r */
        vfmadd213pd {rn-sae}, %zmm7, %zmm8, %zmm11

/* p36 = (c6*r+c5)*r^2+c4*r+c3 */
        vfmadd213pd {rn-sae}, %zmm9, %zmm8, %zmm15

/* p16 = ((c6*r+c5)*r^2+c4*r+c3)*r^3+c2*r^2+r = exp(r)-1 */
        vfmadd213pd {rn-sae}, %zmm11, %zmm10, %zmm15

/* Th*(exp(r)-1)+Th=Th*exp(r) */
        vfmadd213pd {rn-sae}, %zmm14, %zmm14, %zmm15

/* exp(x) = 2^M*Th*exp(r) */
        vscalefpd {rn-sae}, %zmm1, %zmm15, %zmm1


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

        lea       64(%rsp,%r12,8), %rdi
        lea       128(%rsp,%r12,8), %rsi

        call      __svml_dexp_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_exp8,@function
	.size	__svml_exp8,.-__svml_exp8
..LN__svml_exp8.0:

.L_2__routine_start___svml_dexp_cout_rare_internal_1:

	.align    16,0x90

__svml_dexp_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    6(%rdi), %edx
        andl      $32752, %edx
        shrl      $4, %edx
        movsd     (%rdi), %xmm6
        movsd     %xmm6, -8(%rsp)
        cmpl      $2047, %edx
        je        .LBL_2_17


        cmpl      $970, %edx
        jle       .LBL_2_15


        movsd     1080+_imldExpHATab(%rip), %xmm0
        comisd    %xmm6, %xmm0
        jb        .LBL_2_14


        comisd    1096+_imldExpHATab(%rip), %xmm6
        jb        .LBL_2_13


        movsd     1024+_imldExpHATab(%rip), %xmm0
        movaps    %xmm6, %xmm5
        mulsd     %xmm6, %xmm0
        lea       _imldExpHATab(%rip), %r10
        movsd     %xmm0, -24(%rsp)
        movsd     -24(%rsp), %xmm1
        movq      1136+_imldExpHATab(%rip), %rdx
        movq      %rdx, -8(%rsp)
        addsd     1032+_imldExpHATab(%rip), %xmm1
        movsd     %xmm1, -16(%rsp)
        movsd     -16(%rsp), %xmm2
        movl      -16(%rsp), %r8d
        movl      %r8d, %ecx
        andl      $63, %r8d
        subsd     1032+_imldExpHATab(%rip), %xmm2
        movsd     %xmm2, -24(%rsp)
        lea       1(%r8,%r8), %r9d
        movsd     -24(%rsp), %xmm3
        lea       (%r8,%r8), %edi
        mulsd     1104+_imldExpHATab(%rip), %xmm3
        movsd     -24(%rsp), %xmm4
        subsd     %xmm3, %xmm5
        mulsd     1112+_imldExpHATab(%rip), %xmm4
        movsd     1072+_imldExpHATab(%rip), %xmm2
        subsd     %xmm4, %xmm5
        mulsd     %xmm5, %xmm2
        shrl      $6, %ecx
        addsd     1064+_imldExpHATab(%rip), %xmm2
        comisd    1088+_imldExpHATab(%rip), %xmm6
        mulsd     %xmm5, %xmm2
        movsd     (%r10,%rdi,8), %xmm0
        lea       1023(%rcx), %edx
        addsd     1056+_imldExpHATab(%rip), %xmm2
        mulsd     %xmm5, %xmm2
        addsd     1048+_imldExpHATab(%rip), %xmm2
        mulsd     %xmm5, %xmm2
        addsd     1040+_imldExpHATab(%rip), %xmm2
        mulsd     %xmm5, %xmm2
        mulsd     %xmm5, %xmm2
        addsd     %xmm5, %xmm2
        addsd     (%r10,%r9,8), %xmm2
        mulsd     %xmm0, %xmm2
        jb        .LBL_2_9


        andl      $2047, %edx
        addsd     %xmm0, %xmm2
        cmpl      $2046, %edx
        ja        .LBL_2_8


        movq      1136+_imldExpHATab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm0, %xmm2
        movsd     %xmm2, (%rsi)
        ret

.LBL_2_8:

        decl      %edx
        andl      $2047, %edx
        movzwl    -2(%rsp), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm0, %xmm2
        mulsd     1152+_imldExpHATab(%rip), %xmm2
        movsd     %xmm2, (%rsi)
        ret

.LBL_2_9:

        addl      $1083, %ecx
        andl      $2047, %ecx
        movl      %ecx, %eax
        movzwl    -2(%rsp), %edx
        shll      $4, %eax
        andl      $-32753, %edx
        orl       %eax, %edx
        movw      %dx, -2(%rsp)
        movsd     -8(%rsp), %xmm1
        mulsd     %xmm1, %xmm2
        mulsd     %xmm0, %xmm1
        movaps    %xmm1, %xmm0
        addsd     %xmm2, %xmm0
        cmpl      $50, %ecx
        ja        .LBL_2_11


        mulsd     1160+_imldExpHATab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        jmp       .LBL_2_12

.LBL_2_11:

        movsd     %xmm0, -72(%rsp)
        movsd     -72(%rsp), %xmm0
        subsd     %xmm0, %xmm1
        movsd     %xmm1, -64(%rsp)
        movsd     -64(%rsp), %xmm1
        addsd     %xmm2, %xmm1
        movsd     %xmm1, -64(%rsp)
        movsd     -72(%rsp), %xmm2
        mulsd     1168+_imldExpHATab(%rip), %xmm2
        movsd     %xmm2, -56(%rsp)
        movsd     -72(%rsp), %xmm4
        movsd     -56(%rsp), %xmm3
        addsd     %xmm3, %xmm4
        movsd     %xmm4, -48(%rsp)
        movsd     -48(%rsp), %xmm6
        movsd     -56(%rsp), %xmm5
        subsd     %xmm5, %xmm6
        movsd     %xmm6, -40(%rsp)
        movsd     -72(%rsp), %xmm8
        movsd     -40(%rsp), %xmm7
        subsd     %xmm7, %xmm8
        movsd     %xmm8, -32(%rsp)
        movsd     -64(%rsp), %xmm10
        movsd     -32(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -32(%rsp)
        movsd     -40(%rsp), %xmm11
        mulsd     1160+_imldExpHATab(%rip), %xmm11
        movsd     %xmm11, -40(%rsp)
        movsd     -32(%rsp), %xmm12
        mulsd     1160+_imldExpHATab(%rip), %xmm12
        movsd     %xmm12, -32(%rsp)
        movsd     -40(%rsp), %xmm14
        movsd     -32(%rsp), %xmm13
        addsd     %xmm13, %xmm14
        movsd     %xmm14, (%rsi)

.LBL_2_12:

        movl      $4, %eax
        ret

.LBL_2_13:

        movsd     1120+_imldExpHATab(%rip), %xmm0
        movl      $4, %eax
        mulsd     %xmm0, %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_14:

        movsd     1128+_imldExpHATab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm0, %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_15:

        movsd     1144+_imldExpHATab(%rip), %xmm0
        addsd     %xmm6, %xmm0
        movsd     %xmm0, (%rsi)


        ret

.LBL_2_17:

        movb      -1(%rsp), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_19

.LBL_2_18:

        mulsd     %xmm6, %xmm6
        movsd     %xmm6, (%rsi)
        ret

.LBL_2_19:

        testl     $1048575, -4(%rsp)
        jne       .LBL_2_18


        cmpl      $0, -8(%rsp)
        jne       .LBL_2_18


        movq      1136+_imldExpHATab(%rip), %rdx
        movq      %rdx, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dexp_cout_rare_internal,@function
	.size	__svml_dexp_cout_rare_internal,.-__svml_dexp_cout_rare_internal
..LN__svml_dexp_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dexp_data_internal_avx512:
	.long	0
	.long	1072693248
	.long	1828292879
	.long	1072739672
	.long	1014845819
	.long	1072788152
	.long	1853186616
	.long	1072838778
	.long	171030293
	.long	1072891646
	.long	1276261410
	.long	1072946854
	.long	3577096743
	.long	1073004506
	.long	3712504873
	.long	1073064711
	.long	1719614413
	.long	1073127582
	.long	1944781191
	.long	1073193236
	.long	1110089947
	.long	1073261797
	.long	2191782032
	.long	1073333393
	.long	2572866477
	.long	1073408159
	.long	3716502172
	.long	1073486235
	.long	3707479175
	.long	1073567768
	.long	2728693978
	.long	1073652911
	.long	0
	.long	0
	.long	1568897901
	.long	1016568486
	.long	3936719688
	.long	3162512149
	.long	3819481236
	.long	1016499965
	.long	1303423926
	.long	1015238005
	.long	2804567149
	.long	1015390024
	.long	3145379760
	.long	1014403278
	.long	3793507337
	.long	1016095713
	.long	3210617384
	.long	3163796463
	.long	3108873501
	.long	3162190556
	.long	3253791412
	.long	1015920431
	.long	730975783
	.long	1014083580
	.long	2462790535
	.long	1015814775
	.long	816778419
	.long	1014197934
	.long	2789017511
	.long	1014276997
	.long	2413007344
	.long	3163551506
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
	.long	16368
	.long	1123549184
	.long	16368
	.long	1123549184
	.long	16368
	.long	1123549184
	.long	16368
	.long	1123549184
	.long	16368
	.long	1123549184
	.long	16368
	.long	1123549184
	.long	16368
	.long	1123549184
	.long	16368
	.long	1123549184
	.long	4277811695
	.long	1072049730
	.long	4277811695
	.long	1072049730
	.long	4277811695
	.long	1072049730
	.long	4277811695
	.long	1072049730
	.long	4277811695
	.long	1072049730
	.long	4277811695
	.long	1072049730
	.long	4277811695
	.long	1072049730
	.long	4277811695
	.long	1072049730
	.long	993624127
	.long	1014676638
	.long	993624127
	.long	1014676638
	.long	993624127
	.long	1014676638
	.long	993624127
	.long	1014676638
	.long	993624127
	.long	1014676638
	.long	993624127
	.long	1014676638
	.long	993624127
	.long	1014676638
	.long	993624127
	.long	1014676638
	.long	4294967295
	.long	3221225471
	.long	4294967295
	.long	3221225471
	.long	4294967295
	.long	3221225471
	.long	4294967295
	.long	3221225471
	.long	4294967295
	.long	3221225471
	.long	4294967295
	.long	3221225471
	.long	4294967295
	.long	3221225471
	.long	4294967295
	.long	3221225471
	.long	4010615101
	.long	1062678197
	.long	4010615101
	.long	1062678197
	.long	4010615101
	.long	1062678197
	.long	4010615101
	.long	1062678197
	.long	4010615101
	.long	1062678197
	.long	4010615101
	.long	1062678197
	.long	4010615101
	.long	1062678197
	.long	4010615101
	.long	1062678197
	.long	64584704
	.long	1065422950
	.long	64584704
	.long	1065422950
	.long	64584704
	.long	1065422950
	.long	64584704
	.long	1065422950
	.long	64584704
	.long	1065422950
	.long	64584704
	.long	1065422950
	.long	64584704
	.long	1065422950
	.long	64584704
	.long	1065422950
	.long	1270150640
	.long	1067799894
	.long	1270150640
	.long	1067799894
	.long	1270150640
	.long	1067799894
	.long	1270150640
	.long	1067799894
	.long	1270150640
	.long	1067799894
	.long	1270150640
	.long	1067799894
	.long	1270150640
	.long	1067799894
	.long	1270150640
	.long	1067799894
	.long	1421150370
	.long	1069897045
	.long	1421150370
	.long	1069897045
	.long	1421150370
	.long	1069897045
	.long	1421150370
	.long	1069897045
	.long	1421150370
	.long	1069897045
	.long	1421150370
	.long	1069897045
	.long	1421150370
	.long	1069897045
	.long	1421150370
	.long	1069897045
	.long	9762
	.long	1071644672
	.long	9762
	.long	1071644672
	.long	9762
	.long	1071644672
	.long	9762
	.long	1071644672
	.long	9762
	.long	1071644672
	.long	9762
	.long	1071644672
	.long	9762
	.long	1071644672
	.long	9762
	.long	1071644672
	.long	4294967152
	.long	1072693247
	.long	4294967152
	.long	1072693247
	.long	4294967152
	.long	1072693247
	.long	4294967152
	.long	1072693247
	.long	4294967152
	.long	1072693247
	.long	4294967152
	.long	1072693247
	.long	4294967152
	.long	1072693247
	.long	4294967152
	.long	1072693247
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	4294967295
	.long	2147483647
	.long	1287323204
	.long	1082531232
	.long	1287323204
	.long	1082531232
	.long	1287323204
	.long	1082531232
	.long	1287323204
	.long	1082531232
	.long	1287323204
	.long	1082531232
	.long	1287323204
	.long	1082531232
	.long	1287323204
	.long	1082531232
	.long	1287323204
	.long	1082531232
	.long	0
	.long	1005584384
	.long	0
	.long	1005584384
	.long	0
	.long	1005584384
	.long	0
	.long	1005584384
	.long	0
	.long	1005584384
	.long	0
	.long	1005584384
	.long	0
	.long	1005584384
	.long	0
	.long	1005584384
	.long	120
	.long	0
	.long	120
	.long	0
	.long	120
	.long	0
	.long	120
	.long	0
	.long	120
	.long	0
	.long	120
	.long	0
	.long	120
	.long	0
	.long	120
	.long	0
	.long	32752
	.long	0
	.long	32752
	.long	0
	.long	32752
	.long	0
	.long	32752
	.long	0
	.long	32752
	.long	0
	.long	32752
	.long	0
	.long	32752
	.long	0
	.long	32752
	.long	0
	.type	__svml_dexp_data_internal_avx512,@object
	.size	__svml_dexp_data_internal_avx512,1280
	.align 32
_imldExpHATab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	1048019041
	.long	1072704666
	.long	2631457885
	.long	3161546771
	.long	3541402996
	.long	1072716208
	.long	896005651
	.long	1015861842
	.long	410360776
	.long	1072727877
	.long	1642514529
	.long	1012987726
	.long	1828292879
	.long	1072739672
	.long	1568897901
	.long	1016568486
	.long	852742562
	.long	1072751596
	.long	1882168529
	.long	1010744893
	.long	3490863953
	.long	1072763649
	.long	707771662
	.long	3163903570
	.long	2930322912
	.long	1072775834
	.long	3117806614
	.long	3163670819
	.long	1014845819
	.long	1072788152
	.long	3936719688
	.long	3162512149
	.long	3949972341
	.long	1072800603
	.long	1058231231
	.long	1015777676
	.long	828946858
	.long	1072813191
	.long	1044000608
	.long	1016786167
	.long	2288159958
	.long	1072825915
	.long	1151779725
	.long	1015705409
	.long	1853186616
	.long	1072838778
	.long	3819481236
	.long	1016499965
	.long	1709341917
	.long	1072851781
	.long	2552227826
	.long	1015039787
	.long	4112506593
	.long	1072864925
	.long	1829350193
	.long	1015216097
	.long	2799960843
	.long	1072878213
	.long	1913391796
	.long	1015756674
	.long	171030293
	.long	1072891646
	.long	1303423926
	.long	1015238005
	.long	2992903935
	.long	1072905224
	.long	1574172746
	.long	1016061241
	.long	926591435
	.long	1072918951
	.long	3427487848
	.long	3163704045
	.long	887463927
	.long	1072932827
	.long	1049900754
	.long	3161575912
	.long	1276261410
	.long	1072946854
	.long	2804567149
	.long	1015390024
	.long	569847338
	.long	1072961034
	.long	1209502043
	.long	3159926671
	.long	1617004845
	.long	1072975368
	.long	1623370769
	.long	1011049453
	.long	3049340112
	.long	1072989858
	.long	3667985273
	.long	1013894369
	.long	3577096743
	.long	1073004506
	.long	3145379760
	.long	1014403278
	.long	1990012071
	.long	1073019314
	.long	7447438
	.long	3163526196
	.long	1453150082
	.long	1073034283
	.long	3171891295
	.long	3162037958
	.long	917841882
	.long	1073049415
	.long	419288974
	.long	1016280325
	.long	3712504873
	.long	1073064711
	.long	3793507337
	.long	1016095713
	.long	363667784
	.long	1073080175
	.long	728023093
	.long	1016345318
	.long	2956612997
	.long	1073095806
	.long	1005538728
	.long	3163304901
	.long	2186617381
	.long	1073111608
	.long	2018924632
	.long	3163803357
	.long	1719614413
	.long	1073127582
	.long	3210617384
	.long	3163796463
	.long	1013258799
	.long	1073143730
	.long	3094194670
	.long	3160631279
	.long	3907805044
	.long	1073160053
	.long	2119843535
	.long	3161988964
	.long	1447192521
	.long	1073176555
	.long	508946058
	.long	3162904882
	.long	1944781191
	.long	1073193236
	.long	3108873501
	.long	3162190556
	.long	919555682
	.long	1073210099
	.long	2882956373
	.long	1013312481
	.long	2571947539
	.long	1073227145
	.long	4047189812
	.long	3163777462
	.long	2604962541
	.long	1073244377
	.long	3631372142
	.long	3163870288
	.long	1110089947
	.long	1073261797
	.long	3253791412
	.long	1015920431
	.long	2568320822
	.long	1073279406
	.long	1509121860
	.long	1014756995
	.long	2966275557
	.long	1073297207
	.long	2339118633
	.long	3160254904
	.long	2682146384
	.long	1073315202
	.long	586480042
	.long	3163702083
	.long	2191782032
	.long	1073333393
	.long	730975783
	.long	1014083580
	.long	2069751141
	.long	1073351782
	.long	576856675
	.long	3163014404
	.long	2990417245
	.long	1073370371
	.long	3552361237
	.long	3163667409
	.long	1434058175
	.long	1073389163
	.long	1853053619
	.long	1015310724
	.long	2572866477
	.long	1073408159
	.long	2462790535
	.long	1015814775
	.long	3092190715
	.long	1073427362
	.long	1457303226
	.long	3159737305
	.long	4076559943
	.long	1073446774
	.long	950899508
	.long	3160987380
	.long	2420883922
	.long	1073466398
	.long	174054861
	.long	1014300631
	.long	3716502172
	.long	1073486235
	.long	816778419
	.long	1014197934
	.long	777507147
	.long	1073506289
	.long	3507050924
	.long	1015341199
	.long	3706687593
	.long	1073526560
	.long	1821514088
	.long	1013410604
	.long	1242007932
	.long	1073547053
	.long	1073740399
	.long	3163532637
	.long	3707479175
	.long	1073567768
	.long	2789017511
	.long	1014276997
	.long	64696965
	.long	1073588710
	.long	3586233004
	.long	1015962192
	.long	863738719
	.long	1073609879
	.long	129252895
	.long	3162690849
	.long	3884662774
	.long	1073631278
	.long	1614448851
	.long	1014281732
	.long	2728693978
	.long	1073652911
	.long	2413007344
	.long	3163551506
	.long	3999357479
	.long	1073674779
	.long	1101668360
	.long	1015989180
	.long	1533953344
	.long	1073696886
	.long	835814894
	.long	1015702697
	.long	2174652632
	.long	1073719233
	.long	1301400989
	.long	1014466875
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
	.long	4277811695
	.long	1082535490
	.long	3715808466
	.long	3230016299
	.long	3576508497
	.long	3230091536
	.long	4277796864
	.long	1065758274
	.long	3164486458
	.long	1025308570
	.long	1
	.long	1048576
	.long	4294967295
	.long	2146435071
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	1073741824
	.long	0
	.long	1009778688
	.long	0
	.long	1106771968
	.type	_imldExpHATab,@object
	.size	_imldExpHATab,1176
