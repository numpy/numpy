/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *  log10(x) = VGETEXP(x)*log10(2) + log10(VGETMANT(x))
 *  *       VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *  *   mx = VGETMANT(x) is in [1,2) for all x>=0
 *  *   log10(mx) = -log10(RCP(mx)) + log10(1 +(mx*RCP(mx)-1))
 *  *      RCP(mx) is rounded to 4 fractional bits,
 *  *      and the table lookup for log(RCP(mx)) is based on a small permute instruction
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_log108_z0_0:

	.align    16,0x90
	.globl __svml_log108

__svml_log108:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovaps   %zmm0, %zmm7

/* GetMant(x), normalized to [1,2) for x>=0, NaN for x<0 */
        vgetmantpd $8, {sae}, %zmm7, %zmm6
        vmovups   128+__svml_dlog10_data_internal_avx512(%rip), %zmm3
        vmovups   512+__svml_dlog10_data_internal_avx512(%rip), %zmm12
        vmovups   640+__svml_dlog10_data_internal_avx512(%rip), %zmm13

/* Start polynomial evaluation */
        vmovups   256+__svml_dlog10_data_internal_avx512(%rip), %zmm10
        vmovups   320+__svml_dlog10_data_internal_avx512(%rip), %zmm1
        vmovups   384+__svml_dlog10_data_internal_avx512(%rip), %zmm11
        vmovups   448+__svml_dlog10_data_internal_avx512(%rip), %zmm14

/* Prepare exponent correction: DblRcp<0.75? */
        vmovups   192+__svml_dlog10_data_internal_avx512(%rip), %zmm2

/* Table lookup */
        vmovups   __svml_dlog10_data_internal_avx512(%rip), %zmm5

/* GetExp(x) */
        vgetexppd {sae}, %zmm7, %zmm0

/* DblRcp ~ 1/Mantissa */
        vrcp14pd  %zmm6, %zmm8

/* x<=0? */
        vfpclasspd $94, %zmm7, %k0

/* round DblRcp to 4 fractional bits (RN mode, no Precision exception) */
        vrndscalepd $88, {sae}, %zmm8, %zmm4
        vmovups   576+__svml_dlog10_data_internal_avx512(%rip), %zmm8
        kmovw     %k0, %edx

/* Reduced argument: R = DblRcp*Mantissa - 1 */
        vfmsub213pd {rn-sae}, %zmm3, %zmm4, %zmm6
        vcmppd    $17, {sae}, %zmm2, %zmm4, %k1
        vfmadd231pd {rn-sae}, %zmm6, %zmm12, %zmm8
        vmovups   704+__svml_dlog10_data_internal_avx512(%rip), %zmm12
        vfmadd231pd {rn-sae}, %zmm6, %zmm10, %zmm1
        vfmadd231pd {rn-sae}, %zmm6, %zmm11, %zmm14
        vmovups   768+__svml_dlog10_data_internal_avx512(%rip), %zmm2

/* R^2 */
        vmulpd    {rn-sae}, %zmm6, %zmm6, %zmm15
        vfmadd231pd {rn-sae}, %zmm6, %zmm13, %zmm12

/* Prepare table index */
        vpsrlq    $48, %zmm4, %zmm9

/* add 1 to Expon if DblRcp<0.75 */
        vaddpd    {rn-sae}, %zmm3, %zmm0, %zmm0{%k1}
        vmulpd    {rn-sae}, %zmm15, %zmm15, %zmm13
        vfmadd213pd {rn-sae}, %zmm14, %zmm15, %zmm1
        vfmadd213pd {rn-sae}, %zmm12, %zmm15, %zmm8
        vpermt2pd 64+__svml_dlog10_data_internal_avx512(%rip), %zmm9, %zmm5

/* polynomial */
        vfmadd213pd {rn-sae}, %zmm8, %zmm13, %zmm1
        vfmadd213pd {rn-sae}, %zmm2, %zmm6, %zmm1
        vfmadd213pd {rn-sae}, %zmm5, %zmm1, %zmm6
        vmovups   832+__svml_dlog10_data_internal_avx512(%rip), %zmm1
        vfmadd213pd {rn-sae}, %zmm6, %zmm1, %zmm0
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

        vmovups   %zmm7, 64(%rsp)
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

        call      __svml_dlog10_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log108,@function
	.size	__svml_log108,.-__svml_log108
..LN__svml_log108.0:

.L_2__routine_start___svml_dlog10_cout_rare_internal_1:

	.align    16,0x90

__svml_dlog10_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    6(%rdi), %edx
        andl      $32752, %edx
        cmpl      $32752, %edx
        je        .LBL_2_12


        movsd     (%rdi), %xmm2
        xorl      %ecx, %ecx
        movsd     %xmm2, -8(%rsp)
        movzwl    -2(%rsp), %edx
        testl     $32752, %edx
        jne       .LBL_2_4


        movsd     1600+__dlog10_la_CoutTab(%rip), %xmm0
        movl      $-60, %ecx
        mulsd     %xmm0, %xmm2
        movsd     %xmm2, -8(%rsp)

.LBL_2_4:

        movsd     1608+__dlog10_la_CoutTab(%rip), %xmm0
        comisd    %xmm0, %xmm2
        jbe       .LBL_2_8


        movaps    %xmm2, %xmm1
        subsd     .L_2il0floatpacket.89(%rip), %xmm1
        movsd     %xmm1, -16(%rsp)
        andb      $127, -9(%rsp)
        movsd     -16(%rsp), %xmm0
        comisd    1592+__dlog10_la_CoutTab(%rip), %xmm0
        jbe       .LBL_2_7


        movsd     %xmm2, -16(%rsp)
        pxor      %xmm7, %xmm7
        movzwl    -10(%rsp), %edi
        lea       __dlog10_la_CoutTab(%rip), %r10
        andl      $-32753, %edi
        addl      $16368, %edi
        movw      %di, -10(%rsp)
        movsd     -16(%rsp), %xmm3
        movaps    %xmm3, %xmm1
        movaps    %xmm3, %xmm2
        movsd     1688+__dlog10_la_CoutTab(%rip), %xmm5
        movzwl    -2(%rsp), %edx
        andl      $32752, %edx
        addsd     1576+__dlog10_la_CoutTab(%rip), %xmm1
        addsd     1584+__dlog10_la_CoutTab(%rip), %xmm2
        movsd     %xmm1, -24(%rsp)
        movl      -24(%rsp), %r8d
        movsd     %xmm2, -24(%rsp)
        andl      $127, %r8d
        movsd     -24(%rsp), %xmm8
        movsd     1560+__dlog10_la_CoutTab(%rip), %xmm9
        movsd     1568+__dlog10_la_CoutTab(%rip), %xmm0
        shrl      $4, %edx
        subsd     1584+__dlog10_la_CoutTab(%rip), %xmm8
        lea       (%r8,%r8,2), %r9d
        movsd     (%r10,%r9,8), %xmm6
        lea       -1023(%rcx,%rdx), %ecx
        cvtsi2sd  %ecx, %xmm7
        subsd     %xmm8, %xmm3
        mulsd     %xmm6, %xmm8
        mulsd     %xmm7, %xmm9
        subsd     1624+__dlog10_la_CoutTab(%rip), %xmm8
        mulsd     %xmm3, %xmm6
        mulsd     %xmm0, %xmm7
        addsd     8(%r10,%r9,8), %xmm9
        addsd     16(%r10,%r9,8), %xmm7
        addsd     %xmm8, %xmm9
        movaps    %xmm8, %xmm4
        addsd     %xmm6, %xmm4
        mulsd     %xmm4, %xmm5
        addsd     1680+__dlog10_la_CoutTab(%rip), %xmm5
        mulsd     %xmm4, %xmm5
        addsd     1672+__dlog10_la_CoutTab(%rip), %xmm5
        mulsd     %xmm4, %xmm5
        addsd     1664+__dlog10_la_CoutTab(%rip), %xmm5
        mulsd     %xmm4, %xmm5
        addsd     1656+__dlog10_la_CoutTab(%rip), %xmm5
        mulsd     %xmm4, %xmm5
        addsd     1648+__dlog10_la_CoutTab(%rip), %xmm5
        mulsd     %xmm4, %xmm5
        addsd     1640+__dlog10_la_CoutTab(%rip), %xmm5
        mulsd     %xmm4, %xmm5
        addsd     1632+__dlog10_la_CoutTab(%rip), %xmm5
        mulsd     %xmm5, %xmm8
        mulsd     %xmm6, %xmm5
        addsd     %xmm5, %xmm7
        addsd     %xmm6, %xmm7
        addsd     %xmm7, %xmm8
        addsd     %xmm8, %xmm9
        movsd     %xmm9, (%rsi)
        ret

.LBL_2_7:

        movsd     1624+__dlog10_la_CoutTab(%rip), %xmm0
        mulsd     %xmm0, %xmm1
        movsd     1688+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1680+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1672+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1664+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1656+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1648+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1640+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1632+__dlog10_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     %xmm1, %xmm2
        movsd     %xmm2, (%rsi)
        ret

.LBL_2_8:

        ucomisd   %xmm0, %xmm2
        jp        .LBL_2_9
        je        .LBL_2_11

.LBL_2_9:

        divsd     %xmm0, %xmm0
        movsd     %xmm0, (%rsi)
        movl      $1, %eax


        ret

.LBL_2_11:

        movsd     1616+__dlog10_la_CoutTab(%rip), %xmm1
        movl      $2, %eax
        xorps     .L_2il0floatpacket.88(%rip), %xmm1
        divsd     %xmm0, %xmm1
        movsd     %xmm1, (%rsi)
        ret

.LBL_2_12:

        movb      7(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_14

.LBL_2_13:

        movsd     (%rdi), %xmm0
        mulsd     %xmm0, %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_14:

        testl     $1048575, 4(%rdi)
        jne       .LBL_2_13


        cmpl      $0, (%rdi)
        jne       .LBL_2_13


        movsd     1608+__dlog10_la_CoutTab(%rip), %xmm0
        movl      $1, %eax
        divsd     %xmm0, %xmm0
        movsd     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dlog10_cout_rare_internal,@function
	.size	__svml_dlog10_cout_rare_internal,.-__svml_dlog10_cout_rare_internal
..LN__svml_dlog10_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dlog10_data_internal_avx512:
	.long	0
	.long	0
	.long	721479184
	.long	3214603769
	.long	3590975466
	.long	3215601833
	.long	1438937368
	.long	3216186160
	.long	948324365
	.long	3216559896
	.long	2869828094
	.long	3216915393
	.long	516509563
	.long	3217142759
	.long	2145647618
	.long	3217304702
	.long	733771779
	.long	1069546492
	.long	3513866211
	.long	1069249052
	.long	3459676924
	.long	1068963280
	.long	1085767695
	.long	1068688295
	.long	3613830132
	.long	1068347678
	.long	1803457173
	.long	1067836310
	.long	3436756955
	.long	1067234191
	.long	930630721
	.long	1066155272
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
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	0
	.long	1072168960
	.long	675808112
	.long	1068024536
	.long	675808112
	.long	1068024536
	.long	675808112
	.long	1068024536
	.long	675808112
	.long	1068024536
	.long	675808112
	.long	1068024536
	.long	675808112
	.long	1068024536
	.long	675808112
	.long	1068024536
	.long	675808112
	.long	1068024536
	.long	2516752404
	.long	3215710221
	.long	2516752404
	.long	3215710221
	.long	2516752404
	.long	3215710221
	.long	2516752404
	.long	3215710221
	.long	2516752404
	.long	3215710221
	.long	2516752404
	.long	3215710221
	.long	2516752404
	.long	3215710221
	.long	2516752404
	.long	3215710221
	.long	4085995682
	.long	1068483574
	.long	4085995682
	.long	1068483574
	.long	4085995682
	.long	1068483574
	.long	4085995682
	.long	1068483574
	.long	4085995682
	.long	1068483574
	.long	4085995682
	.long	1068483574
	.long	4085995682
	.long	1068483574
	.long	4085995682
	.long	1068483574
	.long	879025280
	.long	3216148390
	.long	879025280
	.long	3216148390
	.long	879025280
	.long	3216148390
	.long	879025280
	.long	3216148390
	.long	879025280
	.long	3216148390
	.long	879025280
	.long	3216148390
	.long	879025280
	.long	3216148390
	.long	879025280
	.long	3216148390
	.long	2004821977
	.long	1068907618
	.long	2004821977
	.long	1068907618
	.long	2004821977
	.long	1068907618
	.long	2004821977
	.long	1068907618
	.long	2004821977
	.long	1068907618
	.long	2004821977
	.long	1068907618
	.long	2004821977
	.long	1068907618
	.long	2004821977
	.long	1068907618
	.long	356255395
	.long	3216755579
	.long	356255395
	.long	3216755579
	.long	356255395
	.long	3216755579
	.long	356255395
	.long	3216755579
	.long	356255395
	.long	3216755579
	.long	356255395
	.long	3216755579
	.long	356255395
	.long	3216755579
	.long	356255395
	.long	3216755579
	.long	1668235916
	.long	1069713319
	.long	1668235916
	.long	1069713319
	.long	1668235916
	.long	1069713319
	.long	1668235916
	.long	1069713319
	.long	1668235916
	.long	1069713319
	.long	1668235916
	.long	1069713319
	.long	1668235916
	.long	1069713319
	.long	1668235916
	.long	1069713319
	.long	354870491
	.long	3217804155
	.long	354870491
	.long	3217804155
	.long	354870491
	.long	3217804155
	.long	354870491
	.long	3217804155
	.long	354870491
	.long	3217804155
	.long	354870491
	.long	3217804155
	.long	354870491
	.long	3217804155
	.long	354870491
	.long	3217804155
	.long	354870542
	.long	1071369083
	.long	354870542
	.long	1071369083
	.long	354870542
	.long	1071369083
	.long	354870542
	.long	1071369083
	.long	354870542
	.long	1071369083
	.long	354870542
	.long	1071369083
	.long	354870542
	.long	1071369083
	.long	354870542
	.long	1071369083
	.long	1352628735
	.long	1070810131
	.long	1352628735
	.long	1070810131
	.long	1352628735
	.long	1070810131
	.long	1352628735
	.long	1070810131
	.long	1352628735
	.long	1070810131
	.long	1352628735
	.long	1070810131
	.long	1352628735
	.long	1070810131
	.long	1352628735
	.long	1070810131
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
	.long	0
	.long	1048576
	.long	0
	.long	1048576
	.long	0
	.long	1048576
	.long	0
	.long	1048576
	.long	0
	.long	1048576
	.long	0
	.long	1048576
	.long	0
	.long	1048576
	.long	0
	.long	1048576
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.long	4294967295
	.long	2146435071
	.type	__svml_dlog10_data_internal_avx512,@object
	.size	__svml_dlog10_data_internal_avx512,1152
	.align 32
__dlog10_la_CoutTab:
	.long	0
	.long	1071366144
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1071337728
	.long	184549376
	.long	1065092008
	.long	2099961998
	.long	3178897324
	.long	0
	.long	1071309312
	.long	931135488
	.long	1066155272
	.long	2365712557
	.long	3178155773
	.long	0
	.long	1071280896
	.long	603979776
	.long	1066752445
	.long	709057215
	.long	1031474920
	.long	0
	.long	1071252480
	.long	3437232128
	.long	1067234191
	.long	1515412199
	.long	3179085970
	.long	0
	.long	1071238272
	.long	1105723392
	.long	1067464226
	.long	153915826
	.long	3178000698
	.long	0
	.long	1071209856
	.long	3328442368
	.long	1067711223
	.long	3899912278
	.long	3177135692
	.long	0
	.long	1071181440
	.long	341835776
	.long	1067962480
	.long	2509208190
	.long	3176531222
	.long	0
	.long	1071167232
	.long	2884632576
	.long	1068089751
	.long	1030636902
	.long	1025224143
	.long	0
	.long	1071138816
	.long	3613917184
	.long	1068347678
	.long	3527163461
	.long	3177529532
	.long	0
	.long	1071124608
	.long	3549954048
	.long	1068478374
	.long	3498894081
	.long	3173000425
	.long	0
	.long	1071096192
	.long	1623785472
	.long	1068621140
	.long	2889825554
	.long	3176375375
	.long	0
	.long	1071081984
	.long	1085800448
	.long	1068688295
	.long	4015256301
	.long	3177184346
	.long	0
	.long	1071053568
	.long	3652976640
	.long	1068824490
	.long	3600693529
	.long	3175753877
	.long	0
	.long	1071039360
	.long	1592393728
	.long	1068893555
	.long	231073830
	.long	3177087939
	.long	0
	.long	1071025152
	.long	3459645440
	.long	1068963280
	.long	1740576090
	.long	1029619435
	.long	0
	.long	1070996736
	.long	3774611456
	.long	1069104765
	.long	3858552785
	.long	1028603845
	.long	0
	.long	1070982528
	.long	845086720
	.long	1069176552
	.long	3138879731
	.long	1029120443
	.long	0
	.long	1070968320
	.long	3513843712
	.long	1069249052
	.long	2107125367
	.long	1029044389
	.long	0
	.long	1070954112
	.long	434503680
	.long	1069322282
	.long	3827602229
	.long	1028932700
	.long	0
	.long	1070939904
	.long	3613851648
	.long	1069396254
	.long	1223751955
	.long	3176465139
	.long	0
	.long	1070911488
	.long	733741056
	.long	1069546492
	.long	1625232067
	.long	1029570781
	.long	0
	.long	1070897280
	.long	1511620608
	.long	1069585154
	.long	3044605139
	.long	1028090775
	.long	0
	.long	1070883072
	.long	1337196544
	.long	1069623706
	.long	2602639001
	.long	3175938675
	.long	0
	.long	1070868864
	.long	2572533760
	.long	1069662670
	.long	3067107955
	.long	1022933137
	.long	0
	.long	1070854656
	.long	559611904
	.long	1069702056
	.long	764145786
	.long	3174041535
	.long	0
	.long	1070840448
	.long	485818368
	.long	1069741872
	.long	2037567072
	.long	3175580956
	.long	0
	.long	1070826240
	.long	259604480
	.long	1069782128
	.long	4012068429
	.long	1027865895
	.long	0
	.long	1070812032
	.long	3454042112
	.long	1069822833
	.long	2867680007
	.long	3174202478
	.long	0
	.long	1070797824
	.long	2188754944
	.long	1069863999
	.long	2538655286
	.long	3175840981
	.long	0
	.long	1070783616
	.long	2965241856
	.long	1069905635
	.long	1338936972
	.long	3176093950
	.long	0
	.long	1070769408
	.long	966279168
	.long	1069947753
	.long	1774547674
	.long	3175051484
	.long	0
	.long	1070755200
	.long	1604042752
	.long	1069990363
	.long	2557470738
	.long	3174667448
	.long	0
	.long	1070740992
	.long	3417833472
	.long	1070033477
	.long	2268255117
	.long	3175678264
	.long	0
	.long	1070740992
	.long	3417833472
	.long	1070033477
	.long	2268255117
	.long	3175678264
	.long	0
	.long	1070726784
	.long	2451292160
	.long	1070077108
	.long	3757728941
	.long	1027943275
	.long	0
	.long	1070712576
	.long	929644544
	.long	1070121268
	.long	899045708
	.long	1027944939
	.long	0
	.long	1070698368
	.long	3057254400
	.long	1070165969
	.long	3880649376
	.long	3172972504
	.long	0
	.long	1070684160
	.long	2231091200
	.long	1070211226
	.long	521319256
	.long	1027600177
	.long	0
	.long	1070684160
	.long	2231091200
	.long	1070211226
	.long	521319256
	.long	1027600177
	.long	0
	.long	1070669952
	.long	2620162048
	.long	1070257052
	.long	1385613369
	.long	3176104036
	.long	0
	.long	1070655744
	.long	2096726016
	.long	1070303462
	.long	3138305819
	.long	3173646777
	.long	0
	.long	1070641536
	.long	944717824
	.long	1070350471
	.long	1065120110
	.long	1027539054
	.long	0
	.long	1070641536
	.long	944717824
	.long	1070350471
	.long	1065120110
	.long	1027539054
	.long	0
	.long	1070627328
	.long	1985789952
	.long	1070398094
	.long	3635943864
	.long	3173136490
	.long	0
	.long	1070613120
	.long	2123825152
	.long	1070446348
	.long	1125219725
	.long	3175615738
	.long	0
	.long	1070598912
	.long	1078378496
	.long	1070495250
	.long	603852726
	.long	3174570526
	.long	0
	.long	1070598912
	.long	1078378496
	.long	1070495250
	.long	603852726
	.long	3174570526
	.long	0
	.long	1070573312
	.long	1537933312
	.long	1070544817
	.long	998069198
	.long	1026662908
	.long	0
	.long	1070544896
	.long	733773824
	.long	1070595068
	.long	4061058002
	.long	3174036009
	.long	0
	.long	1070544896
	.long	733773824
	.long	1070595068
	.long	4061058002
	.long	3174036009
	.long	0
	.long	1070516480
	.long	3897544704
	.long	1070621058
	.long	951856294
	.long	1026731877
	.long	0
	.long	1070516480
	.long	3897544704
	.long	1070621058
	.long	951856294
	.long	1026731877
	.long	0
	.long	1070488064
	.long	493535232
	.long	1070646897
	.long	3852369308
	.long	3173264746
	.long	0
	.long	1070459648
	.long	463249408
	.long	1070673107
	.long	2853152111
	.long	3174564937
	.long	0
	.long	1070459648
	.long	463249408
	.long	1070673107
	.long	2853152111
	.long	3174564937
	.long	0
	.long	1070431232
	.long	3186585600
	.long	1070699699
	.long	1874718356
	.long	3174139933
	.long	0
	.long	1070431232
	.long	3186585600
	.long	1070699699
	.long	1874718356
	.long	3174139933
	.long	0
	.long	1070402816
	.long	1525858304
	.long	1070726686
	.long	3039843523
	.long	1024724665
	.long	0
	.long	1070402816
	.long	1525858304
	.long	1070726686
	.long	3039843523
	.long	1024724665
	.long	0
	.long	1070374400
	.long	3425300480
	.long	1070754078
	.long	1303046649
	.long	1022401701
	.long	0
	.long	1070374400
	.long	3425300480
	.long	1070754078
	.long	1303046649
	.long	1022401701
	.long	0
	.long	1070345984
	.long	1980465152
	.long	1070781889
	.long	3188656319
	.long	1027271390
	.long	0
	.long	1070345984
	.long	1980465152
	.long	1070781889
	.long	3188656319
	.long	1027271390
	.long	0
	.long	1070317568
	.long	1352630272
	.long	1070810131
	.long	3090895658
	.long	3174564915
	.long	1352630272
	.long	1070810131
	.long	3090895658
	.long	3174564915
	.long	64
	.long	1120927744
	.long	0
	.long	1096810496
	.long	0
	.long	1064828928
	.long	0
	.long	1135607808
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	1071366144
	.long	3207479559
	.long	1062894188
	.long	3698831637
	.long	3220339442
	.long	3700832817
	.long	1073506818
	.long	1691624569
	.long	3221787401
	.long	2065628764
	.long	1075227551
	.long	1770847080
	.long	3223701774
	.long	3786517112
	.long	1077250450
	.long	1316351650
	.long	3225793313
	.type	__dlog10_la_CoutTab,@object
	.size	__dlog10_la_CoutTab,1696
	.align 16
.L_2il0floatpacket.88:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.88,@object
	.size	.L_2il0floatpacket.88,16
	.align 8
.L_2il0floatpacket.89:
	.long	0x00000000,0x3ff00000
	.type	.L_2il0floatpacket.89,@object
	.size	.L_2il0floatpacket.89,8
