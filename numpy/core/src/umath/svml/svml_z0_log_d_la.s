/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *
 *  *  log(x) = -log(Rcp) + log(Rcp*x),
 *  *    where Rcp ~ 1/x (accuracy ~9 bits, obtained by rounding HW approximation to 1+9 mantissa bits)
 *  *
 *  *   Reduced argument R=Rcp*x-1 is used to approximate log(1+R) as polynomial
 *  *
 *  *   log(Rcp) = exponent_Rcp*log(2) + log(mantissa_Rcp)
 *  *     -log(mantissa_Rcp) is obtained from a lookup table, accessed by a 9-bit index
 *  
 */


	.text
.L_2__routine_start___svml_log8_z0_0:

	.align    16,0x90
	.globl __svml_log8

__svml_log8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vgetmantpd $8, {sae}, %zmm0, %zmm6
        vmovups   128+__svml_dlog_data_internal_avx512(%rip), %zmm4
        vmovups   384+__svml_dlog_data_internal_avx512(%rip), %zmm11
        vmovups   448+__svml_dlog_data_internal_avx512(%rip), %zmm14
        vmovups   512+__svml_dlog_data_internal_avx512(%rip), %zmm12
        vmovups   256+__svml_dlog_data_internal_avx512(%rip), %zmm10

/* Start polynomial evaluation */
        vmovups   320+__svml_dlog_data_internal_avx512(%rip), %zmm1
        vmovups   640+__svml_dlog_data_internal_avx512(%rip), %zmm13
        vmovups   704+__svml_dlog_data_internal_avx512(%rip), %zmm15

/* Prepare exponent correction: DblRcp<0.75? */
        vmovups   192+__svml_dlog_data_internal_avx512(%rip), %zmm3

/* Table lookup */
        vmovups   __svml_dlog_data_internal_avx512(%rip), %zmm5

/* GetExp(x) */
        vgetexppd {sae}, %zmm0, %zmm7

/* DblRcp ~ 1/Mantissa */
        vrcp14pd  %zmm6, %zmm8

/* x<=0? */
        vfpclasspd $94, %zmm0, %k0

/* round DblRcp to 4 fractional bits (RN mode, no Precision exception) */
        vrndscalepd $88, {sae}, %zmm8, %zmm2
        kmovw     %k0, %edx

/* Reduced argument: R = DblRcp*Mantissa - 1 */
        vfmsub213pd {rn-sae}, %zmm4, %zmm2, %zmm6
        vcmppd    $17, {sae}, %zmm3, %zmm2, %k1
        vfmadd231pd {rn-sae}, %zmm6, %zmm11, %zmm14
        vmovups   576+__svml_dlog_data_internal_avx512(%rip), %zmm11
        vfmadd231pd {rn-sae}, %zmm6, %zmm10, %zmm1
        vfmadd231pd {rn-sae}, %zmm6, %zmm13, %zmm15

/* R^2 */
        vmulpd    {rn-sae}, %zmm6, %zmm6, %zmm8
        vfmadd231pd {rn-sae}, %zmm6, %zmm12, %zmm11

/* Prepare table index */
        vpsrlq    $48, %zmm2, %zmm9

/* add 1 to Expon if DblRcp<0.75 */
        vaddpd    {rn-sae}, %zmm4, %zmm7, %zmm7{%k1}
        vmulpd    {rn-sae}, %zmm8, %zmm8, %zmm12
        vfmadd213pd {rn-sae}, %zmm14, %zmm8, %zmm1
        vfmadd213pd {rn-sae}, %zmm15, %zmm8, %zmm11
        vpermt2pd 64+__svml_dlog_data_internal_avx512(%rip), %zmm9, %zmm5

/* polynomial */
        vfmadd213pd {rn-sae}, %zmm11, %zmm12, %zmm1
        vfmadd213pd {rn-sae}, %zmm6, %zmm8, %zmm1
        vaddpd    {rn-sae}, %zmm1, %zmm5, %zmm2
        vmovups   768+__svml_dlog_data_internal_avx512(%rip), %zmm1
        vfmadd213pd {rn-sae}, %zmm2, %zmm1, %zmm7
        testl     %edx, %edx
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        vmovaps   %zmm7, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm7, 128(%rsp)
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
        vmovups   128(%rsp), %zmm7
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

        call      __svml_dlog_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log8,@function
	.size	__svml_log8,.-__svml_log8
..LN__svml_log8.0:

.L_2__routine_start___svml_dlog_cout_rare_internal_1:

	.align    16,0x90

__svml_dlog_cout_rare_internal:


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


        mulsd     1600+__dlog_la_CoutTab(%rip), %xmm2
        movl      $-60, %ecx
        movsd     %xmm2, -8(%rsp)

.LBL_2_4:

        movsd     1608+__dlog_la_CoutTab(%rip), %xmm0
        comisd    %xmm0, %xmm2
        jbe       .LBL_2_8


        movsd     .L_2il0floatpacket.80(%rip), %xmm3
        movaps    %xmm2, %xmm1
        subsd     %xmm3, %xmm1
        movsd     %xmm1, -16(%rsp)
        andb      $127, -9(%rsp)
        movsd     -16(%rsp), %xmm0
        comisd    1592+__dlog_la_CoutTab(%rip), %xmm0
        jbe       .LBL_2_7


        movsd     %xmm2, -16(%rsp)
        pxor      %xmm7, %xmm7
        movzwl    -10(%rsp), %edi
        lea       __dlog_la_CoutTab(%rip), %r10
        andl      $-32753, %edi
        addl      $16368, %edi
        movw      %di, -10(%rsp)
        movsd     -16(%rsp), %xmm4
        movaps    %xmm4, %xmm1
        movaps    %xmm4, %xmm2
        movsd     1672+__dlog_la_CoutTab(%rip), %xmm8
        movzwl    -2(%rsp), %edx
        andl      $32752, %edx
        addsd     1576+__dlog_la_CoutTab(%rip), %xmm1
        addsd     1584+__dlog_la_CoutTab(%rip), %xmm2
        movsd     %xmm1, -24(%rsp)
        movl      -24(%rsp), %r8d
        movsd     %xmm2, -24(%rsp)
        andl      $127, %r8d
        movsd     -24(%rsp), %xmm5
        movsd     1560+__dlog_la_CoutTab(%rip), %xmm9
        movsd     1568+__dlog_la_CoutTab(%rip), %xmm0
        shrl      $4, %edx
        subsd     1584+__dlog_la_CoutTab(%rip), %xmm5
        lea       (%r8,%r8,2), %r9d
        movsd     (%r10,%r9,8), %xmm6
        lea       -1023(%rcx,%rdx), %ecx
        cvtsi2sd  %ecx, %xmm7
        subsd     %xmm5, %xmm4
        mulsd     %xmm6, %xmm5
        mulsd     %xmm7, %xmm9
        subsd     %xmm3, %xmm5
        mulsd     %xmm4, %xmm6
        mulsd     %xmm0, %xmm7
        addsd     8(%r10,%r9,8), %xmm9
        addsd     16(%r10,%r9,8), %xmm7
        addsd     %xmm5, %xmm9
        addsd     %xmm6, %xmm7
        movaps    %xmm5, %xmm3
        addsd     %xmm6, %xmm3
        mulsd     %xmm3, %xmm8
        addsd     1664+__dlog_la_CoutTab(%rip), %xmm8
        mulsd     %xmm3, %xmm8
        addsd     1656+__dlog_la_CoutTab(%rip), %xmm8
        mulsd     %xmm3, %xmm8
        addsd     1648+__dlog_la_CoutTab(%rip), %xmm8
        mulsd     %xmm3, %xmm8
        addsd     1640+__dlog_la_CoutTab(%rip), %xmm8
        mulsd     %xmm3, %xmm8
        addsd     1632+__dlog_la_CoutTab(%rip), %xmm8
        mulsd     %xmm3, %xmm8
        addsd     1624+__dlog_la_CoutTab(%rip), %xmm8
        mulsd     %xmm3, %xmm8
        mulsd     %xmm3, %xmm8
        addsd     %xmm7, %xmm8
        addsd     %xmm8, %xmm9
        movsd     %xmm9, (%rsi)
        ret

.LBL_2_7:

        movsd     1672+__dlog_la_CoutTab(%rip), %xmm0
        mulsd     %xmm1, %xmm0
        addsd     1664+__dlog_la_CoutTab(%rip), %xmm0
        mulsd     %xmm1, %xmm0
        addsd     1656+__dlog_la_CoutTab(%rip), %xmm0
        mulsd     %xmm1, %xmm0
        addsd     1648+__dlog_la_CoutTab(%rip), %xmm0
        mulsd     %xmm1, %xmm0
        addsd     1640+__dlog_la_CoutTab(%rip), %xmm0
        mulsd     %xmm1, %xmm0
        addsd     1632+__dlog_la_CoutTab(%rip), %xmm0
        mulsd     %xmm1, %xmm0
        addsd     1624+__dlog_la_CoutTab(%rip), %xmm0
        mulsd     %xmm1, %xmm0
        mulsd     %xmm1, %xmm0
        addsd     %xmm1, %xmm0
        movsd     %xmm0, (%rsi)
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

        movsd     1616+__dlog_la_CoutTab(%rip), %xmm1
        movl      $2, %eax
        xorps     .L_2il0floatpacket.79(%rip), %xmm1
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


        movsd     1608+__dlog_la_CoutTab(%rip), %xmm0
        movl      $1, %eax
        divsd     %xmm0, %xmm0
        movsd     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dlog_cout_rare_internal,@function
	.size	__svml_dlog_cout_rare_internal,.-__svml_dlog_cout_rare_internal
..LN__svml_dlog_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dlog_data_internal_avx512:
	.long	0
	.long	0
	.long	3222364838
	.long	3215919664
	.long	1848308454
	.long	3216910087
	.long	1890030548
	.long	3217424176
	.long	3348797986
	.long	3217854455
	.long	2880153102
	.long	3218171740
	.long	3256639170
	.long	3218366859
	.long	4139503623
	.long	3218553303
	.long	288676754
	.long	1070754146
	.long	1823716217
	.long	1070569756
	.long	1015732707
	.long	1070240749
	.long	1800682930
	.long	1069924160
	.long	2183652476
	.long	1069619086
	.long	1566132617
	.long	1069101918
	.long	3047045639
	.long	1068533144
	.long	2313634884
	.long	1067467101
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
	.long	815627376
	.long	1069318605
	.long	815627376
	.long	1069318605
	.long	815627376
	.long	1069318605
	.long	815627376
	.long	1069318605
	.long	815627376
	.long	1069318605
	.long	815627376
	.long	1069318605
	.long	815627376
	.long	1069318605
	.long	815627376
	.long	1069318605
	.long	2123607906
	.long	3217033013
	.long	2123607906
	.long	3217033013
	.long	2123607906
	.long	3217033013
	.long	2123607906
	.long	3217033013
	.long	2123607906
	.long	3217033013
	.long	2123607906
	.long	3217033013
	.long	2123607906
	.long	3217033013
	.long	2123607906
	.long	3217033013
	.long	2632876527
	.long	1069697314
	.long	2632876527
	.long	1069697314
	.long	2632876527
	.long	1069697314
	.long	2632876527
	.long	1069697314
	.long	2632876527
	.long	1069697314
	.long	2632876527
	.long	1069697314
	.long	2632876527
	.long	1069697314
	.long	2632876527
	.long	1069697314
	.long	4213758726
	.long	3217380691
	.long	4213758726
	.long	3217380691
	.long	4213758726
	.long	3217380691
	.long	4213758726
	.long	3217380691
	.long	4213758726
	.long	3217380691
	.long	4213758726
	.long	3217380691
	.long	4213758726
	.long	3217380691
	.long	4213758726
	.long	3217380691
	.long	2580324188
	.long	1070176665
	.long	2580324188
	.long	1070176665
	.long	2580324188
	.long	1070176665
	.long	2580324188
	.long	1070176665
	.long	2580324188
	.long	1070176665
	.long	2580324188
	.long	1070176665
	.long	2580324188
	.long	1070176665
	.long	2580324188
	.long	1070176665
	.long	787901
	.long	3218079744
	.long	787901
	.long	3218079744
	.long	787901
	.long	3218079744
	.long	787901
	.long	3218079744
	.long	787901
	.long	3218079744
	.long	787901
	.long	3218079744
	.long	787901
	.long	3218079744
	.long	787901
	.long	3218079744
	.long	1431655526
	.long	1070945621
	.long	1431655526
	.long	1070945621
	.long	1431655526
	.long	1070945621
	.long	1431655526
	.long	1070945621
	.long	1431655526
	.long	1070945621
	.long	1431655526
	.long	1070945621
	.long	1431655526
	.long	1070945621
	.long	1431655526
	.long	1070945621
	.long	4294967238
	.long	3219128319
	.long	4294967238
	.long	3219128319
	.long	4294967238
	.long	3219128319
	.long	4294967238
	.long	3219128319
	.long	4294967238
	.long	3219128319
	.long	4294967238
	.long	3219128319
	.long	4294967238
	.long	3219128319
	.long	4294967238
	.long	3219128319
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
	.type	__svml_dlog_data_internal_avx512,@object
	.size	__svml_dlog_data_internal_avx512,1088
	.align 32
__dlog_la_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1072660480
	.long	1486880768
	.long	1066410070
	.long	1813744607
	.long	3179892593
	.long	0
	.long	1072629760
	.long	377487360
	.long	1067416219
	.long	919019713
	.long	3179241129
	.long	0
	.long	1072599040
	.long	1513619456
	.long	1067944025
	.long	874573033
	.long	3178512940
	.long	0
	.long	1072570368
	.long	3221749760
	.long	1068427825
	.long	4181665006
	.long	3177478212
	.long	0
	.long	1072541696
	.long	4162322432
	.long	1068708823
	.long	627020255
	.long	1028629941
	.long	0
	.long	1072513024
	.long	183107584
	.long	1068957907
	.long	2376703469
	.long	1030233118
	.long	0
	.long	1072486400
	.long	1053425664
	.long	1069192557
	.long	696277142
	.long	1030474863
	.long	0
	.long	1072459776
	.long	3996123136
	.long	1069430535
	.long	2630798680
	.long	1028792016
	.long	0
	.long	1072435200
	.long	3452764160
	.long	1069600382
	.long	624954044
	.long	3177101741
	.long	0
	.long	1072409600
	.long	207650816
	.long	1069717971
	.long	3272735636
	.long	3175176575
	.long	0
	.long	1072386048
	.long	2647228416
	.long	1069827627
	.long	3594228712
	.long	1029303785
	.long	0
	.long	1072362496
	.long	2712010752
	.long	1069938736
	.long	3653242769
	.long	3176839013
	.long	0
	.long	1072338944
	.long	374439936
	.long	1070051337
	.long	4072775574
	.long	3176577495
	.long	0
	.long	1072316416
	.long	3707174912
	.long	1070160474
	.long	1486946159
	.long	1023930920
	.long	0
	.long	1072294912
	.long	1443954688
	.long	1070265993
	.long	293532967
	.long	3176278277
	.long	0
	.long	1072273408
	.long	127762432
	.long	1070372856
	.long	3404145447
	.long	3177023955
	.long	0
	.long	1072252928
	.long	2053832704
	.long	1070475911
	.long	1575076358
	.long	1029048544
	.long	0
	.long	1072232448
	.long	3194093568
	.long	1070580248
	.long	1864169120
	.long	1026866084
	.long	0
	.long	1072212992
	.long	3917201408
	.long	1070638340
	.long	2362145246
	.long	3175606197
	.long	0
	.long	1072193536
	.long	3417112576
	.long	1070689116
	.long	70087871
	.long	3174183577
	.long	0
	.long	1072175104
	.long	4226777088
	.long	1070737793
	.long	1620410586
	.long	3174700065
	.long	0
	.long	1072156672
	.long	3168870400
	.long	1070787042
	.long	311238082
	.long	1025781772
	.long	0
	.long	1072139264
	.long	2150580224
	.long	1070834092
	.long	1664262457
	.long	3175299224
	.long	0
	.long	1072120832
	.long	4095672320
	.long	1070884491
	.long	1657121015
	.long	3174674199
	.long	0
	.long	1072104448
	.long	2595577856
	.long	1070929805
	.long	2014006823
	.long	3175423830
	.long	0
	.long	1072087040
	.long	3747176448
	.long	1070978493
	.long	144991708
	.long	3171552042
	.long	0
	.long	1072070656
	.long	1050435584
	.long	1071024840
	.long	3386227432
	.long	1027876916
	.long	0
	.long	1072055296
	.long	255516672
	.long	1071068760
	.long	2637594316
	.long	1028049573
	.long	0
	.long	1072038912
	.long	1640783872
	.long	1071116120
	.long	893247007
	.long	1028452162
	.long	0
	.long	1072023552
	.long	2940411904
	.long	1071161011
	.long	813240633
	.long	1027664048
	.long	0
	.long	1072009216
	.long	882917376
	.long	1071203348
	.long	2376597551
	.long	3175828767
	.long	0
	.long	1071993856
	.long	213966848
	.long	1071249188
	.long	2977204125
	.long	1028350609
	.long	0
	.long	1071979520
	.long	2921504768
	.long	1071292428
	.long	523218347
	.long	1028007004
	.long	0
	.long	1071965184
	.long	3186655232
	.long	1071336119
	.long	2352907891
	.long	1026967097
	.long	0
	.long	1071951872
	.long	2653364224
	.long	1071377101
	.long	2453418583
	.long	3174349512
	.long	0
	.long	1071938560
	.long	3759783936
	.long	1071418487
	.long	3685870403
	.long	3175415611
	.long	0
	.long	1071925248
	.long	2468364288
	.long	1071460286
	.long	1578908842
	.long	3175510517
	.long	0
	.long	1071911936
	.long	81903616
	.long	1071502506
	.long	770710269
	.long	1026742353
	.long	0
	.long	1071899648
	.long	2799321088
	.long	1071541858
	.long	3822266185
	.long	1028434427
	.long	0
	.long	1071886336
	.long	2142265344
	.long	1071584911
	.long	175901806
	.long	3173871540
	.long	0
	.long	1071874048
	.long	2944024576
	.long	1071625048
	.long	2747360403
	.long	1027672159
	.long	0
	.long	1071862784
	.long	3434301440
	.long	1071653426
	.long	4194662196
	.long	3173893003
	.long	0
	.long	1071850496
	.long	1547755520
	.long	1071673870
	.long	4248764681
	.long	3172759087
	.long	0
	.long	1071839232
	.long	4246986752
	.long	1071692786
	.long	2840205638
	.long	3174430911
	.long	0
	.long	1071826944
	.long	3418390528
	.long	1071713619
	.long	3041880823
	.long	1025440860
	.long	0
	.long	1071816704
	.long	4143093760
	.long	1071731139
	.long	2727587401
	.long	3173965207
	.long	0
	.long	1071805440
	.long	3121326080
	.long	1071750582
	.long	3173887692
	.long	3174190163
	.long	0
	.long	1071794176
	.long	1852893184
	.long	1071770207
	.long	3951060252
	.long	1027348295
	.long	0
	.long	1071783936
	.long	3636379648
	.long	1071788208
	.long	1684924001
	.long	3174777086
	.long	0
	.long	1071773696
	.long	516505600
	.long	1071806366
	.long	429181199
	.long	3173211033
	.long	0
	.long	1071763456
	.long	4186185728
	.long	1071824681
	.long	2044904577
	.long	3174967132
	.long	0
	.long	1071753216
	.long	877596672
	.long	1071843159
	.long	1396318105
	.long	3173959727
	.long	0
	.long	1071742976
	.long	2912784384
	.long	1071861800
	.long	448136789
	.long	3174814192
	.long	0
	.long	1071733760
	.long	3722825728
	.long	1071878720
	.long	714165913
	.long	3173439560
	.long	0
	.long	1071723520
	.long	2522374144
	.long	1071897682
	.long	3227240353
	.long	3173394323
	.long	0
	.long	1071714304
	.long	4165410816
	.long	1071914895
	.long	1365684961
	.long	3174365060
	.long	0
	.long	1071705088
	.long	3477135360
	.long	1071932251
	.long	368482985
	.long	3174140821
	.long	0
	.long	1071695872
	.long	2079455232
	.long	1071949752
	.long	1320576317
	.long	1026822714
	.long	0
	.long	1071687680
	.long	851795968
	.long	1071965432
	.long	3702467026
	.long	1025224125
	.long	0
	.long	1071678464
	.long	647743488
	.long	1071983213
	.long	772992109
	.long	3174038459
	.long	0
	.long	1071670272
	.long	26537984
	.long	1071999146
	.long	2360214276
	.long	3174861275
	.long	0
	.long	1071661056
	.long	1547061248
	.long	1072017216
	.long	2886781435
	.long	1026423395
	.long	0
	.long	1071652864
	.long	2854492160
	.long	1072033410
	.long	215631550
	.long	1025638968
	.long	0
	.long	1071644672
	.long	4277811200
	.long	1072049730
	.long	2479318832
	.long	1026487127
	.long	4277811200
	.long	1072049730
	.long	2479318832
	.long	1026487127
	.long	64
	.long	1120927744
	.long	0
	.long	1094713344
	.long	0
	.long	1065615360
	.long	0
	.long	1135607808
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	3219128320
	.long	1431655955
	.long	1070945621
	.long	610
	.long	3218079744
	.long	2545118337
	.long	1070176665
	.long	1378399119
	.long	3217380693
	.long	612435357
	.long	1069697472
	.long	94536557
	.long	3217031348
	.type	__dlog_la_CoutTab,@object
	.size	__dlog_la_CoutTab,1680
	.align 16
.L_2il0floatpacket.79:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.79,@object
	.size	.L_2il0floatpacket.79,16
	.align 8
.L_2il0floatpacket.80:
	.long	0x00000000,0x3ff00000
	.type	.L_2il0floatpacket.80,@object
	.size	.L_2il0floatpacket.80,8
