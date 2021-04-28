/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *
 *  *   Compute log(x+sqrt(x*x-1)) using RSQRT14/RSQRT28 for starting the
 *  *   square root approximation, and small table lookups for log (mapping to
 *  *   AVX3 permute instructions).
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_acosh8_z0_0:

	.align    16,0x90
	.globl __svml_acosh8

__svml_acosh8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_dacosh_data_internal_avx512(%rip), %zmm5

/* polynomial computation for small inputs */
        vmovups   576+__svml_dacosh_data_internal_avx512(%rip), %zmm13
        vmovups   640+__svml_dacosh_data_internal_avx512(%rip), %zmm14

/*
 * sqrt(1+x^2) ~ Sh + Sl + Sh*Eh*poly_s
 * poly_s = c1+c2*Eh+c3*Eh^2
 */
        vmovups   704+__svml_dacosh_data_internal_avx512(%rip), %zmm1
        vmovups   832+__svml_dacosh_data_internal_avx512(%rip), %zmm2
        vmovups   896+__svml_dacosh_data_internal_avx512(%rip), %zmm6

/* very large inputs ? */
        vmovups   448+__svml_dacosh_data_internal_avx512(%rip), %zmm15

/* out of range inputs? */
        vmovups   512+__svml_dacosh_data_internal_avx512(%rip), %zmm3

/* not a very small input ? */
        vmovups   384+__svml_dacosh_data_internal_avx512(%rip), %zmm10
        vmovaps   %zmm0, %zmm12

/* x^2 - 1 */
        vmovaps   %zmm5, %zmm11
        vfmsub231pd {rn-sae}, %zmm12, %zmm12, %zmm11
        vcmppd    $21, {sae}, %zmm15, %zmm12, %k2
        vcmppd    $22, {sae}, %zmm3, %zmm12, %k0
        vcmppd    $18, {sae}, %zmm5, %zmm12, %k1
        vrsqrt14pd %zmm11, %zmm4
        vcmppd    $21, {sae}, %zmm10, %zmm11, %k3
        vfmadd231pd {rn-sae}, %zmm11, %zmm13, %zmm14
        vmovups   768+__svml_dacosh_data_internal_avx512(%rip), %zmm13

/* Sh ~sqrt(-1+x^2) */
        vmulpd    {rn-sae}, %zmm4, %zmm11, %zmm9
        vmulpd    {rn-sae}, %zmm11, %zmm14, %zmm8

/* Sh+x */
        vaddpd    {rn-sae}, %zmm12, %zmm9, %zmm15
        kmovw     %k0, %edx
        kmovw     %k1, %eax

/* Shh */
        vsubpd    {rn-sae}, %zmm12, %zmm15, %zmm14

/* (Yh*R0)_low */
        vmovaps   %zmm11, %zmm0
        orl       %eax, %edx

/* rel. error term: Eh=1-Sh*R0 */
        vmovaps   %zmm5, %zmm7
        vfmsub213pd {rn-sae}, %zmm9, %zmm4, %zmm0
        vfnmadd231pd {rn-sae}, %zmm9, %zmm4, %zmm7

/* rel. error term: Eh=(1-Sh*R0)-Sl*R0 */
        vfnmadd231pd {rn-sae}, %zmm0, %zmm4, %zmm7

/* Shl */
        vsubpd    {rn-sae}, %zmm14, %zmm9, %zmm4
        vmovups   1472+__svml_dacosh_data_internal_avx512(%rip), %zmm14
        vfmadd231pd {rn-sae}, %zmm7, %zmm1, %zmm13
        vfmadd213pd {rn-sae}, %zmm2, %zmm7, %zmm13
        vfmadd213pd {rn-sae}, %zmm6, %zmm7, %zmm13

/* Sh*Eh */
        vmulpd    {rn-sae}, %zmm7, %zmm9, %zmm7

/* Sl + Sh*Eh*poly_s */
        vfmadd213pd {rn-sae}, %zmm0, %zmm13, %zmm7

/* polynomials */
        vmovups   1344+__svml_dacosh_data_internal_avx512(%rip), %zmm13

/* polynomial computation for small inputs */
        vaddpd    {rn-sae}, %zmm7, %zmm9, %zmm0

/* Xin0+Sl+Sh*Eh*poly_s ~ x+sqrt(1+x^2) */
        vaddpd    {rn-sae}, %zmm7, %zmm15, %zmm6
        vfmadd231pd {rn-sae}, %zmm0, %zmm8, %zmm0

/* fixup for very large inputs */
        vmovups   1216+__svml_dacosh_data_internal_avx512(%rip), %zmm8

/* Sl_high */
        vsubpd    {rn-sae}, %zmm15, %zmm6, %zmm9
        vmovups   1536+__svml_dacosh_data_internal_avx512(%rip), %zmm15
        vmulpd    {rn-sae}, %zmm8, %zmm12, %zmm6{%k2}

/* Sl_l */
        vsubpd    {rn-sae}, %zmm9, %zmm7, %zmm3
        vrcp14pd  %zmm6, %zmm1

/* Xin_low */
        vaddpd    {rn-sae}, %zmm4, %zmm3, %zmm7

/* Table lookups */
        vmovups   __svml_dacosh_data_internal_avx512(%rip), %zmm3

/* round reciprocal to 1+4b mantissas */
        vpaddq    1088+__svml_dacosh_data_internal_avx512(%rip), %zmm1, %zmm2

/* fixup for very large inputs */
        vxorpd    %zmm7, %zmm7, %zmm7{%k2}
        vmovups   1408+__svml_dacosh_data_internal_avx512(%rip), %zmm1
        vandpd    1152+__svml_dacosh_data_internal_avx512(%rip), %zmm2, %zmm8
        vmovups   128+__svml_dacosh_data_internal_avx512(%rip), %zmm2

/* Prepare table index */
        vpsrlq    $48, %zmm8, %zmm9

/* reduced argument for log(): (Rcp*Xin-1)+Rcp*Xin_low */
        vfmsub231pd {rn-sae}, %zmm8, %zmm6, %zmm5

/* exponents */
        vgetexppd {sae}, %zmm8, %zmm4
        vmovups   1280+__svml_dacosh_data_internal_avx512(%rip), %zmm6
        vpermt2pd 64+__svml_dacosh_data_internal_avx512(%rip), %zmm9, %zmm3
        vpermt2pd 192+__svml_dacosh_data_internal_avx512(%rip), %zmm9, %zmm2
        vsubpd    {rn-sae}, %zmm6, %zmm4, %zmm4{%k2}
        vfmadd231pd {rn-sae}, %zmm8, %zmm7, %zmm5
        vmovups   1600+__svml_dacosh_data_internal_avx512(%rip), %zmm6
        vmovups   1664+__svml_dacosh_data_internal_avx512(%rip), %zmm7

/* -K*L2H + Th */
        vmovups   1920+__svml_dacosh_data_internal_avx512(%rip), %zmm8

/* -K*L2L + Tl */
        vmovups   1984+__svml_dacosh_data_internal_avx512(%rip), %zmm9
        vfmadd231pd {rn-sae}, %zmm5, %zmm13, %zmm1
        vmovups   1792+__svml_dacosh_data_internal_avx512(%rip), %zmm13
        vfnmadd231pd {rn-sae}, %zmm4, %zmm8, %zmm3
        vfnmadd213pd {rn-sae}, %zmm2, %zmm9, %zmm4
        vfmadd213pd {rn-sae}, %zmm14, %zmm5, %zmm1
        vmovups   1728+__svml_dacosh_data_internal_avx512(%rip), %zmm2
        vmovups   1856+__svml_dacosh_data_internal_avx512(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm15, %zmm5, %zmm1

/* R^2 */
        vmulpd    {rn-sae}, %zmm5, %zmm5, %zmm15
        vfmadd213pd {rn-sae}, %zmm6, %zmm5, %zmm1
        vfmadd213pd {rn-sae}, %zmm7, %zmm5, %zmm1
        vfmadd213pd {rn-sae}, %zmm2, %zmm5, %zmm1
        vfmadd213pd {rn-sae}, %zmm13, %zmm5, %zmm1
        vfmadd213pd {rn-sae}, %zmm14, %zmm5, %zmm1

/* Tl + R^2*Poly */
        vfmadd213pd {rn-sae}, %zmm4, %zmm15, %zmm1

/* R+Tl + R^2*Poly */
        vaddpd    {rn-sae}, %zmm5, %zmm1, %zmm5
        vaddpd    {rn-sae}, %zmm5, %zmm3, %zmm0{%k3}
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

        vmovups   %zmm12, 64(%rsp)
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

        call      __svml_dacosh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_acosh8,@function
	.size	__svml_acosh8,.-__svml_acosh8
..LN__svml_acosh8.0:

.L_2__routine_start___svml_dacosh_cout_rare_internal_1:

	.align    16,0x90

__svml_dacosh_cout_rare_internal:


	.cfi_startproc
..L53:

        movzwl    6(%rdi), %eax
        andl      $32752, %eax
        movsd     (%rdi), %xmm0
        cmpl      $32752, %eax
        je        .LBL_2_9

.LBL_2_2:

        movsd     %xmm0, -8(%rsp)


        cmpl      $0, -8(%rsp)
        jne       .LBL_2_8


        cmpl      $1072693248, -4(%rsp)
        je        .LBL_2_7


        cmpl      $2146435072, -4(%rsp)
        jne       .LBL_2_8


        movsd     %xmm0, (%rsi)
        xorl      %eax, %eax
        ret

.LBL_2_7:

        movq      8+__dacosh_la_CoutTab(%rip), %rax
        movq      %rax, (%rsi)
        xorl      %eax, %eax
        ret

.LBL_2_8:

        movsd     16+__dacosh_la_CoutTab(%rip), %xmm0
        movl      $1, %eax
        mulsd     8+__dacosh_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_9:

        testl     $1048575, 4(%rdi)
        jne       .LBL_2_12


        cmpl      $0, (%rdi)
        je        .LBL_2_2

.LBL_2_12:

        mulsd     %xmm0, %xmm0
        xorl      %eax, %eax
        movsd     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dacosh_cout_rare_internal,@function
	.size	__svml_dacosh_cout_rare_internal,.-__svml_dacosh_cout_rare_internal
..LN__svml_dacosh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dacosh_data_internal_avx512:
	.long	0
	.long	0
	.long	3222405120
	.long	3215919664
	.long	1848311808
	.long	3216910087
	.long	1890025472
	.long	3217424176
	.long	3348791296
	.long	3217854455
	.long	2880159744
	.long	3218171740
	.long	3256631296
	.long	3218366859
	.long	4139499520
	.long	3218553303
	.long	3971973120
	.long	3218731811
	.long	3348791296
	.long	3218903031
	.long	1605304320
	.long	3219067535
	.long	3827638272
	.long	3219177074
	.long	1584414720
	.long	3219253343
	.long	860823552
	.long	3219326935
	.long	3896934400
	.long	3219398031
	.long	643547136
	.long	3219466797
	.long	0
	.long	0
	.long	3496399314
	.long	1028893491
	.long	720371772
	.long	1026176044
	.long	1944193543
	.long	3175338952
	.long	634920691
	.long	3175752108
	.long	1664625295
	.long	1029304828
	.long	192624563
	.long	3177103997
	.long	3796653051
	.long	3176138396
	.long	3062724207
	.long	3176680434
	.long	634920691
	.long	3176800684
	.long	1913570380
	.long	3174806221
	.long	825194088
	.long	3176465773
	.long	2335489660
	.long	3172599741
	.long	2497625109
	.long	1029604288
	.long	914782743
	.long	1029350199
	.long	3743595607
	.long	3175525305
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
	.long	0
	.long	1055916032
	.long	0
	.long	1055916032
	.long	0
	.long	1055916032
	.long	0
	.long	1055916032
	.long	0
	.long	1055916032
	.long	0
	.long	1055916032
	.long	0
	.long	1055916032
	.long	0
	.long	1055916032
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
	.long	0
	.long	1608515584
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
	.long	246350567
	.long	1068708642
	.long	246350567
	.long	1068708642
	.long	246350567
	.long	1068708642
	.long	246350567
	.long	1068708642
	.long	246350567
	.long	1068708642
	.long	246350567
	.long	1068708642
	.long	246350567
	.long	1068708642
	.long	246350567
	.long	1068708642
	.long	1431445118
	.long	3217380693
	.long	1431445118
	.long	3217380693
	.long	1431445118
	.long	3217380693
	.long	1431445118
	.long	3217380693
	.long	1431445118
	.long	3217380693
	.long	1431445118
	.long	3217380693
	.long	1431445118
	.long	3217380693
	.long	1431445118
	.long	3217380693
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	26490386
	.long	1070694400
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	20643840
	.long	1070858240
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
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
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
	.long	0
	.long	1071120384
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
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	32768
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	4294901760
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1069547520
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	0
	.long	1074790400
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1075921768
	.long	3216615856
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	1847891832
	.long	1069318246
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	2315602889
	.long	3217031163
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	4145174257
	.long	1069697314
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	1436264246
	.long	3217380693
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	2579396527
	.long	1070176665
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	4294966373
	.long	3218079743
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	1431655617
	.long	1070945621
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
	.long	0
	.long	3219128320
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
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.long	3164471296
	.long	1031600026
	.type	__svml_dacosh_data_internal_avx512,@object
	.size	__svml_dacosh_data_internal_avx512,2048
	.align 8
__dacosh_la_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	2146435072
	.long	0
	.long	4293918720
	.type	__dacosh_la_CoutTab,@object
	.size	__dacosh_la_CoutTab,32
