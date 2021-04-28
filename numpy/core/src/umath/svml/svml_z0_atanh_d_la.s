/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *
 *  *   Compute 0.5*[log(1+x)-log(1-x)], using small table
 *  *   lookups that map to AVX3 permute instructions
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_atanh8_z0_0:

	.align    16,0x90
	.globl __svml_atanh8

__svml_atanh8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_datanh_data_internal_avx512(%rip), %zmm15

/* round reciprocals to 1+4b mantissas */
        vmovups   384+__svml_datanh_data_internal_avx512(%rip), %zmm6
        vmovups   448+__svml_datanh_data_internal_avx512(%rip), %zmm9
        vmovaps   %zmm0, %zmm2
        vandpd    320+__svml_datanh_data_internal_avx512(%rip), %zmm2, %zmm13

/* 1+y */
        vaddpd    {rn-sae}, %zmm15, %zmm13, %zmm0

/* 1-y */
        vsubpd    {rn-sae}, %zmm13, %zmm15, %zmm4
        vxorpd    %zmm13, %zmm2, %zmm1

/* Yp_high */
        vsubpd    {rn-sae}, %zmm15, %zmm0, %zmm7

/* -Ym_high */
        vsubpd    {rn-sae}, %zmm15, %zmm4, %zmm12

/* RcpP ~ 1/Yp */
        vrcp14pd  %zmm0, %zmm3

/* RcpM ~ 1/Ym */
        vrcp14pd  %zmm4, %zmm5

/* input outside (-1, 1) ? */
        vcmppd    $21, {sae}, %zmm15, %zmm13, %k0
        vpaddq    %zmm6, %zmm3, %zmm11
        vpaddq    %zmm6, %zmm5, %zmm10

/* Yp_low */
        vsubpd    {rn-sae}, %zmm7, %zmm13, %zmm8
        vandpd    %zmm9, %zmm11, %zmm14
        vandpd    %zmm9, %zmm10, %zmm3

/* Ym_low */
        vaddpd    {rn-sae}, %zmm12, %zmm13, %zmm12

/* Reduced argument: Rp = (RcpP*Yp - 1)+RcpP*Yp_low */
        vfmsub213pd {rn-sae}, %zmm15, %zmm14, %zmm0

/* Reduced argument: Rm = (RcpM*Ym - 1)+RcpM*Ym_low */
        vfmsub231pd {rn-sae}, %zmm3, %zmm4, %zmm15

/* exponents */
        vgetexppd {sae}, %zmm14, %zmm5
        vgetexppd {sae}, %zmm3, %zmm4

/* Table lookups */
        vmovups   __svml_datanh_data_internal_avx512(%rip), %zmm9
        vmovups   64+__svml_datanh_data_internal_avx512(%rip), %zmm13
        vmovups   128+__svml_datanh_data_internal_avx512(%rip), %zmm7
        vfmadd231pd {rn-sae}, %zmm14, %zmm8, %zmm0
        vfnmadd231pd {rn-sae}, %zmm3, %zmm12, %zmm15

/* Prepare table index */
        vpsrlq    $48, %zmm14, %zmm11
        vpsrlq    $48, %zmm3, %zmm8
        vmovups   192+__svml_datanh_data_internal_avx512(%rip), %zmm14

/* polynomials */
        vmovups   512+__svml_datanh_data_internal_avx512(%rip), %zmm3

/* Km-Kp */
        vsubpd    {rn-sae}, %zmm5, %zmm4, %zmm5
        vmovups   576+__svml_datanh_data_internal_avx512(%rip), %zmm4
        kmovw     %k0, %edx
        vmovaps   %zmm11, %zmm10
        vmovaps   %zmm4, %zmm6
        vpermi2pd %zmm13, %zmm9, %zmm10
        vpermi2pd %zmm14, %zmm7, %zmm11
        vpermt2pd %zmm13, %zmm8, %zmm9
        vpermt2pd %zmm14, %zmm8, %zmm7
        vmovups   640+__svml_datanh_data_internal_avx512(%rip), %zmm8
        vfmadd231pd {rn-sae}, %zmm0, %zmm3, %zmm6
        vfmadd231pd {rn-sae}, %zmm15, %zmm3, %zmm4
        vmovups   832+__svml_datanh_data_internal_avx512(%rip), %zmm13
        vmovups   896+__svml_datanh_data_internal_avx512(%rip), %zmm14
        vfmadd213pd {rn-sae}, %zmm8, %zmm0, %zmm6
        vfmadd213pd {rn-sae}, %zmm8, %zmm15, %zmm4
        vmovups   1024+__svml_datanh_data_internal_avx512(%rip), %zmm8
        vsubpd    {rn-sae}, %zmm11, %zmm7, %zmm12

/* table values */
        vsubpd    {rn-sae}, %zmm10, %zmm9, %zmm3
        vmovups   704+__svml_datanh_data_internal_avx512(%rip), %zmm7
        vmovups   768+__svml_datanh_data_internal_avx512(%rip), %zmm9

/* K*L2H + Th */
        vmovups   1152+__svml_datanh_data_internal_avx512(%rip), %zmm10

/* K*L2L + Tl */
        vmovups   1216+__svml_datanh_data_internal_avx512(%rip), %zmm11
        vfmadd213pd {rn-sae}, %zmm7, %zmm0, %zmm6
        vfmadd213pd {rn-sae}, %zmm7, %zmm15, %zmm4
        vmovups   960+__svml_datanh_data_internal_avx512(%rip), %zmm7
        vfmadd231pd {rn-sae}, %zmm5, %zmm10, %zmm3
        vfmadd213pd {rn-sae}, %zmm12, %zmm11, %zmm5
        vfmadd213pd {rn-sae}, %zmm9, %zmm0, %zmm6
        vfmadd213pd {rn-sae}, %zmm9, %zmm15, %zmm4
        vfmadd213pd {rn-sae}, %zmm13, %zmm0, %zmm6
        vfmadd213pd {rn-sae}, %zmm13, %zmm15, %zmm4
        vfmadd213pd {rn-sae}, %zmm14, %zmm0, %zmm6
        vfmadd213pd {rn-sae}, %zmm14, %zmm15, %zmm4
        vfmadd213pd {rn-sae}, %zmm7, %zmm0, %zmm6
        vfmadd213pd {rn-sae}, %zmm7, %zmm15, %zmm4
        vfmadd213pd {rn-sae}, %zmm8, %zmm0, %zmm6
        vfmadd213pd {rn-sae}, %zmm8, %zmm15, %zmm4

/* (K*L2L + Tl) + Rp*PolyP */
        vfmadd213pd {rn-sae}, %zmm5, %zmm0, %zmm6
        vorpd     1088+__svml_datanh_data_internal_avx512(%rip), %zmm1, %zmm0

/* (K*L2L + Tl) + Rp*PolyP -Rm*PolyM */
        vfnmadd213pd {rn-sae}, %zmm6, %zmm15, %zmm4
        vaddpd    {rn-sae}, %zmm4, %zmm3, %zmm1
        vmulpd    {rn-sae}, %zmm0, %zmm1, %zmm0
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

        vmovups   %zmm2, 64(%rsp)
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

        call      __svml_datanh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atanh8,@function
	.size	__svml_atanh8,.-__svml_atanh8
..LN__svml_atanh8.0:

.L_2__routine_start___svml_datanh_cout_rare_internal_1:

	.align    16,0x90

__svml_datanh_cout_rare_internal:


	.cfi_startproc
..L53:

        movzwl    6(%rdi), %eax
        andl      $32752, %eax
        movsd     (%rdi), %xmm0
        movb      7(%rdi), %dl
        andb      $127, %dl
        movsd     %xmm0, -8(%rsp)
        cmpl      $32752, %eax
        je        .LBL_2_6

.LBL_2_2:

        cmpl      $0, -8(%rsp)
        jne       .LBL_2_5


        movb      %dl, -1(%rsp)
        cmpl      $1072693248, -4(%rsp)
        jne       .LBL_2_5


        divsd     8+__datanh_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        movl      $2, %eax
        ret

.LBL_2_5:

        movsd     8+__datanh_la_CoutTab(%rip), %xmm0
        movl      $1, %eax
        mulsd     16+__datanh_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_6:

        testl     $1048575, 4(%rdi)
        jne       .LBL_2_8


        cmpl      $0, (%rdi)
        je        .LBL_2_2

.LBL_2_8:

        mulsd     %xmm0, %xmm0
        xorl      %eax, %eax
        movsd     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_datanh_cout_rare_internal,@function
	.size	__svml_datanh_cout_rare_internal,.-__svml_datanh_cout_rare_internal
..LN__svml_datanh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_datanh_data_internal_avx512:
	.long	0
	.long	0
	.long	3222274048
	.long	1068436016
	.long	1848246272
	.long	1069426439
	.long	1890058240
	.long	1069940528
	.long	3348824064
	.long	1070370807
	.long	2880143360
	.long	1070688092
	.long	3256647680
	.long	1070883211
	.long	4139515904
	.long	1071069655
	.long	3971973120
	.long	1071248163
	.long	3348791296
	.long	1071419383
	.long	1605304320
	.long	1071583887
	.long	3827646464
	.long	1071693426
	.long	1584414720
	.long	1071769695
	.long	860815360
	.long	1071843287
	.long	3896934400
	.long	1071914383
	.long	643547136
	.long	1071983149
	.long	0
	.long	0
	.long	399283991
	.long	1030105702
	.long	1028718588
	.long	1030642877
	.long	3808918910
	.long	3177909005
	.long	4136237123
	.long	3177805716
	.long	3462654649
	.long	1029900033
	.long	2051171366
	.long	3177225921
	.long	2396640771
	.long	3177708721
	.long	3062724207
	.long	1029196786
	.long	634920691
	.long	1029317036
	.long	1913570380
	.long	1027322573
	.long	1734886604
	.long	3177545033
	.long	2335489660
	.long	1025116093
	.long	3046154741
	.long	1029750303
	.long	914782743
	.long	3176833847
	.long	3743595607
	.long	1028041657
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
	.long	1087603010
	.long	1069318621
	.long	1087603010
	.long	1069318621
	.long	1087603010
	.long	1069318621
	.long	1087603010
	.long	1069318621
	.long	1087603010
	.long	1069318621
	.long	1087603010
	.long	1069318621
	.long	1087603010
	.long	1069318621
	.long	1087603010
	.long	1069318621
	.long	3090058096
	.long	3217033020
	.long	3090058096
	.long	3217033020
	.long	3090058096
	.long	3217033020
	.long	3090058096
	.long	3217033020
	.long	3090058096
	.long	3217033020
	.long	3090058096
	.long	3217033020
	.long	3090058096
	.long	3217033020
	.long	3090058096
	.long	3217033020
	.long	2566904552
	.long	1069697314
	.long	2566904552
	.long	1069697314
	.long	2566904552
	.long	1069697314
	.long	2566904552
	.long	1069697314
	.long	2566904552
	.long	1069697314
	.long	2566904552
	.long	1069697314
	.long	2566904552
	.long	1069697314
	.long	2566904552
	.long	1069697314
	.long	4168213957
	.long	3217380691
	.long	4168213957
	.long	3217380691
	.long	4168213957
	.long	3217380691
	.long	4168213957
	.long	3217380691
	.long	4168213957
	.long	3217380691
	.long	4168213957
	.long	3217380691
	.long	4168213957
	.long	3217380691
	.long	4168213957
	.long	3217380691
	.long	2580363594
	.long	1070176665
	.long	2580363594
	.long	1070176665
	.long	2580363594
	.long	1070176665
	.long	2580363594
	.long	1070176665
	.long	2580363594
	.long	1070176665
	.long	2580363594
	.long	1070176665
	.long	2580363594
	.long	1070176665
	.long	2580363594
	.long	1070176665
	.long	797185
	.long	3218079744
	.long	797185
	.long	3218079744
	.long	797185
	.long	3218079744
	.long	797185
	.long	3218079744
	.long	797185
	.long	3218079744
	.long	797185
	.long	3218079744
	.long	797185
	.long	3218079744
	.long	797185
	.long	3218079744
	.long	1431655522
	.long	1070945621
	.long	1431655522
	.long	1070945621
	.long	1431655522
	.long	1070945621
	.long	1431655522
	.long	1070945621
	.long	1431655522
	.long	1070945621
	.long	1431655522
	.long	1070945621
	.long	1431655522
	.long	1070945621
	.long	1431655522
	.long	1070945621
	.long	4294967237
	.long	3219128319
	.long	4294967237
	.long	3219128319
	.long	4294967237
	.long	3219128319
	.long	4294967237
	.long	3219128319
	.long	4294967237
	.long	3219128319
	.long	4294967237
	.long	3219128319
	.long	4294967237
	.long	3219128319
	.long	4294967237
	.long	3219128319
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
	.type	__svml_datanh_data_internal_avx512,@object
	.size	__svml_datanh_data_internal_avx512,1280
	.align 8
__datanh_la_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	2146435072
	.long	0
	.long	4293918720
	.type	__datanh_la_CoutTab,@object
	.size	__datanh_la_CoutTab,32
