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
.L_2__routine_start___svml_atanhf16_z0_0:

	.align    16,0x90
	.globl __svml_atanhf16

__svml_atanhf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_satanh_data_internal_avx512(%rip), %zmm4

/* round reciprocals to 1+5b mantissas */
        vmovups   384+__svml_satanh_data_internal_avx512(%rip), %zmm14
        vmovups   448+__svml_satanh_data_internal_avx512(%rip), %zmm1
        vmovaps   %zmm0, %zmm11
        vandps    320+__svml_satanh_data_internal_avx512(%rip), %zmm11, %zmm6

/* 1+y */
        vaddps    {rn-sae}, %zmm4, %zmm6, %zmm9

/* 1-y */
        vsubps    {rn-sae}, %zmm6, %zmm4, %zmm8
        vxorps    %zmm6, %zmm11, %zmm10

/* Yp_high */
        vsubps    {rn-sae}, %zmm4, %zmm9, %zmm2

/* -Ym_high */
        vsubps    {rn-sae}, %zmm4, %zmm8, %zmm5

/* RcpP ~ 1/Yp */
        vrcp14ps  %zmm9, %zmm12

/* RcpM ~ 1/Ym */
        vrcp14ps  %zmm8, %zmm13

/* input outside (-1, 1) ? */
        vcmpps    $21, {sae}, %zmm4, %zmm6, %k0
        vpaddd    %zmm14, %zmm12, %zmm15
        vpaddd    %zmm14, %zmm13, %zmm0

/* Yp_low */
        vsubps    {rn-sae}, %zmm2, %zmm6, %zmm3
        vandps    %zmm1, %zmm15, %zmm7
        vandps    %zmm1, %zmm0, %zmm12

/* Ym_low */
        vaddps    {rn-sae}, %zmm5, %zmm6, %zmm5

/* Reduced argument: Rp = (RcpP*Yp - 1)+RcpP*Yp_low */
        vfmsub213ps {rn-sae}, %zmm4, %zmm7, %zmm9

/* Reduced argument: Rm = (RcpM*Ym - 1)+RcpM*Ym_low */
        vfmsub231ps {rn-sae}, %zmm12, %zmm8, %zmm4
        vmovups   128+__svml_satanh_data_internal_avx512(%rip), %zmm8
        vmovups   192+__svml_satanh_data_internal_avx512(%rip), %zmm13

/* exponents */
        vgetexpps {sae}, %zmm7, %zmm15
        vfmadd231ps {rn-sae}, %zmm7, %zmm3, %zmm9

/* Table lookups */
        vmovups   __svml_satanh_data_internal_avx512(%rip), %zmm6
        vgetexpps {sae}, %zmm12, %zmm14
        vfnmadd231ps {rn-sae}, %zmm12, %zmm5, %zmm4

/* Prepare table index */
        vpsrld    $18, %zmm7, %zmm3
        vpsrld    $18, %zmm12, %zmm2
        vmovups   64+__svml_satanh_data_internal_avx512(%rip), %zmm7
        vmovups   640+__svml_satanh_data_internal_avx512(%rip), %zmm12

/* Km-Kp */
        vsubps    {rn-sae}, %zmm15, %zmm14, %zmm1
        kmovw     %k0, %edx
        vmovaps   %zmm3, %zmm0
        vpermi2ps %zmm13, %zmm8, %zmm3
        vpermt2ps %zmm13, %zmm2, %zmm8
        vpermi2ps %zmm7, %zmm6, %zmm0
        vpermt2ps %zmm7, %zmm2, %zmm6
        vsubps    {rn-sae}, %zmm3, %zmm8, %zmm5

/* K*L2H + Th */
        vmovups   832+__svml_satanh_data_internal_avx512(%rip), %zmm2

/* K*L2L + Tl */
        vmovups   896+__svml_satanh_data_internal_avx512(%rip), %zmm3

/* polynomials */
        vmovups   512+__svml_satanh_data_internal_avx512(%rip), %zmm7
        vmovups   704+__svml_satanh_data_internal_avx512(%rip), %zmm13

/* table values */
        vsubps    {rn-sae}, %zmm0, %zmm6, %zmm0
        vfmadd231ps {rn-sae}, %zmm1, %zmm2, %zmm0
        vfmadd213ps {rn-sae}, %zmm5, %zmm3, %zmm1
        vmovups   576+__svml_satanh_data_internal_avx512(%rip), %zmm3
        vmovaps   %zmm3, %zmm2
        vfmadd231ps {rn-sae}, %zmm9, %zmm7, %zmm2
        vfmadd231ps {rn-sae}, %zmm4, %zmm7, %zmm3
        vfmadd213ps {rn-sae}, %zmm12, %zmm9, %zmm2
        vfmadd213ps {rn-sae}, %zmm12, %zmm4, %zmm3
        vfmadd213ps {rn-sae}, %zmm13, %zmm9, %zmm2
        vfmadd213ps {rn-sae}, %zmm13, %zmm4, %zmm3

/* (K*L2L + Tl) + Rp*PolyP */
        vfmadd213ps {rn-sae}, %zmm1, %zmm9, %zmm2
        vorps     768+__svml_satanh_data_internal_avx512(%rip), %zmm10, %zmm9

/* (K*L2L + Tl) + Rp*PolyP -Rm*PolyM */
        vfnmadd213ps {rn-sae}, %zmm2, %zmm4, %zmm3
        vaddps    {rn-sae}, %zmm3, %zmm0, %zmm4
        vmulps    {rn-sae}, %zmm9, %zmm4, %zmm0
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

        vmovups   %zmm11, 64(%rsp)
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

        call      __svml_satanh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atanhf16,@function
	.size	__svml_atanhf16,.-__svml_atanhf16
..LN__svml_atanhf16.0:

.L_2__routine_start___svml_satanh_cout_rare_internal_1:

	.align    16,0x90

__svml_satanh_cout_rare_internal:


	.cfi_startproc
..L53:

        movzwl    2(%rdi), %edx
        movss     (%rdi), %xmm1
        andl      $32640, %edx
        movb      3(%rdi), %al
        andb      $127, %al
        movss     %xmm1, -8(%rsp)
        movb      %al, -5(%rsp)
        cmpl      $32640, %edx
        je        .LBL_2_6


        cmpl      $1065353216, -8(%rsp)
        jne       .LBL_2_4


        divss     4+__satanh_la__imlsAtanhTab(%rip), %xmm1
        movss     %xmm1, (%rsi)
        movl      $2, %eax
        ret

.LBL_2_4:

        movss     8+__satanh_la__imlsAtanhTab(%rip), %xmm0
        movl      $1, %eax
        mulss     4+__satanh_la__imlsAtanhTab(%rip), %xmm0
        movss     %xmm0, (%rsi)


        ret

.LBL_2_6:

        cmpl      $2139095040, -8(%rsp)
        jne       .LBL_2_8


        movss     4+__satanh_la__imlsAtanhTab(%rip), %xmm0
        movl      $1, %eax
        mulss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret

.LBL_2_8:

        mulss     (%rdi), %xmm1
        xorl      %eax, %eax
        movss     %xmm1, (%rsi)


        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_satanh_cout_rare_internal,@function
	.size	__svml_satanh_cout_rare_internal,.-__svml_satanh_cout_rare_internal
..LN__svml_satanh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_satanh_data_internal_avx512:
	.long	0
	.long	1023148032
	.long	1031274496
	.long	1035436032
	.long	1039204352
	.long	1041547264
	.long	1043333120
	.long	1045069824
	.long	1046773760
	.long	1048428544
	.long	1049313280
	.long	1050099712
	.long	1050869760
	.long	1051623424
	.long	1052360704
	.long	1053089792
	.long	1053794304
	.long	1054482432
	.long	1055162368
	.long	1055825920
	.long	1056481280
	.long	1057042432
	.long	1057353728
	.long	1057660928
	.long	1057964032
	.long	1058263040
	.long	1058553856
	.long	1058840576
	.long	1059123200
	.long	1059397632
	.long	1059672064
	.long	1059942400
	.long	0
	.long	925287326
	.long	950209537
	.long	928156389
	.long	954265029
	.long	3098231288
	.long	3083833176
	.long	949397309
	.long	3045295702
	.long	940324527
	.long	3089323092
	.long	945994465
	.long	952492302
	.long	954130348
	.long	954989406
	.long	3102096543
	.long	3093041984
	.long	947354573
	.long	3053684310
	.long	936642948
	.long	3099086888
	.long	3098368602
	.long	946194506
	.long	952357621
	.long	943635681
	.long	3097619830
	.long	3080925892
	.long	3078186319
	.long	3093311347
	.long	955801008
	.long	934582639
	.long	3099571146
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
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	131072
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	4294705152
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	3196061712
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	1051373854
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
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
	.type	__svml_satanh_data_internal_avx512,@object
	.size	__svml_satanh_data_internal_avx512,960
	.align 4
__satanh_la__imlsAtanhTab:
	.long	1065353216
	.long	0
	.long	2139095040
	.type	__satanh_la__imlsAtanhTab,@object
	.size	__satanh_la__imlsAtanhTab,12
