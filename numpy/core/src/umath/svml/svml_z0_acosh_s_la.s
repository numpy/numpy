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
.L_2__routine_start___svml_acoshf16_z0_0:

	.align    16,0x90
	.globl __svml_acoshf16

__svml_acoshf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   256+__svml_sacosh_data_internal_avx512(%rip), %zmm1

/*
 * sqrt(1+x^2) ~ Sh + Sl + Sh*Eh*poly_s
 * poly_s = c1+c2*Eh
 */
        vmovups   640+__svml_sacosh_data_internal_avx512(%rip), %zmm13
        vmovups   704+__svml_sacosh_data_internal_avx512(%rip), %zmm15

/* polynomial computation for small inputs */
        vmovups   576+__svml_sacosh_data_internal_avx512(%rip), %zmm9

/* very large inputs ? */
        vmovups   448+__svml_sacosh_data_internal_avx512(%rip), %zmm10

/* out of range inputs? */
        vmovups   512+__svml_sacosh_data_internal_avx512(%rip), %zmm11

/* not a very small input ? */
        vmovups   384+__svml_sacosh_data_internal_avx512(%rip), %zmm6
        vmovaps   %zmm0, %zmm8

/* x^2 - 1 */
        vmovaps   %zmm1, %zmm7
        vfmsub231ps {rn-sae}, %zmm8, %zmm8, %zmm7
        vcmpps    $21, {sae}, %zmm10, %zmm8, %k2
        vcmpps    $22, {sae}, %zmm11, %zmm8, %k0
        vcmpps    $18, {sae}, %zmm1, %zmm8, %k1
        vrsqrt14ps %zmm7, %zmm12
        vcmpps    $21, {sae}, %zmm6, %zmm7, %k3
        vmulps    {rn-sae}, %zmm9, %zmm7, %zmm4

/* Sh ~sqrt(-1+x^2) */
        vmulps    {rn-sae}, %zmm12, %zmm7, %zmm5
        kmovw     %k0, %edx
        kmovw     %k1, %eax

/* Sh+x */
        vaddps    {rn-sae}, %zmm8, %zmm5, %zmm9

/* (Yh*R0)_low */
        vmovaps   %zmm7, %zmm0
        orl       %eax, %edx

/* rel. error term: Eh=1-Sh*R0 */
        vmovaps   %zmm1, %zmm14
        vfmsub213ps {rn-sae}, %zmm5, %zmm12, %zmm0
        vfnmadd231ps {rn-sae}, %zmm5, %zmm12, %zmm14

/* rel. error term: Eh=(1-Sh*R0)-Sl*R0 */
        vfnmadd231ps {rn-sae}, %zmm0, %zmm12, %zmm14

/* Sh*Eh */
        vmulps    {rn-sae}, %zmm14, %zmm5, %zmm3
        vfmadd231ps {rn-sae}, %zmm14, %zmm13, %zmm15

/* Sl + Sh*Eh*poly_s */
        vfmadd213ps {rn-sae}, %zmm0, %zmm15, %zmm3

/* Shh */
        vsubps    {rn-sae}, %zmm8, %zmm9, %zmm15

/* polynomial computation for small inputs */
        vaddps    {rn-sae}, %zmm3, %zmm5, %zmm0

/* Xin0+Sl+Sh*Eh*poly_s ~ x+sqrt(1+x^2) */
        vaddps    {rn-sae}, %zmm3, %zmm9, %zmm2

/* Shl */
        vsubps    {rn-sae}, %zmm15, %zmm5, %zmm10
        vfmadd231ps {rn-sae}, %zmm0, %zmm4, %zmm0

/* fixup for very large inputs */
        vmovups   896+__svml_sacosh_data_internal_avx512(%rip), %zmm4

/* Sl_high */
        vsubps    {rn-sae}, %zmm9, %zmm2, %zmm5

/* polynomial */
        vmovups   1024+__svml_sacosh_data_internal_avx512(%rip), %zmm9
        vmulps    {rn-sae}, %zmm4, %zmm8, %zmm2{%k2}

/* -K*L2L + Tl */
        vmovups   1280+__svml_sacosh_data_internal_avx512(%rip), %zmm4

/* Sl_l */
        vsubps    {rn-sae}, %zmm5, %zmm3, %zmm3
        vrcp14ps  %zmm2, %zmm11
        vmovups   128+__svml_sacosh_data_internal_avx512(%rip), %zmm5

/* Xin_low */
        vaddps    {rn-sae}, %zmm10, %zmm3, %zmm13

/* round reciprocal to 1+4b mantissas */
        vpaddd    768+__svml_sacosh_data_internal_avx512(%rip), %zmm11, %zmm12
        vmovups   1152+__svml_sacosh_data_internal_avx512(%rip), %zmm10
        vandps    832+__svml_sacosh_data_internal_avx512(%rip), %zmm12, %zmm14

/* fixup for very large inputs */
        vxorps    %zmm13, %zmm13, %zmm13{%k2}

/* reduced argument for log(): (Rcp*Xin-1)+Rcp*Xin_low */
        vfmsub231ps {rn-sae}, %zmm14, %zmm2, %zmm1

/* exponents */
        vgetexpps {sae}, %zmm14, %zmm12
        vmovups   960+__svml_sacosh_data_internal_avx512(%rip), %zmm2

/* Prepare table index */
        vpsrld    $18, %zmm14, %zmm3
        vfmadd231ps {rn-sae}, %zmm14, %zmm13, %zmm1
        vmovups   1088+__svml_sacosh_data_internal_avx512(%rip), %zmm13

/* Table lookups */
        vmovups   __svml_sacosh_data_internal_avx512(%rip), %zmm14
        vsubps    {rn-sae}, %zmm2, %zmm12, %zmm12{%k2}
        vpermt2ps 192+__svml_sacosh_data_internal_avx512(%rip), %zmm3, %zmm5
        vpermt2ps 64+__svml_sacosh_data_internal_avx512(%rip), %zmm3, %zmm14

/* R^2 */
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm11

/* -K*L2H + Th */
        vmovups   1216+__svml_sacosh_data_internal_avx512(%rip), %zmm2
        vfmadd231ps {rn-sae}, %zmm1, %zmm9, %zmm13
        vfnmadd231ps {rn-sae}, %zmm12, %zmm2, %zmm14
        vfnmadd213ps {rn-sae}, %zmm5, %zmm4, %zmm12
        vfmadd213ps {rn-sae}, %zmm10, %zmm1, %zmm13

/* Tl + R^2*Poly */
        vfmadd213ps {rn-sae}, %zmm12, %zmm11, %zmm13

/* R+Tl + R^2*Poly */
        vaddps    {rn-sae}, %zmm1, %zmm13, %zmm1
        vaddps    {rn-sae}, %zmm1, %zmm14, %zmm0{%k3}
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

        call      __svml_sacosh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_acoshf16,@function
	.size	__svml_acoshf16,.-__svml_acoshf16
..LN__svml_acoshf16.0:

.L_2__routine_start___svml_sacosh_cout_rare_internal_1:

	.align    16,0x90

__svml_sacosh_cout_rare_internal:


	.cfi_startproc
..L53:

        movl      (%rdi), %eax
        movl      %eax, -8(%rsp)
        andl      $2139095040, %eax
        cmpl      $2139095040, %eax
        je        .LBL_2_8

.LBL_2_2:

        movl      -8(%rsp), %eax
        cmpl      $1065353216, %eax
        je        .LBL_2_7


        cmpl      $2139095040, %eax
        jne       .LBL_2_5


        movl      8+__sacosh_la__iml_sacosh_cout_tab(%rip), %eax
        movl      %eax, (%rsi)
        xorl      %eax, %eax
        ret

.LBL_2_5:

        movss     8+__sacosh_la__iml_sacosh_cout_tab(%rip), %xmm0
        movl      $1, %eax
        mulss     4+__sacosh_la__iml_sacosh_cout_tab(%rip), %xmm0
        movss     %xmm0, (%rsi)


        ret

.LBL_2_7:

        movl      4+__sacosh_la__iml_sacosh_cout_tab(%rip), %eax
        movl      %eax, (%rsi)
        xorl      %eax, %eax
        ret

.LBL_2_8:

        testl     $8388607, -8(%rsp)
        je        .LBL_2_2


        movss     -8(%rsp), %xmm0
        xorl      %eax, %eax
        mulss     -8(%rsp), %xmm0
        movss     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sacosh_cout_rare_internal,@function
	.size	__svml_sacosh_cout_rare_internal,.-__svml_sacosh_cout_rare_internal
..LN__svml_sacosh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_sacosh_data_internal_avx512:
	.long	0
	.long	3170631680
	.long	3178790912
	.long	3182919680
	.long	3186704384
	.long	3189022720
	.long	3190816768
	.long	3192561664
	.long	3194257408
	.long	3195912192
	.long	3196796928
	.long	3197583360
	.long	3198357504
	.long	3199111168
	.long	3199848448
	.long	3200569344
	.long	3201277952
	.long	3201966080
	.long	3202646016
	.long	3203309568
	.long	3203960832
	.long	3204524032
	.long	3204837376
	.long	3205146624
	.long	3205447680
	.long	3205744640
	.long	3206037504
	.long	3206324224
	.long	3206606848
	.long	3206883328
	.long	3207155712
	.long	3207424000
	.long	2147483648
	.long	3072770974
	.long	943319038
	.long	3075640037
	.long	930648533
	.long	3089726480
	.long	936349528
	.long	944943494
	.long	897812054
	.long	3087808175
	.long	941839444
	.long	3093478113
	.long	937982919
	.long	931430736
	.long	924853521
	.long	3075349253
	.long	945558336
	.long	3094838221
	.long	906200662
	.long	3084126596
	.long	3088015279
	.long	3089451852
	.long	3093678154
	.long	938521645
	.long	3091119329
	.long	3090949395
	.long	933442244
	.long	930702671
	.long	945827699
	.long	913590776
	.long	3082066287
	.long	3087046763
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
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	964689920
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	1593835520
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	2139095039
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	3190466014
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
	.long	1052770304
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
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1040187392
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
	.long	1082130432
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
	.type	__svml_sacosh_data_internal_avx512,@object
	.size	__svml_sacosh_data_internal_avx512,1344
	.align 4
__sacosh_la__iml_sacosh_cout_tab:
	.long	1065353216
	.long	0
	.long	2139095040
	.type	__sacosh_la__iml_sacosh_cout_tab,@object
	.size	__sacosh_la__iml_sacosh_cout_tab,12
