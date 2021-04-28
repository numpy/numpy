/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *     
 *      ( optimized for throughput, with small table lookup, works when HW FMA is available )
 *     
 *       Implementation reduces argument x to |R|<pi/64
 *       32-entry tables used to store high and low parts of tan(x0)
 *       Argument x = N*pi + x0 + (R);   x0 = k*pi/32, with k in {0, 1, ..., 31}
 *       (very large arguments reduction resolved in _vsreduction_core.i)
 *       Compute result as (tan(x0) + tan(R))/(1-tan(x0)*tan(R))
 *       _HA_ version keeps extra precision for numerator, denominator, and during
 *       final NR-iteration computing quotient.
 *     
 *     
 */


	.text
.L_2__routine_start___svml_tanf16_z0_0:

	.align    16,0x90
	.globl __svml_tanf16

__svml_tanf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        xorl      %edx, %edx

/* Large values check */
        vmovups   768+__svml_stan_data_internal(%rip), %zmm10

/*
 * ----------------------------------------------------------
 * Main path
 * ----------------------------------------------------------
 * start arg. reduction
 */
        vmovups   1088+__svml_stan_data_internal(%rip), %zmm1
        vmovups   64+__svml_stan_data_internal(%rip), %zmm4
        vmovups   128+__svml_stan_data_internal(%rip), %zmm2
        vmovups   192+__svml_stan_data_internal(%rip), %zmm3
        vmovaps   %zmm0, %zmm11
        vandps    960+__svml_stan_data_internal(%rip), %zmm11, %zmm0
        vcmpps    $22, {sae}, %zmm10, %zmm0, %k1
        vmovups   __svml_stan_data_internal(%rip), %zmm10

/*
 * ----------------------------------------------------------
 * End of main path
 * ----------------------------------------------------------
 */
        kortestw  %k1, %k1
        vfmadd213ps {rn-sae}, %zmm1, %zmm11, %zmm10
        vsubps    {rn-sae}, %zmm1, %zmm10, %zmm5
        vfnmadd213ps {rn-sae}, %zmm11, %zmm5, %zmm4
        vfnmadd231ps {rn-sae}, %zmm5, %zmm2, %zmm4
        vfnmadd213ps {rn-sae}, %zmm4, %zmm3, %zmm5
        jne       .LBL_1_12

.LBL_1_2:


/* Table lookup */
        vmovups   384+__svml_stan_data_internal(%rip), %zmm3
        vmovups   640+__svml_stan_data_internal(%rip), %zmm0
        vmulps    {rn-sae}, %zmm5, %zmm5, %zmm1
        vpermt2ps 448+__svml_stan_data_internal(%rip), %zmm10, %zmm3
        vmovups   704+__svml_stan_data_internal(%rip), %zmm10
        vfmadd231ps {rn-sae}, %zmm1, %zmm10, %zmm0
        vmulps    {rn-sae}, %zmm5, %zmm0, %zmm4
        vfmadd213ps {rn-sae}, %zmm5, %zmm1, %zmm4

/*
 * Computer Denominator:
 * sDenominator - sDlow ~= 1-(sTh+sTl)*(sP+sPlow)
 */
        vmovups   1152+__svml_stan_data_internal(%rip), %zmm5
        vmulps    {rn-sae}, %zmm4, %zmm3, %zmm7

/*
 * Compute Numerator:
 * sNumerator + sNlow ~= sTh+sTl+sP+sPlow
 */
        vaddps    {rn-sae}, %zmm3, %zmm4, %zmm8
        vsubps    {rn-sae}, %zmm7, %zmm5, %zmm9
        vsubps    {rn-sae}, %zmm3, %zmm8, %zmm2

/*
 * Now computes (sNumerator + sNlow)/(sDenominator - sDlow)
 * Choose NR iteration instead of hardware division
 */
        vrcp14ps  %zmm9, %zmm14
        vsubps    {rn-sae}, %zmm5, %zmm9, %zmm6
        vsubps    {rn-sae}, %zmm2, %zmm4, %zmm13
        vmulps    {rn-sae}, %zmm8, %zmm14, %zmm15
        vaddps    {rn-sae}, %zmm7, %zmm6, %zmm12

/* One NR iteration to refine sQuotient */
        vfmsub213ps {rn-sae}, %zmm8, %zmm15, %zmm9
        vfnmadd213ps {rn-sae}, %zmm9, %zmm15, %zmm12
        vsubps    {rn-sae}, %zmm13, %zmm12, %zmm0
        vfnmadd213ps {rn-sae}, %zmm15, %zmm14, %zmm0
        testl     %edx, %edx
        jne       .LBL_1_4

.LBL_1_3:


/* no invcbrt in libm, so taking it out here */
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_4:

        vmovups   %zmm11, 64(%rsp)
        vmovups   %zmm0, 128(%rsp)
        je        .LBL_1_3


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

.LBL_1_8:

        btl       %r12d, %r13d
        jc        .LBL_1_11

.LBL_1_9:

        incl      %r12d
        cmpl      $16, %r12d
        jl        .LBL_1_8


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
        jmp       .LBL_1_3
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x68, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x60, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x78, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x70, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x58, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x50, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x48, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x40, 0xff, 0xff, 0xff, 0x22

.LBL_1_11:

        lea       64(%rsp,%r12,4), %rdi
        lea       128(%rsp,%r12,4), %rsi

        call      __svml_stan_cout_rare_internal
        jmp       .LBL_1_9
	.cfi_restore 4
	.cfi_restore 5
	.cfi_restore 12
	.cfi_restore 13
	.cfi_restore 122
	.cfi_restore 123
	.cfi_restore 124
	.cfi_restore 125

.LBL_1_12:

        vmovups   1024+__svml_stan_data_internal(%rip), %zmm6

/*
 * Get the (2^a / 2pi) mod 1 values from the table.
 * Because VLANG doesn't have I-type gather, we need a trivial cast
 */
        lea       __svml_stan_reduction_data_internal(%rip), %rax
        vmovups   %zmm5, (%rsp)
        vandps    %zmm0, %zmm6, %zmm14
        vcmpps    $0, {sae}, %zmm6, %zmm14, %k0

/*
 * Break the P_xxx and m into 16-bit chunks ready for
 * the long multiplication via 16x16->32 multiplications
 */
        vmovups   .L_2il0floatpacket.18(%rip), %zmm6
        kxnorw    %k0, %k0, %k2
        kxnorw    %k0, %k0, %k3
        kmovw     %k0, %edx
        vpandd    .L_2il0floatpacket.15(%rip), %zmm11, %zmm5
        vpsrld    $23, %zmm5, %zmm7
        vpslld    $1, %zmm7, %zmm8
        vpaddd    %zmm7, %zmm8, %zmm9
        vpslld    $2, %zmm9, %zmm4
        vpxord    %zmm3, %zmm3, %zmm3
        vgatherdps (%rax,%zmm4), %zmm3{%k2}
        kxnorw    %k0, %k0, %k2
        vpsrld    $16, %zmm3, %zmm5
        vpxord    %zmm15, %zmm15, %zmm15
        vpxord    %zmm2, %zmm2, %zmm2
        vgatherdps 4(%rax,%zmm4), %zmm15{%k3}
        vgatherdps 8(%rax,%zmm4), %zmm2{%k2}
        vpsrld    $16, %zmm2, %zmm13

/*
 * Also get the significand as an integer
 * NB: adding in the integer bit is wrong for denorms!
 * To make this work for denorms we should do something slightly different
 */
        vpandd    .L_2il0floatpacket.16(%rip), %zmm11, %zmm0
        vpaddd    .L_2il0floatpacket.17(%rip), %zmm0, %zmm1
        vpsrld    $16, %zmm15, %zmm0
        vpsrld    $16, %zmm1, %zmm8
        vpandd    %zmm6, %zmm3, %zmm9
        vpandd    %zmm6, %zmm15, %zmm12
        vpandd    %zmm6, %zmm2, %zmm7
        vpandd    %zmm6, %zmm1, %zmm14

/* Now do the big multiplication and carry propagation */
        vpmulld   %zmm9, %zmm8, %zmm4
        vpmulld   %zmm0, %zmm8, %zmm3
        vpmulld   %zmm12, %zmm8, %zmm2
        vpmulld   %zmm13, %zmm8, %zmm1
        vpmulld   %zmm7, %zmm8, %zmm8
        vpmulld   %zmm5, %zmm14, %zmm7
        vpmulld   %zmm9, %zmm14, %zmm5
        vpmulld   %zmm0, %zmm14, %zmm9
        vpmulld   %zmm12, %zmm14, %zmm0
        vpmulld   %zmm13, %zmm14, %zmm12
        vpsrld    $16, %zmm12, %zmm14
        vpsrld    $16, %zmm0, %zmm13
        vpsrld    $16, %zmm9, %zmm15
        vpsrld    $16, %zmm5, %zmm12
        vpsrld    $16, %zmm8, %zmm8
        vpaddd    %zmm14, %zmm1, %zmm1
        vpaddd    %zmm13, %zmm2, %zmm2
        vpaddd    %zmm15, %zmm3, %zmm15
        vpaddd    %zmm12, %zmm4, %zmm3
        vpandd    %zmm6, %zmm0, %zmm13
        vpaddd    %zmm1, %zmm13, %zmm4
        vpaddd    %zmm4, %zmm8, %zmm14
        vpsrld    $16, %zmm14, %zmm0
        vpandd    %zmm6, %zmm9, %zmm9
        vpaddd    %zmm2, %zmm9, %zmm1
        vpaddd    %zmm1, %zmm0, %zmm8

/*
 * Now round at the 2^-8 bit position for reduction mod pi/2^7
 * instead of the original 2pi (but still with the same 2pi scaling).
 * Use a shifter of 2^15 + 2^14.
 * The N we get is our final version; it has an offset of
 * 2^8 because of the implicit integer bit, and anyway for negative
 * starting value it's a 2s complement thing. But we need to mask
 * off the exponent part anyway so it's fine.
 */
        vmovups   .L_2il0floatpacket.21(%rip), %zmm1
        vpandd    %zmm6, %zmm7, %zmm7
        vpaddd    %zmm3, %zmm7, %zmm13
        vpsrld    $16, %zmm8, %zmm3
        vpandd    %zmm6, %zmm5, %zmm5
        vpaddd    %zmm15, %zmm5, %zmm2
        vpaddd    %zmm2, %zmm3, %zmm15
        vpsrld    $16, %zmm15, %zmm12
        vpaddd    %zmm13, %zmm12, %zmm5

/* Assemble reduced argument from the pieces */
        vpandd    %zmm6, %zmm14, %zmm9
        vpandd    %zmm6, %zmm15, %zmm7
        vpslld    $16, %zmm5, %zmm6
        vpslld    $16, %zmm8, %zmm5
        vpaddd    %zmm7, %zmm6, %zmm4
        vpaddd    %zmm9, %zmm5, %zmm9
        vpsrld    $9, %zmm4, %zmm6

/*
 * We want to incorporate the original sign now too.
 * Do it here for convenience in getting the right N value,
 * though we could wait right to the end if we were prepared
 * to modify the sign of N later too.
 * So get the appropriate sign mask now (or sooner).
 */
        vpandd    .L_2il0floatpacket.19(%rip), %zmm11, %zmm0
        vpandd    .L_2il0floatpacket.24(%rip), %zmm9, %zmm13
        vpslld    $5, %zmm13, %zmm14

/*
 * Create floating-point high part, implicitly adding integer bit 1
 * Incorporate overall sign at this stage too.
 */
        vpxord    .L_2il0floatpacket.20(%rip), %zmm0, %zmm8
        vpord     %zmm8, %zmm6, %zmm2
        vaddps    {rn-sae}, %zmm2, %zmm1, %zmm12
        vsubps    {rn-sae}, %zmm1, %zmm12, %zmm3
        vsubps    {rn-sae}, %zmm3, %zmm2, %zmm7

/*
 * Create floating-point low and medium parts, respectively
 * lo_17, ... lo_0, 0, ..., 0
 * hi_8, ... hi_0, lo_31, ..., lo_18
 * then subtract off the implicitly added integer bits,
 * 2^-46 and 2^-23, respectively.
 * Put the original sign into all of them at this stage.
 */
        vpxord    .L_2il0floatpacket.23(%rip), %zmm0, %zmm6
        vpord     %zmm6, %zmm14, %zmm15
        vpandd    .L_2il0floatpacket.26(%rip), %zmm4, %zmm4
        vsubps    {rn-sae}, %zmm6, %zmm15, %zmm8
        vandps    .L_2il0floatpacket.29(%rip), %zmm11, %zmm15
        vpsrld    $18, %zmm9, %zmm6

/*
 * If the magnitude of the input is <= 2^-20, then
 * just pass through the input, since no reduction will be needed and
 * the main path will only work accurately if the reduced argument is
 * about >= 2^-40 (which it is for all large pi multiples)
 */
        vmovups   .L_2il0floatpacket.30(%rip), %zmm14
        vcmpps    $26, {sae}, %zmm14, %zmm15, %k3
        vcmpps    $22, {sae}, %zmm14, %zmm15, %k2
        vpxord    .L_2il0floatpacket.25(%rip), %zmm0, %zmm1
        vpslld    $14, %zmm4, %zmm0
        vpord     %zmm6, %zmm0, %zmm0
        vpord     %zmm1, %zmm0, %zmm4
        vsubps    {rn-sae}, %zmm1, %zmm4, %zmm2
        vpternlogd $255, %zmm6, %zmm6, %zmm6

/* Now add them up into 2 reasonably aligned pieces */
        vaddps    {rn-sae}, %zmm2, %zmm7, %zmm13
        vsubps    {rn-sae}, %zmm13, %zmm7, %zmm7
        vaddps    {rn-sae}, %zmm7, %zmm2, %zmm3

/*
 * The output is _VRES_R (high) + _VRES_E (low), and the integer part is _VRES_IND
 * Set sRp2 = _VRES_R^2 and then resume the original code.
 */
        vmovups   .L_2il0floatpacket.31(%rip), %zmm2
        vaddps    {rn-sae}, %zmm8, %zmm3, %zmm1
        vmovups   .L_2il0floatpacket.28(%rip), %zmm8

/* Grab our final N value as an integer, appropriately masked mod 2^8 */
        vpandd    .L_2il0floatpacket.22(%rip), %zmm12, %zmm5

/*
 * Now multiply those numbers all by 2 pi, reasonably accurately.
 * (RHi + RLo) * (pi_lead + pi_trail) ~=
 * RHi * pi_lead + (RHi * pi_trail + RLo * pi_lead)
 */
        vmovups   .L_2il0floatpacket.27(%rip), %zmm12
        vmulps    {rn-sae}, %zmm12, %zmm13, %zmm0
        vmovaps   %zmm12, %zmm9
        vfmsub213ps {rn-sae}, %zmm0, %zmm13, %zmm9
        vfmadd213ps {rn-sae}, %zmm9, %zmm8, %zmm13
        vmovaps   %zmm6, %zmm8
        vfmadd213ps {rn-sae}, %zmm13, %zmm12, %zmm1
        vpandnd   %zmm15, %zmm15, %zmm8{%k3}
        vpandnd   %zmm15, %zmm15, %zmm6{%k2}
        vandps    %zmm11, %zmm6, %zmm14
        vandps    %zmm0, %zmm8, %zmm15
        vandps    %zmm1, %zmm8, %zmm12
        vorps     %zmm15, %zmm14, %zmm6
        vpsrld    $31, %zmm6, %zmm3
        vpsubd    %zmm3, %zmm2, %zmm4
        vpaddd    %zmm4, %zmm5, %zmm7
        vpsrld    $2, %zmm7, %zmm13
        vpslld    $2, %zmm13, %zmm9

/*
 * ----------------------------------------------------------
 * End of large arguments path
 * ----------------------------------------------------------
 * Merge results from main and large paths:
 */
        vblendmps %zmm13, %zmm10, %zmm10{%k1}
        vpsubd    %zmm9, %zmm5, %zmm5
        vmovups   .L_2il0floatpacket.32(%rip), %zmm9
        vcvtdq2ps {rn-sae}, %zmm5, %zmm0
        vmovups   .L_2il0floatpacket.33(%rip), %zmm5
        vfmadd231ps {rn-sae}, %zmm0, %zmm5, %zmm12
        vmovups   (%rsp), %zmm5
        vaddps    {rn-sae}, %zmm6, %zmm12, %zmm6
        vfmadd213ps {rn-sae}, %zmm6, %zmm9, %zmm0
        vblendmps %zmm0, %zmm5, %zmm5{%k1}
        jmp       .LBL_1_2
	.align    16,0x90

	.cfi_endproc

	.type	__svml_tanf16,@function
	.size	__svml_tanf16,.-__svml_tanf16
..LN__svml_tanf16.0:

.L_2__routine_start___svml_stan_cout_rare_internal_1:

	.align    16,0x90

__svml_stan_cout_rare_internal:


	.cfi_startproc
..L63:

        xorl      %eax, %eax
        movl      (%rdi), %edx
        movzwl    2(%rdi), %ecx
        movl      %edx, -8(%rsp)
        andl      $32640, %ecx
        shrl      $24, %edx
        andl      $127, %edx
        movb      %dl, -5(%rsp)
        cmpl      $32640, %ecx
        je        .LBL_2_3


        ret

.LBL_2_3:

        cmpl      $2139095040, -8(%rsp)
        jne       .LBL_2_5


        movss     (%rdi), %xmm0
        movl      $1, %eax
        mulss     __stan_la__vmlsTanTab(%rip), %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_5:

        movss     (%rdi), %xmm0
        mulss     (%rdi), %xmm0
        movss     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_stan_cout_rare_internal,@function
	.size	__svml_stan_cout_rare_internal,.-__svml_stan_cout_rare_internal
..LN__svml_stan_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
.L_2il0floatpacket.15:
	.long	0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000
	.type	.L_2il0floatpacket.15,@object
	.size	.L_2il0floatpacket.15,64
	.align 64
.L_2il0floatpacket.16:
	.long	0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff
	.type	.L_2il0floatpacket.16,@object
	.size	.L_2il0floatpacket.16,64
	.align 64
.L_2il0floatpacket.17:
	.long	0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000
	.type	.L_2il0floatpacket.17,@object
	.size	.L_2il0floatpacket.17,64
	.align 64
.L_2il0floatpacket.18:
	.long	0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff
	.type	.L_2il0floatpacket.18,@object
	.size	.L_2il0floatpacket.18,64
	.align 64
.L_2il0floatpacket.19:
	.long	0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000
	.type	.L_2il0floatpacket.19,@object
	.size	.L_2il0floatpacket.19,64
	.align 64
.L_2il0floatpacket.20:
	.long	0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000
	.type	.L_2il0floatpacket.20,@object
	.size	.L_2il0floatpacket.20,64
	.align 64
.L_2il0floatpacket.21:
	.long	0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000
	.type	.L_2il0floatpacket.21,@object
	.size	.L_2il0floatpacket.21,64
	.align 64
.L_2il0floatpacket.22:
	.long	0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff
	.type	.L_2il0floatpacket.22,@object
	.size	.L_2il0floatpacket.22,64
	.align 64
.L_2il0floatpacket.23:
	.long	0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000
	.type	.L_2il0floatpacket.23,@object
	.size	.L_2il0floatpacket.23,64
	.align 64
.L_2il0floatpacket.24:
	.long	0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff
	.type	.L_2il0floatpacket.24,@object
	.size	.L_2il0floatpacket.24,64
	.align 64
.L_2il0floatpacket.25:
	.long	0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000
	.type	.L_2il0floatpacket.25,@object
	.size	.L_2il0floatpacket.25,64
	.align 64
.L_2il0floatpacket.26:
	.long	0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff
	.type	.L_2il0floatpacket.26,@object
	.size	.L_2il0floatpacket.26,64
	.align 64
.L_2il0floatpacket.27:
	.long	0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb
	.type	.L_2il0floatpacket.27,@object
	.size	.L_2il0floatpacket.27,64
	.align 64
.L_2il0floatpacket.28:
	.long	0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e
	.type	.L_2il0floatpacket.28,@object
	.size	.L_2il0floatpacket.28,64
	.align 64
.L_2il0floatpacket.29:
	.long	0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff
	.type	.L_2il0floatpacket.29,@object
	.size	.L_2il0floatpacket.29,64
	.align 64
.L_2il0floatpacket.30:
	.long	0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000
	.type	.L_2il0floatpacket.30,@object
	.size	.L_2il0floatpacket.30,64
	.align 64
.L_2il0floatpacket.31:
	.long	0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002,0x00000002
	.type	.L_2il0floatpacket.31,@object
	.size	.L_2il0floatpacket.31,64
	.align 64
.L_2il0floatpacket.32:
	.long	0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb,0x3cc90fdb
	.type	.L_2il0floatpacket.32,@object
	.size	.L_2il0floatpacket.32,64
	.align 64
.L_2il0floatpacket.33:
	.long	0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e,0xb03bbd2e
	.type	.L_2il0floatpacket.33,@object
	.size	.L_2il0floatpacket.33,64
	.align 64
__svml_stan_data_internal:
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1092811139
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	1036586970
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	832708968
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	633484485
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	832708608
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	708075802
	.long	2147483648
	.long	1036629468
	.long	1045147567
	.long	1050366018
	.long	1054086093
	.long	1057543609
	.long	1059786177
	.long	1062344705
	.long	1065353216
	.long	1067186156
	.long	1069519047
	.long	1072658590
	.long	1075479162
	.long	1079179983
	.long	1084284919
	.long	1092776803
	.long	4286578687
	.long	3240260451
	.long	3231768567
	.long	3226663631
	.long	3222962810
	.long	3220142238
	.long	3217002695
	.long	3214669804
	.long	3212836864
	.long	3209828353
	.long	3207269825
	.long	3205027257
	.long	3201569741
	.long	3197849666
	.long	3192631215
	.long	3184113116
	.long	2147483648
	.long	826651354
	.long	791306928
	.long	2989111746
	.long	2982175258
	.long	2992568675
	.long	850100121
	.long	850281093
	.long	0
	.long	861435400
	.long	840342808
	.long	3003924160
	.long	3016492578
	.long	865099790
	.long	856723932
	.long	3025444934
	.long	4085252096
	.long	877961286
	.long	3004207580
	.long	3012583438
	.long	869008930
	.long	856440512
	.long	2987826456
	.long	3008919048
	.long	0
	.long	2997764741
	.long	2997583769
	.long	845085027
	.long	834691610
	.long	841628098
	.long	2938790576
	.long	2974135002
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1051372198
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1040758920
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	1059256707
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
	.long	2147483648
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
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
	.long	2139095040
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
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1174470656
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	1070137344
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	972922880
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	866263040
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	741630234
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	2801216749
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	3183752116
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	1065353212
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	3202070443
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1008677739
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1070141403
	.long	3007036718
	.long	0
	.long	0
	.long	0
	.long	1065353216
	.long	0
	.long	0
	.long	1051372765
	.long	0
	.long	1069935515
	.long	853435276
	.long	0
	.long	1019812401
	.long	797871386
	.long	1065353216
	.long	975043072
	.long	1019820333
	.long	1051400329
	.long	1015569723
	.long	1069729628
	.long	2999697034
	.long	0
	.long	1028208956
	.long	816029531
	.long	1065353216
	.long	991832832
	.long	1028240852
	.long	1051479824
	.long	1023251493
	.long	1069523740
	.long	860164016
	.long	0
	.long	1033310670
	.long	827321128
	.long	1065353216
	.long	1001540608
	.long	1033364538
	.long	1051617929
	.long	1028458464
	.long	1069317853
	.long	2977958621
	.long	0
	.long	1036629468
	.long	826649990
	.long	1065353216
	.long	1008660256
	.long	1036757738
	.long	1051807326
	.long	1032162226
	.long	1069111966
	.long	3009745511
	.long	0
	.long	1039964354
	.long	2964214364
	.long	1065353216
	.long	1014578464
	.long	1040201797
	.long	1052059423
	.long	1034708638
	.long	1068906078
	.long	848017692
	.long	0
	.long	1041753444
	.long	2982519524
	.long	1065353216
	.long	1018446032
	.long	1041972480
	.long	1052374628
	.long	1037453248
	.long	1068700191
	.long	3004118141
	.long	0
	.long	1043443277
	.long	2985501265
	.long	1065353216
	.long	1022797056
	.long	1043793882
	.long	1052746889
	.long	1039915463
	.long	1068494303
	.long	857455223
	.long	0
	.long	1045147567
	.long	791292384
	.long	1065353216
	.long	1025642520
	.long	1045675728
	.long	1053195814
	.long	1041590498
	.long	1068288416
	.long	2992986704
	.long	0
	.long	1046868583
	.long	833925599
	.long	1065353216
	.long	1028557712
	.long	1047628490
	.long	1053716836
	.long	1043186017
	.long	1068082528
	.long	863082593
	.long	0
	.long	1048592340
	.long	2988940902
	.long	1065353216
	.long	1031831496
	.long	1049119700
	.long	1054310701
	.long	1044788971
	.long	1067876641
	.long	837040812
	.long	0
	.long	1049473154
	.long	2972885556
	.long	1065353216
	.long	1033689040
	.long	1050184288
	.long	1054999523
	.long	1046698028
	.long	1067670754
	.long	3006826934
	.long	0
	.long	1050366018
	.long	2989112046
	.long	1065353216
	.long	1035760784
	.long	1051302645
	.long	1055777031
	.long	1048635818
	.long	1067464866
	.long	853854846
	.long	0
	.long	1051272279
	.long	817367088
	.long	1065353216
	.long	1038057984
	.long	1052482025
	.long	1056656040
	.long	1049723582
	.long	1067258979
	.long	2999277465
	.long	0
	.long	1052193360
	.long	2986510371
	.long	1065353216
	.long	1040390392
	.long	1053730424
	.long	1057307751
	.long	1050943059
	.long	1067053091
	.long	860373800
	.long	0
	.long	1053130765
	.long	2987705281
	.long	1065353216
	.long	1041784404
	.long	1055056706
	.long	1057868403
	.long	1052298273
	.long	1066847204
	.long	2974604846
	.long	0
	.long	1054086093
	.long	2982175058
	.long	1065353216
	.long	1043312844
	.long	1056470731
	.long	1058502663
	.long	1053852727
	.long	1066641317
	.long	3009535726
	.long	0
	.long	1055061049
	.long	2985572766
	.long	1065353216
	.long	1044984860
	.long	1057474074
	.long	1059214863
	.long	1055565854
	.long	1066435429
	.long	848437261
	.long	0
	.long	1056057456
	.long	844263924
	.long	1065353216
	.long	1046810746
	.long	1058286064
	.long	1060014844
	.long	1057227928
	.long	1066229542
	.long	3003908357
	.long	0
	.long	1057020941
	.long	2987700082
	.long	1065353216
	.long	1048689044
	.long	1059160627
	.long	1060914481
	.long	1058313864
	.long	1066023654
	.long	857665008
	.long	0
	.long	1057543609
	.long	2992568718
	.long	1065353216
	.long	1049773965
	.long	1060105673
	.long	1061932376
	.long	1059565214
	.long	1065817767
	.long	2992147565
	.long	0
	.long	1058080175
	.long	854607280
	.long	1065353216
	.long	1050955490
	.long	1061130203
	.long	1063075792
	.long	1060964899
	.long	1065611879
	.long	863292377
	.long	0
	.long	1058631876
	.long	848316488
	.long	1065353216
	.long	1052241912
	.long	1062244476
	.long	1064374250
	.long	1062608877
	.long	1065405992
	.long	838719090
	.long	0
	.long	1059200055
	.long	2987155932
	.long	1065353216
	.long	1053642609
	.long	1063460266
	.long	1065596017
	.long	1064468970
	.long	1065046993
	.long	848647046
	.long	0
	.long	1059786177
	.long	850099898
	.long	1065353216
	.long	1055168194
	.long	1064791104
	.long	1066427841
	.long	1065988022
	.long	1064635218
	.long	854274415
	.long	0
	.long	1060391849
	.long	2998448362
	.long	1065353216
	.long	1056830711
	.long	1065802920
	.long	1067373883
	.long	1067237086
	.long	1064223444
	.long	2998857895
	.long	0
	.long	1061018831
	.long	852661766
	.long	1073741824
	.long	3202769007
	.long	1066608086
	.long	1068453481
	.long	1068697612
	.long	1063811669
	.long	2991727995
	.long	0
	.long	1061669068
	.long	2986407194
	.long	1073741824
	.long	3200789612
	.long	1067498217
	.long	1069688111
	.long	1070408903
	.long	1063399894
	.long	2971248290
	.long	0
	.long	1062344705
	.long	850280824
	.long	1073741824
	.long	3198626104
	.long	1068485666
	.long	1071103306
	.long	1072410651
	.long	1062988119
	.long	839209514
	.long	0
	.long	1063048126
	.long	826671880
	.long	1073741824
	.long	3196257989
	.long	1069584946
	.long	1072731698
	.long	1074256640
	.long	1062576344
	.long	848856831
	.long	0
	.long	1063781982
	.long	845614362
	.long	1073741824
	.long	3191263702
	.long	1070813191
	.long	1074178145
	.long	1075661786
	.long	1062164569
	.long	854484200
	.long	0
	.long	1064549237
	.long	855412877
	.long	1073741824
	.long	3183449264
	.long	1072190735
	.long	1075269479
	.long	1077331464
	.long	1061752795
	.long	2998648110
	.long	1065353216
	.long	3196839438
	.long	839748996
	.long	1056964608
	.long	3187152817
	.long	3179496939
	.long	1025375660
	.long	3159543663
	.long	1061341020
	.long	2991308426
	.long	1065353216
	.long	3196528703
	.long	2993207654
	.long	1056964608
	.long	3187565865
	.long	3178961235
	.long	1025040649
	.long	3158667440
	.long	1060929245
	.long	2969570013
	.long	1065353216
	.long	3196220448
	.long	839617357
	.long	1048576000
	.long	1039897640
	.long	3178234548
	.long	1024731756
	.long	3157936127
	.long	1060517470
	.long	839629084
	.long	1065353216
	.long	3195769474
	.long	2972943314
	.long	1048576000
	.long	1039520273
	.long	3177530035
	.long	1024452069
	.long	3157392148
	.long	1060105695
	.long	849066615
	.long	1065353216
	.long	3195162227
	.long	824230882
	.long	1048576000
	.long	1039159939
	.long	3176846430
	.long	1024176063
	.long	3156719803
	.long	1059693920
	.long	854693985
	.long	1065353216
	.long	3194559300
	.long	837912886
	.long	1048576000
	.long	1038816139
	.long	3176182519
	.long	1023917626
	.long	3156100775
	.long	1059282146
	.long	2998438326
	.long	1065353216
	.long	3193960492
	.long	2976936506
	.long	1048576000
	.long	1038488404
	.long	3175537158
	.long	1023672824
	.long	3155484691
	.long	1058870371
	.long	2990888857
	.long	1065353216
	.long	3193365611
	.long	837021697
	.long	1048576000
	.long	1038176293
	.long	3174909264
	.long	1023428141
	.long	3154717848
	.long	1058458596
	.long	2966216238
	.long	1065353216
	.long	3192774465
	.long	2981011604
	.long	1048576000
	.long	1037879388
	.long	3174297790
	.long	1023026096
	.long	3154246903
	.long	1058046821
	.long	840048653
	.long	1065353216
	.long	3192186872
	.long	2982847435
	.long	1048576000
	.long	1037597300
	.long	3173701765
	.long	1022609285
	.long	3153191752
	.long	1057635046
	.long	849276400
	.long	1065353216
	.long	3191602652
	.long	2972865050
	.long	1048576000
	.long	1037329660
	.long	3173120241
	.long	1022242934
	.long	3152466531
	.long	1057223271
	.long	854903769
	.long	1065353216
	.long	3191021630
	.long	838792638
	.long	1048576000
	.long	1037076124
	.long	3172552332
	.long	1021893801
	.long	3151682133
	.long	1056658385
	.long	840258438
	.long	1065353216
	.long	3190443633
	.long	2979855596
	.long	1048576000
	.long	1036836369
	.long	3171997189
	.long	1021543079
	.long	3150495127
	.long	1055834836
	.long	2990469287
	.long	1065353216
	.long	3189868496
	.long	2981611511
	.long	1048576000
	.long	1036610091
	.long	3171453986
	.long	1021220110
	.long	3149437649
	.long	1055011286
	.long	2962859682
	.long	1065353216
	.long	3189296055
	.long	2950857776
	.long	1048576000
	.long	1036397006
	.long	3170921933
	.long	1020942892
	.long	3148919762
	.long	1054187736
	.long	840468223
	.long	1065353216
	.long	3188726149
	.long	2955915960
	.long	1048576000
	.long	1036196851
	.long	3169906765
	.long	1020660679
	.long	3147905210
	.long	1053364187
	.long	2990259502
	.long	1065353216
	.long	3188158621
	.long	2978622083
	.long	1048576000
	.long	1036009378
	.long	3168882838
	.long	1020421234
	.long	3147436656
	.long	1052540637
	.long	2961181405
	.long	1065353216
	.long	3187515595
	.long	789904544
	.long	1048576000
	.long	1035834359
	.long	3167876891
	.long	1020189995
	.long	3146799430
	.long	1051717087
	.long	840678007
	.long	1065353216
	.long	3186389132
	.long	2974324164
	.long	1048576000
	.long	1035671582
	.long	3166887590
	.long	1019957287
	.long	3145677161
	.long	1050893538
	.long	2990049718
	.long	1065353216
	.long	3185266517
	.long	821445502
	.long	1048576000
	.long	1035520850
	.long	3165913616
	.long	1019751749
	.long	3143905397
	.long	1050069988
	.long	2957827630
	.long	1065353216
	.long	3184147455
	.long	823956970
	.long	1048576000
	.long	1035381982
	.long	3164953691
	.long	1019591684
	.long	3143870825
	.long	1049246438
	.long	840887792
	.long	1065353216
	.long	3183031657
	.long	2948197632
	.long	1048576000
	.long	1035254815
	.long	3164006661
	.long	1019406069
	.long	3141406886
	.long	1048269777
	.long	831869830
	.long	1065353216
	.long	3181918839
	.long	829265530
	.long	1048576000
	.long	1035139196
	.long	3163071263
	.long	1019275107
	.long	3141473894
	.long	1046622678
	.long	2954471074
	.long	1065353216
	.long	3180808717
	.long	2974758491
	.long	1048576000
	.long	1035034991
	.long	3161787608
	.long	1019131285
	.long	3139614851
	.long	1044975579
	.long	2981870894
	.long	1065353216
	.long	3179701015
	.long	2951749952
	.long	1048576000
	.long	1034942077
	.long	3159956688
	.long	1019002541
	.long	3137649644
	.long	1043328479
	.long	832289399
	.long	1065353216
	.long	3177908479
	.long	2968441398
	.long	1048576000
	.long	1034860345
	.long	3158142289
	.long	1018906717
	.long	3137336762
	.long	1041681380
	.long	2949439022
	.long	1065353216
	.long	3175701100
	.long	2963548093
	.long	1048576000
	.long	1034789701
	.long	3156342344
	.long	1018810804
	.long	3133887847
	.long	1039881169
	.long	823481222
	.long	1065353216
	.long	3173496918
	.long	2969038481
	.long	1048576000
	.long	1034730062
	.long	3154554595
	.long	1018750428
	.long	3136028910
	.long	1036586971
	.long	2973482286
	.long	1065353216
	.long	3171295395
	.long	2968300352
	.long	1048576000
	.long	1034681361
	.long	3151437839
	.long	1018664053
	.long	3123383004
	.long	1033292772
	.long	2941050414
	.long	1065353216
	.long	3167298168
	.long	808398440
	.long	1048576000
	.long	1034643540
	.long	3147899215
	.long	1018610153
	.long	943964915
	.long	1028198363
	.long	2965093678
	.long	1065353216
	.long	3162902549
	.long	2950073902
	.long	1048576000
	.long	1034616555
	.long	3143016255
	.long	1018603598
	.long	3133555092
	.long	1019809755
	.long	2956705070
	.long	1065353216
	.long	3154512883
	.long	803361198
	.long	1048576000
	.long	1034600377
	.long	3134618720
	.long	1018580133
	.long	3134056577
	.long	0
	.long	0
	.long	1065353216
	.long	0
	.long	0
	.long	1048576000
	.long	1034594987
	.long	0
	.long	1018552971
	.long	0
	.long	3167293403
	.long	809221422
	.long	1065353216
	.long	1007029235
	.long	2950844846
	.long	1048576000
	.long	1034600377
	.long	987135072
	.long	1018580133
	.long	986572929
	.long	3175682011
	.long	817610030
	.long	1065353216
	.long	1015418901
	.long	802590254
	.long	1048576000
	.long	1034616555
	.long	995532607
	.long	1018603598
	.long	986071444
	.long	3180776420
	.long	793566766
	.long	1065353216
	.long	1019814520
	.long	2955882088
	.long	1048576000
	.long	1034643540
	.long	1000415567
	.long	1018610153
	.long	3091448562
	.long	3184070619
	.long	825998638
	.long	1065353216
	.long	1023811747
	.long	820816704
	.long	1048576000
	.long	1034681361
	.long	1003954191
	.long	1018664053
	.long	975899356
	.long	3187364817
	.long	2970964870
	.long	1065353216
	.long	1026013270
	.long	821554833
	.long	1048576000
	.long	1034730062
	.long	1007070947
	.long	1018750428
	.long	988545262
	.long	3189165028
	.long	801955374
	.long	1065353216
	.long	1028217452
	.long	816064445
	.long	1048576000
	.long	1034789701
	.long	1008858696
	.long	1018810804
	.long	986404199
	.long	3190812127
	.long	2979773047
	.long	1065353216
	.long	1030424831
	.long	820957750
	.long	1048576000
	.long	1034860345
	.long	1010658641
	.long	1018906717
	.long	989853114
	.long	3192459227
	.long	834387246
	.long	1065353216
	.long	1032217367
	.long	804266304
	.long	1048576000
	.long	1034942077
	.long	1012473040
	.long	1019002541
	.long	990165996
	.long	3194106326
	.long	806987426
	.long	1065353216
	.long	1033325069
	.long	827274843
	.long	1048576000
	.long	1035034991
	.long	1014303960
	.long	1019131285
	.long	992131203
	.long	3195753425
	.long	2979353478
	.long	1065353216
	.long	1034435191
	.long	2976749178
	.long	1048576000
	.long	1035139196
	.long	1015587615
	.long	1019275107
	.long	993990246
	.long	3196730086
	.long	2988371440
	.long	1065353216
	.long	1035548009
	.long	800713984
	.long	1048576000
	.long	1035254815
	.long	1016523013
	.long	1019406069
	.long	993923238
	.long	3197553636
	.long	810343982
	.long	1065353216
	.long	1036663807
	.long	2971440618
	.long	1048576000
	.long	1035381982
	.long	1017470043
	.long	1019591684
	.long	996387177
	.long	3198377186
	.long	842566070
	.long	1065353216
	.long	1037782869
	.long	2968929150
	.long	1048576000
	.long	1035520850
	.long	1018429968
	.long	1019751749
	.long	996421749
	.long	3199200735
	.long	2988161655
	.long	1065353216
	.long	1038905484
	.long	826840516
	.long	1048576000
	.long	1035671582
	.long	1019403942
	.long	1019957287
	.long	998193513
	.long	3200024285
	.long	813697757
	.long	1065353216
	.long	1040031947
	.long	2937388192
	.long	1048576000
	.long	1035834359
	.long	1020393243
	.long	1020189995
	.long	999315782
	.long	3200847835
	.long	842775854
	.long	1065353216
	.long	1040674973
	.long	831138435
	.long	1048576000
	.long	1036009378
	.long	1021399190
	.long	1020421234
	.long	999953008
	.long	3201671384
	.long	2987951871
	.long	1065353216
	.long	1041242501
	.long	808432312
	.long	1048576000
	.long	1036196851
	.long	1022423117
	.long	1020660679
	.long	1000421562
	.long	3202494934
	.long	815376034
	.long	1065353216
	.long	1041812407
	.long	803374128
	.long	1048576000
	.long	1036397006
	.long	1023438285
	.long	1020942892
	.long	1001436114
	.long	3203318484
	.long	842985639
	.long	1065353216
	.long	1042384848
	.long	834127863
	.long	1048576000
	.long	1036610091
	.long	1023970338
	.long	1021220110
	.long	1001954001
	.long	3204142033
	.long	2987742086
	.long	1065353216
	.long	1042959985
	.long	832371948
	.long	1048576000
	.long	1036836369
	.long	1024513541
	.long	1021543079
	.long	1003011479
	.long	3204706919
	.long	3002387417
	.long	1065353216
	.long	1043537982
	.long	2986276286
	.long	1048576000
	.long	1037076124
	.long	1025068684
	.long	1021893801
	.long	1004198485
	.long	3205118694
	.long	2996760048
	.long	1065353216
	.long	1044119004
	.long	825381402
	.long	1048576000
	.long	1037329660
	.long	1025636593
	.long	1022242934
	.long	1004982883
	.long	3205530469
	.long	2987532301
	.long	1065353216
	.long	1044703224
	.long	835363787
	.long	1048576000
	.long	1037597300
	.long	1026218117
	.long	1022609285
	.long	1005708104
	.long	3205942244
	.long	818732590
	.long	1065353216
	.long	1045290817
	.long	833527956
	.long	1048576000
	.long	1037879388
	.long	1026814142
	.long	1023026096
	.long	1006763255
	.long	3206354019
	.long	843405209
	.long	1065353216
	.long	1045881963
	.long	2984505345
	.long	1048576000
	.long	1038176293
	.long	1027425616
	.long	1023428141
	.long	1007234200
	.long	3206765794
	.long	850954678
	.long	1065353216
	.long	1046476844
	.long	829452858
	.long	1048576000
	.long	1038488404
	.long	1028053510
	.long	1023672824
	.long	1008001043
	.long	3207177568
	.long	3002177633
	.long	1065353216
	.long	1047075652
	.long	2985396534
	.long	1048576000
	.long	1038816139
	.long	1028698871
	.long	1023917626
	.long	1008617127
	.long	3207589343
	.long	2996550263
	.long	1065353216
	.long	1047678579
	.long	2971714530
	.long	1048576000
	.long	1039159939
	.long	1029362782
	.long	1024176063
	.long	1009236155
	.long	3208001118
	.long	2987112732
	.long	1065353216
	.long	1048285826
	.long	825459666
	.long	1048576000
	.long	1039520273
	.long	1030046387
	.long	1024452069
	.long	1009908500
	.long	3208412893
	.long	822086365
	.long	1065353216
	.long	1048736800
	.long	2987101005
	.long	1048576000
	.long	1039897640
	.long	1030750900
	.long	1024731756
	.long	1010452479
	.long	3208824668
	.long	843824778
	.long	1065353216
	.long	1049045055
	.long	845724006
	.long	1056964608
	.long	3187565865
	.long	1031477587
	.long	1025040649
	.long	1011183792
	.long	3209236443
	.long	851164462
	.long	0
	.long	3212836864
	.long	725680128
	.long	1073741824
	.long	3003121664
	.long	3221225472
	.long	1076541384
	.long	3226821083
	.long	3209648217
	.long	3001967848
	.long	0
	.long	3212032885
	.long	3002896525
	.long	1073741824
	.long	3183449264
	.long	3219674383
	.long	1075269479
	.long	3224815112
	.long	3210059992
	.long	2996340479
	.long	0
	.long	3211265630
	.long	2993098010
	.long	1073741824
	.long	3191263702
	.long	3218296839
	.long	1074178145
	.long	3223145434
	.long	3210471767
	.long	2986693162
	.long	0
	.long	3210531774
	.long	2974155528
	.long	1073741824
	.long	3196257989
	.long	3217068594
	.long	1072731698
	.long	3221740288
	.long	3210883542
	.long	823764642
	.long	0
	.long	3209828353
	.long	2997764472
	.long	1073741824
	.long	3198626104
	.long	3215969314
	.long	1071103306
	.long	3219894299
	.long	3211295317
	.long	844244347
	.long	0
	.long	3209152716
	.long	838923546
	.long	1073741824
	.long	3200789612
	.long	3214981865
	.long	1069688111
	.long	3217892551
	.long	3211707092
	.long	851374247
	.long	0
	.long	3208502479
	.long	3000145414
	.long	1073741824
	.long	3202769007
	.long	3214091734
	.long	1068453481
	.long	3216181260
	.long	3212118866
	.long	3001758063
	.long	0
	.long	3207875497
	.long	850964714
	.long	1065353216
	.long	1056830711
	.long	3213286568
	.long	1067373883
	.long	3214720734
	.long	3212530641
	.long	2996130694
	.long	0
	.long	3207269825
	.long	2997583546
	.long	1065353216
	.long	1055168194
	.long	3212274752
	.long	1066427841
	.long	3213471670
	.long	3212889640
	.long	2986202738
	.long	0
	.long	3206683703
	.long	839672284
	.long	1065353216
	.long	1053642609
	.long	3210943914
	.long	1065596017
	.long	3211952618
	.long	3213095527
	.long	3010776025
	.long	0
	.long	3206115524
	.long	2995800136
	.long	1065353216
	.long	1052241912
	.long	3209728124
	.long	1064374250
	.long	3210092525
	.long	3213301415
	.long	844663917
	.long	0
	.long	3205563823
	.long	3002090928
	.long	1065353216
	.long	1050955490
	.long	3208613851
	.long	1063075792
	.long	3208448547
	.long	3213507302
	.long	3005148656
	.long	0
	.long	3205027257
	.long	845085070
	.long	1065353216
	.long	1049773965
	.long	3207589321
	.long	1061932376
	.long	3207048862
	.long	3213713190
	.long	856424709
	.long	0
	.long	3204504589
	.long	840216434
	.long	1065353216
	.long	1048689044
	.long	3206644275
	.long	1060914481
	.long	3205797512
	.long	3213919077
	.long	2995920909
	.long	0
	.long	3203541104
	.long	2991747572
	.long	1065353216
	.long	1046810746
	.long	3205769712
	.long	1060014844
	.long	3204711576
	.long	3214124965
	.long	862052078
	.long	0
	.long	3202544697
	.long	838089118
	.long	1065353216
	.long	1044984860
	.long	3204957722
	.long	1059214863
	.long	3203049502
	.long	3214330852
	.long	827121198
	.long	0
	.long	3201569741
	.long	834691410
	.long	1065353216
	.long	1043312844
	.long	3203954379
	.long	1058502663
	.long	3201336375
	.long	3214536739
	.long	3007857448
	.long	0
	.long	3200614413
	.long	840221633
	.long	1065353216
	.long	1041784404
	.long	3202540354
	.long	1057868403
	.long	3199781921
	.long	3214742627
	.long	851793817
	.long	0
	.long	3199677008
	.long	839026723
	.long	1065353216
	.long	1040390392
	.long	3201214072
	.long	1057307751
	.long	3198426707
	.long	3214948514
	.long	3001338494
	.long	0
	.long	3198755927
	.long	2964850736
	.long	1065353216
	.long	1038057984
	.long	3199965673
	.long	1056656040
	.long	3197207230
	.long	3215154402
	.long	859343286
	.long	0
	.long	3197849666
	.long	841628398
	.long	1065353216
	.long	1035760784
	.long	3198786293
	.long	1055777031
	.long	3196119466
	.long	3215360289
	.long	2984524460
	.long	0
	.long	3196956802
	.long	825401908
	.long	1065353216
	.long	1033689040
	.long	3197667936
	.long	1054999523
	.long	3194181676
	.long	3215566176
	.long	3010566241
	.long	0
	.long	3196075988
	.long	841457254
	.long	1065353216
	.long	1031831496
	.long	3196603348
	.long	1054310701
	.long	3192272619
	.long	3215772064
	.long	845503056
	.long	0
	.long	3194352231
	.long	2981409247
	.long	1065353216
	.long	1028557712
	.long	3195112138
	.long	1053716836
	.long	3190669665
	.long	3215977951
	.long	3004938871
	.long	0
	.long	3192631215
	.long	2938776032
	.long	1065353216
	.long	1025642520
	.long	3193159376
	.long	1053195814
	.long	3189074146
	.long	3216183839
	.long	856634493
	.long	0
	.long	3190926925
	.long	838017617
	.long	1065353216
	.long	1022797056
	.long	3191277530
	.long	1052746889
	.long	3187399111
	.long	3216389726
	.long	2995501340
	.long	0
	.long	3189237092
	.long	835035876
	.long	1065353216
	.long	1018446032
	.long	3189456128
	.long	1052374628
	.long	3184936896
	.long	3216595614
	.long	862261863
	.long	0
	.long	3187448002
	.long	816730716
	.long	1065353216
	.long	1014578464
	.long	3187685445
	.long	1052059423
	.long	3182192286
	.long	3216801501
	.long	830474973
	.long	0
	.long	3184113116
	.long	2974133638
	.long	1065353216
	.long	1008660256
	.long	3184241386
	.long	1051807326
	.long	3179645874
	.long	3217007388
	.long	3007647664
	.long	0
	.long	3180794318
	.long	2974804776
	.long	1065353216
	.long	1001540608
	.long	3180848186
	.long	1051617929
	.long	3175942112
	.long	3217213276
	.long	852213386
	.long	0
	.long	3175692604
	.long	2963513179
	.long	1065353216
	.long	991832832
	.long	3175724500
	.long	1051479824
	.long	3170735141
	.long	3217419163
	.long	3000918924
	.long	0
	.long	3167296049
	.long	2945355034
	.long	1065353216
	.long	975043072
	.long	3167303981
	.long	1051400329
	.long	3163053371
	.type	__svml_stan_data_internal,@object
	.size	__svml_stan_data_internal,7232
	.align 64
__svml_stan_reduction_data_internal:
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
	.long	1
	.long	0
	.long	0
	.long	2
	.long	0
	.long	0
	.long	5
	.long	0
	.long	0
	.long	10
	.long	0
	.long	0
	.long	20
	.long	0
	.long	0
	.long	40
	.long	0
	.long	0
	.long	81
	.long	0
	.long	0
	.long	162
	.long	0
	.long	0
	.long	325
	.long	0
	.long	0
	.long	651
	.long	0
	.long	0
	.long	1303
	.long	0
	.long	0
	.long	2607
	.long	0
	.long	0
	.long	5215
	.long	0
	.long	0
	.long	10430
	.long	0
	.long	0
	.long	20860
	.long	0
	.long	0
	.long	41721
	.long	0
	.long	0
	.long	83443
	.long	0
	.long	0
	.long	166886
	.long	0
	.long	0
	.long	333772
	.long	0
	.long	0
	.long	667544
	.long	0
	.long	0
	.long	1335088
	.long	0
	.long	0
	.long	2670176
	.long	0
	.long	0
	.long	5340353
	.long	0
	.long	0
	.long	10680707
	.long	0
	.long	0
	.long	21361414
	.long	0
	.long	0
	.long	42722829
	.long	0
	.long	0
	.long	85445659
	.long	0
	.long	0
	.long	170891318
	.long	0
	.long	0
	.long	341782637
	.long	0
	.long	0
	.long	683565275
	.long	0
	.long	0
	.long	1367130551
	.long	0
	.long	0
	.long	2734261102
	.long	0
	.long	1
	.long	1173554908
	.long	0
	.long	2
	.long	2347109817
	.long	0
	.long	5
	.long	399252338
	.long	0
	.long	10
	.long	798504676
	.long	0
	.long	20
	.long	1597009353
	.long	0
	.long	40
	.long	3194018707
	.long	0
	.long	81
	.long	2093070119
	.long	0
	.long	162
	.long	4186140238
	.long	0
	.long	325
	.long	4077313180
	.long	0
	.long	651
	.long	3859659065
	.long	0
	.long	1303
	.long	3424350834
	.long	0
	.long	2607
	.long	2553734372
	.long	0
	.long	5215
	.long	812501448
	.long	0
	.long	10430
	.long	1625002897
	.long	0
	.long	20860
	.long	3250005794
	.long	0
	.long	41721
	.long	2205044292
	.long	0
	.long	83443
	.long	115121288
	.long	0
	.long	166886
	.long	230242576
	.long	0
	.long	333772
	.long	460485152
	.long	0
	.long	667544
	.long	920970305
	.long	0
	.long	1335088
	.long	1841940610
	.long	0
	.long	2670176
	.long	3683881221
	.long	0
	.long	5340353
	.long	3072795146
	.long	0
	.long	10680707
	.long	1850622997
	.long	0
	.long	21361414
	.long	3701245994
	.long	0
	.long	42722829
	.long	3107524692
	.long	0
	.long	85445659
	.long	1920082089
	.long	0
	.long	170891318
	.long	3840164178
	.long	0
	.long	341782637
	.long	3385361061
	.long	0
	.long	683565275
	.long	2475754826
	.long	0
	.long	1367130551
	.long	656542356
	.long	0
	.long	2734261102
	.long	1313084713
	.long	1
	.long	1173554908
	.long	2626169427
	.long	2
	.long	2347109817
	.long	957371559
	.long	5
	.long	399252338
	.long	1914743119
	.long	10
	.long	798504676
	.long	3829486239
	.long	20
	.long	1597009353
	.long	3364005183
	.long	40
	.long	3194018707
	.long	2433043071
	.long	81
	.long	2093070119
	.long	571118846
	.long	162
	.long	4186140238
	.long	1142237692
	.long	325
	.long	4077313180
	.long	2284475384
	.long	651
	.long	3859659065
	.long	273983472
	.long	1303
	.long	3424350834
	.long	547966945
	.long	2607
	.long	2553734372
	.long	1095933890
	.long	5215
	.long	812501448
	.long	2191867780
	.long	10430
	.long	1625002897
	.long	88768265
	.long	20860
	.long	3250005794
	.long	177536531
	.long	41721
	.long	2205044292
	.long	355073063
	.long	83443
	.long	115121288
	.long	710146126
	.long	166886
	.long	230242576
	.long	1420292253
	.long	333772
	.long	460485152
	.long	2840584506
	.long	667544
	.long	920970305
	.long	1386201717
	.long	1335088
	.long	1841940610
	.long	2772403434
	.long	2670176
	.long	3683881221
	.long	1249839573
	.long	5340353
	.long	3072795146
	.long	2499679147
	.long	10680707
	.long	1850622997
	.long	704390999
	.long	21361414
	.long	3701245994
	.long	1408781999
	.long	42722829
	.long	3107524692
	.long	2817563999
	.long	85445659
	.long	1920082089
	.long	1340160702
	.long	170891318
	.long	3840164178
	.long	2680321405
	.long	341782637
	.long	3385361061
	.long	1065675514
	.long	683565275
	.long	2475754826
	.long	2131351028
	.long	1367130551
	.long	656542356
	.long	4262702056
	.long	2734261102
	.long	1313084713
	.long	4230436817
	.long	1173554908
	.long	2626169427
	.long	4165906339
	.long	2347109817
	.long	957371559
	.long	4036845383
	.long	399252338
	.long	1914743119
	.long	3778723471
	.long	798504676
	.long	3829486239
	.long	3262479647
	.long	1597009353
	.long	3364005183
	.long	2229991998
	.long	3194018707
	.long	2433043071
	.long	165016701
	.long	2093070119
	.long	571118846
	.long	330033402
	.long	4186140238
	.long	1142237692
	.long	660066805
	.long	4077313180
	.long	2284475384
	.long	1320133610
	.long	3859659065
	.long	273983472
	.long	2640267220
	.long	3424350834
	.long	547966945
	.long	985567145
	.long	2553734372
	.long	1095933890
	.long	1971134291
	.long	812501448
	.long	2191867780
	.long	3942268582
	.long	1625002897
	.long	88768265
	.long	3589569869
	.long	3250005794
	.long	177536531
	.long	2884172442
	.long	2205044292
	.long	355073063
	.long	1473377588
	.long	115121288
	.long	710146126
	.long	2946755177
	.long	230242576
	.long	1420292253
	.long	1598543059
	.long	460485152
	.long	2840584506
	.long	3197086118
	.long	920970305
	.long	1386201717
	.long	2099204941
	.long	1841940610
	.long	2772403434
	.long	4198409883
	.long	3683881221
	.long	1249839573
	.long	4101852471
	.long	3072795146
	.long	2499679147
	.long	3908737646
	.long	1850622997
	.long	704390999
	.long	3522507997
	.long	3701245994
	.long	1408781999
	.long	2750048699
	.long	3107524692
	.long	2817563999
	.long	1205130103
	.long	1920082089
	.long	1340160702
	.long	2410260206
	.long	3840164178
	.long	2680321405
	.long	525553116
	.long	3385361061
	.long	1065675514
	.long	1051106232
	.long	2475754826
	.long	2131351028
	.long	2102212464
	.long	656542356
	.long	4262702056
	.long	4204424928
	.long	1313084713
	.long	4230436817
	.long	4113882560
	.long	2626169427
	.long	4165906339
	.long	3932797825
	.long	957371559
	.long	4036845383
	.long	3570628355
	.long	1914743119
	.long	3778723471
	.long	2846289414
	.long	3829486239
	.long	3262479647
	.long	1397611533
	.long	3364005183
	.long	2229991998
	.long	2795223067
	.long	2433043071
	.long	165016701
	.long	1295478838
	.long	571118846
	.long	330033402
	.long	2590957677
	.long	1142237692
	.long	660066805
	.long	886948059
	.long	2284475384
	.long	1320133610
	.long	1773896118
	.long	273983472
	.long	2640267220
	.long	3547792237
	.long	547966945
	.long	985567145
	.long	2800617179
	.long	1095933890
	.long	1971134291
	.long	1306267062
	.long	2191867780
	.long	3942268582
	.long	2612534124
	.long	88768265
	.long	3589569869
	.long	930100952
	.long	177536531
	.long	2884172442
	.long	1860201905
	.long	355073063
	.long	1473377588
	.long	3720403810
	.long	710146126
	.long	2946755177
	.long	3145840325
	.long	1420292253
	.long	1598543059
	.long	1996713354
	.long	2840584506
	.long	3197086118
	.long	3993426708
	.long	1386201717
	.long	2099204941
	.long	3691886121
	.long	2772403434
	.long	4198409883
	.long	3088804946
	.long	1249839573
	.long	4101852471
	.long	1882642597
	.long	2499679147
	.long	3908737646
	.long	3765285194
	.long	704390999
	.long	3522507997
	.long	3235603093
	.long	1408781999
	.long	2750048699
	.long	2176238891
	.long	2817563999
	.long	1205130103
	.long	57510486
	.long	1340160702
	.long	2410260206
	.long	115020972
	.long	2680321405
	.long	525553116
	.long	230041945
	.long	1065675514
	.long	1051106232
	.long	460083891
	.long	2131351028
	.long	2102212464
	.long	920167782
	.long	4262702056
	.long	4204424928
	.long	1840335564
	.long	4230436817
	.long	4113882560
	.long	3680671129
	.long	4165906339
	.long	3932797825
	.long	3066374962
	.long	4036845383
	.long	3570628355
	.long	1837782628
	.long	3778723471
	.long	2846289414
	.long	3675565257
	.long	3262479647
	.long	1397611533
	.long	3056163219
	.long	2229991998
	.long	2795223067
	.long	1817359143
	.long	165016701
	.long	1295478838
	.long	3634718287
	.long	330033402
	.long	2590957677
	.long	2974469278
	.long	660066805
	.long	886948059
	.long	1653971260
	.long	1320133610
	.long	1773896118
	.long	3307942520
	.long	2640267220
	.long	3547792237
	.long	2320917745
	.long	985567145
	.long	2800617179
	.long	346868194
	.long	1971134291
	.long	1306267062
	.long	693736388
	.long	3942268582
	.long	2612534124
	.long	1387472776
	.long	3589569869
	.long	930100952
	.long	2774945552
	.long	2884172442
	.long	1860201905
	.long	1254923809
	.long	1473377588
	.long	3720403810
	.long	2509847619
	.long	2946755177
	.long	3145840325
	.long	724727943
	.long	1598543059
	.long	1996713354
	.long	1449455886
	.long	3197086118
	.long	3993426708
	.long	2898911772
	.long	2099204941
	.long	3691886121
	.long	1502856249
	.long	4198409883
	.long	3088804946
	.long	3005712498
	.long	4101852471
	.long	1882642597
	.long	1716457700
	.long	3908737646
	.long	3765285194
	.long	3432915400
	.long	3522507997
	.long	3235603093
	.long	2570863504
	.long	2750048699
	.long	2176238891
	.long	846759712
	.long	1205130103
	.long	57510486
	.long	1693519425
	.long	2410260206
	.long	115020972
	.long	3387038850
	.long	525553116
	.long	230041945
	.long	2479110404
	.long	1051106232
	.long	460083891
	.long	663253512
	.long	2102212464
	.long	920167782
	.long	1326507024
	.long	4204424928
	.long	1840335564
	.long	2653014048
	.long	4113882560
	.long	3680671129
	.long	1011060801
	.long	3932797825
	.long	3066374962
	.long	2022121603
	.long	3570628355
	.long	1837782628
	.long	4044243207
	.long	2846289414
	.long	3675565257
	.long	3793519119
	.long	1397611533
	.long	3056163219
	.long	3292070943
	.long	2795223067
	.long	1817359143
	.long	2289174591
	.long	1295478838
	.long	3634718287
	.long	283381887
	.long	2590957677
	.long	2974469278
	.long	566763775
	.type	__svml_stan_reduction_data_internal,@object
	.size	__svml_stan_reduction_data_internal,3072
	.align 4
__stan_la__vmlsTanTab:
	.long	0
	.long	2139095040
	.type	__stan_la__vmlsTanTab,@object
	.size	__stan_la__vmlsTanTab,8
