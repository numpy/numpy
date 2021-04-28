/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *  1) Range reduction to [-Pi/2; +Pi/2] interval
 *     a) Grab sign from source argument and save it.
 *     b) Remove sign using AND operation
 *     c) Getting octant Y by 1/Pi multiplication
 *     d) Add "Right Shifter" value
 *     e) Treat obtained value as integer for destination sign setting.
 *        Shift first bit of this value to the last (sign) position
 *     f) Change destination sign if source sign is negative
 *        using XOR operation.
 *     g) Subtract "Right Shifter" value
 *     h) Subtract Y*PI from X argument, where PI divided to 4 parts:
 *        X = X - Y*PI1 - Y*PI2 - Y*PI3 - Y*PI4;
 *  2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
 *     a) Calculate X^2 = X * X
 *     b) Calculate polynomial:
 *        R = X + X * X^2 * (A3 + x^2 * (A5 + ......
 *  3) Destination sign setting
 *     a) Set shifted destination sign using XOR operation:
 *        R = XOR( R, S );
 * 
 */


	.text
.L_2__routine_start___svml_sinf16_z0_0:

	.align    16,0x90
	.globl __svml_sinf16

__svml_sinf16:


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
        vmovups   4096+__svml_ssin_data_internal(%rip), %zmm7
        vmovups   5248+__svml_ssin_data_internal(%rip), %zmm2
        vmovups   5312+__svml_ssin_data_internal(%rip), %zmm3
        vmovups   4800+__svml_ssin_data_internal(%rip), %zmm6
        vmovups   4864+__svml_ssin_data_internal(%rip), %zmm4
        vmovups   4928+__svml_ssin_data_internal(%rip), %zmm5
        vmovups   5184+__svml_ssin_data_internal(%rip), %zmm10
        vmovups   5120+__svml_ssin_data_internal(%rip), %zmm14
        vmovups   5056+__svml_ssin_data_internal(%rip), %zmm12
        vmovups   4992+__svml_ssin_data_internal(%rip), %zmm13
        vmovaps   %zmm0, %zmm11

/* b) Remove sign using AND operation */
        vandps    %zmm7, %zmm11, %zmm1

/*
 * f) Change destination sign if source sign is negative
 * using XOR operation.
 */
        vandnps   %zmm11, %zmm7, %zmm0

/*
 * c) Getting octant Y by 1/Pi multiplication
 * d) Add "Right Shifter" value
 */
        vfmadd213ps {rn-sae}, %zmm3, %zmm1, %zmm2

/* g) Subtract "Right Shifter" value */
        vsubps    {rn-sae}, %zmm3, %zmm2, %zmm8

/*
 * e) Treat obtained value as integer for destination sign setting.
 * Shift first bit of this value to the last (sign) position
 */
        vpslld    $31, %zmm2, %zmm9

/* Check for large and special values */
        vmovups   4160+__svml_ssin_data_internal(%rip), %zmm2

/*
 * h) Subtract Y*PI from X argument, where PI divided to 4 parts:
 * X = X - Y*PI1 - Y*PI2 - Y*PI3;
 */
        vfnmadd213ps {rn-sae}, %zmm1, %zmm8, %zmm6
        vcmpps    $18, {sae}, %zmm2, %zmm1, %k1
        vfnmadd231ps {rn-sae}, %zmm8, %zmm4, %zmm6
        vfnmadd213ps {rn-sae}, %zmm6, %zmm5, %zmm8

/*
 * 2) Polynomial (minimax for sin within [-Pi/2; +Pi/2] interval)
 * a) Calculate X^2 = X * X
 * b) Calculate polynomial:
 * R = X + X * X^2 * (A3 + x^2 * (A5 + ......
 */
        vmulps    {rn-sae}, %zmm8, %zmm8, %zmm15
        vxorps    %zmm9, %zmm8, %zmm8
        vfmadd231ps {rn-sae}, %zmm15, %zmm10, %zmm14
        vpternlogd $255, %zmm10, %zmm10, %zmm10
        vfmadd213ps {rn-sae}, %zmm12, %zmm15, %zmm14
        vfmadd213ps {rn-sae}, %zmm13, %zmm15, %zmm14
        vmulps    {rn-sae}, %zmm15, %zmm14, %zmm9
        vfmadd213ps {rn-sae}, %zmm8, %zmm8, %zmm9

/*
 * 3) Destination sign setting
 * a) Set shifted destination sign using XOR operation:
 * R = XOR( R, S );
 */
        vxorps    %zmm0, %zmm9, %zmm0
        vpandnd   %zmm1, %zmm1, %zmm10{%k1}
        vptestmd  %zmm10, %zmm10, %k0
        kortestw  %k0, %k0
        jne       .LBL_1_12

.LBL_1_2:

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

        call      __svml_ssin_cout_rare_internal
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

        vmovups   4224+__svml_ssin_data_internal(%rip), %zmm8

/*
 * Get the (2^a / 2pi) mod 1 values from the table.
 * Because VLANG doesn't have I-type gather, we need a trivial cast
 */
        lea       __svml_ssin_reduction_data_internal(%rip), %rax
        vmovups   %zmm0, (%rsp)
        vandps    %zmm1, %zmm8, %zmm6
        vpternlogd $255, %zmm1, %zmm1, %zmm1
        vcmpps    $4, {sae}, %zmm8, %zmm6, %k1

/* ..................... Table look-up ........................ */
        lea       __svml_ssin_data_internal(%rip), %rcx
        vpandd    .L_2il0floatpacket.21(%rip), %zmm11, %zmm12
        vpandnd   %zmm6, %zmm6, %zmm1{%k1}
        vpsrld    $23, %zmm12, %zmm13
        vptestmd  %zmm1, %zmm1, %k0

/*
 * Break the P_xxx and m into 16-bit chunks ready for
 * the long multiplication via 16x16->32 multiplications
 */
        vmovups   .L_2il0floatpacket.24(%rip), %zmm6
        vpslld    $1, %zmm13, %zmm4
        kxnorw    %k0, %k0, %k3
        kxnorw    %k0, %k0, %k1
        kxnorw    %k0, %k0, %k2
        kmovw     %k0, %edx
        vpaddd    %zmm13, %zmm4, %zmm9
        vpslld    $2, %zmm9, %zmm5
        vpxord    %zmm3, %zmm3, %zmm3
        vpxord    %zmm14, %zmm14, %zmm14
        vpxord    %zmm8, %zmm8, %zmm8
        vgatherdps 4(%rax,%zmm5), %zmm3{%k3}
        vgatherdps 8(%rax,%zmm5), %zmm14{%k1}
        vgatherdps (%rax,%zmm5), %zmm8{%k2}
        vpsrld    $16, %zmm3, %zmm15
        vpsrld    $16, %zmm14, %zmm13
        vpsrld    $16, %zmm8, %zmm7
        kxnorw    %k0, %k0, %k1

/*
 * Also get the significand as an integer
 * NB: adding in the integer bit is wrong for denorms!
 * To make this work for denorms we should do something slightly different
 */
        vpandd    .L_2il0floatpacket.22(%rip), %zmm11, %zmm2
        vpaddd    .L_2il0floatpacket.23(%rip), %zmm2, %zmm1
        vpsrld    $16, %zmm1, %zmm9
        vpandd    %zmm6, %zmm8, %zmm0
        vpandd    %zmm6, %zmm3, %zmm12
        vpandd    %zmm6, %zmm14, %zmm5
        vpandd    %zmm6, %zmm1, %zmm14

/* Now do the big multiplication and carry propagation */
        vpmulld   %zmm0, %zmm9, %zmm4
        vpmulld   %zmm15, %zmm9, %zmm3
        vpmulld   %zmm12, %zmm9, %zmm2
        vpmulld   %zmm13, %zmm9, %zmm1
        vpmulld   %zmm5, %zmm9, %zmm8
        vpmulld   %zmm15, %zmm14, %zmm9
        vpmulld   %zmm13, %zmm14, %zmm15
        vpmulld   %zmm7, %zmm14, %zmm7
        vpmulld   %zmm0, %zmm14, %zmm5
        vpmulld   %zmm12, %zmm14, %zmm0
        vpsrld    $16, %zmm15, %zmm14
        vpsrld    $16, %zmm9, %zmm15
        vpsrld    $16, %zmm0, %zmm13
        vpsrld    $16, %zmm5, %zmm12
        vpaddd    %zmm15, %zmm3, %zmm3
        vpaddd    %zmm14, %zmm1, %zmm1
        vpaddd    %zmm13, %zmm2, %zmm14
        vpaddd    %zmm12, %zmm4, %zmm2
        vpandd    %zmm6, %zmm5, %zmm4
        vpaddd    %zmm3, %zmm4, %zmm3
        vpsrld    $16, %zmm8, %zmm4
        vpandd    %zmm6, %zmm0, %zmm0
        vpaddd    %zmm1, %zmm0, %zmm13
        vpandd    %zmm6, %zmm7, %zmm7
        vpaddd    %zmm2, %zmm7, %zmm0
        vpaddd    %zmm13, %zmm4, %zmm7
        vpandd    %zmm6, %zmm9, %zmm12
        vpsrld    $16, %zmm7, %zmm9
        vpaddd    %zmm14, %zmm12, %zmm5
        vpaddd    %zmm5, %zmm9, %zmm1
        vpsrld    $16, %zmm1, %zmm2
        vpslld    $16, %zmm1, %zmm13
        vpaddd    %zmm3, %zmm2, %zmm15
        vpsrld    $16, %zmm15, %zmm8
        vpaddd    %zmm0, %zmm8, %zmm14

/* Assemble reduced argument from the pieces */
        vpandd    %zmm6, %zmm7, %zmm12
        vpandd    %zmm6, %zmm15, %zmm4
        vpslld    $16, %zmm14, %zmm6
        vpaddd    %zmm12, %zmm13, %zmm13
        vpaddd    %zmm4, %zmm6, %zmm9

/*
 * Now round at the 2^-8 bit position for reduction mod pi/2^7
 * instead of the original 2pi (but still with the same 2pi scaling).
 * Use a shifter of 2^15 + 2^14.
 * The N we get is our final version; it has an offset of
 * 2^8 because of the implicit integer bit, and anyway for negative
 * starting value it's a 2s complement thing. But we need to mask
 * off the exponent part anyway so it's fine.
 */
        vmovups   .L_2il0floatpacket.27(%rip), %zmm6
        vpsrld    $9, %zmm9, %zmm2
        vpandd    .L_2il0floatpacket.30(%rip), %zmm13, %zmm15
        vpslld    $5, %zmm15, %zmm1
        vpsrld    $18, %zmm13, %zmm13

/*
 * We want to incorporate the original sign now too.
 * Do it here for convenience in getting the right N value,
 * though we could wait right to the end if we were prepared
 * to modify the sign of N later too.
 * So get the appropriate sign mask now (or sooner).
 */
        vpandd    .L_2il0floatpacket.25(%rip), %zmm11, %zmm5

/*
 * Create floating-point high part, implicitly adding integer bit 1
 * Incorporate overall sign at this stage too.
 */
        vpxord    .L_2il0floatpacket.26(%rip), %zmm5, %zmm8
        vpord     %zmm8, %zmm2, %zmm3
        vaddps    {rn-sae}, %zmm3, %zmm6, %zmm7
        vsubps    {rn-sae}, %zmm6, %zmm7, %zmm0
        vsubps    {rn-sae}, %zmm0, %zmm3, %zmm14
        vandps    .L_2il0floatpacket.35(%rip), %zmm11, %zmm3
        vpternlogd $255, %zmm0, %zmm0, %zmm0

/*
 * Create floating-point low and medium parts, respectively
 * lo_17, ... lo_0, 0, ..., 0
 * hi_8, ... hi_0, lo_31, ..., lo_18
 * then subtract off the implicitly added integer bits,
 * 2^-46 and 2^-23, respectively.
 * Put the original sign into all of them at this stage.
 */
        vpxord    .L_2il0floatpacket.29(%rip), %zmm5, %zmm8
        vpord     %zmm8, %zmm1, %zmm2
        vpandd    .L_2il0floatpacket.32(%rip), %zmm9, %zmm9
        vsubps    {rn-sae}, %zmm8, %zmm2, %zmm15
        vpxord    .L_2il0floatpacket.31(%rip), %zmm5, %zmm2
        vpslld    $14, %zmm9, %zmm5
        vpord     %zmm13, %zmm5, %zmm5
        vpord     %zmm2, %zmm5, %zmm9
        vsubps    {rn-sae}, %zmm2, %zmm9, %zmm8

/*
 * Now multiply those numbers all by 2 pi, reasonably accurately.
 * (RHi + RLo) * (pi_lead + pi_trail) ~=
 * RHi * pi_lead + (RHi * pi_trail + RLo * pi_lead)
 */
        vmovups   .L_2il0floatpacket.33(%rip), %zmm9
        vmovups   .L_2il0floatpacket.34(%rip), %zmm2

/* Now add them up into 2 reasonably aligned pieces */
        vaddps    {rn-sae}, %zmm8, %zmm14, %zmm6
        vsubps    {rn-sae}, %zmm6, %zmm14, %zmm14
        vmovaps   %zmm9, %zmm5
        vaddps    {rn-sae}, %zmm14, %zmm8, %zmm13

/*
 * If the magnitude of the input is <= 2^-20, then
 * just pass through the input, since no reduction will be needed and
 * the main path will only work accurately if the reduced argument is
 * about >= 2^-40 (which it is for all large pi multiples)
 */
        vmovups   .L_2il0floatpacket.36(%rip), %zmm8
        vaddps    {rn-sae}, %zmm15, %zmm13, %zmm14
        vpternlogd $255, %zmm15, %zmm15, %zmm15
        vcmpps    $26, {sae}, %zmm8, %zmm3, %k2
        vcmpps    $22, {sae}, %zmm8, %zmm3, %k3

/* Grab our final N value as an integer, appropriately masked mod 2^8 */
        vpandd    .L_2il0floatpacket.28(%rip), %zmm7, %zmm4
        vmulps    {rn-sae}, %zmm9, %zmm6, %zmm7
        vfmsub213ps {rn-sae}, %zmm7, %zmm6, %zmm5
        vfmadd213ps {rn-sae}, %zmm5, %zmm2, %zmm6
        vfmadd213ps {rn-sae}, %zmm6, %zmm9, %zmm14
        vpslld    $4, %zmm4, %zmm9
        vpandnd   %zmm3, %zmm3, %zmm15{%k2}
        vpandnd   %zmm3, %zmm3, %zmm0{%k3}
        kxnorw    %k0, %k0, %k2
        kxnorw    %k0, %k0, %k3
        vandps    %zmm7, %zmm15, %zmm12
        vandps    %zmm11, %zmm0, %zmm1
        vandps    %zmm14, %zmm15, %zmm2
        vorps     %zmm12, %zmm1, %zmm13

/* ............... Polynomial approximation ................... */
        vmovups   4352+__svml_ssin_data_internal(%rip), %zmm12

/*
 * The output is _VRES_R (high) + _VRES_E (low), and the integer part is _VRES_IND
 * Set sRp2 = _VRES_R^2 and then resume the original code.
 */
        vmulps    {rn-sae}, %zmm13, %zmm13, %zmm3

/* ................. Reconstruction: res_hi ................... */
        vmovaps   %zmm13, %zmm8
        vmovaps   %zmm13, %zmm4
        vpxord    %zmm5, %zmm5, %zmm5
        vpxord    %zmm7, %zmm7, %zmm7
        vgatherdps 4(%rcx,%zmm9), %zmm5{%k2}
        vgatherdps 12(%rcx,%zmm9), %zmm7{%k3}
        vfmadd213ps {rn-sae}, %zmm5, %zmm7, %zmm8
        vsubps    {rn-sae}, %zmm8, %zmm5, %zmm1
        vfmadd231ps {rn-sae}, %zmm13, %zmm7, %zmm1
        vpxord    %zmm0, %zmm0, %zmm0
        vgatherdps (%rcx,%zmm9), %zmm0{%k1}

/* ................. Reconstruction: res_lo=corr+polS+polC+res_lo0 ................... */
        kxnorw    %k0, %k0, %k1
        vfmadd132ps {rn-sae}, %zmm0, %zmm8, %zmm4
        vsubps    {rn-sae}, %zmm4, %zmm8, %zmm6
        vfmadd231ps {rn-sae}, %zmm0, %zmm13, %zmm6
        vaddps    {rn-sae}, %zmm1, %zmm6, %zmm8
        vaddps    {rn-sae}, %zmm7, %zmm0, %zmm1
        vmovups   4288+__svml_ssin_data_internal(%rip), %zmm6
        vmovups   4480+__svml_ssin_data_internal(%rip), %zmm0
        vmovups   4416+__svml_ssin_data_internal(%rip), %zmm7
        vfmadd231ps {rn-sae}, %zmm3, %zmm12, %zmm6
        vfmadd231ps {rn-sae}, %zmm3, %zmm0, %zmm7
        vmulps    {rn-sae}, %zmm3, %zmm6, %zmm14
        vmulps    {rn-sae}, %zmm3, %zmm7, %zmm0
        vmulps    {rn-sae}, %zmm13, %zmm14, %zmm6
        vfnmadd213ps {rn-sae}, %zmm1, %zmm5, %zmm13
        vfmadd213ps {rn-sae}, %zmm8, %zmm13, %zmm6
        vpxord    %zmm3, %zmm3, %zmm3
        vgatherdps 8(%rcx,%zmm9), %zmm3{%k1}
        vfmadd213ps {rn-sae}, %zmm3, %zmm13, %zmm2
        vfmadd213ps {rn-sae}, %zmm2, %zmm5, %zmm0
        vaddps    {rn-sae}, %zmm6, %zmm0, %zmm2

/* .................. Final reconstruction ................... */
        vaddps    {rn-sae}, %zmm2, %zmm4, %zmm4

/*
 * ----------------------------------------------------------
 * End of large arguments path
 * ----------------------------------------------------------
 * Merge results from main and large paths:
 */
        vpandnd   (%rsp), %zmm10, %zmm0
        vpandd    %zmm10, %zmm4, %zmm10
        vpord     %zmm10, %zmm0, %zmm0
        jmp       .LBL_1_2
	.align    16,0x90

	.cfi_endproc

	.type	__svml_sinf16,@function
	.size	__svml_sinf16,.-__svml_sinf16
..LN__svml_sinf16.0:

.L_2__routine_start___svml_ssin_cout_rare_internal_1:

	.align    16,0x90

__svml_ssin_cout_rare_internal:


	.cfi_startproc
..L63:

        movl      (%rdi), %edx
        movzwl    2(%rdi), %eax
        movl      %edx, -8(%rsp)
        andl      $32640, %eax
        shrl      $24, %edx
        andl      $127, %edx
        movss     (%rdi), %xmm1
        cmpl      $32640, %eax
        jne       .LBL_2_6


        movb      %dl, -5(%rsp)
        cmpl      $2139095040, -8(%rsp)
        jne       .LBL_2_4


        movss     __ssin_la__vmlsSinHATab(%rip), %xmm0
        movl      $1, %eax
        mulss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret

.LBL_2_4:

        mulss     (%rdi), %xmm1
        xorl      %eax, %eax
        movss     %xmm1, (%rsi)


        ret

.LBL_2_6:

        xorl      %eax, %eax
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_ssin_cout_rare_internal,@function
	.size	__svml_ssin_cout_rare_internal,.-__svml_ssin_cout_rare_internal
..LN__svml_ssin_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
.L_2il0floatpacket.21:
	.long	0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000,0x7f800000
	.type	.L_2il0floatpacket.21,@object
	.size	.L_2il0floatpacket.21,64
	.align 64
.L_2il0floatpacket.22:
	.long	0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff,0x007fffff
	.type	.L_2il0floatpacket.22,@object
	.size	.L_2il0floatpacket.22,64
	.align 64
.L_2il0floatpacket.23:
	.long	0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000,0x00800000
	.type	.L_2il0floatpacket.23,@object
	.size	.L_2il0floatpacket.23,64
	.align 64
.L_2il0floatpacket.24:
	.long	0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff,0x0000ffff
	.type	.L_2il0floatpacket.24,@object
	.size	.L_2il0floatpacket.24,64
	.align 64
.L_2il0floatpacket.25:
	.long	0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000,0x80000000
	.type	.L_2il0floatpacket.25,@object
	.size	.L_2il0floatpacket.25,64
	.align 64
.L_2il0floatpacket.26:
	.long	0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000,0x3f800000
	.type	.L_2il0floatpacket.26,@object
	.size	.L_2il0floatpacket.26,64
	.align 64
.L_2il0floatpacket.27:
	.long	0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000,0x47400000
	.type	.L_2il0floatpacket.27,@object
	.size	.L_2il0floatpacket.27,64
	.align 64
.L_2il0floatpacket.28:
	.long	0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff,0x000000ff
	.type	.L_2il0floatpacket.28,@object
	.size	.L_2il0floatpacket.28,64
	.align 64
.L_2il0floatpacket.29:
	.long	0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000,0x28800000
	.type	.L_2il0floatpacket.29,@object
	.size	.L_2il0floatpacket.29,64
	.align 64
.L_2il0floatpacket.30:
	.long	0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff,0x0003ffff
	.type	.L_2il0floatpacket.30,@object
	.size	.L_2il0floatpacket.30,64
	.align 64
.L_2il0floatpacket.31:
	.long	0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000,0x34000000
	.type	.L_2il0floatpacket.31,@object
	.size	.L_2il0floatpacket.31,64
	.align 64
.L_2il0floatpacket.32:
	.long	0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff,0x000001ff
	.type	.L_2il0floatpacket.32,@object
	.size	.L_2il0floatpacket.32,64
	.align 64
.L_2il0floatpacket.33:
	.long	0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb,0x40c90fdb
	.type	.L_2il0floatpacket.33,@object
	.size	.L_2il0floatpacket.33,64
	.align 64
.L_2il0floatpacket.34:
	.long	0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e,0xb43bbd2e
	.type	.L_2il0floatpacket.34,@object
	.size	.L_2il0floatpacket.34,64
	.align 64
.L_2il0floatpacket.35:
	.long	0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff
	.type	.L_2il0floatpacket.35,@object
	.size	.L_2il0floatpacket.35,64
	.align 64
.L_2il0floatpacket.36:
	.long	0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000,0x35800000
	.type	.L_2il0floatpacket.36,@object
	.size	.L_2il0floatpacket.36,64
	.align 64
__svml_ssin_data_internal:
	.long	0
	.long	0
	.long	0
	.long	1065353216
	.long	3114133471
	.long	1019808432
	.long	2953169304
	.long	1065353216
	.long	3130909128
	.long	1028193072
	.long	2968461951
	.long	1065353216
	.long	3140588184
	.long	1033283845
	.long	2975014497
	.long	1065353216
	.long	3147680113
	.long	1036565814
	.long	2960495349
	.long	1065353216
	.long	3153489468
	.long	1039839859
	.long	2970970319
	.long	1065353216
	.long	3157349634
	.long	1041645699
	.long	837346836
	.long	1065353216
	.long	3161536011
	.long	1043271842
	.long	823224313
	.long	1065353216
	.long	3164432432
	.long	1044891074
	.long	2967836285
	.long	1065353216
	.long	3167161428
	.long	1046502419
	.long	833086710
	.long	1065353216
	.long	3170205956
	.long	1048104908
	.long	2971391005
	.long	1065353216
	.long	3172229004
	.long	1049136787
	.long	824999326
	.long	1065353216
	.long	3174063957
	.long	1049927729
	.long	846027248
	.long	1065353216
	.long	3176053642
	.long	1050712805
	.long	2990442912
	.long	1065353216
	.long	3178196862
	.long	1051491540
	.long	2988789250
	.long	1065353216
	.long	3179887378
	.long	1052263466
	.long	2993707942
	.long	1065353216
	.long	3181110540
	.long	1053028117
	.long	836097324
	.long	1065353216
	.long	3182408396
	.long	1053785034
	.long	829045603
	.long	1065353216
	.long	3183780163
	.long	1054533760
	.long	840832460
	.long	1065353216
	.long	3185225016
	.long	1055273845
	.long	2983839604
	.long	1065353216
	.long	3186742084
	.long	1056004842
	.long	2986287417
	.long	1065353216
	.long	3188000746
	.long	1056726311
	.long	2978016425
	.long	1065353216
	.long	3188830103
	.long	1057201213
	.long	2992349186
	.long	1065353216
	.long	3189694133
	.long	1057551771
	.long	2998815566
	.long	1065353216
	.long	3190592315
	.long	1057896922
	.long	2991207143
	.long	1065353216
	.long	3191524108
	.long	1058236458
	.long	852349230
	.long	1065353216
	.long	3192488951
	.long	1058570176
	.long	2982650867
	.long	1065353216
	.long	3193486263
	.long	1058897873
	.long	848430348
	.long	1065353216
	.long	3194515443
	.long	1059219353
	.long	841032635
	.long	1065353216
	.long	3195575871
	.long	1059534422
	.long	2986574659
	.long	1065353216
	.long	3196363278
	.long	1059842890
	.long	2998350134
	.long	1065353216
	.long	3196923773
	.long	1060144571
	.long	2997759282
	.long	1065353216
	.long	3197498906
	.long	1060439283
	.long	844097402
	.long	1065353216
	.long	1044518635
	.long	1060726850
	.long	2994798599
	.long	1056964608
	.long	1043311911
	.long	1061007097
	.long	832220140
	.long	1056964608
	.long	1042078039
	.long	1061279856
	.long	851442039
	.long	1056964608
	.long	1040817765
	.long	1061544963
	.long	850481524
	.long	1056964608
	.long	1038876298
	.long	1061802258
	.long	848897600
	.long	1056964608
	.long	1036254719
	.long	1062051586
	.long	847147240
	.long	1056964608
	.long	1033584979
	.long	1062292797
	.long	806113028
	.long	1056964608
	.long	1029938589
	.long	1062525745
	.long	848357914
	.long	1056964608
	.long	1024416170
	.long	1062750291
	.long	2994560960
	.long	1056964608
	.long	1013387058
	.long	1062966298
	.long	841166280
	.long	1056964608
	.long	3152590408
	.long	1063173637
	.long	851900755
	.long	1056964608
	.long	3169472868
	.long	1063372184
	.long	3001545765
	.long	1056964608
	.long	3176031322
	.long	1063561817
	.long	823789818
	.long	1056964608
	.long	3180617215
	.long	1063742424
	.long	2998678409
	.long	1056964608
	.long	3183612120
	.long	1063913895
	.long	3001754476
	.long	1056964608
	.long	3186639787
	.long	1064076126
	.long	854796500
	.long	1056964608
	.long	3188684717
	.long	1064229022
	.long	2995991516
	.long	1056964608
	.long	1035072335
	.long	1064372488
	.long	840880349
	.long	1048576000
	.long	1031957395
	.long	1064506439
	.long	851742225
	.long	1048576000
	.long	1025835404
	.long	1064630795
	.long	2996018466
	.long	1048576000
	.long	1015605553
	.long	1064745479
	.long	846006572
	.long	1048576000
	.long	3152414341
	.long	1064850424
	.long	2987244005
	.long	1048576000
	.long	3170705253
	.long	1064945565
	.long	851856985
	.long	1048576000
	.long	3177244920
	.long	1065030846
	.long	855602635
	.long	1048576000
	.long	1027359369
	.long	1065106216
	.long	2989610635
	.long	1040187392
	.long	1018299420
	.long	1065171628
	.long	2969000681
	.long	1040187392
	.long	3140071849
	.long	1065227044
	.long	3002197507
	.long	1040187392
	.long	3168602920
	.long	1065272429
	.long	838093129
	.long	1040187392
	.long	1010124837
	.long	1065307757
	.long	852498564
	.long	1031798784
	.long	3160150850
	.long	1065333007
	.long	836655967
	.long	1031798784
	.long	3151746369
	.long	1065348163
	.long	814009613
	.long	1023410176
	.long	0
	.long	1065353216
	.long	0
	.long	0
	.long	1004262721
	.long	1065348163
	.long	814009613
	.long	3170893824
	.long	1012667202
	.long	1065333007
	.long	836655967
	.long	3179282432
	.long	3157608485
	.long	1065307757
	.long	852498564
	.long	3179282432
	.long	1021119272
	.long	1065272429
	.long	838093129
	.long	3187671040
	.long	992588201
	.long	1065227044
	.long	3002197507
	.long	3187671040
	.long	3165783068
	.long	1065171628
	.long	2969000681
	.long	3187671040
	.long	3174843017
	.long	1065106216
	.long	2989610635
	.long	3187671040
	.long	1029761272
	.long	1065030846
	.long	855602635
	.long	3196059648
	.long	1023221605
	.long	1064945565
	.long	851856985
	.long	3196059648
	.long	1004930693
	.long	1064850424
	.long	2987244005
	.long	3196059648
	.long	3163089201
	.long	1064745479
	.long	846006572
	.long	3196059648
	.long	3173319052
	.long	1064630795
	.long	2996018466
	.long	3196059648
	.long	3179441043
	.long	1064506439
	.long	851742225
	.long	3196059648
	.long	3182555983
	.long	1064372488
	.long	840880349
	.long	3196059648
	.long	1041201069
	.long	1064229022
	.long	2995991516
	.long	3204448256
	.long	1039156139
	.long	1064076126
	.long	854796500
	.long	3204448256
	.long	1036128472
	.long	1063913895
	.long	3001754476
	.long	3204448256
	.long	1033133567
	.long	1063742424
	.long	2998678409
	.long	3204448256
	.long	1028547674
	.long	1063561817
	.long	823789818
	.long	3204448256
	.long	1021989220
	.long	1063372184
	.long	3001545765
	.long	3204448256
	.long	1005106760
	.long	1063173637
	.long	851900755
	.long	3204448256
	.long	3160870706
	.long	1062966298
	.long	841166280
	.long	3204448256
	.long	3171899818
	.long	1062750291
	.long	2994560960
	.long	3204448256
	.long	3177422237
	.long	1062525745
	.long	848357914
	.long	3204448256
	.long	3181068627
	.long	1062292797
	.long	806113028
	.long	3204448256
	.long	3183738367
	.long	1062051586
	.long	847147240
	.long	3204448256
	.long	3186359946
	.long	1061802258
	.long	848897600
	.long	3204448256
	.long	3188301413
	.long	1061544963
	.long	850481524
	.long	3204448256
	.long	3189561687
	.long	1061279856
	.long	851442039
	.long	3204448256
	.long	3190795559
	.long	1061007097
	.long	832220140
	.long	3204448256
	.long	3192002283
	.long	1060726850
	.long	2994798599
	.long	3204448256
	.long	1050015258
	.long	1060439283
	.long	844097402
	.long	3212836864
	.long	1049440125
	.long	1060144571
	.long	2997759282
	.long	3212836864
	.long	1048879630
	.long	1059842890
	.long	2998350134
	.long	3212836864
	.long	1048092223
	.long	1059534422
	.long	2986574659
	.long	3212836864
	.long	1047031795
	.long	1059219353
	.long	841032635
	.long	3212836864
	.long	1046002615
	.long	1058897873
	.long	848430348
	.long	3212836864
	.long	1045005303
	.long	1058570176
	.long	2982650867
	.long	3212836864
	.long	1044040460
	.long	1058236458
	.long	852349230
	.long	3212836864
	.long	1043108667
	.long	1057896922
	.long	2991207143
	.long	3212836864
	.long	1042210485
	.long	1057551771
	.long	2998815566
	.long	3212836864
	.long	1041346455
	.long	1057201213
	.long	2992349186
	.long	3212836864
	.long	1040517098
	.long	1056726311
	.long	2978016425
	.long	3212836864
	.long	1039258436
	.long	1056004842
	.long	2986287417
	.long	3212836864
	.long	1037741368
	.long	1055273845
	.long	2983839604
	.long	3212836864
	.long	1036296515
	.long	1054533760
	.long	840832460
	.long	3212836864
	.long	1034924748
	.long	1053785034
	.long	829045603
	.long	3212836864
	.long	1033626892
	.long	1053028117
	.long	836097324
	.long	3212836864
	.long	1032403730
	.long	1052263466
	.long	2993707942
	.long	3212836864
	.long	1030713214
	.long	1051491540
	.long	2988789250
	.long	3212836864
	.long	1028569994
	.long	1050712805
	.long	2990442912
	.long	3212836864
	.long	1026580309
	.long	1049927729
	.long	846027248
	.long	3212836864
	.long	1024745356
	.long	1049136787
	.long	824999326
	.long	3212836864
	.long	1022722308
	.long	1048104908
	.long	2971391005
	.long	3212836864
	.long	1019677780
	.long	1046502419
	.long	833086710
	.long	3212836864
	.long	1016948784
	.long	1044891074
	.long	2967836285
	.long	3212836864
	.long	1014052363
	.long	1043271842
	.long	823224313
	.long	3212836864
	.long	1009865986
	.long	1041645699
	.long	837346836
	.long	3212836864
	.long	1006005820
	.long	1039839859
	.long	2970970319
	.long	3212836864
	.long	1000196465
	.long	1036565814
	.long	2960495349
	.long	3212836864
	.long	993104536
	.long	1033283845
	.long	2975014497
	.long	3212836864
	.long	983425480
	.long	1028193072
	.long	2968461951
	.long	3212836864
	.long	966649823
	.long	1019808432
	.long	2953169304
	.long	3212836864
	.long	0
	.long	0
	.long	0
	.long	3212836864
	.long	966649823
	.long	3167292080
	.long	805685656
	.long	3212836864
	.long	983425480
	.long	3175676720
	.long	820978303
	.long	3212836864
	.long	993104536
	.long	3180767493
	.long	827530849
	.long	3212836864
	.long	1000196465
	.long	3184049462
	.long	813011701
	.long	3212836864
	.long	1006005820
	.long	3187323507
	.long	823486671
	.long	3212836864
	.long	1009865986
	.long	3189129347
	.long	2984830484
	.long	3212836864
	.long	1014052363
	.long	3190755490
	.long	2970707961
	.long	3212836864
	.long	1016948784
	.long	3192374722
	.long	820352637
	.long	3212836864
	.long	1019677780
	.long	3193986067
	.long	2980570358
	.long	3212836864
	.long	1022722308
	.long	3195588556
	.long	823907357
	.long	3212836864
	.long	1024745356
	.long	3196620435
	.long	2972482974
	.long	3212836864
	.long	1026580309
	.long	3197411377
	.long	2993510896
	.long	3212836864
	.long	1028569994
	.long	3198196453
	.long	842959264
	.long	3212836864
	.long	1030713214
	.long	3198975188
	.long	841305602
	.long	3212836864
	.long	1032403730
	.long	3199747114
	.long	846224294
	.long	3212836864
	.long	1033626892
	.long	3200511765
	.long	2983580972
	.long	3212836864
	.long	1034924748
	.long	3201268682
	.long	2976529251
	.long	3212836864
	.long	1036296515
	.long	3202017408
	.long	2988316108
	.long	3212836864
	.long	1037741368
	.long	3202757493
	.long	836355956
	.long	3212836864
	.long	1039258436
	.long	3203488490
	.long	838803769
	.long	3212836864
	.long	1040517098
	.long	3204209959
	.long	830532777
	.long	3212836864
	.long	1041346455
	.long	3204684861
	.long	844865538
	.long	3212836864
	.long	1042210485
	.long	3205035419
	.long	851331918
	.long	3212836864
	.long	1043108667
	.long	3205380570
	.long	843723495
	.long	3212836864
	.long	1044040460
	.long	3205720106
	.long	2999832878
	.long	3212836864
	.long	1045005303
	.long	3206053824
	.long	835167219
	.long	3212836864
	.long	1046002615
	.long	3206381521
	.long	2995913996
	.long	3212836864
	.long	1047031795
	.long	3206703001
	.long	2988516283
	.long	3212836864
	.long	1048092223
	.long	3207018070
	.long	839091011
	.long	3212836864
	.long	1048879630
	.long	3207326538
	.long	850866486
	.long	3212836864
	.long	1049440125
	.long	3207628219
	.long	850275634
	.long	3212836864
	.long	1050015258
	.long	3207922931
	.long	2991581050
	.long	3212836864
	.long	3192002283
	.long	3208210498
	.long	847314951
	.long	3204448256
	.long	3190795559
	.long	3208490745
	.long	2979703788
	.long	3204448256
	.long	3189561687
	.long	3208763504
	.long	2998925687
	.long	3204448256
	.long	3188301413
	.long	3209028611
	.long	2997965172
	.long	3204448256
	.long	3186359946
	.long	3209285906
	.long	2996381248
	.long	3204448256
	.long	3183738367
	.long	3209535234
	.long	2994630888
	.long	3204448256
	.long	3181068627
	.long	3209776445
	.long	2953596676
	.long	3204448256
	.long	3177422237
	.long	3210009393
	.long	2995841562
	.long	3204448256
	.long	3171899818
	.long	3210233939
	.long	847077312
	.long	3204448256
	.long	3160870706
	.long	3210449946
	.long	2988649928
	.long	3204448256
	.long	1005106760
	.long	3210657285
	.long	2999384403
	.long	3204448256
	.long	1021989220
	.long	3210855832
	.long	854062117
	.long	3204448256
	.long	1028547674
	.long	3211045465
	.long	2971273466
	.long	3204448256
	.long	1033133567
	.long	3211226072
	.long	851194761
	.long	3204448256
	.long	1036128472
	.long	3211397543
	.long	854270828
	.long	3204448256
	.long	1039156139
	.long	3211559774
	.long	3002280148
	.long	3204448256
	.long	1041201069
	.long	3211712670
	.long	848507868
	.long	3204448256
	.long	3182555983
	.long	3211856136
	.long	2988363997
	.long	3196059648
	.long	3179441043
	.long	3211990087
	.long	2999225873
	.long	3196059648
	.long	3173319052
	.long	3212114443
	.long	848534818
	.long	3196059648
	.long	3163089201
	.long	3212229127
	.long	2993490220
	.long	3196059648
	.long	1004930693
	.long	3212334072
	.long	839760357
	.long	3196059648
	.long	1023221605
	.long	3212429213
	.long	2999340633
	.long	3196059648
	.long	1029761272
	.long	3212514494
	.long	3003086283
	.long	3196059648
	.long	3174843017
	.long	3212589864
	.long	842126987
	.long	3187671040
	.long	3165783068
	.long	3212655276
	.long	821517033
	.long	3187671040
	.long	992588201
	.long	3212710692
	.long	854713859
	.long	3187671040
	.long	1021119272
	.long	3212756077
	.long	2985576777
	.long	3187671040
	.long	3157608485
	.long	3212791405
	.long	2999982212
	.long	3179282432
	.long	1012667202
	.long	3212816655
	.long	2984139615
	.long	3179282432
	.long	1004262721
	.long	3212831811
	.long	2961493261
	.long	3170893824
	.long	0
	.long	3212836864
	.long	0
	.long	0
	.long	3151746369
	.long	3212831811
	.long	2961493261
	.long	1023410176
	.long	3160150850
	.long	3212816655
	.long	2984139615
	.long	1031798784
	.long	1010124837
	.long	3212791405
	.long	2999982212
	.long	1031798784
	.long	3168602920
	.long	3212756077
	.long	2985576777
	.long	1040187392
	.long	3140071849
	.long	3212710692
	.long	854713859
	.long	1040187392
	.long	1018299420
	.long	3212655276
	.long	821517033
	.long	1040187392
	.long	1027359369
	.long	3212589864
	.long	842126987
	.long	1040187392
	.long	3177244920
	.long	3212514494
	.long	3003086283
	.long	1048576000
	.long	3170705253
	.long	3212429213
	.long	2999340633
	.long	1048576000
	.long	3152414341
	.long	3212334072
	.long	839760357
	.long	1048576000
	.long	1015605553
	.long	3212229127
	.long	2993490220
	.long	1048576000
	.long	1025835404
	.long	3212114443
	.long	848534818
	.long	1048576000
	.long	1031957395
	.long	3211990087
	.long	2999225873
	.long	1048576000
	.long	1035072335
	.long	3211856136
	.long	2988363997
	.long	1048576000
	.long	3188684717
	.long	3211712670
	.long	848507868
	.long	1056964608
	.long	3186639787
	.long	3211559774
	.long	3002280148
	.long	1056964608
	.long	3183612120
	.long	3211397543
	.long	854270828
	.long	1056964608
	.long	3180617215
	.long	3211226072
	.long	851194761
	.long	1056964608
	.long	3176031322
	.long	3211045465
	.long	2971273466
	.long	1056964608
	.long	3169472868
	.long	3210855832
	.long	854062117
	.long	1056964608
	.long	3152590408
	.long	3210657285
	.long	2999384403
	.long	1056964608
	.long	1013387058
	.long	3210449946
	.long	2988649928
	.long	1056964608
	.long	1024416170
	.long	3210233939
	.long	847077312
	.long	1056964608
	.long	1029938589
	.long	3210009393
	.long	2995841562
	.long	1056964608
	.long	1033584979
	.long	3209776445
	.long	2953596676
	.long	1056964608
	.long	1036254719
	.long	3209535234
	.long	2994630888
	.long	1056964608
	.long	1038876298
	.long	3209285906
	.long	2996381248
	.long	1056964608
	.long	1040817765
	.long	3209028611
	.long	2997965172
	.long	1056964608
	.long	1042078039
	.long	3208763504
	.long	2998925687
	.long	1056964608
	.long	1043311911
	.long	3208490745
	.long	2979703788
	.long	1056964608
	.long	1044518635
	.long	3208210498
	.long	847314951
	.long	1056964608
	.long	3197498906
	.long	3207922931
	.long	2991581050
	.long	1065353216
	.long	3196923773
	.long	3207628219
	.long	850275634
	.long	1065353216
	.long	3196363278
	.long	3207326538
	.long	850866486
	.long	1065353216
	.long	3195575871
	.long	3207018070
	.long	839091011
	.long	1065353216
	.long	3194515443
	.long	3206703001
	.long	2988516283
	.long	1065353216
	.long	3193486263
	.long	3206381521
	.long	2995913996
	.long	1065353216
	.long	3192488951
	.long	3206053824
	.long	835167219
	.long	1065353216
	.long	3191524108
	.long	3205720106
	.long	2999832878
	.long	1065353216
	.long	3190592315
	.long	3205380570
	.long	843723495
	.long	1065353216
	.long	3189694133
	.long	3205035419
	.long	851331918
	.long	1065353216
	.long	3188830103
	.long	3204684861
	.long	844865538
	.long	1065353216
	.long	3188000746
	.long	3204209959
	.long	830532777
	.long	1065353216
	.long	3186742084
	.long	3203488490
	.long	838803769
	.long	1065353216
	.long	3185225016
	.long	3202757493
	.long	836355956
	.long	1065353216
	.long	3183780163
	.long	3202017408
	.long	2988316108
	.long	1065353216
	.long	3182408396
	.long	3201268682
	.long	2976529251
	.long	1065353216
	.long	3181110540
	.long	3200511765
	.long	2983580972
	.long	1065353216
	.long	3179887378
	.long	3199747114
	.long	846224294
	.long	1065353216
	.long	3178196862
	.long	3198975188
	.long	841305602
	.long	1065353216
	.long	3176053642
	.long	3198196453
	.long	842959264
	.long	1065353216
	.long	3174063957
	.long	3197411377
	.long	2993510896
	.long	1065353216
	.long	3172229004
	.long	3196620435
	.long	2972482974
	.long	1065353216
	.long	3170205956
	.long	3195588556
	.long	823907357
	.long	1065353216
	.long	3167161428
	.long	3193986067
	.long	2980570358
	.long	1065353216
	.long	3164432432
	.long	3192374722
	.long	820352637
	.long	1065353216
	.long	3161536011
	.long	3190755490
	.long	2970707961
	.long	1065353216
	.long	3157349634
	.long	3189129347
	.long	2984830484
	.long	1065353216
	.long	3153489468
	.long	3187323507
	.long	823486671
	.long	1065353216
	.long	3147680113
	.long	3184049462
	.long	813011701
	.long	1065353216
	.long	3140588184
	.long	3180767493
	.long	827530849
	.long	1065353216
	.long	3130909128
	.long	3175676720
	.long	820978303
	.long	1065353216
	.long	3114133471
	.long	3167292080
	.long	805685656
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
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
	.long	1176256512
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
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	3190467243
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
	.long	1007192156
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
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1026206332
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	1078525952
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	981311488
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	874651648
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	750018842
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	1078530011
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	3015425326
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	2809605357
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	3190467238
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	1007191910
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	3109009407
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	909041400
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
	.long	1050868099
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
	.type	__svml_ssin_data_internal,@object
	.size	__svml_ssin_data_internal,5376
	.align 64
__svml_ssin_reduction_data_internal:
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
	.type	__svml_ssin_reduction_data_internal,@object
	.size	__svml_ssin_reduction_data_internal,3072
	.align 4
__ssin_la__vmlsSinHATab:
	.long	0
	.long	2139095040
	.type	__ssin_la__vmlsSinHATab,@object
	.size	__ssin_la__vmlsSinHATab,8
