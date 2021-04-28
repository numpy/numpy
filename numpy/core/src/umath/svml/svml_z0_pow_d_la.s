/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *       1) Calculating log2|x|
 * 
 *           Here we use the following formula.
 *           Let |x|=2^k1*X1, where k1 is integer, 1<=X1<2.
 *           Let C ~= 1/ln(2),
 *           Rcp1 ~= 1/X1,   X2=Rcp1*X1,
 *           Rcp2 ~= 1/X2,   X3=Rcp2*X2,
 *           Rcp3 ~= 1/X3,   Rcp3C ~= C/X3.
 *           Then
 * 
 *               log2|x| = k1 + log2(1/Rcp1) + log2(1/Rcp2) + log2(C/Rcp3C) +
 *                       + log2(X1*Rcp1*Rcp2*Rcp3C/C),
 * 
 *           where X1*Rcp1*Rcp2*Rcp3C = C*(1+q), q is very small.
 * 
 *           The values of Rcp1, log2(1/Rcp1), Rcp2, log2(1/Rcp2),
 *           Rcp3C, log2(C/Rcp3C) are taken from tables.
 *           Values of Rcp1, Rcp2, Rcp3C are such that RcpC=Rcp1*Rcp2*Rcp3C
 *           is exactly represented in target precision.
 * 
 *           log2(X1*Rcp1*Rcp2*Rcp3C/C) = log2(1+q) = ln(1+q)/ln2 =
 *               = 1/(ln2)*q - 1/(2ln2)*q^2 + 1/(3ln2)*q^3 - ... =
 *               = 1/(C*ln2)*cq - 1/(2*C^2*ln2)*cq^2 + 1/(3*C^3*ln2)*cq^3 - ... =
 *               = (1 + a1)*cq + a2*cq^2 + a3*cq^3 + ...,
 *           where
 *               cq=X1*Rcp1*Rcp2*Rcp3C-C,
 *               a1=1/(C*ln(2))-1 is small,
 *               a2=1/(2*C^2*ln2),
 *               a3=1/(3*C^3*ln2),
 *               ...
 *               We get 3 parts of log2 result: HH+HL+HLL ~= log2|x|.
 * 
 *           2)  Calculation of y*(HH+HL+HLL).
 *               Split y into YHi+YLo.
 *               Get high PH and medium PL parts of y*log2|x|.
 *               Get low PLL part of y*log2|x|.
 *               Now we have PH+PL+PLL ~= y*log2|x|.
 * 
 *           3) Calculation of 2^(PH+PL+PLL).
 * 
 *               Mathematical idea of computing 2^(PH+PL+PLL) is the following.
 *               Let's represent PH+PL+PLL in the form N + j/2^expK + Z,
 *               where expK=7 in this implementation, N and j are integers,
 *               0<=j<=2^expK-1, |Z|<2^(-expK-1). Hence
 * 
 *                   2^(PH+PL+PLL) ~= 2^N * 2^(j/2^expK) * 2^Z,
 * 
 *               where 2^(j/2^expK) is stored in a table, and
 * 
 *                   2^Z ~= 1 + B1*Z + B2*Z^2 ... + B5*Z^5.
 * 
 *               We compute 2^(PH+PL+PLL) as follows.
 * 
 *               Break PH into PHH + PHL, where PHH = N + j/2^expK.
 *               Z = PHL + PL + PLL
 *               Exp2Poly = B1*Z + B2*Z^2 ... + B5*Z^5
 *               Get 2^(j/2^expK) from table in the form THI+TLO.
 *               Now we have 2^(PH+PL+PLL) ~= 2^N * (THI + TLO) * (1 + Exp2Poly).
 * 
 *               Get significand of 2^(PH+PL+PLL) in the form ResHi+ResLo:
 *               ResHi := THI
 *               ResLo := THI * Exp2Poly + TLO
 * 
 *               Get exponent ERes of the result:
 *               Res := ResHi + ResLo:
 *               Result := ex(Res) + N
 * 
 * 
 */


	.text
.L_2__routine_start___svml_pow8_z0_0:

	.align    16,0x90
	.globl __svml_pow8

__svml_pow8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $256, %rsp
        vgetmantpd $10, {sae}, %zmm0, %zmm10

/* Reduced argument: R = (DblRcp*Mantissa - 1) */
        vmovups   640+__svml_dpow_data_internal_avx512(%rip), %zmm15

/* Table lookup */
        vmovups   __svml_dpow_data_internal_avx512(%rip), %zmm2
        vmovups   128+__svml_dpow_data_internal_avx512(%rip), %zmm3
        vmovups   256+__svml_dpow_data_internal_avx512(%rip), %zmm5
        vmovups   384+__svml_dpow_data_internal_avx512(%rip), %zmm4

/* Prepare exponent correction: DblRcp<1.5?  -- was 0.75 in initial AVX3 version, which used Mantissa in [1,2) */
        vmovups   704+__svml_dpow_data_internal_avx512(%rip), %zmm14

/* GetExp(x) */
        vgetexppd {sae}, %zmm0, %zmm9

/* P8_9 */
        vmovups   960+__svml_dpow_data_internal_avx512(%rip), %zmm7

/* DblRcp ~ 1/Mantissa */
        vrcp14pd  %zmm10, %zmm12

/* x<=0 or Inf/NaN? */
        vfpclasspd $223, %zmm0, %k0

/* round DblRcp to 5 fractional bits (RN mode, no Precision exception) */
        vrndscalepd $88, {sae}, %zmm12, %zmm13
        vmovups   896+__svml_dpow_data_internal_avx512(%rip), %zmm12
        kmovw     %k0, %edx
        vfmsub213pd {rn-sae}, %zmm15, %zmm13, %zmm10
        vcmppd    $17, {sae}, %zmm14, %zmm13, %k1
        vfmadd231pd {rn-sae}, %zmm10, %zmm12, %zmm7

/* Prepare table index */
        vpsrlq    $47, %zmm13, %zmm8
        vmovups   1024+__svml_dpow_data_internal_avx512(%rip), %zmm13
        vpermt2pd 64+__svml_dpow_data_internal_avx512(%rip), %zmm8, %zmm2
        vpermt2pd 192+__svml_dpow_data_internal_avx512(%rip), %zmm8, %zmm3
        vpermt2pd 320+__svml_dpow_data_internal_avx512(%rip), %zmm8, %zmm5
        vpermt2pd 448+__svml_dpow_data_internal_avx512(%rip), %zmm8, %zmm4

/* add 1 to Expon if DblRcp<1.5 */
        vaddpd    {rn-sae}, %zmm15, %zmm9, %zmm9{%k1}
        vmovaps   %zmm1, %zmm11
        vpsllq    $59, %zmm8, %zmm1

/* R^2 */
        vmulpd    {rn-sae}, %zmm10, %zmm10, %zmm8
        vpmovq2m  %zmm1, %k2

/* y Inf/NaN? */
        vfpclasspd $153, %zmm11, %k3

/* P6_7 */
        vmovups   1088+__svml_dpow_data_internal_avx512(%rip), %zmm1
        vblendmpd %zmm3, %zmm2, %zmm14{%k2}
        vblendmpd %zmm4, %zmm5, %zmm6{%k2}

/* P4_5 */
        vmovups   1216+__svml_dpow_data_internal_avx512(%rip), %zmm2
        vfmadd231pd {rn-sae}, %zmm10, %zmm13, %zmm1
        vmovups   832+__svml_dpow_data_internal_avx512(%rip), %zmm4
        vmovups   768+__svml_dpow_data_internal_avx512(%rip), %zmm3

/* Expon + Th */
        vaddpd    {rn-sae}, %zmm9, %zmm14, %zmm15
        vmovups   1152+__svml_dpow_data_internal_avx512(%rip), %zmm9

/* P6_9 */
        vfmadd213pd {rn-sae}, %zmm1, %zmm8, %zmm7

/* P2_3 */
        vmovups   1344+__svml_dpow_data_internal_avx512(%rip), %zmm1

/* P1_2 */
        vmovups   1920+__svml_dpow_data_internal_avx512(%rip), %zmm14
        vfmadd231pd {rn-sae}, %zmm10, %zmm9, %zmm2
        kmovw     %k3, %eax

/* P4_9 */
        vfmadd213pd {rn-sae}, %zmm2, %zmm8, %zmm7

/* R2l */
        vmovaps   %zmm10, %zmm5
        orl       %eax, %edx
        vfmsub213pd {rn-sae}, %zmm8, %zmm10, %zmm5

/* Tl + R2l*c2h */
        vfmadd213pd {rn-sae}, %zmm6, %zmm4, %zmm5
        vmovups   1280+__svml_dpow_data_internal_avx512(%rip), %zmm6
        vfmadd231pd {rn-sae}, %zmm10, %zmm6, %zmm1

/* Expon + Th+ R*c1h */
        vmovaps   %zmm15, %zmm12
        vfmadd231pd {rn-sae}, %zmm10, %zmm3, %zmm12

/* P2_9 */
        vfmadd213pd {rn-sae}, %zmm1, %zmm8, %zmm7

/* (R*c1h)_h */
        vsubpd    {rn-sae}, %zmm15, %zmm12, %zmm9

/* Tl + R2l*c2h + R2*P2_9 */
        vfmadd231pd {rn-sae}, %zmm8, %zmm7, %zmm5
        vmovups   1408+__svml_dpow_data_internal_avx512(%rip), %zmm7

/* (R*c1h)_l */
        vfmsub231pd {rn-sae}, %zmm3, %zmm10, %zmm9

/* Expon + Th+ R*c1h + R2*c2h */
        vmovaps   %zmm12, %zmm13
        vfmadd231pd {rn-sae}, %zmm8, %zmm4, %zmm13

/* R*c1l + (R*c1h)_l */
        vfmadd213pd {rn-sae}, %zmm9, %zmm7, %zmm10
        vmovups   1728+__svml_dpow_data_internal_avx512(%rip), %zmm9

/* High2 + Tlh */
        vaddpd    {rn-sae}, %zmm5, %zmm13, %zmm6

/* (R2*c2h)_h */
        vsubpd    {rn-sae}, %zmm12, %zmm13, %zmm2

/* P3_4 */
        vmovups   1792+__svml_dpow_data_internal_avx512(%rip), %zmm12

/* y*High */
        vmulpd    {rz-sae}, %zmm11, %zmm6, %zmm3

/* (R2*c2h)_l */
        vfmsub213pd {rn-sae}, %zmm2, %zmm4, %zmm8
        vsubpd    {rn-sae}, %zmm13, %zmm6, %zmm1

/* (y*High)_low */
        vfmsub213pd {rz-sae}, %zmm3, %zmm11, %zmm6

/* Tll */
        vsubpd    {rn-sae}, %zmm1, %zmm5, %zmm4

/* R*c1l + (R*c1h)_l+(R2*c2h)_l */
        vaddpd    {rn-sae}, %zmm8, %zmm10, %zmm10
        vmovups   1472+__svml_dpow_data_internal_avx512(%rip), %zmm1
        vmovups   1600+__svml_dpow_data_internal_avx512(%rip), %zmm8

/* Tll + R*c1l + (R*c1h)_l */
        vaddpd    {rn-sae}, %zmm10, %zmm4, %zmm5
        vaddpd    {rd-sae}, %zmm1, %zmm3, %zmm2

/*
 * /
 * exp2 computation starts here
 */
        vreducepd $65, {sae}, %zmm3, %zmm4

/* Zl = y*Tll + Zl */
        vfmadd213pd {rz-sae}, %zmm6, %zmm11, %zmm5

/* P5_6 */
        vmovups   1664+__svml_dpow_data_internal_avx512(%rip), %zmm1
        vmovups   1856+__svml_dpow_data_internal_avx512(%rip), %zmm10
        vaddpd    {rn-sae}, %zmm5, %zmm4, %zmm7
        vandpd    2176+__svml_dpow_data_internal_avx512(%rip), %zmm3, %zmm3

/* Table lookup: The, Tle/The */
        vmovups   512+__svml_dpow_data_internal_avx512(%rip), %zmm4

/*
 * scaled result
 * Filter very large |y*log2(x)| and scale final result for LRB2
 */
        vmovups   2240+__svml_dpow_data_internal_avx512(%rip), %zmm5

/* ensure |R|<2 even for special cases */
        vandpd    1536+__svml_dpow_data_internal_avx512(%rip), %zmm7, %zmm15
        vpermt2pd 576+__svml_dpow_data_internal_avx512(%rip), %zmm2, %zmm4
        vcmppd    $22, {sae}, %zmm5, %zmm3, %k0

/* Re^2 */
        vmulpd    {rn-sae}, %zmm15, %zmm15, %zmm13

/* R*The */
        vmulpd    {rn-sae}, %zmm4, %zmm15, %zmm7
        vfmadd231pd {rn-sae}, %zmm15, %zmm8, %zmm1
        vfmadd231pd {rn-sae}, %zmm15, %zmm9, %zmm12
        vfmadd231pd {rn-sae}, %zmm15, %zmm10, %zmm14
        vpsllq    $48, %zmm2, %zmm2
        vfmadd213pd {rn-sae}, %zmm12, %zmm13, %zmm1
        vandpd    2304+__svml_dpow_data_internal_avx512(%rip), %zmm2, %zmm2
        kmovw     %k0, %ecx
        vfmadd213pd {rn-sae}, %zmm14, %zmm13, %zmm1

/* The + The*R*poly */
        vfmadd213pd {rn-sae}, %zmm4, %zmm7, %zmm1
        orl       %ecx, %edx
        vmulpd    {rn-sae}, %zmm2, %zmm1, %zmm1
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
        vmovups   %zmm11, 128(%rsp)
        vmovups   %zmm1, 192(%rsp)
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
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x28, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x20, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x38, 0xff, 0xff, 0xff, 0x22
        movl      %eax, %r12d
        movq      %r13, 48(%rsp)
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x30, 0xff, 0xff, 0xff, 0x22
        movl      %edx, %r13d
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x18, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22

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
        vmovups   192(%rsp), %zmm1
        movq      40(%rsp), %rsi
	.cfi_restore 4
        movq      32(%rsp), %rdi
	.cfi_restore 5
        movq      56(%rsp), %r12
	.cfi_restore 12
        movq      48(%rsp), %r13
	.cfi_restore 13
        jmp       .LBL_1_2
	.cfi_escape 0x10, 0x04, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x28, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x05, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x20, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0c, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x38, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0x0d, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x30, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfa, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x18, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfb, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x10, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfc, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x08, 0xff, 0xff, 0xff, 0x22
	.cfi_escape 0x10, 0xfd, 0x00, 0x0e, 0x38, 0x1c, 0x0d, 0xc0, 0xff, 0xff, 0xff, 0x1a, 0x0d, 0x00, 0xff, 0xff, 0xff, 0x22

.LBL_1_10:

        lea       64(%rsp,%r12,8), %rdi
        lea       128(%rsp,%r12,8), %rsi
        lea       192(%rsp,%r12,8), %rdx

        call      __svml_dpow_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_pow8,@function
	.size	__svml_pow8,.-__svml_pow8
..LN__svml_pow8.0:

.L_2__routine_start___svml_dpow_cout_rare_internal_1:

	.align    16,0x90

__svml_dpow_cout_rare_internal:


	.cfi_startproc
..L53:

        pushq     %r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r15
	.cfi_def_cfa_offset 32
	.cfi_offset 15, -32
        pushq     %rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
        movq      %rdx, %r8
        movsd     6824+__dpow_la_CoutTab(%rip), %xmm1
        xorl      %eax, %eax
        movsd     (%rdi), %xmm3
        movsd     (%rsi), %xmm0
        mulsd     %xmm1, %xmm3
        mulsd     %xmm1, %xmm0
        movsd     %xmm3, -72(%rsp)
        movsd     %xmm0, -16(%rsp)
        movzwl    -66(%rsp), %r13d
        movzwl    -10(%rsp), %edx
        andl      $32752, %r13d
        movzbl    -65(%rsp), %r12d
        andl      $32752, %edx
        movb      -9(%rsp), %bl
        andl      $128, %r12d
        andb      $-128, %bl
        shrl      $4, %r13d
        shrl      $4, %edx
        shrl      $7, %r12d
        shrb      $7, %bl
        testl     $1048575, -68(%rsp)
        jne       .LBL_2_4


        cmpl      $0, -72(%rsp)
        jne       .LBL_2_4


        movb      $1, %r11b
        jmp       .LBL_2_5

.LBL_2_4:

        xorb      %r11b, %r11b

.LBL_2_5:

        movl      -12(%rsp), %ecx
        movl      -16(%rsp), %edi
        andl      $1048575, %ecx
        jne       .LBL_2_8


        testl     %edi, %edi
        jne       .LBL_2_8


        movl      $1, %r10d
        jmp       .LBL_2_9

.LBL_2_8:

        xorl      %r10d, %r10d

.LBL_2_9:

        movl      %edx, %r9d
        xorl      %esi, %esi
        cmpl      $2047, %edx
        setne     %sil
        shll      $20, %r9d
        orl       %ecx, %r9d
        movl      %edi, %ecx
        orl       %r9d, %ecx
        je        .LBL_2_24


        lea       -1023(%rdx), %ecx
        cmpl      $1023, %edx
        jl        .LBL_2_23


        testl     %esi, %esi
        je        .LBL_2_23


        cmpl      $20, %ecx
        jg        .LBL_2_16


        movl      %r9d, %r15d
        shll      %cl, %r15d
        shll      $12, %r15d
        orl       %edi, %r15d
        je        .LBL_2_15

.LBL_2_14:

        xorl      %r9d, %r9d
        jmp       .LBL_2_21

.LBL_2_15:

        lea       -1012(%rdx), %ecx
        shll      %cl, %r9d
        notl      %r9d
        shrl      $31, %r9d
        incl      %r9d
        jmp       .LBL_2_21

.LBL_2_16:

        cmpl      $53, %ecx
        jge       .LBL_2_20


        lea       -1012(%rdx), %ecx
        shll      %cl, %edi
        testl     $2147483647, %edi
        jne       .LBL_2_14


        notl      %edi
        shrl      $31, %edi
        lea       1(%rdi), %r9d
        jmp       .LBL_2_21

.LBL_2_20:

        movl      $2, %r9d

.LBL_2_21:

        testl     %r12d, %r12d
        jne       .LBL_2_29


        cmpl      $1023, %r13d
        jne       .LBL_2_29
        jmp       .LBL_2_57

.LBL_2_23:

        xorl      %r9d, %r9d
        jmp       .LBL_2_25

.LBL_2_24:

        movl      $2, %r9d

.LBL_2_25:

        testl     %r12d, %r12d
        jne       .LBL_2_27


        cmpl      $1023, %r13d
        je        .LBL_2_74

.LBL_2_27:

        testl     %edx, %edx
        jne       .LBL_2_29


        testl     %r10d, %r10d
        jne       .LBL_2_55

.LBL_2_29:

        cmpl      $2047, %r13d
        je        .LBL_2_31

.LBL_2_30:

        movb      $1, %cl
        jmp       .LBL_2_32

.LBL_2_31:

        xorb      %cl, %cl

.LBL_2_32:

        orb       %cl, %r11b
        je        .LBL_2_54


        orl       %esi, %r10d
        je        .LBL_2_54


        movsd     6816+__dpow_la_CoutTab(%rip), %xmm2
        ucomisd   %xmm2, %xmm3
        jp        .LBL_2_35
        je        .LBL_2_71

.LBL_2_35:

        ucomisd   6832+__dpow_la_CoutTab(%rip), %xmm3
        jp        .LBL_2_36
        je        .LBL_2_68

.LBL_2_36:

        testb     %cl, %cl
        je        .LBL_2_47


        testl     %esi, %esi
        je        .LBL_2_47


        comisd    %xmm2, %xmm3
        ja        .LBL_2_40


        testl     %r9d, %r9d
        je        .LBL_2_46

.LBL_2_40:

        lea       6824+__dpow_la_CoutTab(%rip), %rcx
        andl      %r12d, %r9d
        movsd     %xmm3, -72(%rsp)
        andb      $127, -65(%rsp)
        movsd     (%rcx,%r9,8), %xmm0
        xorl      %ecx, %ecx
        testl     %r13d, %r13d
        jne       .LBL_2_42


        movsd     -72(%rsp), %xmm2
        movl      $-200, %ecx
        mulsd     6864+__dpow_la_CoutTab(%rip), %xmm2
        movsd     %xmm2, -72(%rsp)
        jmp       .LBL_2_43

.LBL_2_42:

        movsd     -72(%rsp), %xmm2

.LBL_2_43:

        movzwl    -66(%rsp), %esi
        pxor      %xmm7, %xmm7
        andl      $32752, %esi
        shrl      $4, %esi
        movl      -68(%rsp), %r9d
        shll      $20, %esi
        andl      $1048575, %r9d
        movsd     %xmm2, -56(%rsp)
        orl       %r9d, %esi
        movzwl    -50(%rsp), %edi
        addl      $-1072152576, %esi
        andl      $-32753, %edi
        addl      $16368, %edi
        movw      %di, -50(%rsp)
        sarl      $20, %esi
        movl      -52(%rsp), %r10d
        addl      %ecx, %esi
        lea       __dpow_la_CoutTab(%rip), %rcx
        andl      $1032192, %r10d
        addl      $16384, %r10d
        shrl      $15, %r10d
        movsd     -56(%rsp), %xmm2
        movsd     (%rcx,%r10,8), %xmm5
        addl      %r10d, %r10d
        movaps    %xmm5, %xmm6
        movsd     6856+__dpow_la_CoutTab(%rip), %xmm14
        mulsd     %xmm2, %xmm6
        cvtsi2sd  %esi, %xmm7
        mulsd     %xmm2, %xmm14
        addsd     264(%rcx,%r10,8), %xmm7
        movsd     %xmm6, -48(%rsp)
        movsd     %xmm14, -32(%rsp)
        movl      -44(%rsp), %r11d
        andl      $64512, %r11d
        movsd     -32(%rsp), %xmm15
        addl      $1024, %r11d
        shrl      $11, %r11d
        subsd     -56(%rsp), %xmm15
        movsd     792(%rcx,%r11,8), %xmm12
        addl      %r11d, %r11d
        mulsd     %xmm12, %xmm6
        addsd     1056(%rcx,%r11,8), %xmm7
        mulsd     %xmm12, %xmm5
        movsd     %xmm15, -24(%rsp)
        movsd     -32(%rsp), %xmm4
        movsd     -24(%rsp), %xmm3
        movsd     %xmm6, -40(%rsp)
        subsd     %xmm3, %xmm4
        movl      -36(%rsp), %r12d
        andl      $4080, %r12d
        addl      $16, %r12d
        movsd     %xmm4, -32(%rsp)
        shrl      $5, %r12d
        movsd     -32(%rsp), %xmm12
        movsd     1584(%rcx,%r12,8), %xmm13
        addl      %r12d, %r12d
        mulsd     %xmm13, %xmm5
        subsd     %xmm12, %xmm2
        addsd     2616(%rcx,%r12,8), %xmm7
        mulsd     %xmm13, %xmm6
        movsd     %xmm2, -24(%rsp)
        movaps    %xmm6, %xmm8
        movsd     6856+__dpow_la_CoutTab(%rip), %xmm2
        mulsd     %xmm5, %xmm2
        subsd     6848+__dpow_la_CoutTab(%rip), %xmm8
        movsd     -32(%rsp), %xmm3
        movsd     -24(%rsp), %xmm4
        movsd     %xmm2, -32(%rsp)
        movsd     -32(%rsp), %xmm13
        movsd     272(%rcx,%r10,8), %xmm11
        subsd     %xmm5, %xmm13
        movsd     %xmm13, -24(%rsp)
        movsd     -32(%rsp), %xmm2
        movsd     -24(%rsp), %xmm14
        movsd     1064(%rcx,%r11,8), %xmm10
        subsd     %xmm14, %xmm2
        movsd     %xmm2, -32(%rsp)
        movaps    %xmm3, %xmm2
        movsd     -32(%rsp), %xmm15
        movsd     2624(%rcx,%r12,8), %xmm9
        subsd     %xmm15, %xmm5
        movsd     %xmm5, -24(%rsp)
        movsd     -32(%rsp), %xmm5
        mulsd     %xmm5, %xmm2
        mulsd     %xmm4, %xmm5
        subsd     %xmm6, %xmm2
        movaps    %xmm7, %xmm6
        addsd     %xmm5, %xmm2
        addsd     %xmm8, %xmm6
        movsd     -24(%rsp), %xmm12
        mulsd     %xmm12, %xmm3
        mulsd     %xmm12, %xmm4
        addsd     %xmm3, %xmm2
        movsd     %xmm6, -32(%rsp)
        addsd     %xmm4, %xmm2
        movsd     -32(%rsp), %xmm3
        subsd     %xmm3, %xmm7
        addsd     %xmm8, %xmm7
        movsd     %xmm7, -24(%rsp)
        movsd     -32(%rsp), %xmm4
        movsd     %xmm4, -64(%rsp)
        movzwl    -58(%rsp), %ecx
        andl      $32752, %ecx
        shrl      $4, %ecx
        addl      %edx, %ecx
        movsd     -24(%rsp), %xmm3
        cmpl      $2057, %ecx
        jge       .LBL_2_67


        cmpl      $1984, %ecx
        jg        .LBL_2_58


        movsd     %xmm1, -32(%rsp)
        movsd     -32(%rsp), %xmm1
        addsd     6808+__dpow_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -32(%rsp)
        movsd     -32(%rsp), %xmm2
        mulsd     %xmm0, %xmm2
        jmp       .LBL_2_56

.LBL_2_46:

        movsd     %xmm2, -32(%rsp)
        movl      $1, %eax
        movsd     -32(%rsp), %xmm1
        movsd     -32(%rsp), %xmm0
        divsd     %xmm0, %xmm1
        movsd     %xmm1, -32(%rsp)
        movsd     -32(%rsp), %xmm2
        jmp       .LBL_2_56

.LBL_2_47:

        cmpl      $1023, %r13d
        jge       .LBL_2_51


        testb     %bl, %bl
        je        .LBL_2_56


        movaps    %xmm0, %xmm2
        mulsd     %xmm0, %xmm2
        jmp       .LBL_2_56

.LBL_2_51:

        testb     %bl, %bl
        je        .LBL_2_53


        lea       6824+__dpow_la_CoutTab(%rip), %rdx
        andl      %r12d, %r9d
        mulsd     (%rdx,%r9,8), %xmm2
        jmp       .LBL_2_56

.LBL_2_53:

        mulsd     %xmm3, %xmm3
        lea       6824+__dpow_la_CoutTab(%rip), %rdx
        mulsd     %xmm0, %xmm3
        andl      %r12d, %r9d
        movaps    %xmm3, %xmm2
        mulsd     (%rdx,%r9,8), %xmm2
        jmp       .LBL_2_56

.LBL_2_54:

        movaps    %xmm3, %xmm2
        addsd     %xmm0, %xmm2
        jmp       .LBL_2_56

.LBL_2_55:

        movq      6824+__dpow_la_CoutTab(%rip), %rdx
        addsd     %xmm0, %xmm3
        movsd     %xmm3, -32(%rsp)
        movq      %rdx, -24(%rsp)
        movb      -25(%rsp), %cl
        movb      -17(%rsp), %bl
        andb      $-128, %cl
        andb      $127, %bl
        orb       %cl, %bl
        movb      %bl, -17(%rsp)
        movsd     -24(%rsp), %xmm2
        movsd     -24(%rsp), %xmm0
        mulsd     %xmm0, %xmm2

.LBL_2_56:

        movsd     %xmm2, (%r8)
	.cfi_restore 3
        popq      %rbx
	.cfi_def_cfa_offset 32
	.cfi_restore 15
        popq      %r15
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12
	.cfi_def_cfa_offset 8
        ret
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	.cfi_offset 12, -16
	.cfi_offset 13, -24
	.cfi_offset 15, -32

.LBL_2_57:

        testb     %r11b, %r11b
        je        .LBL_2_30
        jmp       .LBL_2_55

.LBL_2_58:

        lea       6752+__dpow_la_CoutTab(%rip), %rdx
        movsd     (%rdx), %xmm6
        lea       4688+__dpow_la_CoutTab(%rip), %rcx
        movsd     -64(%rsp), %xmm7
        addsd     %xmm2, %xmm8
        addsd     %xmm9, %xmm10
        addsd     %xmm3, %xmm2
        mulsd     %xmm8, %xmm6
        addsd     %xmm10, %xmm11
        addsd     -8(%rdx), %xmm6
        addsd     %xmm2, %xmm11
        mulsd     %xmm8, %xmm6
        movaps    %xmm11, %xmm9
        addsd     -16(%rdx), %xmm6
        mulsd     %xmm8, %xmm6
        addsd     -24(%rdx), %xmm6
        mulsd     %xmm8, %xmm6
        addsd     %xmm6, %xmm4
        lea       -32(%rsp), %rbx
        movsd     %xmm4, (%rbx)
        movsd     (%rbx), %xmm8
        subsd     %xmm8, %xmm7
        addsd     %xmm6, %xmm7
        lea       -24(%rsp), %rsi
        movsd     %xmm7, (%rsi)
        movsd     (%rbx), %xmm12
        movsd     (%rsi), %xmm5
        addsd     %xmm12, %xmm9
        movsd     %xmm9, (%rbx)
        movsd     (%rbx), %xmm10
        subsd     %xmm10, %xmm12
        addsd     %xmm11, %xmm12
        movsd     104(%rdx), %xmm11
        movsd     %xmm12, (%rsi)
        movsd     (%rbx), %xmm3
        mulsd     %xmm3, %xmm11
        movsd     (%rsi), %xmm4
        movsd     %xmm11, (%rbx)
        addsd     %xmm5, %xmm4
        movsd     (%rbx), %xmm13
        lea       -16(%rsp), %rdi
        movsd     (%rdi), %xmm11
        subsd     %xmm3, %xmm13
        mulsd     (%rdi), %xmm4
        movsd     %xmm13, (%rsi)
        movaps    %xmm11, %xmm6
        movsd     (%rbx), %xmm15
        movsd     (%rsi), %xmm14
        mulsd     104(%rdx), %xmm6
        subsd     %xmm14, %xmm15
        movsd     %xmm15, (%rbx)
        movsd     (%rbx), %xmm2
        movsd     88(%rdx), %xmm5
        subsd     %xmm2, %xmm3
        movsd     %xmm3, (%rsi)
        movsd     (%rbx), %xmm3
        movsd     (%rsi), %xmm2
        movsd     %xmm6, (%rbx)
        movsd     (%rbx), %xmm7
        subsd     (%rdi), %xmm7
        movsd     %xmm7, (%rsi)
        movsd     (%rbx), %xmm9
        movsd     (%rsi), %xmm8
        subsd     %xmm8, %xmm9
        movsd     %xmm9, (%rbx)
        movsd     (%rbx), %xmm10
        subsd     %xmm10, %xmm11
        movsd     %xmm11, (%rsi)
        movsd     (%rbx), %xmm6
        movaps    %xmm6, %xmm14
        mulsd     %xmm3, %xmm14
        mulsd     %xmm2, %xmm6
        addsd     %xmm14, %xmm5
        movsd     (%rsi), %xmm15
        mulsd     %xmm15, %xmm2
        mulsd     %xmm15, %xmm3
        addsd     %xmm2, %xmm6
        movsd     %xmm5, (%rbx)
        addsd     %xmm3, %xmm6
        movsd     (%rbx), %xmm12
        addsd     %xmm4, %xmm6
        subsd     88(%rdx), %xmm12
        movsd     %xmm12, -88(%rsp)
        movsd     -88(%rsp), %xmm13
        movsd     40(%rdx), %xmm3
        subsd     %xmm13, %xmm14
        movsd     %xmm14, -80(%rsp)
        movsd     -80(%rsp), %xmm5
        movl      (%rbx), %edx
        movl      %edx, %esi
        andl      $127, %edx
        addsd     %xmm6, %xmm5
        mulsd     %xmm5, %xmm3
        addl      %edx, %edx
        movsd     -8(%rcx,%rdx,8), %xmm2
        sarl      $7, %esi
        lea       6784+__dpow_la_CoutTab(%rip), %r9
        addsd     (%r9), %xmm3
        mulsd     %xmm5, %xmm3
        addsd     -8(%r9), %xmm3
        mulsd     %xmm5, %xmm3
        addsd     -16(%r9), %xmm3
        mulsd     %xmm5, %xmm3
        addsd     -24(%r9), %xmm3
        mulsd     %xmm5, %xmm3
        mulsd     %xmm2, %xmm3
        addsd     (%rcx,%rdx,8), %xmm3
        movaps    %xmm3, %xmm4
        addsd     %xmm2, %xmm4
        movsd     %xmm4, -72(%rsp)
        movzwl    -66(%rsp), %ecx
        movl      %ecx, %ebx
        andl      $32752, %ebx
        shrl      $4, %ebx
        lea       -1023(%rsi,%rbx), %edx
        cmpl      $1024, %edx
        jge       .LBL_2_66


        cmpl      $-1022, %edx
        jl        .LBL_2_61


        andl      $-32753, %ecx
        lea       1023(%rdx), %edx
        andl      $2047, %edx
        shll      $4, %edx
        orl       %edx, %ecx
        movw      %cx, -66(%rsp)
        movsd     -72(%rsp), %xmm2
        mulsd     %xmm0, %xmm2
        movsd     %xmm2, -72(%rsp)
        jmp       .LBL_2_56

.LBL_2_61:

        cmpl      $-1032, %edx
        jl        .LBL_2_63


        lea       -32(%rsp), %rcx
        movsd     %xmm4, (%rcx)
        addl      $1223, %esi
        movsd     (%rcx), %xmm1
        andl      $2047, %esi
        lea       6824+__dpow_la_CoutTab(%rip), %rbx
        movq      (%rbx), %rdx
        subsd     %xmm1, %xmm2
        movq      %rdx, -64(%rsp)
        addsd     %xmm2, %xmm3
        lea       -24(%rsp), %rdi
        movsd     %xmm3, (%rdi)
        movsd     (%rcx), %xmm7
        movsd     32(%rbx), %xmm2
        mulsd     %xmm7, %xmm2
        movsd     (%rdi), %xmm9
        movsd     %xmm2, (%rcx)
        movsd     (%rcx), %xmm3
        shrq      $48, %rdx
        subsd     %xmm7, %xmm3
        movsd     %xmm3, (%rdi)
        andl      $-32753, %edx
        movsd     (%rcx), %xmm5
        movsd     (%rdi), %xmm4
        shll      $4, %esi
        subsd     %xmm4, %xmm5
        movsd     %xmm5, (%rcx)
        orl       %esi, %edx
        lea       -32(%rsp), %rsi
        movsd     (%rsi), %xmm6
        movw      %dx, -58(%rsp)
        subsd     %xmm6, %xmm7
        movsd     %xmm7, (%rdi)
        movsd     (%rsi), %xmm11
        movsd     (%rdi), %xmm12
        movsd     -64(%rsp), %xmm10
        addsd     %xmm9, %xmm12
        mulsd     %xmm10, %xmm11
        mulsd     %xmm10, %xmm12
        movsd     48(%rbx), %xmm8
        addsd     %xmm11, %xmm12
        mulsd     %xmm8, %xmm0
        movq      -16(%rbx), %rcx
        movq      %rcx, (%rsi)
        lea       -32(%rsp), %rcx
        movsd     (%rcx), %xmm14
        movsd     (%rcx), %xmm13
        mulsd     %xmm13, %xmm14
        mulsd     %xmm12, %xmm0
        movsd     %xmm14, (%rcx)
        movsd     (%rcx), %xmm15
        addsd     %xmm15, %xmm0
        movaps    %xmm0, %xmm2
        movsd     %xmm2, -72(%rsp)
        jmp       .LBL_2_56

.LBL_2_63:

        cmpl      $-1084, %edx
        jl        .LBL_2_65


        addl      $1223, %esi
        andl      $2047, %esi
        lea       6830+__dpow_la_CoutTab(%rip), %rcx
        movzwl    (%rcx), %edx
        shll      $4, %esi
        andl      $-32753, %edx
        movsd     %xmm1, -64(%rsp)
        orl       %esi, %edx
        movw      %dx, -58(%rsp)
        movsd     42(%rcx), %xmm2
        movsd     -64(%rsp), %xmm1
        mulsd     %xmm2, %xmm0
        mulsd     %xmm1, %xmm4
        movq      -22(%rcx), %rcx
        movq      %rcx, -32(%rsp)
        mulsd     %xmm4, %xmm0
        lea       -32(%rsp), %rcx
        movsd     (%rcx), %xmm4
        movsd     (%rcx), %xmm3
        mulsd     %xmm3, %xmm4
        movsd     %xmm4, (%rcx)
        movsd     (%rcx), %xmm5
        subsd     %xmm5, %xmm0
        movaps    %xmm0, %xmm2
        movsd     %xmm2, -72(%rsp)
        jmp       .LBL_2_56

.LBL_2_65:

        movq      6808+__dpow_la_CoutTab(%rip), %rdx
        movq      %rdx, -32(%rsp)
        lea       -32(%rsp), %rdx
        movsd     (%rdx), %xmm2
        movsd     (%rdx), %xmm1
        mulsd     %xmm1, %xmm2
        movsd     %xmm2, (%rdx)
        movsd     (%rdx), %xmm3
        mulsd     %xmm3, %xmm0
        movaps    %xmm0, %xmm2
        movsd     %xmm2, -72(%rsp)
        jmp       .LBL_2_56

.LBL_2_66:

        movq      6800+__dpow_la_CoutTab(%rip), %rdx
        movq      %rdx, -32(%rsp)
        lea       -32(%rsp), %rdx
        movsd     (%rdx), %xmm2
        movsd     (%rdx), %xmm1
        mulsd     %xmm1, %xmm2
        movsd     %xmm2, (%rdx)
        movsd     (%rdx), %xmm3
        mulsd     %xmm3, %xmm0
        movaps    %xmm0, %xmm2
        movsd     %xmm2, -72(%rsp)
        jmp       .LBL_2_56

.LBL_2_67:

        movb      -57(%rsp), %dl
        lea       6800+__dpow_la_CoutTab(%rip), %rcx
        andb      $-128, %dl
        shrb      $7, %dl
        xorb      %dl, %bl
        movzbl    %bl, %ebx
        movsd     (%rcx,%rbx,8), %xmm2
        mulsd     %xmm2, %xmm2
        mulsd     %xmm0, %xmm2
        jmp       .LBL_2_56

.LBL_2_68:

        testl     %r9d, %r9d
        jne       .LBL_2_70


        testl     %esi, %esi
        jne       .LBL_2_36

.LBL_2_70:

        lea       6824+__dpow_la_CoutTab(%rip), %rdx
        andl      $1, %r9d
        movsd     (%rdx,%r9,8), %xmm2
        jmp       .LBL_2_56

.LBL_2_71:

        mulsd     %xmm3, %xmm3
        testb     %bl, %bl
        je        .LBL_2_73


        lea       6824+__dpow_la_CoutTab(%rip), %rax
        andl      %r12d, %r9d
        movsd     (%rax,%r9,8), %xmm2
        movl      $1, %eax
        divsd     %xmm3, %xmm2
        jmp       .LBL_2_56

.LBL_2_73:

        lea       6824+__dpow_la_CoutTab(%rip), %rdx
        andl      %r12d, %r9d
        movsd     (%rdx,%r9,8), %xmm2
        mulsd     %xmm3, %xmm2
        jmp       .LBL_2_56

.LBL_2_74:

        testb     %r11b, %r11b
        jne       .LBL_2_55


        testl     %edx, %edx
        jne       .LBL_2_30


        testl     %r10d, %r10d
        je        .LBL_2_30
        jmp       .LBL_2_55
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dpow_cout_rare_internal,@function
	.size	__svml_dpow_cout_rare_internal,.-__svml_dpow_cout_rare_internal
..LN__svml_dpow_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dpow_data_internal_avx512:
	.long	0
	.long	0
	.long	1972240384
	.long	3215375059
	.long	4207476736
	.long	3216401398
	.long	2363129856
	.long	3217067096
	.long	972816384
	.long	3217408026
	.long	766836736
	.long	3217739614
	.long	3103948800
	.long	3218062358
	.long	2869821440
	.long	3218228231
	.long	1271726080
	.long	3218381432
	.long	3449618432
	.long	3218530849
	.long	2300510208
	.long	3218676666
	.long	4147675136
	.long	3218819051
	.long	3761438720
	.long	3218958163
	.long	1758134272
	.long	3219094149
	.long	14041088
	.long	3219177733
	.long	513138688
	.long	3219242801
	.long	1904279552
	.long	1071288313
	.long	180338688
	.long	1071163544
	.long	1751498752
	.long	1071041295
	.long	2999894016
	.long	1070921467
	.long	3815833600
	.long	1070803966
	.long	1399062528
	.long	1070688704
	.long	2835742720
	.long	1070555096
	.long	2818572288
	.long	1070333031
	.long	180322304
	.long	1070114968
	.long	704610304
	.long	1069900764
	.long	3265986560
	.long	1069690285
	.long	3908239360
	.long	1069419290
	.long	2530213888
	.long	1069012484
	.long	2785017856
	.long	1068612402
	.long	3386507264
	.long	1067938708
	.long	2250244096
	.long	1066877934
	.long	0
	.long	0
	.long	650173971
	.long	3177165030
	.long	3428024929
	.long	3174241916
	.long	1628324029
	.long	1026060711
	.long	804943611
	.long	1028963376
	.long	518075456
	.long	1027828752
	.long	1462134616
	.long	1028126172
	.long	384118417
	.long	3174884873
	.long	1227618047
	.long	3176893182
	.long	446961290
	.long	3175726255
	.long	2998207852
	.long	3176597684
	.long	2742536172
	.long	3173319968
	.long	3242321520
	.long	1029042433
	.long	1690697745
	.long	3174775608
	.long	4137858450
	.long	1027958429
	.long	2514005062
	.long	1029694520
	.long	804943611
	.long	1027914800
	.long	2871266960
	.long	3173412044
	.long	3679462403
	.long	1027724294
	.long	2476829589
	.long	1026974179
	.long	1572243234
	.long	3176241050
	.long	2514550597
	.long	3175960347
	.long	1207415416
	.long	1029642824
	.long	531120703
	.long	3174459378
	.long	894287639
	.long	1029609779
	.long	1133539114
	.long	1029069062
	.long	1763539348
	.long	1029327721
	.long	1658032750
	.long	3171241178
	.long	825146242
	.long	3176213734
	.long	831162967
	.long	1028990787
	.long	1128763360
	.long	3176457556
	.long	896504796
	.long	3175699769
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
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
	.long	0
	.long	1073217536
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
	.long	1697350398
	.long	3219592519
	.long	1697350398
	.long	3219592519
	.long	1697350398
	.long	3219592519
	.long	1697350398
	.long	3219592519
	.long	1697350398
	.long	3219592519
	.long	1697350398
	.long	3219592519
	.long	1697350398
	.long	3219592519
	.long	1697350398
	.long	3219592519
	.long	4114041750
	.long	1069844377
	.long	4114041750
	.long	1069844377
	.long	4114041750
	.long	1069844377
	.long	4114041750
	.long	1069844377
	.long	4114041750
	.long	1069844377
	.long	4114041750
	.long	1069844377
	.long	4114041750
	.long	1069844377
	.long	4114041750
	.long	1069844377
	.long	2589302621
	.long	3217496037
	.long	2589302621
	.long	3217496037
	.long	2589302621
	.long	3217496037
	.long	2589302621
	.long	3217496037
	.long	2589302621
	.long	3217496037
	.long	2589302621
	.long	3217496037
	.long	2589302621
	.long	3217496037
	.long	2589302621
	.long	3217496037
	.long	4238449923
	.long	1070227829
	.long	4238449923
	.long	1070227829
	.long	4238449923
	.long	1070227829
	.long	4238449923
	.long	1070227829
	.long	4238449923
	.long	1070227829
	.long	4238449923
	.long	1070227829
	.long	4238449923
	.long	1070227829
	.long	4238449923
	.long	1070227829
	.long	3163535583
	.long	3217999625
	.long	3163535583
	.long	3217999625
	.long	3163535583
	.long	3217999625
	.long	3163535583
	.long	3217999625
	.long	3163535583
	.long	3217999625
	.long	3163535583
	.long	3217999625
	.long	3163535583
	.long	3217999625
	.long	3163535583
	.long	3217999625
	.long	1357918834
	.long	1070757740
	.long	1357918834
	.long	1070757740
	.long	1357918834
	.long	1070757740
	.long	1357918834
	.long	1070757740
	.long	1357918834
	.long	1070757740
	.long	1357918834
	.long	1070757740
	.long	1357918834
	.long	1070757740
	.long	1357918834
	.long	1070757740
	.long	1697368554
	.long	3218543943
	.long	1697368554
	.long	3218543943
	.long	1697368554
	.long	3218543943
	.long	1697368554
	.long	3218543943
	.long	1697368554
	.long	3218543943
	.long	1697368554
	.long	3218543943
	.long	1697368554
	.long	3218543943
	.long	1697368554
	.long	3218543943
	.long	3694789628
	.long	1071564553
	.long	3694789628
	.long	1071564553
	.long	3694789628
	.long	1071564553
	.long	3694789628
	.long	1071564553
	.long	3694789628
	.long	1071564553
	.long	3694789628
	.long	1071564553
	.long	3694789628
	.long	1071564553
	.long	3694789628
	.long	1071564553
	.long	3757988711
	.long	1013148509
	.long	3757988711
	.long	1013148509
	.long	3757988711
	.long	1013148509
	.long	3757988711
	.long	1013148509
	.long	3757988711
	.long	1013148509
	.long	3757988711
	.long	1013148509
	.long	3757988711
	.long	1013148509
	.long	3757988711
	.long	1013148509
	.long	3999174959
	.long	1014462451
	.long	3999174959
	.long	1014462451
	.long	3999174959
	.long	1014462451
	.long	3999174959
	.long	1014462451
	.long	3999174959
	.long	1014462451
	.long	3999174959
	.long	1014462451
	.long	3999174959
	.long	1014462451
	.long	3999174959
	.long	1014462451
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
	.long	4119604569
	.long	1059365335
	.long	4119604569
	.long	1059365335
	.long	4119604569
	.long	1059365335
	.long	4119604569
	.long	1059365335
	.long	4119604569
	.long	1059365335
	.long	4119604569
	.long	1059365335
	.long	4119604569
	.long	1059365335
	.long	4119604569
	.long	1059365335
	.long	662950521
	.long	1062590279
	.long	662950521
	.long	1062590279
	.long	662950521
	.long	1062590279
	.long	662950521
	.long	1062590279
	.long	662950521
	.long	1062590279
	.long	662950521
	.long	1062590279
	.long	662950521
	.long	1062590279
	.long	662950521
	.long	1062590279
	.long	454355882
	.long	1065595565
	.long	454355882
	.long	1065595565
	.long	454355882
	.long	1065595565
	.long	454355882
	.long	1065595565
	.long	454355882
	.long	1065595565
	.long	454355882
	.long	1065595565
	.long	454355882
	.long	1065595565
	.long	454355882
	.long	1065595565
	.long	3568144057
	.long	1068264200
	.long	3568144057
	.long	1068264200
	.long	3568144057
	.long	1068264200
	.long	3568144057
	.long	1068264200
	.long	3568144057
	.long	1068264200
	.long	3568144057
	.long	1068264200
	.long	3568144057
	.long	1068264200
	.long	3568144057
	.long	1068264200
	.long	4286862669
	.long	1070514109
	.long	4286862669
	.long	1070514109
	.long	4286862669
	.long	1070514109
	.long	4286862669
	.long	1070514109
	.long	4286862669
	.long	1070514109
	.long	4286862669
	.long	1070514109
	.long	4286862669
	.long	1070514109
	.long	4286862669
	.long	1070514109
	.long	4277811595
	.long	1072049730
	.long	4277811595
	.long	1072049730
	.long	4277811595
	.long	1072049730
	.long	4277811595
	.long	1072049730
	.long	4277811595
	.long	1072049730
	.long	4277811595
	.long	1072049730
	.long	4277811595
	.long	1072049730
	.long	4277811595
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
	.long	248
	.long	0
	.long	248
	.long	0
	.long	248
	.long	0
	.long	248
	.long	0
	.long	248
	.long	0
	.long	248
	.long	0
	.long	248
	.long	0
	.long	248
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
	.long	1083173888
	.long	0
	.long	1083173888
	.long	0
	.long	1083173888
	.long	0
	.long	1083173888
	.long	0
	.long	1083173888
	.long	0
	.long	1083173888
	.long	0
	.long	1083173888
	.long	0
	.long	1083173888
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.long	0
	.long	2146435072
	.type	__svml_dpow_data_internal_avx512,@object
	.size	__svml_dpow_data_internal_avx512,2368
	.align 32
__dpow_la_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	1072629696
	.long	0
	.long	1072569856
	.long	0
	.long	1072513472
	.long	0
	.long	1072460224
	.long	0
	.long	1072409856
	.long	0
	.long	1072362112
	.long	0
	.long	1072316864
	.long	0
	.long	1072273792
	.long	0
	.long	1072232896
	.long	0
	.long	1072193920
	.long	0
	.long	1072156736
	.long	0
	.long	1072121280
	.long	0
	.long	1072087424
	.long	0
	.long	1072054976
	.long	0
	.long	1072023936
	.long	0
	.long	1071994176
	.long	0
	.long	1071965696
	.long	0
	.long	1071938304
	.long	0
	.long	1071911936
	.long	0
	.long	1071886656
	.long	0
	.long	1071862272
	.long	0
	.long	1071838848
	.long	0
	.long	1071816256
	.long	0
	.long	1071794496
	.long	0
	.long	1071773440
	.long	0
	.long	1071753152
	.long	0
	.long	1071733504
	.long	0
	.long	1071714560
	.long	0
	.long	1071696256
	.long	0
	.long	1071678528
	.long	0
	.long	1071661312
	.long	0
	.long	1071644672
	.long	0
	.long	0
	.long	0
	.long	0
	.long	2686386176
	.long	1067891457
	.long	1949948785
	.long	1027381598
	.long	1341652992
	.long	1068918120
	.long	2376679344
	.long	1026589938
	.long	2182004736
	.long	1069583575
	.long	297009671
	.long	1026900933
	.long	1687183360
	.long	1069924424
	.long	2120169064
	.long	1026082260
	.long	53207040
	.long	1070255920
	.long	3737096550
	.long	1026438963
	.long	3818315776
	.long	1070578756
	.long	677794872
	.long	1028109305
	.long	2429726720
	.long	1070744485
	.long	3907638365
	.long	1027382133
	.long	2702757888
	.long	1070897876
	.long	1929563302
	.long	1027984695
	.long	2465140736
	.long	1071047207
	.long	243175481
	.long	1026641700
	.long	2657701888
	.long	1071193041
	.long	3841377895
	.long	1028504382
	.long	658427904
	.long	1071335525
	.long	161357665
	.long	1028306250
	.long	539168768
	.long	1071474585
	.long	2531816708
	.long	1025043792
	.long	2658430976
	.long	1071610420
	.long	2178519328
	.long	1028288112
	.long	1355743232
	.long	1071694102
	.long	3943781029
	.long	1028003666
	.long	1854838784
	.long	1071759170
	.long	1812291414
	.long	1027042047
	.long	473251840
	.long	3218771869
	.long	1330616404
	.long	3175482613
	.long	2315530240
	.long	3218647330
	.long	3482179716
	.long	3175726112
	.long	3886694400
	.long	3218525081
	.long	3584491563
	.long	3175164762
	.long	1568866304
	.long	3218405023
	.long	3528175174
	.long	3174626157
	.long	4172640256
	.long	3218287637
	.long	3760034354
	.long	3171774178
	.long	3545214976
	.long	3218172213
	.long	881689765
	.long	3173077446
	.long	2121375744
	.long	3218038698
	.long	549802690
	.long	3174897014
	.long	492560384
	.long	3217816668
	.long	239252792
	.long	3173483664
	.long	155754496
	.long	3217598893
	.long	1693604438
	.long	3175909818
	.long	4285202432
	.long	3217384365
	.long	127148739
	.long	3175942199
	.long	41181184
	.long	3217174003
	.long	3260046653
	.long	3174058211
	.long	2465087488
	.long	3216902292
	.long	4241850247
	.long	3175110025
	.long	1101037568
	.long	3216495763
	.long	3170347605
	.long	3176066808
	.long	3478798336
	.long	3216096373
	.long	329155479
	.long	3175972274
	.long	3246555136
	.long	3215423741
	.long	4071576371
	.long	3174315914
	.long	830078976
	.long	3214361213
	.long	1258533012
	.long	3175547121
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	1072689152
	.long	0
	.long	1072685056
	.long	0
	.long	1072681024
	.long	0
	.long	1072676992
	.long	0
	.long	1072672960
	.long	0
	.long	1072668928
	.long	0
	.long	1072664960
	.long	0
	.long	1072660992
	.long	0
	.long	1072657024
	.long	0
	.long	1072653056
	.long	0
	.long	1072649152
	.long	0
	.long	1072645248
	.long	0
	.long	1072641344
	.long	0
	.long	1072637440
	.long	0
	.long	1072710976
	.long	0
	.long	1072709888
	.long	0
	.long	1072708864
	.long	0
	.long	1072707776
	.long	0
	.long	1072706752
	.long	0
	.long	1072705664
	.long	0
	.long	1072704640
	.long	0
	.long	1072703616
	.long	0
	.long	1072702528
	.long	0
	.long	1072701504
	.long	0
	.long	1072700480
	.long	0
	.long	1072699456
	.long	0
	.long	1072698368
	.long	0
	.long	1072697344
	.long	0
	.long	1072696320
	.long	0
	.long	1072695296
	.long	0
	.long	1072694272
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	0
	.long	2754084864
	.long	1063721742
	.long	2557931335
	.long	1028226920
	.long	3228041216
	.long	1064771801
	.long	930662348
	.long	1027873525
	.long	2323251200
	.long	1065436614
	.long	2596299912
	.long	1027915217
	.long	1641152512
	.long	1065811444
	.long	1188689655
	.long	1027383036
	.long	895221760
	.long	1066187001
	.long	2918954073
	.long	1026717129
	.long	3962896384
	.long	1066482539
	.long	1338190555
	.long	1024402868
	.long	2071330816
	.long	1066668054
	.long	2834125591
	.long	1027573772
	.long	830078976
	.long	1066853925
	.long	1683363035
	.long	1027948302
	.long	1828782080
	.long	1067040153
	.long	874130859
	.long	1026348678
	.long	2395996160
	.long	1067226740
	.long	1724975876
	.long	1028585613
	.long	3558866944
	.long	1067410669
	.long	2189961434
	.long	1027936707
	.long	2542927872
	.long	1067522658
	.long	3621009110
	.long	1028493916
	.long	4208394240
	.long	1067614973
	.long	2777386350
	.long	1028255456
	.long	3217162240
	.long	1067707465
	.long	772669574
	.long	1028516547
	.long	824377344
	.long	3214460051
	.long	1593617402
	.long	3175722247
	.long	830078976
	.long	3214361213
	.long	1258533012
	.long	3175547121
	.long	4002480128
	.long	3214268096
	.long	1397883555
	.long	3175764245
	.long	2914385920
	.long	3214169062
	.long	3775067953
	.long	3175176772
	.long	1460142080
	.long	3214075761
	.long	1592372614
	.long	3175907032
	.long	219152384
	.long	3213976530
	.long	1716511551
	.long	3175540921
	.long	3419144192
	.long	3213880645
	.long	1128677462
	.long	3174560569
	.long	3320446976
	.long	3213693490
	.long	2965227743
	.long	3172454196
	.long	677904384
	.long	3213494440
	.long	4029390031
	.long	3174409513
	.long	1290797056
	.long	3213306911
	.long	1477436787
	.long	3173730612
	.long	2800877568
	.long	3213119200
	.long	4281418519
	.long	3173304523
	.long	3692822528
	.long	3212931307
	.long	751117103
	.long	3175382448
	.long	2547253248
	.long	3212626079
	.long	2419265147
	.long	3175328924
	.long	1836580864
	.long	3212249540
	.long	1456335141
	.long	3175441338
	.long	3438542848
	.long	3211872634
	.long	3721652080
	.long	3176073447
	.long	4278714368
	.long	3211202435
	.long	836003693
	.long	3174279974
	.long	926941184
	.long	3210154597
	.long	4249864733
	.long	3174015648
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1073741824
	.long	1073157447
	.long	0
	.long	1073157401
	.long	0
	.long	1073157355
	.long	3221225472
	.long	1073157308
	.long	2147483648
	.long	1073157262
	.long	2147483648
	.long	1073157216
	.long	1073741824
	.long	1073157170
	.long	1073741824
	.long	1073157124
	.long	0
	.long	1073157078
	.long	3221225472
	.long	1073157031
	.long	3221225472
	.long	1073156985
	.long	2147483648
	.long	1073156939
	.long	2147483648
	.long	1073156893
	.long	1073741824
	.long	1073156847
	.long	1073741824
	.long	1073156801
	.long	0
	.long	1073156755
	.long	0
	.long	1073156709
	.long	3221225472
	.long	1073156662
	.long	3221225472
	.long	1073156616
	.long	2147483648
	.long	1073156570
	.long	2147483648
	.long	1073156524
	.long	2147483648
	.long	1073156478
	.long	1073741824
	.long	1073156432
	.long	1073741824
	.long	1073156386
	.long	0
	.long	1073156340
	.long	0
	.long	1073156294
	.long	0
	.long	1073156248
	.long	3221225472
	.long	1073156201
	.long	3221225472
	.long	1073156155
	.long	2147483648
	.long	1073156109
	.long	2147483648
	.long	1073156063
	.long	2147483648
	.long	1073156017
	.long	1073741824
	.long	1073155971
	.long	1073741824
	.long	1073155925
	.long	1073741824
	.long	1073155879
	.long	1073741824
	.long	1073155833
	.long	0
	.long	1073155787
	.long	0
	.long	1073155741
	.long	0
	.long	1073155695
	.long	0
	.long	1073155649
	.long	3221225472
	.long	1073155602
	.long	3221225472
	.long	1073155556
	.long	3221225472
	.long	1073155510
	.long	3221225472
	.long	1073155464
	.long	3221225472
	.long	1073155418
	.long	2147483648
	.long	1073155372
	.long	2147483648
	.long	1073155326
	.long	2147483648
	.long	1073155280
	.long	2147483648
	.long	1073155234
	.long	2147483648
	.long	1073155188
	.long	2147483648
	.long	1073155142
	.long	2147483648
	.long	1073155096
	.long	2147483648
	.long	1073155050
	.long	2147483648
	.long	1073155004
	.long	1073741824
	.long	1073154958
	.long	1073741824
	.long	1073154912
	.long	1073741824
	.long	1073154866
	.long	1073741824
	.long	1073154820
	.long	1073741824
	.long	1073154774
	.long	1073741824
	.long	1073154728
	.long	1073741824
	.long	1073154682
	.long	2147483648
	.long	1073158995
	.long	1073741824
	.long	1073158972
	.long	1073741824
	.long	1073158949
	.long	0
	.long	1073158926
	.long	0
	.long	1073158903
	.long	3221225472
	.long	1073158879
	.long	3221225472
	.long	1073158856
	.long	2147483648
	.long	1073158833
	.long	2147483648
	.long	1073158810
	.long	1073741824
	.long	1073158787
	.long	1073741824
	.long	1073158764
	.long	0
	.long	1073158741
	.long	0
	.long	1073158718
	.long	3221225472
	.long	1073158694
	.long	3221225472
	.long	1073158671
	.long	2147483648
	.long	1073158648
	.long	2147483648
	.long	1073158625
	.long	1073741824
	.long	1073158602
	.long	1073741824
	.long	1073158579
	.long	0
	.long	1073158556
	.long	0
	.long	1073158533
	.long	3221225472
	.long	1073158509
	.long	3221225472
	.long	1073158486
	.long	2147483648
	.long	1073158463
	.long	2147483648
	.long	1073158440
	.long	1073741824
	.long	1073158417
	.long	1073741824
	.long	1073158394
	.long	1073741824
	.long	1073158371
	.long	0
	.long	1073158348
	.long	0
	.long	1073158325
	.long	3221225472
	.long	1073158301
	.long	3221225472
	.long	1073158278
	.long	2147483648
	.long	1073158255
	.long	2147483648
	.long	1073158232
	.long	2147483648
	.long	1073158209
	.long	1073741824
	.long	1073158186
	.long	1073741824
	.long	1073158163
	.long	0
	.long	1073158140
	.long	0
	.long	1073158117
	.long	3221225472
	.long	1073158093
	.long	3221225472
	.long	1073158070
	.long	3221225472
	.long	1073158047
	.long	2147483648
	.long	1073158024
	.long	2147483648
	.long	1073158001
	.long	1073741824
	.long	1073157978
	.long	1073741824
	.long	1073157955
	.long	1073741824
	.long	1073157932
	.long	0
	.long	1073157909
	.long	0
	.long	1073157886
	.long	3221225472
	.long	1073157862
	.long	3221225472
	.long	1073157839
	.long	3221225472
	.long	1073157816
	.long	2147483648
	.long	1073157793
	.long	2147483648
	.long	1073157770
	.long	2147483648
	.long	1073157747
	.long	1073741824
	.long	1073157724
	.long	1073741824
	.long	1073157701
	.long	0
	.long	1073157678
	.long	0
	.long	1073157655
	.long	0
	.long	1073157632
	.long	3221225472
	.long	1073157608
	.long	3221225472
	.long	1073157585
	.long	3221225472
	.long	1073157562
	.long	2147483648
	.long	1073157539
	.long	2147483648
	.long	1073157516
	.long	2147483648
	.long	1073157493
	.long	1073741824
	.long	1073157470
	.long	1073741824
	.long	1073157447
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1342177280
	.long	1057431575
	.long	1679773494
	.long	1024039205
	.long	989855744
	.long	1058476078
	.long	3244478756
	.long	1024589954
	.long	209715200
	.long	1059147828
	.long	152199156
	.long	1027874535
	.long	2449473536
	.long	1059526748
	.long	2343302255
	.long	1022283036
	.long	1560281088
	.long	1059903632
	.long	4038848719
	.long	1027337824
	.long	4282384384
	.long	1060196455
	.long	2325104861
	.long	1027595231
	.long	1665138688
	.long	1060384909
	.long	2934027888
	.long	1026982347
	.long	3263168512
	.long	1060574392
	.long	3208451390
	.long	1027670758
	.long	3980394496
	.long	1060763881
	.long	863587004
	.long	1026973426
	.long	2470445056
	.long	1060952352
	.long	1027097864
	.long	1028644619
	.long	1296039936
	.long	1061141853
	.long	2016162954
	.long	1025089894
	.long	3107979264
	.long	1061244623
	.long	970842239
	.long	1028172704
	.long	3722444800
	.long	1061339379
	.long	2640304163
	.long	1027825546
	.long	2959081472
	.long	1061433626
	.long	306547692
	.long	1028101690
	.long	2631925760
	.long	1061528388
	.long	747377661
	.long	1028120913
	.long	794820608
	.long	1061622641
	.long	3406550266
	.long	1028182206
	.long	3825205248
	.long	1061717408
	.long	3705775220
	.long	1027201825
	.long	916455424
	.long	1061811667
	.long	1432750358
	.long	1028165990
	.long	3011510272
	.long	1061906440
	.long	3361908688
	.long	1027438936
	.long	3330277376
	.long	1062000704
	.long	3560665332
	.long	1027805882
	.long	3082813440
	.long	1062094971
	.long	2539531329
	.long	1028011583
	.long	3747610624
	.long	1062189753
	.long	2232403651
	.long	1025658467
	.long	1218445312
	.long	1062245757
	.long	396499622
	.long	1025861782
	.long	1086324736
	.long	1062293151
	.long	2757240868
	.long	1026731615
	.long	2047868928
	.long	1062340290
	.long	2226191703
	.long	1027982328
	.long	580911104
	.long	1062387431
	.long	1252857417
	.long	1028280924
	.long	1887436800
	.long	1062434829
	.long	659583454
	.long	1025370904
	.long	4186963968
	.long	1062481972
	.long	3587661750
	.long	1028188900
	.long	738197504
	.long	1062529374
	.long	3240696709
	.long	1027025093
	.long	2511339520
	.long	1062576520
	.long	2884432087
	.long	1028614554
	.long	1859125248
	.long	1062623668
	.long	2402099113
	.long	1025699109
	.long	4148166656
	.long	1062671073
	.long	2335237504
	.long	1026835951
	.long	2970615808
	.long	1062718224
	.long	3698719430
	.long	1027808594
	.long	3662675968
	.long	1062765376
	.long	2704653673
	.long	1027603403
	.long	1929379840
	.long	1062812530
	.long	761521627
	.long	1027109120
	.long	3273654272
	.long	1062859941
	.long	470528098
	.long	1027977181
	.long	1019215872
	.long	1062907098
	.long	3704635566
	.long	1027707215
	.long	635437056
	.long	1062954256
	.long	3676592927
	.long	1027502983
	.long	2122317824
	.long	1063001415
	.long	1497197375
	.long	1028267547
	.long	2529165312
	.long	1063048832
	.long	3425827878
	.long	1022000476
	.long	3498049536
	.long	1063095994
	.long	1982476393
	.long	1026289596
	.long	2043674624
	.long	1063143158
	.long	2502680620
	.long	1028471295
	.long	2463105024
	.long	1063190323
	.long	991567028
	.long	1027421239
	.long	460324864
	.long	1063237490
	.long	1461814384
	.long	1026181618
	.long	920125440
	.long	1063270489
	.long	1613472693
	.long	1027845558
	.long	3956277248
	.long	1063294073
	.long	93449747
	.long	1028284502
	.long	1487405056
	.long	1063317659
	.long	1336931403
	.long	1026834156
	.long	2102919168
	.long	1063341245
	.long	319680825
	.long	1027392710
	.long	1508376576
	.long	1063364832
	.long	2474643583
	.long	1027776685
	.long	3999268864
	.long	1063388419
	.long	3104004650
	.long	1024627034
	.long	985137152
	.long	1063412008
	.long	550153379
	.long	1026678253
	.long	1056440320
	.long	1063435597
	.long	672168391
	.long	1027731310
	.long	4213702656
	.long	1063459186
	.long	1805142399
	.long	1026660459
	.long	2772434944
	.long	1063482905
	.long	2448602160
	.long	1028404887
	.long	3528458240
	.long	1063506496
	.long	3457943394
	.long	1027665063
	.long	3075473408
	.long	1063530088
	.long	121314862
	.long	1027996294
	.long	1414004736
	.long	1063553681
	.long	94774013
	.long	1028053481
	.long	2839019520
	.long	1063577274
	.long	1263902834
	.long	1028588748
	.long	3056074752
	.long	1063600868
	.long	369708558
	.long	1028257136
	.long	2065170432
	.long	1063624463
	.long	1634529849
	.long	1027810905
	.long	1769996288
	.long	3210227157
	.long	1054279927
	.long	3174741313
	.long	2442133504
	.long	3210203373
	.long	2067107398
	.long	3175167430
	.long	456130560
	.long	3210179845
	.long	4142755806
	.long	3170825152
	.long	2302672896
	.long	3210156060
	.long	1526169727
	.long	3175523413
	.long	1524629504
	.long	3210132531
	.long	2442955053
	.long	3175425591
	.long	251658240
	.long	3210108746
	.long	2154729168
	.long	3175535488
	.long	681574400
	.long	3210085216
	.long	4275862891
	.long	3176027230
	.long	584056832
	.long	3210061430
	.long	4255852476
	.long	3173565530
	.long	2221932544
	.long	3210037899
	.long	2498876736
	.long	3175149504
	.long	3297771520
	.long	3210014112
	.long	1851620949
	.long	3175688865
	.long	1849688064
	.long	3209990581
	.long	2923055509
	.long	3171310641
	.long	4099932160
	.long	3209966793
	.long	2427653201
	.long	3173037457
	.long	3858759680
	.long	3209943261
	.long	1550068012
	.long	3173027359
	.long	2987393024
	.long	3209919473
	.long	4127650534
	.long	3175851613
	.long	3954180096
	.long	3209895940
	.long	442055840
	.long	3174771669
	.long	4257218560
	.long	3209872151
	.long	4113960829
	.long	3175350854
	.long	2135949312
	.long	3209848618
	.long	2076166727
	.long	3175229825
	.long	3613392896
	.long	3209824828
	.long	3476091171
	.long	3171604778
	.long	2699034624
	.long	3209801294
	.long	1765290157
	.long	3173591669
	.long	1053818880
	.long	3209777504
	.long	3761837094
	.long	3175683182
	.long	1346371584
	.long	3209753969
	.long	1459626820
	.long	3176031561
	.long	875560960
	.long	3209730178
	.long	2402361097
	.long	3174909319
	.long	2375024640
	.long	3209706642
	.long	687754918
	.long	3174943382
	.long	1858076672
	.long	3209674565
	.long	252333183
	.long	3175531572
	.long	2975858688
	.long	3209627492
	.long	1334776821
	.long	3174591557
	.long	2430599168
	.long	3209579907
	.long	1326030186
	.long	3173486707
	.long	1665138688
	.long	3209532833
	.long	737674412
	.long	3174401557
	.long	2122317824
	.long	3209485758
	.long	3987168834
	.long	3175346908
	.long	815792128
	.long	3209438171
	.long	3526910672
	.long	3176068855
	.long	3686793216
	.long	3209391094
	.long	587265932
	.long	3174950865
	.long	429916160
	.long	3209343506
	.long	3143915816
	.long	3175955609
	.long	1417674752
	.long	3209296428
	.long	2918285701
	.long	3174860756
	.long	505413632
	.long	3209248838
	.long	436607152
	.long	3175743066
	.long	3904897024
	.long	3209201758
	.long	2867787430
	.long	3173594277
	.long	4229955584
	.long	3209154678
	.long	3971699810
	.long	3174682560
	.long	2556428288
	.long	3209107086
	.long	3215049067
	.long	3174495054
	.long	998244352
	.long	3209060005
	.long	2424883713
	.long	3173182748
	.long	1667235840
	.long	3209012411
	.long	762177973
	.long	3175232288
	.long	2518679552
	.long	3208965328
	.long	282609672
	.long	3175635057
	.long	1237319680
	.long	3208917733
	.long	1502777354
	.long	3174942228
	.long	203423744
	.long	3208870649
	.long	4128371954
	.long	3175884977
	.long	392167424
	.long	3208823564
	.long	306802084
	.long	3175724146
	.long	2642411520
	.long	3208775966
	.long	2960876517
	.long	3173143647
	.long	945815552
	.long	3208728880
	.long	1800251929
	.long	3170106484
	.long	1241513984
	.long	3208681281
	.long	2675524524
	.long	3173521837
	.long	3904897024
	.long	3208625826
	.long	83988225
	.long	3175795858
	.long	3477078016
	.long	3208531649
	.long	1575792028
	.long	3175657512
	.long	2537553920
	.long	3208436447
	.long	1662079495
	.long	3175916253
	.long	2634022912
	.long	3208342267
	.long	2818347875
	.long	3174383619
	.long	2080374784
	.long	3208247062
	.long	1081767985
	.long	3175779040
	.long	2696937472
	.long	3208152879
	.long	2443744157
	.long	3175275915
	.long	1459617792
	.long	3208058695
	.long	790904149
	.long	3174713637
	.long	3670016000
	.long	3207963485
	.long	581064731
	.long	3173466591
	.long	2952790016
	.long	3207869298
	.long	1008918738
	.long	3171724149
	.long	377487360
	.long	3207775110
	.long	1606538461
	.long	3175837201
	.long	1052770304
	.long	3207679896
	.long	2534546984
	.long	3175060122
	.long	2298478592
	.long	3207577425
	.long	2154814426
	.long	3172198942
	.long	117440512
	.long	3207386992
	.long	1374248651
	.long	3174502065
	.long	1342177280
	.long	3207198603
	.long	4280579335
	.long	3175188313
	.long	3154116608
	.long	3207010211
	.long	3334926656
	.long	3174829419
	.long	2189426688
	.long	3206819769
	.long	3100885346
	.long	3175936751
	.long	746586112
	.long	3206631372
	.long	315615614
	.long	3173018851
	.long	4043309056
	.long	3206340535
	.long	274116456
	.long	3175970612
	.long	268435456
	.long	3205959634
	.long	691182319
	.long	3173304996
	.long	603979776
	.long	3205582822
	.long	112661265
	.long	3170010307
	.long	4194304000
	.long	3204915176
	.long	3717748378
	.long	3174284044
	.long	2885681152
	.long	3203858420
	.long	192153543
	.long	3175961815
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	2851812149
	.long	1072698941
	.long	2595802551
	.long	1016815913
	.long	1048019041
	.long	1072704666
	.long	1398474845
	.long	3161559171
	.long	3899555717
	.long	1072710421
	.long	427280750
	.long	3163595548
	.long	3541402996
	.long	1072716208
	.long	2759177317
	.long	1015903202
	.long	702412510
	.long	1072722027
	.long	3803266086
	.long	3163328991
	.long	410360776
	.long	1072727877
	.long	1269990655
	.long	1013024446
	.long	3402036099
	.long	1072733758
	.long	405889333
	.long	1016154232
	.long	1828292879
	.long	1072739672
	.long	1255956746
	.long	1016636974
	.long	728909815
	.long	1072745618
	.long	383930225
	.long	1016078044
	.long	852742562
	.long	1072751596
	.long	667253586
	.long	1010842135
	.long	2952712987
	.long	1072757606
	.long	3293494651
	.long	3161168877
	.long	3490863953
	.long	1072763649
	.long	960797497
	.long	3163997456
	.long	3228316108
	.long	1072769725
	.long	3010241991
	.long	3159471380
	.long	2930322912
	.long	1072775834
	.long	2599499422
	.long	3163762623
	.long	3366293073
	.long	1072781976
	.long	3119426313
	.long	1015169130
	.long	1014845819
	.long	1072788152
	.long	3117910645
	.long	3162607681
	.long	948735466
	.long	1072794361
	.long	3516338027
	.long	3163623459
	.long	3949972341
	.long	1072800603
	.long	2068408548
	.long	1015962444
	.long	2214878420
	.long	1072806880
	.long	892270087
	.long	3164164998
	.long	828946858
	.long	1072813191
	.long	10642492
	.long	1016988014
	.long	586995997
	.long	1072819536
	.long	41662347
	.long	3163676568
	.long	2288159958
	.long	1072825915
	.long	2169144468
	.long	1015924597
	.long	2440944790
	.long	1072832329
	.long	2492769773
	.long	1015196030
	.long	1853186616
	.long	1072838778
	.long	3066496370
	.long	1016705150
	.long	1337108031
	.long	1072845262
	.long	3203724452
	.long	1015726421
	.long	1709341917
	.long	1072851781
	.long	2571168217
	.long	1015201075
	.long	3790955393
	.long	1072858335
	.long	2352942461
	.long	3164228666
	.long	4112506593
	.long	1072864925
	.long	2947355221
	.long	1015419624
	.long	3504003472
	.long	1072871551
	.long	3594001059
	.long	3158379228
	.long	2799960843
	.long	1072878213
	.long	1423655380
	.long	1016070727
	.long	2839424854
	.long	1072884911
	.long	1171596163
	.long	1014090255
	.long	171030293
	.long	1072891646
	.long	3526460132
	.long	1015477354
	.long	4232894513
	.long	1072898416
	.long	2383938684
	.long	1015717095
	.long	2992903935
	.long	1072905224
	.long	2218154405
	.long	1016276769
	.long	1603444721
	.long	1072912069
	.long	1548633640
	.long	3163249902
	.long	926591435
	.long	1072918951
	.long	3208833761
	.long	3163962090
	.long	1829099622
	.long	1072925870
	.long	1016661180
	.long	3164509581
	.long	887463927
	.long	1072932827
	.long	3596744162
	.long	3161842742
	.long	3272845541
	.long	1072939821
	.long	928852419
	.long	3164536824
	.long	1276261410
	.long	1072946854
	.long	300981947
	.long	1015732745
	.long	78413852
	.long	1072953925
	.long	4183226867
	.long	3164065827
	.long	569847338
	.long	1072961034
	.long	472945272
	.long	3160339305
	.long	3645941911
	.long	1072968181
	.long	3814685080
	.long	3162621917
	.long	1617004845
	.long	1072975368
	.long	82804943
	.long	1011391354
	.long	3978100823
	.long	1072982593
	.long	3513027190
	.long	1016894539
	.long	3049340112
	.long	1072989858
	.long	3062915824
	.long	1014219171
	.long	4040676318
	.long	1072997162
	.long	4090609238
	.long	1016712034
	.long	3577096743
	.long	1073004506
	.long	2951496418
	.long	1014842263
	.long	2583551245
	.long	1073011890
	.long	3161094195
	.long	1016655067
	.long	1990012071
	.long	1073019314
	.long	3529070563
	.long	3163861769
	.long	2731501122
	.long	1073026778
	.long	1774031854
	.long	3163518597
	.long	1453150082
	.long	1073034283
	.long	498154668
	.long	3162536638
	.long	3395129871
	.long	1073041828
	.long	4025345434
	.long	3163383964
	.long	917841882
	.long	1073049415
	.long	18715564
	.long	1016707884
	.long	3566716925
	.long	1073057042
	.long	1536826855
	.long	1015191009
	.long	3712504873
	.long	1073064711
	.long	88491948
	.long	1016476236
	.long	2321106615
	.long	1073072422
	.long	2171176610
	.long	1010584347
	.long	363667784
	.long	1073080175
	.long	813753949
	.long	1016833785
	.long	3111574537
	.long	1073087969
	.long	2606161479
	.long	3163808322
	.long	2956612997
	.long	1073095806
	.long	2118169750
	.long	3163784129
	.long	885834528
	.long	1073103686
	.long	1973258546
	.long	3163310140
	.long	2186617381
	.long	1073111608
	.long	2270764083
	.long	3164321289
	.long	3561793907
	.long	1073119573
	.long	1157054052
	.long	1012938926
	.long	1719614413
	.long	1073127582
	.long	330458197
	.long	3164331316
	.long	1963711167
	.long	1073135634
	.long	1744767756
	.long	3161622870
	.long	1013258799
	.long	1073143730
	.long	1748797610
	.long	3161177658
	.long	4182873220
	.long	1073151869
	.long	629542646
	.long	3163044879
	.long	3907805044
	.long	1073160053
	.long	2257091225
	.long	3162598983
	.long	1218806132
	.long	1073168282
	.long	1818613051
	.long	3163597017
	.long	1447192521
	.long	1073176555
	.long	1462857171
	.long	3163563097
	.long	1339972927
	.long	1073184873
	.long	167908908
	.long	1016620728
	.long	1944781191
	.long	1073193236
	.long	3993278767
	.long	3162772855
	.long	19972402
	.long	1073201645
	.long	3507899861
	.long	1017057868
	.long	919555682
	.long	1073210099
	.long	3121969534
	.long	1013996802
	.long	1413356050
	.long	1073218599
	.long	1651349290
	.long	3163716742
	.long	2571947539
	.long	1073227145
	.long	3558159063
	.long	3164425245
	.long	1176749997
	.long	1073235738
	.long	2738998779
	.long	3163084420
	.long	2604962541
	.long	1073244377
	.long	2614425274
	.long	3164587768
	.long	3649726105
	.long	1073253063
	.long	4085036346
	.long	1016698050
	.long	1110089947
	.long	1073261797
	.long	1451641638
	.long	1016523249
	.long	380978316
	.long	1073270578
	.long	854188970
	.long	3161511262
	.long	2568320822
	.long	1073279406
	.long	2732824428
	.long	1015401491
	.long	194117574
	.long	1073288283
	.long	777528611
	.long	3164460665
	.long	2966275557
	.long	1073297207
	.long	2176155323
	.long	3160891335
	.long	3418903055
	.long	1073306180
	.long	2527457337
	.long	3161869180
	.long	2682146384
	.long	1073315202
	.long	2082178512
	.long	3164411995
	.long	1892288442
	.long	1073324273
	.long	2446255666
	.long	3163648957
	.long	2191782032
	.long	1073333393
	.long	2960257726
	.long	1014791238
	.long	434316067
	.long	1073342563
	.long	2028358766
	.long	1014506698
	.long	2069751141
	.long	1073351782
	.long	1562170674
	.long	3163773257
	.long	3964284211
	.long	1073361051
	.long	2111583915
	.long	1016475740
	.long	2990417245
	.long	1073370371
	.long	3683467745
	.long	3164417902
	.long	321958744
	.long	1073379742
	.long	3401933766
	.long	1016843134
	.long	1434058175
	.long	1073389163
	.long	251133233
	.long	1016134345
	.long	3218338682
	.long	1073398635
	.long	3404164304
	.long	3163525684
	.long	2572866477
	.long	1073408159
	.long	878562433
	.long	1016570317
	.long	697153126
	.long	1073417735
	.long	1283515428
	.long	3164331765
	.long	3092190715
	.long	1073427362
	.long	814012167
	.long	3160571998
	.long	2380618042
	.long	1073437042
	.long	3149557219
	.long	3164369375
	.long	4076559943
	.long	1073446774
	.long	2119478330
	.long	3161806927
	.long	815859274
	.long	1073456560
	.long	240396590
	.long	3164536019
	.long	2420883922
	.long	1073466398
	.long	2049810052
	.long	1015168464
	.long	1540824585
	.long	1073476290
	.long	1064017010
	.long	3164536266
	.long	3716502172
	.long	1073486235
	.long	2303740125
	.long	1015091301
	.long	1610600570
	.long	1073496235
	.long	3766732298
	.long	1016808759
	.long	777507147
	.long	1073506289
	.long	4282924204
	.long	1016236109
	.long	2483480501
	.long	1073516397
	.long	1216371780
	.long	1014082748
	.long	3706687593
	.long	1073526560
	.long	3521726939
	.long	1014301643
	.long	1432208378
	.long	1073536779
	.long	1401068914
	.long	3163412539
	.long	1242007932
	.long	1073547053
	.long	1132034716
	.long	3164388407
	.long	135105010
	.long	1073557383
	.long	1906148727
	.long	3164424315
	.long	3707479175
	.long	1073567768
	.long	3613079302
	.long	1015213314
	.long	382305176
	.long	1073578211
	.long	2347622376
	.long	3163627201
	.long	64696965
	.long	1073588710
	.long	1768797490
	.long	1016865536
	.long	4076975200
	.long	1073599265
	.long	2029000898
	.long	1016257111
	.long	863738719
	.long	1073609879
	.long	1326992219
	.long	3163661773
	.long	351641897
	.long	1073620550
	.long	2172261526
	.long	3164059175
	.long	3884662774
	.long	1073631278
	.long	2158611599
	.long	1015258761
	.long	4224142467
	.long	1073642065
	.long	3389820385
	.long	1016255778
	.long	2728693978
	.long	1073652911
	.long	396109971
	.long	3164511267
	.long	764307441
	.long	1073663816
	.long	3021057420
	.long	3164378099
	.long	3999357479
	.long	1073674779
	.long	2258941616
	.long	1016973300
	.long	929806999
	.long	1073685803
	.long	3205336643
	.long	1016308133
	.long	1533953344
	.long	1073696886
	.long	769171850
	.long	1016714209
	.long	2912730644
	.long	1073708029
	.long	3490067721
	.long	3164453650
	.long	2174652632
	.long	1073719233
	.long	4087714590
	.long	1015498835
	.long	730821105
	.long	1073730498
	.long	2523232743
	.long	1013115764
	.long	2523158504
	.long	1048167334
	.long	1181303047
	.long	3218484803
	.long	1656151777
	.long	1069842388
	.long	714085080
	.long	3216330823
	.long	4277811695
	.long	1072049730
	.long	4286760335
	.long	1070514109
	.long	3607404736
	.long	1068264200
	.long	1874480759
	.long	1065595563
	.long	3884607281
	.long	1062590591
	.long	0
	.long	2145386496
	.long	0
	.long	1048576
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	3220176896
	.long	0
	.long	1120403456
	.long	1073741824
	.long	1073157447
	.long	33554432
	.long	1101004800
	.long	0
	.long	1282408448
	.long	0
	.long	862978048
	.type	__dpow_la_CoutTab,@object
	.size	__dpow_la_CoutTab,6880
