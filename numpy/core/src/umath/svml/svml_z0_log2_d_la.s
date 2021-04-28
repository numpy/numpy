/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *  log2(x) = VGETEXP(x) + log2(VGETMANT(x))
 *  *       VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *  *   mx = VGETMANT(x) is in [1,2) for all x>=0
 *  *   log2(mx) = -log2(RCP(mx)) + log2(1 +(mx*RCP(mx)-1))
 *  *      RCP(mx) is rounded to 4 fractional bits,
 *  *      and the table lookup for log(RCP(mx)) is based on a small permute instruction
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_log28_z0_0:

	.align    16,0x90
	.globl __svml_log28

__svml_log28:


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
        vmovups   128+__svml_dlog2_data_internal_avx512(%rip), %zmm2
        vmovups   512+__svml_dlog2_data_internal_avx512(%rip), %zmm12
        vmovups   640+__svml_dlog2_data_internal_avx512(%rip), %zmm13

/* Start polynomial evaluation */
        vmovups   256+__svml_dlog2_data_internal_avx512(%rip), %zmm10
        vmovups   320+__svml_dlog2_data_internal_avx512(%rip), %zmm0
        vmovups   384+__svml_dlog2_data_internal_avx512(%rip), %zmm11
        vmovups   448+__svml_dlog2_data_internal_avx512(%rip), %zmm14

/* Prepare exponent correction: DblRcp<0.75? */
        vmovups   192+__svml_dlog2_data_internal_avx512(%rip), %zmm1

/* Table lookup */
        vmovups   __svml_dlog2_data_internal_avx512(%rip), %zmm4

/* GetExp(x) */
        vgetexppd {sae}, %zmm7, %zmm5

/* DblRcp ~ 1/Mantissa */
        vrcp14pd  %zmm6, %zmm8

/* x<=0? */
        vfpclasspd $94, %zmm7, %k0

/* round DblRcp to 4 fractional bits (RN mode, no Precision exception) */
        vrndscalepd $88, {sae}, %zmm8, %zmm3
        vmovups   576+__svml_dlog2_data_internal_avx512(%rip), %zmm8
        kmovw     %k0, %edx

/* Reduced argument: R = DblRcp*Mantissa - 1 */
        vfmsub213pd {rn-sae}, %zmm2, %zmm3, %zmm6
        vcmppd    $17, {sae}, %zmm1, %zmm3, %k1
        vfmadd231pd {rn-sae}, %zmm6, %zmm12, %zmm8
        vmovups   704+__svml_dlog2_data_internal_avx512(%rip), %zmm12
        vfmadd231pd {rn-sae}, %zmm6, %zmm10, %zmm0
        vfmadd231pd {rn-sae}, %zmm6, %zmm11, %zmm14
        vmovups   768+__svml_dlog2_data_internal_avx512(%rip), %zmm1

/* R^2 */
        vmulpd    {rn-sae}, %zmm6, %zmm6, %zmm15
        vfmadd231pd {rn-sae}, %zmm6, %zmm13, %zmm12

/* Prepare table index */
        vpsrlq    $48, %zmm3, %zmm9

/* add 1 to Expon if DblRcp<0.75 */
        vaddpd    {rn-sae}, %zmm2, %zmm5, %zmm5{%k1}
        vmulpd    {rn-sae}, %zmm15, %zmm15, %zmm13
        vfmadd213pd {rn-sae}, %zmm14, %zmm15, %zmm0
        vfmadd213pd {rn-sae}, %zmm12, %zmm15, %zmm8
        vpermt2pd 64+__svml_dlog2_data_internal_avx512(%rip), %zmm9, %zmm4

/* polynomial */
        vfmadd213pd {rn-sae}, %zmm8, %zmm13, %zmm0
        vfmadd213pd {rn-sae}, %zmm1, %zmm6, %zmm0
        vfmadd213pd {rn-sae}, %zmm4, %zmm0, %zmm6
        vaddpd    {rn-sae}, %zmm6, %zmm5, %zmm0
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

        call      __svml_dlog2_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log28,@function
	.size	__svml_log28,.-__svml_log28
..LN__svml_log28.0:

.L_2__routine_start___svml_dlog2_cout_rare_internal_1:

	.align    16,0x90

__svml_dlog2_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      $0, -64(%rsp)
        movsd     -64(%rsp), %xmm0
        movsd     (%rdi), %xmm1
        movups    _zeros.402.0.0.1(%rip), %xmm2
        subsd     %xmm0, %xmm1
        movsd     %xmm1, -8(%rsp)
        movups    %xmm2, -56(%rsp)


        movups    _ones.402.0.0.1(%rip), %xmm0
        movups    %xmm0, -40(%rsp)


        movups    _infs.402.0.0.1(%rip), %xmm0
        movups    %xmm0, -24(%rsp)


        movl      -4(%rsp), %eax
        cmpl      $2146435072, %eax
        jae       .LBL_2_19


        lea       -1072660480(%rax), %edx
        cmpl      $53248, %edx
        jae       .LBL_2_14


        lea       -1072693232(%rax), %edx
        cmpl      $24, %edx
        jae       .LBL_2_13


        movl      -8(%rsp), %edi
        movl      %eax, %ecx
        movl      %edi, %edx
        shll      $11, %ecx
        shrl      $21, %edx
        orl       %edx, %ecx
        addl      $-2147483646, %ecx
        cmpl      $3, %ecx
        jae       .LBL_2_12


        addl      $-1072693248, %eax
        orl       %edi, %eax
        jne       .LBL_2_10


        movq      -56(%rsp), %rax
        movq      %rax, (%rsi)
        jmp       .LBL_2_11

.LBL_2_10:

        movsd     -8(%rsp), %xmm0
        movsd     16+__dlog2_la__Q3(%rip), %xmm2
        movsd     24+__dlog2_la__Q3(%rip), %xmm1
        addsd     -32(%rsp), %xmm0
        mulsd     %xmm0, %xmm2
        mulsd     %xmm0, %xmm1
        addsd     8+__dlog2_la__Q3(%rip), %xmm2
        mulsd     %xmm0, %xmm2
        movsd     %xmm0, -8(%rsp)
        addsd     __dlog2_la__Q3(%rip), %xmm2
        mulsd     %xmm0, %xmm2
        addsd     %xmm1, %xmm2
        movsd     %xmm2, (%rsi)

.LBL_2_11:

        xorl      %eax, %eax
        ret

.LBL_2_12:

        movsd     -8(%rsp), %xmm2
        xorl      %eax, %eax
        movsd     16+__dlog2_la__Q2(%rip), %xmm1
        movsd     24+__dlog2_la__Q2(%rip), %xmm3
        addsd     -32(%rsp), %xmm2
        movaps    %xmm2, %xmm0
        mulsd     %xmm2, %xmm0
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm3
        addsd     __dlog2_la__Q2(%rip), %xmm1
        addsd     8+__dlog2_la__Q2(%rip), %xmm3
        mulsd     %xmm2, %xmm1
        mulsd     %xmm0, %xmm3
        movsd     %xmm2, -8(%rsp)
        addsd     %xmm1, %xmm3
        movl      $0, -8(%rsp)
        movsd     -8(%rsp), %xmm4
        subsd     %xmm4, %xmm2
        addsd     %xmm2, %xmm4
        mulsd     32+__dlog2_la__Q2(%rip), %xmm4
        addsd     %xmm3, %xmm4
        movsd     %xmm4, (%rsi)
        ret

.LBL_2_13:

        movsd     -8(%rsp), %xmm2
        xorl      %eax, %eax
        movsd     72+__dlog2_la__Q1(%rip), %xmm10
        movsd     64+__dlog2_la__Q1(%rip), %xmm4
        movsd     __dlog2_la__TWO_32P(%rip), %xmm5
        movsd     __dlog2_la__TWO_32P(%rip), %xmm7
        movsd     88+__dlog2_la__Q1(%rip), %xmm14
        movsd     __dlog2_la__TWO_32P(%rip), %xmm11
        movsd     __dlog2_la__TWO_32P(%rip), %xmm15
        addsd     -32(%rsp), %xmm2
        movaps    %xmm2, %xmm3
        movaps    %xmm2, %xmm6
        mulsd     %xmm2, %xmm3
        movaps    %xmm2, %xmm0
        mulsd     %xmm2, %xmm5
        mulsd     __dlog2_la__TWO_32(%rip), %xmm6
        mulsd     %xmm3, %xmm10
        mulsd     %xmm3, %xmm4
        addsd     56+__dlog2_la__Q1(%rip), %xmm10
        addsd     48+__dlog2_la__Q1(%rip), %xmm4
        mulsd     %xmm3, %xmm10
        mulsd     %xmm3, %xmm4
        addsd     40+__dlog2_la__Q1(%rip), %xmm10
        addsd     32+__dlog2_la__Q1(%rip), %xmm4
        mulsd     %xmm3, %xmm10
        mulsd     %xmm3, %xmm4
        addsd     24+__dlog2_la__Q1(%rip), %xmm10
        addsd     16+__dlog2_la__Q1(%rip), %xmm4
        mulsd     %xmm3, %xmm10
        mulsd     %xmm2, %xmm4
        movsd     __dlog2_la__TWO_32(%rip), %xmm3
        addsd     %xmm4, %xmm10
        mulsd     %xmm10, %xmm7
        movaps    %xmm3, %xmm8
        mulsd     %xmm10, %xmm8
        movsd     %xmm5, -72(%rsp)
        movsd     -72(%rsp), %xmm1
        movsd     %xmm7, -72(%rsp)
        subsd     %xmm6, %xmm1
        movsd     -72(%rsp), %xmm9
        subsd     %xmm1, %xmm0
        subsd     %xmm8, %xmm9
        movsd     %xmm1, -8(%rsp)
        subsd     %xmm9, %xmm10
        addsd     %xmm9, %xmm14
        addsd     8+__dlog2_la__Q1(%rip), %xmm10
        movaps    %xmm14, %xmm4
        mulsd     %xmm0, %xmm4
        mulsd     %xmm2, %xmm10
        mulsd     %xmm1, %xmm14
        addsd     %xmm0, %xmm1
        addsd     %xmm10, %xmm4
        movaps    %xmm4, %xmm12
        movsd     80+__dlog2_la__Q1(%rip), %xmm9
        addsd     %xmm14, %xmm12
        mulsd     %xmm12, %xmm11
        mulsd     %xmm3, %xmm12
        movsd     %xmm11, -72(%rsp)
        movsd     -72(%rsp), %xmm13
        subsd     %xmm12, %xmm13
        subsd     %xmm13, %xmm4
        addsd     %xmm13, %xmm9
        addsd     %xmm14, %xmm4
        movaps    %xmm9, %xmm13
        addsd     __dlog2_la__Q1(%rip), %xmm4
        addsd     %xmm4, %xmm13
        mulsd     %xmm13, %xmm15
        mulsd     %xmm3, %xmm13
        movsd     %xmm15, -72(%rsp)
        movsd     -72(%rsp), %xmm14
        subsd     %xmm13, %xmm14
        mulsd     %xmm14, %xmm1
        subsd     %xmm14, %xmm9
        addsd     %xmm9, %xmm4
        mulsd     %xmm2, %xmm4
        addsd     %xmm4, %xmm1
        movsd     %xmm1, (%rsi)
        ret

.LBL_2_14:

        movl      %eax, %ecx
        movl      %eax, %edx
        shrl      $20, %ecx
        andl      $1048575, %edx
        addl      $-1023, %ecx
        cmpl      $1048576, %eax
        jae       .LBL_2_18


        movl      %edx, -4(%rsp)
        movl      -8(%rsp), %edx
        orl       %edx, %eax
        jne       .LBL_2_17


        movsd     -32(%rsp), %xmm0
        movl      $2, %eax
        divsd     -56(%rsp), %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_17:

        movsd     -8(%rsp), %xmm0
        mulsd     __dlog2_la__TWO_55(%rip), %xmm0
        movsd     %xmm0, -8(%rsp)
        movl      -4(%rsp), %edx
        movl      %edx, %ecx
        shrl      $20, %ecx
        movl      %edx, %eax
        addl      $-1078, %ecx

.LBL_2_18:

        shrl      $12, %eax
        orl       $1072693248, %edx
        movzbl    %al, %edi
        pxor      %xmm7, %xmm7
        movl      %edx, -4(%rsp)
        lea       __dlog2_la___libm_rcp_table_256(%rip), %rdx
        movsd     -8(%rsp), %xmm10
        pxor      %xmm8, %xmm8
        movl      $0, -8(%rsp)
        lea       __dlog2_la___libm_log2_table_256(%rip), %rax
        movsd     -8(%rsp), %xmm0
        cvtss2sd  (%rdx,%rdi,4), %xmm7
        cvtsi2sd  %ecx, %xmm8
        subsd     %xmm0, %xmm10
        mulsd     %xmm7, %xmm10
        mulsd     %xmm0, %xmm7
        movsd     40+__dlog2_la__P(%rip), %xmm4
        movsd     32+__dlog2_la__P(%rip), %xmm3
        movsd     48+__dlog2_la__P(%rip), %xmm5
        movsd     __dlog2_la__TWO_32(%rip), %xmm6
        shlq      $4, %rdi
        addsd     -32(%rsp), %xmm7
        movaps    %xmm7, %xmm2
        mulsd     %xmm5, %xmm7
        addsd     %xmm10, %xmm2
        mulsd     %xmm5, %xmm10
        addsd     %xmm7, %xmm6
        movaps    %xmm2, %xmm1
        mulsd     %xmm2, %xmm1
        mulsd     %xmm1, %xmm4
        mulsd     %xmm1, %xmm3
        addsd     24+__dlog2_la__P(%rip), %xmm4
        addsd     16+__dlog2_la__P(%rip), %xmm3
        mulsd     %xmm1, %xmm4
        mulsd     %xmm1, %xmm3
        addsd     8+__dlog2_la__P(%rip), %xmm4
        addsd     __dlog2_la__P(%rip), %xmm3
        mulsd     %xmm1, %xmm4
        mulsd     %xmm2, %xmm3
        movsd     %xmm6, -72(%rsp)
        addsd     %xmm3, %xmm4
        addsd     8(%rax,%rdi), %xmm8
        addsd     %xmm4, %xmm10
        movsd     -72(%rsp), %xmm9
        subsd     __dlog2_la__TWO_32(%rip), %xmm9
        subsd     %xmm9, %xmm7
        addsd     %xmm8, %xmm9
        addsd     (%rax,%rdi), %xmm7
        movsd     %xmm9, -8(%rsp)
        xorl      %eax, %eax
        addsd     %xmm7, %xmm10
        addsd     %xmm9, %xmm10
        movsd     %xmm10, (%rsi)
        ret

.LBL_2_19:

        movl      %eax, %edx
        andl      $2147483647, %edx
        cmpl      $2146435072, %edx
        ja        .LBL_2_28


        jne       .LBL_2_22


        cmpl      $0, -8(%rsp)
        jne       .LBL_2_28

.LBL_2_22:

        testl     $-2147483648, %eax
        je        .LBL_2_27


        movl      -8(%rsp), %eax
        orl       %eax, %edx
        movsd     -56(%rsp), %xmm1
        jne       .LBL_2_25


        movsd     -32(%rsp), %xmm0
        movl      $2, %eax
        divsd     %xmm1, %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_25:

        movsd     -24(%rsp), %xmm0
        movl      $1, %eax
        mulsd     %xmm1, %xmm0
        movsd     %xmm0, (%rsi)


        ret

.LBL_2_27:

        movq      -8(%rsp), %rax
        movq      %rax, (%rsi)
        xorl      %eax, %eax
        ret

.LBL_2_28:

        movsd     -8(%rsp), %xmm0
        xorl      %eax, %eax
        mulsd     -40(%rsp), %xmm0
        movsd     %xmm0, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dlog2_cout_rare_internal,@function
	.size	__svml_dlog2_cout_rare_internal,.-__svml_dlog2_cout_rare_internal
..LN__svml_dlog2_cout_rare_internal.1:
	.section .rodata, "a"
	.align 64
	.align 16
_zeros.402.0.0.1:
	.long	0
	.long	0
	.long	0
	.long	2147483648
	.align 16
_ones.402.0.0.1:
	.long	0
	.long	1072693248
	.long	0
	.long	3220176896
	.align 16
_infs.402.0.0.1:
	.long	0
	.long	2146435072
	.long	0
	.long	4293918720

	.section .rodata, "a"
	.space 16, 0x00 	
	.align 64
__svml_dlog2_data_internal_avx512:
	.long	0
	.long	0
	.long	4207481622
	.long	3216401398
	.long	972805768
	.long	3217408026
	.long	3103942666
	.long	3218062358
	.long	1271733131
	.long	3218381432
	.long	2300516105
	.long	3218676666
	.long	3761433103
	.long	3218958163
	.long	14039718
	.long	3219177733
	.long	1904282206
	.long	1071288313
	.long	1751501034
	.long	1071041295
	.long	3815829096
	.long	1070803966
	.long	2835758645
	.long	1070555096
	.long	180337970
	.long	1070114968
	.long	3266000023
	.long	1069690285
	.long	2530196300
	.long	1069012484
	.long	3386464469
	.long	1067938708
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
	.long	3658358034
	.long	1069846603
	.long	3658358034
	.long	1069846603
	.long	3658358034
	.long	1069846603
	.long	3658358034
	.long	1069846603
	.long	3658358034
	.long	1069846603
	.long	3658358034
	.long	1069846603
	.long	3658358034
	.long	1069846603
	.long	3658358034
	.long	1069846603
	.long	1307270350
	.long	3217498040
	.long	1307270350
	.long	3217498040
	.long	1307270350
	.long	3217498040
	.long	1307270350
	.long	3217498040
	.long	1307270350
	.long	3217498040
	.long	1307270350
	.long	3217498040
	.long	1307270350
	.long	3217498040
	.long	1307270350
	.long	3217498040
	.long	1367442963
	.long	1070227827
	.long	1367442963
	.long	1070227827
	.long	1367442963
	.long	1070227827
	.long	1367442963
	.long	1070227827
	.long	1367442963
	.long	1070227827
	.long	1367442963
	.long	1070227827
	.long	1367442963
	.long	1070227827
	.long	1367442963
	.long	1070227827
	.long	3840087116
	.long	3217999623
	.long	3840087116
	.long	3217999623
	.long	3840087116
	.long	3217999623
	.long	3840087116
	.long	3217999623
	.long	3840087116
	.long	3217999623
	.long	3840087116
	.long	3217999623
	.long	3840087116
	.long	3217999623
	.long	3840087116
	.long	3217999623
	.long	1360320794
	.long	1070757740
	.long	1360320794
	.long	1070757740
	.long	1360320794
	.long	1070757740
	.long	1360320794
	.long	1070757740
	.long	1360320794
	.long	1070757740
	.long	1360320794
	.long	1070757740
	.long	1360320794
	.long	1070757740
	.long	1360320794
	.long	1070757740
	.long	1698500493
	.long	3218543943
	.long	1698500493
	.long	3218543943
	.long	1698500493
	.long	3218543943
	.long	1698500493
	.long	3218543943
	.long	1698500493
	.long	3218543943
	.long	1698500493
	.long	3218543943
	.long	1698500493
	.long	3218543943
	.long	1698500493
	.long	3218543943
	.long	3694789279
	.long	1071564553
	.long	3694789279
	.long	1071564553
	.long	3694789279
	.long	1071564553
	.long	3694789279
	.long	1071564553
	.long	3694789279
	.long	1071564553
	.long	3694789279
	.long	1071564553
	.long	3694789279
	.long	1071564553
	.long	3694789279
	.long	1071564553
	.long	1697350356
	.long	3219592519
	.long	1697350356
	.long	3219592519
	.long	1697350356
	.long	3219592519
	.long	1697350356
	.long	3219592519
	.long	1697350356
	.long	3219592519
	.long	1697350356
	.long	3219592519
	.long	1697350356
	.long	3219592519
	.long	1697350356
	.long	3219592519
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
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
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
	.type	__svml_dlog2_data_internal_avx512,@object
	.size	__svml_dlog2_data_internal_avx512,1088
	.align 64
__dlog2_la__Q3:
	.long	3213221364
	.long	1050233568
	.long	1697350398
	.long	3219592519
	.long	3694789629
	.long	1071564553
	.long	0
	.long	1073157447
	.type	__dlog2_la__Q3,@object
	.size	__dlog2_la__Q3,32
	.space 32, 0x00 	
	.align 64
__dlog2_la__Q2:
	.long	3213221358
	.long	1050233568
	.long	1697350398
	.long	3219592519
	.long	3695167823
	.long	1071564553
	.long	1697791998
	.long	3218543943
	.long	0
	.long	1073157447
	.type	__dlog2_la__Q2,@object
	.size	__dlog2_la__Q2,40
	.space 24, 0x00 	
	.align 64
__dlog2_la__Q1:
	.long	3213221370
	.long	1050233568
	.long	3213207734
	.long	3196668640
	.long	3694789628
	.long	1071564553
	.long	1697350743
	.long	3218543943
	.long	1357908666
	.long	1070757740
	.long	3685007067
	.long	3217999625
	.long	96832070
	.long	1070227830
	.long	1140452344
	.long	3217495379
	.long	2606274745
	.long	1069844296
	.long	1933654871
	.long	3217172449
	.long	0
	.long	1073157447
	.long	0
	.long	3219592519
	.type	__dlog2_la__Q1,@object
	.size	__dlog2_la__Q1,96
	.space 32, 0x00 	
	.align 64
__dlog2_la__TWO_32P:
	.long	1048576
	.long	1106247680
	.type	__dlog2_la__TWO_32P,@object
	.size	__dlog2_la__TWO_32P,8
	.space 56, 0x00 	
	.align 64
__dlog2_la__TWO_32:
	.long	0
	.long	1106247680
	.type	__dlog2_la__TWO_32,@object
	.size	__dlog2_la__TWO_32,8
	.space 56, 0x00 	
	.align 64
__dlog2_la__TWO_55:
	.long	0
	.long	1130364928
	.type	__dlog2_la__TWO_55,@object
	.size	__dlog2_la__TWO_55,8
	.space 56, 0x00 	
	.align 64
__dlog2_la___libm_rcp_table_256:
	.long	0x3f7f8000
	.long	0x3f7e8000
	.long	0x3f7d8000
	.long	0x3f7c8000
	.long	0x3f7ba000
	.long	0x3f7aa000
	.long	0x3f79a000
	.long	0x3f78c000
	.long	0x3f77c000
	.long	0x3f76e000
	.long	0x3f75e000
	.long	0x3f750000
	.long	0x3f742000
	.long	0x3f732000
	.long	0x3f724000
	.long	0x3f716000
	.long	0x3f708000
	.long	0x3f6fa000
	.long	0x3f6ec000
	.long	0x3f6de000
	.long	0x3f6d0000
	.long	0x3f6c2000
	.long	0x3f6b6000
	.long	0x3f6a8000
	.long	0x3f69a000
	.long	0x3f68c000
	.long	0x3f680000
	.long	0x3f672000
	.long	0x3f666000
	.long	0x3f658000
	.long	0x3f64c000
	.long	0x3f640000
	.long	0x3f632000
	.long	0x3f626000
	.long	0x3f61a000
	.long	0x3f60e000
	.long	0x3f600000
	.long	0x3f5f4000
	.long	0x3f5e8000
	.long	0x3f5dc000
	.long	0x3f5d0000
	.long	0x3f5c4000
	.long	0x3f5b8000
	.long	0x3f5ae000
	.long	0x3f5a2000
	.long	0x3f596000
	.long	0x3f58a000
	.long	0x3f57e000
	.long	0x3f574000
	.long	0x3f568000
	.long	0x3f55e000
	.long	0x3f552000
	.long	0x3f546000
	.long	0x3f53c000
	.long	0x3f532000
	.long	0x3f526000
	.long	0x3f51c000
	.long	0x3f510000
	.long	0x3f506000
	.long	0x3f4fc000
	.long	0x3f4f2000
	.long	0x3f4e6000
	.long	0x3f4dc000
	.long	0x3f4d2000
	.long	0x3f4c8000
	.long	0x3f4be000
	.long	0x3f4b4000
	.long	0x3f4aa000
	.long	0x3f4a0000
	.long	0x3f496000
	.long	0x3f48c000
	.long	0x3f482000
	.long	0x3f478000
	.long	0x3f46e000
	.long	0x3f464000
	.long	0x3f45c000
	.long	0x3f452000
	.long	0x3f448000
	.long	0x3f43e000
	.long	0x3f436000
	.long	0x3f42c000
	.long	0x3f422000
	.long	0x3f41a000
	.long	0x3f410000
	.long	0x3f408000
	.long	0x3f3fe000
	.long	0x3f3f6000
	.long	0x3f3ec000
	.long	0x3f3e4000
	.long	0x3f3da000
	.long	0x3f3d2000
	.long	0x3f3ca000
	.long	0x3f3c0000
	.long	0x3f3b8000
	.long	0x3f3b0000
	.long	0x3f3a8000
	.long	0x3f39e000
	.long	0x3f396000
	.long	0x3f38e000
	.long	0x3f386000
	.long	0x3f37e000
	.long	0x3f376000
	.long	0x3f36c000
	.long	0x3f364000
	.long	0x3f35c000
	.long	0x3f354000
	.long	0x3f34c000
	.long	0x3f344000
	.long	0x3f33c000
	.long	0x3f334000
	.long	0x3f32e000
	.long	0x3f326000
	.long	0x3f31e000
	.long	0x3f316000
	.long	0x3f30e000
	.long	0x3f306000
	.long	0x3f2fe000
	.long	0x3f2f8000
	.long	0x3f2f0000
	.long	0x3f2e8000
	.long	0x3f2e2000
	.long	0x3f2da000
	.long	0x3f2d2000
	.long	0x3f2cc000
	.long	0x3f2c4000
	.long	0x3f2bc000
	.long	0x3f2b6000
	.long	0x3f2ae000
	.long	0x3f2a8000
	.long	0x3f2a0000
	.long	0x3f29a000
	.long	0x3f292000
	.long	0x3f28c000
	.long	0x3f284000
	.long	0x3f27e000
	.long	0x3f276000
	.long	0x3f270000
	.long	0x3f268000
	.long	0x3f262000
	.long	0x3f25c000
	.long	0x3f254000
	.long	0x3f24e000
	.long	0x3f248000
	.long	0x3f240000
	.long	0x3f23a000
	.long	0x3f234000
	.long	0x3f22e000
	.long	0x3f226000
	.long	0x3f220000
	.long	0x3f21a000
	.long	0x3f214000
	.long	0x3f20e000
	.long	0x3f206000
	.long	0x3f200000
	.long	0x3f1fa000
	.long	0x3f1f4000
	.long	0x3f1ee000
	.long	0x3f1e8000
	.long	0x3f1e2000
	.long	0x3f1dc000
	.long	0x3f1d6000
	.long	0x3f1d0000
	.long	0x3f1ca000
	.long	0x3f1c4000
	.long	0x3f1be000
	.long	0x3f1b8000
	.long	0x3f1b2000
	.long	0x3f1ac000
	.long	0x3f1a6000
	.long	0x3f1a0000
	.long	0x3f19a000
	.long	0x3f194000
	.long	0x3f190000
	.long	0x3f18a000
	.long	0x3f184000
	.long	0x3f17e000
	.long	0x3f178000
	.long	0x3f172000
	.long	0x3f16e000
	.long	0x3f168000
	.long	0x3f162000
	.long	0x3f15c000
	.long	0x3f158000
	.long	0x3f152000
	.long	0x3f14c000
	.long	0x3f148000
	.long	0x3f142000
	.long	0x3f13c000
	.long	0x3f138000
	.long	0x3f132000
	.long	0x3f12c000
	.long	0x3f128000
	.long	0x3f122000
	.long	0x3f11c000
	.long	0x3f118000
	.long	0x3f112000
	.long	0x3f10e000
	.long	0x3f108000
	.long	0x3f104000
	.long	0x3f0fe000
	.long	0x3f0f8000
	.long	0x3f0f4000
	.long	0x3f0ee000
	.long	0x3f0ea000
	.long	0x3f0e6000
	.long	0x3f0e0000
	.long	0x3f0dc000
	.long	0x3f0d6000
	.long	0x3f0d2000
	.long	0x3f0cc000
	.long	0x3f0c8000
	.long	0x3f0c2000
	.long	0x3f0be000
	.long	0x3f0ba000
	.long	0x3f0b4000
	.long	0x3f0b0000
	.long	0x3f0ac000
	.long	0x3f0a6000
	.long	0x3f0a2000
	.long	0x3f09e000
	.long	0x3f098000
	.long	0x3f094000
	.long	0x3f090000
	.long	0x3f08a000
	.long	0x3f086000
	.long	0x3f082000
	.long	0x3f07e000
	.long	0x3f078000
	.long	0x3f074000
	.long	0x3f070000
	.long	0x3f06c000
	.long	0x3f066000
	.long	0x3f062000
	.long	0x3f05e000
	.long	0x3f05a000
	.long	0x3f056000
	.long	0x3f052000
	.long	0x3f04c000
	.long	0x3f048000
	.long	0x3f044000
	.long	0x3f040000
	.long	0x3f03c000
	.long	0x3f038000
	.long	0x3f034000
	.long	0x3f030000
	.long	0x3f02a000
	.long	0x3f026000
	.long	0x3f022000
	.long	0x3f01e000
	.long	0x3f01a000
	.long	0x3f016000
	.long	0x3f012000
	.long	0x3f00e000
	.long	0x3f00a000
	.long	0x3f006000
	.long	0x3f002000
	.type	__dlog2_la___libm_rcp_table_256,@object
	.size	__dlog2_la___libm_rcp_table_256,1024
	.align 64
__dlog2_la___libm_log2_table_256:
	.long	0xfb44c3b7,0x3e1485cb
	.long	0x00000000,0x3f671b0e
	.long	0x06028ac0,0x3e31d5d9
	.long	0x00000000,0x3f815cfe
	.long	0xb8d7240b,0x3df8b9cb
	.long	0x00000000,0x3f8cfee7
	.long	0x0d179106,0x3e38864a
	.long	0x00000000,0x3f94564a
	.long	0xecba1593,0x3e459c6a
	.long	0x00000000,0x3f997723
	.long	0x94120c14,0x3e48d36a
	.long	0x00000000,0x3f9f5923
	.long	0xd2571490,0x3e5410ba
	.long	0x00000000,0x3fa2a094
	.long	0x1dc036a2,0x3e2776b0
	.long	0x00000000,0x3fa53894
	.long	0x78efe2b1,0x3e537229
	.long	0x00000000,0x3fa8324c
	.long	0xfd29dc75,0x3e59c0fa
	.long	0x00000000,0x3faacf54
	.long	0x046734f7,0x3e4636b7
	.long	0x00000000,0x3fadced9
	.long	0xd3b410b8,0x3e6f7950
	.long	0x00000000,0x3fb0387e
	.long	0x00f2200a,0x3e19d1e7
	.long	0x00000000,0x3fb18ac6
	.long	0x8661ba82,0x3e4f09a9
	.long	0x00000000,0x3fb30edd
	.long	0x9367107c,0x3e564d91
	.long	0x00000000,0x3fb463c1
	.long	0x0e4a4ce8,0x3e4e1fd1
	.long	0x00000000,0x3fb5b9e1
	.long	0x3cdb6374,0x3e492cf0
	.long	0x00000000,0x3fb7113f
	.long	0x069c4f7f,0x3e61a364
	.long	0x00000000,0x3fb869dd
	.long	0x427b631b,0x3e6493a6
	.long	0x00000000,0x3fb9c3be
	.long	0xe02b3e8b,0x3e6af2c2
	.long	0x00000000,0x3fbb1ee4
	.long	0x389f4365,0x3e616e1e
	.long	0x00000000,0x3fbc7b52
	.long	0x6a31fd96,0x3e4633b7
	.long	0x00000000,0x3fbdd90a
	.long	0x0508664d,0x3e62ed84
	.long	0x00000000,0x3fbf05d4
	.long	0xaca1905c,0x3e775dcd
	.long	0x00000000,0x3fc032fb
	.long	0x094fbeeb,0x3e753e65
	.long	0x00000000,0x3fc0e3b5
	.long	0x96aa4b17,0x3e671f44
	.long	0x00000000,0x3fc19519
	.long	0x92da5a47,0x3e785566
	.long	0x00000000,0x3fc22dad
	.long	0xbeb7d722,0x3e518efa
	.long	0x00000000,0x3fc2e050
	.long	0xab57551c,0x3e738564
	.long	0x00000000,0x3fc379f7
	.long	0x54a914e3,0x3e55d0da
	.long	0x00000000,0x3fc42ddd
	.long	0xfe974017,0x3e73cd00
	.long	0x00000000,0x3fc4c89b
	.long	0x59064390,0x3e54ffd6
	.long	0x00000000,0x3fc563dc
	.long	0x633ab50f,0x3e67d75e
	.long	0x00000000,0x3fc619a2
	.long	0xc8877e8a,0x3e77e6ce
	.long	0x00000000,0x3fc6b5ff
	.long	0x1ab7837f,0x3e7ecc1f
	.long	0x00000000,0x3fc752e1
	.long	0xf9d5827a,0x3e7cea7c
	.long	0x00000000,0x3fc7f049
	.long	0x64ccd537,0x3e357f7a
	.long	0x00000000,0x3fc8a898
	.long	0xf7c9b05b,0x3e7994ca
	.long	0x00000000,0x3fc94724
	.long	0xa2f56536,0x3e524b8f
	.long	0x00000000,0x3fc9e63a
	.long	0x5edaab42,0x3e7fd640
	.long	0x00000000,0x3fca85d8
	.long	0xd163379a,0x3e625f54
	.long	0x00000000,0x3fcb2602
	.long	0x936acd51,0x3e7ebdc3
	.long	0x00000000,0x3fcbc6b6
	.long	0xcfbc0aa0,0x3e7eee14
	.long	0x00000000,0x3fcc67f7
	.long	0xf73bcdad,0x3e764469
	.long	0x00000000,0x3fcceec4
	.long	0x60971b86,0x3e6eb44e
	.long	0x00000000,0x3fcd9109
	.long	0xcd2052a5,0x3e65fcf4
	.long	0x00000000,0x3fce33dd
	.long	0xc402867b,0x3e61af1e
	.long	0x00000000,0x3fced741
	.long	0xa0c956e4,0x3e61bfbd
	.long	0x00000000,0x3fcf7b36
	.long	0x831e77ff,0x3e85287b
	.long	0x00000000,0x3fd00223
	.long	0xaddfdee2,0x3e7d2fc3
	.long	0x00000000,0x3fd054a4
	.long	0x342052c1,0x3e83724b
	.long	0x00000000,0x3fd0999d
	.long	0x602bcd34,0x3e7b4ec9
	.long	0x00000000,0x3fd0eca6
	.long	0x3aa20ead,0x3e6742da
	.long	0x00000000,0x3fd13ffa
	.long	0x9ecdadf4,0x3e713e82
	.long	0x00000000,0x3fd185a4
	.long	0xfef3031b,0x3e52f27e
	.long	0x00000000,0x3fd1cb83
	.long	0x79e4af8a,0x3e710739
	.long	0x00000000,0x3fd21fa1
	.long	0xe59ad84a,0x3e637301
	.long	0x00000000,0x3fd265f5
	.long	0x3d7dfd9b,0x3e88697c
	.long	0x00000000,0x3fd2baa0
	.long	0x738117b0,0x3e717788
	.long	0x00000000,0x3fd3016b
	.long	0xd3c26a97,0x3e6c5514
	.long	0x00000000,0x3fd3486c
	.long	0x4c4ff246,0x3e8df550
	.long	0x00000000,0x3fd38fa3
	.long	0x40340fa6,0x3e88102d
	.long	0x00000000,0x3fd3e562
	.long	0x4592f4c3,0x3e5f53b6
	.long	0x00000000,0x3fd42d14
	.long	0x8b149a00,0x3e750fc8
	.long	0x00000000,0x3fd474fd
	.long	0xa8f50e5f,0x3e86d01c
	.long	0x00000000,0x3fd4bd1e
	.long	0x7a22a88a,0x3e83c469
	.long	0x00000000,0x3fd50578
	.long	0xdc18b6d2,0x3e79000e
	.long	0x00000000,0x3fd54e0b
	.long	0x7c00250b,0x3e7870f0
	.long	0x00000000,0x3fd596d7
	.long	0xc1c885ae,0x3e8e3dd5
	.long	0x00000000,0x3fd5dfdc
	.long	0xa6ecc47e,0x3e7bf64c
	.long	0x00000000,0x3fd6291c
	.long	0x0bc16c18,0x3e6bdaca
	.long	0x00000000,0x3fd67296
	.long	0x1f925729,0x3e84d25c
	.long	0x00000000,0x3fd6bc4a
	.long	0xf23978de,0x3e5a5fb4
	.long	0x00000000,0x3fd7063a
	.long	0xe2ac7f60,0x3e8e3da0
	.long	0x00000000,0x3fd75064
	.long	0x71f51a7b,0x3e819e20
	.long	0x00000000,0x3fd79acb
	.long	0x5b924234,0x3e5e43ae
	.long	0x00000000,0x3fd7d67c
	.long	0xe9aedf37,0x3e701a2b
	.long	0x00000000,0x3fd8214f
	.long	0xeddd33ea,0x3e6b6f51
	.long	0x00000000,0x3fd86c5f
	.long	0xe732b3c4,0x3e79375f
	.long	0x00000000,0x3fd8b7ac
	.long	0xe47cb9df,0x3e7e831b
	.long	0x00000000,0x3fd8f416
	.long	0xe392d3c8,0x3e8abc37
	.long	0x00000000,0x3fd93fd2
	.long	0x28b35c77,0x3e80852d
	.long	0x00000000,0x3fd98bcd
	.long	0xd29cea36,0x3e69f374
	.long	0x00000000,0x3fd9c8c3
	.long	0xb3d7b0e6,0x3e542981
	.long	0x00000000,0x3fda152f
	.long	0x1fe6d5ac,0x3e8b2bfb
	.long	0x00000000,0x3fda527f
	.long	0xf32e5dda,0x3e6f6f5c
	.long	0x00000000,0x3fda9f5e
	.long	0xd34d0d30,0x3e65abaa
	.long	0x00000000,0x3fdadd0b
	.long	0xe0ac9602,0x3e7bd47f
	.long	0x00000000,0x3fdb2a5d
	.long	0x7c5c784b,0x3e833601
	.long	0x00000000,0x3fdb6867
	.long	0xd7b0821f,0x3e5b887c
	.long	0x00000000,0x3fdbb62f
	.long	0xcd0a8f6a,0x3e83eccc
	.long	0x00000000,0x3fdbf497
	.long	0x7500a4e8,0x3e7ba0e6
	.long	0x00000000,0x3fdc332a
	.long	0xfc8712bb,0x3e885a8b
	.long	0x00000000,0x3fdc819d
	.long	0xf7a3a78f,0x3e81c336
	.long	0x00000000,0x3fdcc090
	.long	0x4ad8a38b,0x3e7846b4
	.long	0x00000000,0x3fdcffae
	.long	0xfcfc3a99,0x3e7db50f
	.long	0x00000000,0x3fdd3ef7
	.long	0x433cccd2,0x3e238809
	.long	0x00000000,0x3fdd8e50
	.long	0x6828fa82,0x3e721b2e
	.long	0x00000000,0x3fddcdfb
	.long	0xfc7c49c1,0x3e82848b
	.long	0x00000000,0x3fde0dd2
	.long	0x1fce4d49,0x3e615144
	.long	0x00000000,0x3fde4dd6
	.long	0x77e2e8fd,0x3e68b88a
	.long	0x00000000,0x3fde8e06
	.long	0x22a961b0,0x3e8fd353
	.long	0x00000000,0x3fdece62
	.long	0x266f2e1f,0x3e72854e
	.long	0x00000000,0x3fdf1f16
	.long	0xc69211fe,0x3e8520c7
	.long	0x00000000,0x3fdf5fd8
	.long	0xaba878d5,0x3e826fcf
	.long	0x00000000,0x3fdfa0c8
	.long	0xcd5b35b1,0x3e726ed8
	.long	0x00000000,0x3fdfe1e6
	.long	0x678a4f1c,0x3e49054d
	.long	0x00000000,0x3fe01199
	.long	0x3bc19f18,0x3e5d23cb
	.long	0x00000000,0x3fe03256
	.long	0x12772acb,0x3e87af35
	.long	0x00000000,0x3fe0532a
	.long	0x2849c88a,0x3e67f5fc
	.long	0x00000000,0x3fe07416
	.long	0x0e574fa1,0x3e84fce2
	.long	0x00000000,0x3fe08cd6
	.long	0x0ec2b5fa,0x3e857044
	.long	0x00000000,0x3fe0adeb
	.long	0xd031f353,0x3e6d5d1d
	.long	0x00000000,0x3fe0cf18
	.long	0xdae75c4d,0x3e99a159
	.long	0x00000000,0x3fe0f05c
	.long	0x1553afb9,0x3e90eaf1
	.long	0x00000000,0x3fe111b9
	.long	0xdcc5c3c7,0x3e8bc6f3
	.long	0x00000000,0x3fe1332e
	.long	0x1109e597,0x3e94ef84
	.long	0x00000000,0x3fe154bb
	.long	0xd82adea4,0x3e937f40
	.long	0x00000000,0x3fe16df5
	.long	0x78336a2f,0x3e96dc5a
	.long	0x00000000,0x3fe18fad
	.long	0x84c2c804,0x3e90935b
	.long	0x00000000,0x3fe1b17e
	.long	0x04fd30aa,0x3e8a837a
	.long	0x00000000,0x3fe1caeb
	.long	0xd924b0ac,0x3e99061d
	.long	0x00000000,0x3fe1ece7
	.long	0xef9b9d01,0x3e7ed700
	.long	0x00000000,0x3fe20efd
	.long	0x202c81ec,0x3e9c6ebb
	.long	0x00000000,0x3fe2289d
	.long	0xfc0d7d76,0x3e936d4d
	.long	0x00000000,0x3fe24adf
	.long	0x9f8756ba,0x3e9a35d7
	.long	0x00000000,0x3fe26d3a
	.long	0xe7c79d83,0x3e584ee3
	.long	0x00000000,0x3fe28710
	.long	0x942065a0,0x3e7d9e0d
	.long	0x00000000,0x3fe2a998
	.long	0x2175afbf,0x3e872108
	.long	0x00000000,0x3fe2c38f
	.long	0xfaf6283c,0x3e9f5809
	.long	0x00000000,0x3fe2e644
	.long	0x411d0507,0x3e871209
	.long	0x00000000,0x3fe3005e
	.long	0x370c24bf,0x3e9c3b11
	.long	0x00000000,0x3fe32341
	.long	0x8044bb5a,0x3e9bce8e
	.long	0x00000000,0x3fe33d7d
	.long	0x698ea854,0x3e98aca0
	.long	0x00000000,0x3fe3608f
	.long	0xd4d873bb,0x3e972a8e
	.long	0x00000000,0x3fe37aee
	.long	0x615e8182,0x3e8f669b
	.long	0x00000000,0x3fe39e2f
	.long	0xdda5b49a,0x3e98d1f4
	.long	0x00000000,0x3fe3b8b1
	.long	0xe72383f7,0x3e9cc279
	.long	0x00000000,0x3fe3dc21
	.long	0x497497f1,0x3e9c6774
	.long	0x00000000,0x3fe3f6c7
	.long	0x801bd0e6,0x3e82c7f6
	.long	0x00000000,0x3fe4117d
	.long	0xd0ee28b4,0x3e9dfdd6
	.long	0x00000000,0x3fe43531
	.long	0xb70d3761,0x3e92d3a2
	.long	0x00000000,0x3fe4500b
	.long	0xe7d6bcb2,0x3e9c8343
	.long	0x00000000,0x3fe46af4
	.long	0x90d43957,0x3e693179
	.long	0x00000000,0x3fe48eef
	.long	0xf12570df,0x3e799eab
	.long	0x00000000,0x3fe4a9fd
	.long	0x1b88755d,0x3e78c5f8
	.long	0x00000000,0x3fe4c51b
	.long	0xdf99a22c,0x3e7369be
	.long	0x00000000,0x3fe4e049
	.long	0x6ed50f62,0x3e88fd93
	.long	0x00000000,0x3fe5049f
	.long	0x978605ff,0x3e9c6042
	.long	0x00000000,0x3fe51ff2
	.long	0xe6c85f4c,0x3e930ae6
	.long	0x00000000,0x3fe53b56
	.long	0xc3275ba6,0x3e92e7b6
	.long	0x00000000,0x3fe556ca
	.long	0x91597938,0x3e9e6401
	.long	0x00000000,0x3fe5724e
	.long	0x448ebb62,0x3e3d2dee
	.long	0x00000000,0x3fe59719
	.long	0x47501b6d,0x3e9b432e
	.long	0x00000000,0x3fe5b2c3
	.long	0x571fa7cd,0x3e83cf9b
	.long	0x00000000,0x3fe5ce7f
	.long	0x7359819d,0x3e8dc615
	.long	0x00000000,0x3fe5ea4b
	.long	0xc78a85ed,0x3e8973c3
	.long	0x00000000,0x3fe60628
	.long	0xc15a9f3d,0x3e77d788
	.long	0x00000000,0x3fe62216
	.long	0x51952736,0x3e9d2538
	.long	0x00000000,0x3fe63e14
	.long	0xde792c07,0x3e94dab9
	.long	0x00000000,0x3fe65a24
	.long	0x5bfa4318,0x3e8f5be1
	.long	0x00000000,0x3fe67645
	.long	0x55090ec8,0x3e903b26
	.long	0x00000000,0x3fe69277
	.long	0xc78b6175,0x3e99236f
	.long	0x00000000,0x3fe6aeba
	.long	0x3a80db6a,0x3e8a1972
	.long	0x00000000,0x3fe6cb0f
	.long	0xf558aa96,0x3e8d43a2
	.long	0x00000000,0x3fe6e775
	.long	0xd9a82f2e,0x3e424ee3
	.long	0x00000000,0x3fe703ed
	.long	0x583878f6,0x3e764d8e
	.long	0x00000000,0x3fe72076
	.long	0xc1150a3e,0x3e379604
	.long	0x00000000,0x3fe73d11
	.long	0xed85584b,0x3e93b229
	.long	0x00000000,0x3fe759bd
	.long	0x451a7b48,0x3e62967a
	.long	0x00000000,0x3fe7767c
	.long	0xc044e72d,0x3e8e12d7
	.long	0x00000000,0x3fe7934c
	.long	0xfdfb6949,0x3e9ca45d
	.long	0x00000000,0x3fe7b02e
	.long	0xff690fce,0x3e9244fc
	.long	0x00000000,0x3fe7c37a
	.long	0x81487a2c,0x3e7e9cea
	.long	0x00000000,0x3fe7e07b
	.long	0xd0ad2d9a,0x3e760da0
	.long	0x00000000,0x3fe7fd8e
	.long	0x096f45d9,0x3e8d7703
	.long	0x00000000,0x3fe81ab3
	.long	0x1b17115b,0x3e78dbee
	.long	0x00000000,0x3fe837eb
	.long	0x7c252ee0,0x3e8dc5a4
	.long	0x00000000,0x3fe85535
	.long	0x42d5123f,0x3e950116
	.long	0x00000000,0x3fe868c6
	.long	0xf11e41be,0x3e4c4eb7
	.long	0x00000000,0x3fe88630
	.long	0xdb2890b4,0x3e7773b8
	.long	0x00000000,0x3fe8a3ac
	.long	0x7ffb4479,0x3e7bc8e7
	.long	0x00000000,0x3fe8c13b
	.long	0x237693b3,0x3e8c388f
	.long	0x00000000,0x3fe8d4fa
	.long	0x45fcf1a0,0x3e731cd4
	.long	0x00000000,0x3fe8f2a9
	.long	0xe4895b91,0x3e9fae07
	.long	0x00000000,0x3fe9106a
	.long	0x42d2824e,0x3e7d140d
	.long	0x00000000,0x3fe9244c
	.long	0x39900f67,0x3e75c3c8
	.long	0x00000000,0x3fe9422e
	.long	0xa314252b,0x3e902422
	.long	0x00000000,0x3fe96023
	.long	0xf2a6b8ef,0x3e7b9be8
	.long	0x00000000,0x3fe97427
	.long	0x1476f5e9,0x3e66b188
	.long	0x00000000,0x3fe9923d
	.long	0xdde10a6f,0x3e93c377
	.long	0x00000000,0x3fe9b066
	.long	0x4944a32c,0x3e817cad
	.long	0x00000000,0x3fe9c48d
	.long	0xc738e7ef,0x3e9927f2
	.long	0x00000000,0x3fe9e2d7
	.long	0xd25cfd94,0x3e7ccc41
	.long	0x00000000,0x3fea0136
	.long	0x2210e81b,0x3e8382fa
	.long	0x00000000,0x3fea1580
	.long	0x1e690ce2,0x3e7752a7
	.long	0x00000000,0x3fea3400
	.long	0x122315d2,0x3e94cfee
	.long	0x00000000,0x3fea4860
	.long	0xce98333b,0x3e80536e
	.long	0x00000000,0x3fea6702
	.long	0xc30f00e9,0x3e838b7e
	.long	0x00000000,0x3fea7b79
	.long	0x05b0c779,0x3e9dc380
	.long	0x00000000,0x3fea9a3c
	.long	0xdd6dd3fe,0x3e6be168
	.long	0x00000000,0x3feab915
	.long	0x26e0d276,0x3e966757
	.long	0x00000000,0x3feacdb0
	.long	0x0aad615c,0x3e880252
	.long	0x00000000,0x3feaecab
	.long	0x7927096a,0x3e607c31
	.long	0x00000000,0x3feb015e
	.long	0x53b3d90e,0x3e596513
	.long	0x00000000,0x3feb161a
	.long	0x8f2f0570,0x3e90ec3a
	.long	0x00000000,0x3feb3545
	.long	0x81193954,0x3e9cb640
	.long	0x00000000,0x3feb4a18
	.long	0x311e7236,0x3e936479
	.long	0x00000000,0x3feb6967
	.long	0x3a42a413,0x3e9210e8
	.long	0x00000000,0x3feb7e52
	.long	0x4a0daeb2,0x3e9a1717
	.long	0x00000000,0x3feb9dc4
	.long	0xce900653,0x3e925bb7
	.long	0x00000000,0x3febb2c7
	.long	0xb5087588,0x3e95dbb8
	.long	0x00000000,0x3febd25d
	.long	0x4a41204c,0x3e8d0aa9
	.long	0x00000000,0x3febe778
	.long	0x69a0d774,0x3e9c772f
	.long	0x00000000,0x3febfc9c
	.long	0x79d0a9a5,0x3e97b6a0
	.long	0x00000000,0x3fec1c65
	.long	0xd26f1a12,0x3e8f7402
	.long	0x00000000,0x3fec31a2
	.long	0x1243bc84,0x3e4db2f1
	.long	0x00000000,0x3fec46e9
	.long	0x477e1755,0x3e80dcc2
	.long	0x00000000,0x3fec66e5
	.long	0xc2f904c1,0x3e8b1e31
	.long	0x00000000,0x3fec7c44
	.long	0x1785b0c4,0x3e8fb619
	.long	0x00000000,0x3fec91ad
	.long	0xedb052ef,0x3e98832d
	.long	0x00000000,0x3fecb1dd
	.long	0x9e373618,0x3e98c822
	.long	0x00000000,0x3fecc75f
	.long	0x32954637,0x3e9a46e7
	.long	0x00000000,0x3fecdceb
	.long	0xf0e6b2a9,0x3e996305
	.long	0x00000000,0x3fecfd50
	.long	0x1a6614ee,0x3e68c160
	.long	0x00000000,0x3fed12f6
	.long	0x09e33b28,0x3e9229c4
	.long	0x00000000,0x3fed28a5
	.long	0xe53b994c,0x3e7f281b
	.long	0x00000000,0x3fed3e5f
	.long	0x1124ac35,0x3e9f27f3
	.long	0x00000000,0x3fed5f08
	.long	0x3d2fdc03,0x3e84e779
	.long	0x00000000,0x3fed74dc
	.long	0x1e93fd97,0x3e416c07
	.long	0x00000000,0x3fed8aba
	.long	0x91b415ef,0x3e654669
	.long	0x00000000,0x3feda0a2
	.long	0x6495f594,0x3e9712d4
	.long	0x00000000,0x3fedc191
	.long	0x40171789,0x3e9ca7b2
	.long	0x00000000,0x3fedd793
	.long	0xe8fc4323,0x3e929afa
	.long	0x00000000,0x3fededa0
	.long	0xeb03bd09,0x3e9a2e96
	.long	0x00000000,0x3fee03b7
	.long	0xca370ea2,0x3e94a63d
	.long	0x00000000,0x3fee19d9
	.long	0xe30512ec,0x3e6a3bcb
	.long	0x00000000,0x3fee3006
	.long	0x411c95ce,0x3e99d3ed
	.long	0x00000000,0x3fee515c
	.long	0x35cfaf8e,0x3e3e5b57
	.long	0x00000000,0x3fee67a4
	.long	0x5669df6a,0x3e9fca71
	.long	0x00000000,0x3fee7df5
	.long	0x04f19d94,0x3e9914e2
	.long	0x00000000,0x3fee9452
	.long	0xcaa19134,0x3e8b511c
	.long	0x00000000,0x3feeaaba
	.long	0x4bb3bfb1,0x3e9ed72f
	.long	0x00000000,0x3feec12c
	.long	0x3c29d75e,0x3e8be8d6
	.long	0x00000000,0x3feed7aa
	.long	0x97da24fd,0x3e9c55d9
	.long	0x00000000,0x3feeee32
	.long	0x983c68ea,0x3e7ddfb1
	.long	0x00000000,0x3fef1014
	.long	0xf4425883,0x3e83ce66
	.long	0x00000000,0x3fef26b8
	.long	0xf7857f23,0x3e9055c3
	.long	0x00000000,0x3fef3d67
	.long	0x2805b525,0x3e9c2223
	.long	0x00000000,0x3fef5421
	.long	0x0c347fcf,0x3e8d59ba
	.long	0x00000000,0x3fef6ae7
	.long	0x7c901c44,0x3e82b110
	.long	0x00000000,0x3fef81b8
	.long	0x74d1b482,0x3e8b1394
	.long	0x00000000,0x3fef9894
	.long	0x36fb9eb2,0x3e9ca75b
	.long	0x00000000,0x3fefaf7b
	.long	0x629b1b7e,0x3e981a0a
	.long	0x00000000,0x3fefc66e
	.long	0x7b8c1116,0x3e54cc20
	.long	0x00000000,0x3fefdd6d
	.long	0xc0babe05,0x3e99ac8b
	.long	0x00000000,0x3feff476
	.type	__dlog2_la___libm_log2_table_256,@object
	.size	__dlog2_la___libm_log2_table_256,4096
	.space 512, 0x00 	
	.align 64
__dlog2_la__P:
	.long	3213235158
	.long	1050233568
	.long	1697350398
	.long	3219592519
	.long	3694740707
	.long	1071564553
	.long	1697260025
	.long	3218543943
	.long	2542794428
	.long	1070757746
	.long	2165113687
	.long	3217999640
	.long	0
	.long	1073157447
	.type	__dlog2_la__P,@object
	.size	__dlog2_la__P,56
