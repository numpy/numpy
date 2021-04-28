/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *      For    0.0    <= x <=  7.0/16.0: atan(x) = atan(0.0) + atan(s), where s=(x-0.0)/(1.0+0.0*x)
 *      For  7.0/16.0 <= x <= 11.0/16.0: atan(x) = atan(0.5) + atan(s), where s=(x-0.5)/(1.0+0.5*x)
 *      For 11.0/16.0 <= x <= 19.0/16.0: atan(x) = atan(1.0) + atan(s), where s=(x-1.0)/(1.0+1.0*x)
 *      For 19.0/16.0 <= x <= 39.0/16.0: atan(x) = atan(1.5) + atan(s), where s=(x-1.5)/(1.0+1.5*x)
 *      For 39.0/16.0 <= x <=    inf   : atan(x) = atan(inf) + atan(s), where s=-1.0/x
 *      Where atan(s) ~= s+s^3*Poly11(s^2) on interval |s|<7.0/0.16.
 * 
 */


	.text
.L_2__routine_start___svml_atan8_z0_0:

	.align    16,0x90
	.globl __svml_atan8

__svml_atan8:


	.cfi_startproc
..L2:

        vmovups   64+__svml_datan_data_internal_avx512(%rip), %zmm4
        vmovups   128+__svml_datan_data_internal_avx512(%rip), %zmm3
        vmovups   256+__svml_datan_data_internal_avx512(%rip), %zmm9

/* saturate X range */
        vmovups   320+__svml_datan_data_internal_avx512(%rip), %zmm7

        vandpd    __svml_datan_data_internal_avx512(%rip), %zmm0, %zmm8

/* R+Rl = DiffX/Y */
        vbroadcastsd .L_2il0floatpacket.14(%rip), %zmm15
        vaddpd    {rn-sae}, %zmm4, %zmm8, %zmm2
        vxorpd    %zmm0, %zmm8, %zmm1
        vcmppd    $29, {sae}, %zmm3, %zmm8, %k2

/* round to 2 bits after binary point */
        vreducepd $40, {sae}, %zmm8, %zmm6
        vsubpd    {rn-sae}, %zmm4, %zmm2, %zmm5

/*
 * if|X|>=MaxThreshold, set DiffX=-1
 * VMSUB(D, DiffX, LargeMask, Zero, One);
 */
        vblendmpd 192+__svml_datan_data_internal_avx512(%rip), %zmm6, %zmm10{%k2}
        vfmadd231pd {rn-sae}, %zmm8, %zmm5, %zmm9
        vmovups   960+__svml_datan_data_internal_avx512(%rip), %zmm5

/* table lookup sequence */
        vmovups   448+__svml_datan_data_internal_avx512(%rip), %zmm6
        vgetmantpd $0, {sae}, %zmm10, %zmm14
        vgetexppd {sae}, %zmm10, %zmm11
        vmovups   1408+__svml_datan_data_internal_avx512(%rip), %zmm10

/*
 * if|X|>=MaxThreshold, set Y=X
 * VMADD(D, Y, LargeMask, X, Zero);
 */
        vminpd    {sae}, %zmm8, %zmm7, %zmm9{%k2}
        vcmppd    $29, {sae}, %zmm5, %zmm2, %k1
        vmovups   576+__svml_datan_data_internal_avx512(%rip), %zmm7
        vmovups   1152+__svml_datan_data_internal_avx512(%rip), %zmm8
        vgetmantpd $0, {sae}, %zmm9, %zmm3
        vgetexppd {sae}, %zmm9, %zmm12
        vmovups   1280+__svml_datan_data_internal_avx512(%rip), %zmm9
        vpermt2pd 512+__svml_datan_data_internal_avx512(%rip), %zmm2, %zmm6
        vsubpd    {rn-sae}, %zmm12, %zmm11, %zmm4
        vpermt2pd 640+__svml_datan_data_internal_avx512(%rip), %zmm2, %zmm7
        vrcp14pd  %zmm3, %zmm13
        vmovups   1344+__svml_datan_data_internal_avx512(%rip), %zmm12
        vmovups   1472+__svml_datan_data_internal_avx512(%rip), %zmm11
        vblendmpd %zmm7, %zmm6, %zmm2{%k1}
        vmulpd    {rn-sae}, %zmm13, %zmm14, %zmm0
        vfnmadd231pd {rn-sae}, %zmm3, %zmm13, %zmm15
        vfnmadd213pd {rn-sae}, %zmm14, %zmm0, %zmm3
        vfmadd213pd {rn-sae}, %zmm15, %zmm15, %zmm15
        vfmadd213pd {rn-sae}, %zmm13, %zmm13, %zmm15
        vfmadd213pd {rn-sae}, %zmm0, %zmm15, %zmm3
        vscalefpd {rn-sae}, %zmm4, %zmm3, %zmm0

/* set table value to Pi/2 for large X */
        vblendmpd 1024+__svml_datan_data_internal_avx512(%rip), %zmm2, %zmm3{%k2}
        vmovups   1216+__svml_datan_data_internal_avx512(%rip), %zmm2

/* polynomial evaluation */
        vmulpd    {rn-sae}, %zmm0, %zmm0, %zmm14
        vmulpd    {rn-sae}, %zmm14, %zmm14, %zmm13
        vmulpd    {rn-sae}, %zmm0, %zmm14, %zmm15
        vfmadd231pd {rn-sae}, %zmm14, %zmm8, %zmm2
        vfmadd231pd {rn-sae}, %zmm14, %zmm9, %zmm12
        vfmadd213pd {rn-sae}, %zmm11, %zmm10, %zmm14
        vfmadd213pd {rn-sae}, %zmm12, %zmm13, %zmm2
        vfmadd213pd {rn-sae}, %zmm14, %zmm13, %zmm2
        vfmadd213pd {rn-sae}, %zmm0, %zmm15, %zmm2
        vaddpd    {rn-sae}, %zmm3, %zmm2, %zmm0
        vxorpd    %zmm1, %zmm0, %zmm0

/* no invcbrt in libm, so taking it out here */
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atan8,@function
	.size	__svml_atan8,.-__svml_atan8
..LN__svml_atan8.0:

.L_2__routine_start___svml_datan_cout_rare_internal_1:

	.align    16,0x90

__svml_datan_cout_rare_internal:


	.cfi_startproc
..L5:

        movzwl    6(%rdi), %r8d
        andl      $32752, %r8d
        shrl      $4, %r8d
        cmpl      $2047, %r8d
        je        .LBL_2_12


        movq      (%rdi), %rdx
        movq      %rdx, -16(%rsp)
        shrq      $56, %rdx
        movb      7(%rdi), %al
        andl      $127, %edx
        movb      %dl, -9(%rsp)
        movsd     -16(%rsp), %xmm0
        shrb      $7, %al
        comisd    1888+__datan_la_CoutTab(%rip), %xmm0
        movl      -12(%rsp), %ecx
        jb        .LBL_2_6


        movsd     1896+__datan_la_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_5


        movl      4(%rdi), %edx
        movl      %ecx, %edi
        andl      $-524288, %ecx
        andl      $-1048576, %edi
        addl      $262144, %ecx
        movaps    %xmm0, %xmm9
        andl      $1048575, %ecx
        movaps    %xmm0, %xmm10
        movsd     %xmm0, -56(%rsp)
        orl       %ecx, %edi
        movl      $0, -56(%rsp)
        andl      $1048575, %edx
        movl      %edi, -52(%rsp)
        lea       __datan_la_CoutTab(%rip), %rcx
        movsd     1928+__datan_la_CoutTab(%rip), %xmm4
        movsd     -56(%rsp), %xmm15
        shll      $20, %r8d
        subsd     -56(%rsp), %xmm9
        mulsd     1928+__datan_la_CoutTab(%rip), %xmm10
        shlb      $7, %al
        mulsd     %xmm9, %xmm4
        movsd     %xmm4, -48(%rsp)
        orl       %edx, %r8d
        movsd     -48(%rsp), %xmm5
        addl      $-1069547520, %r8d
        sarl      $18, %r8d
        subsd     %xmm9, %xmm5
        movsd     %xmm5, -40(%rsp)
        andl      $-2, %r8d
        movsd     -48(%rsp), %xmm7
        movsd     -40(%rsp), %xmm6
        movslq    %r8d, %r8
        subsd     %xmm6, %xmm7
        movsd     %xmm7, -48(%rsp)
        movsd     -48(%rsp), %xmm8
        movsd     1904+__datan_la_CoutTab(%rip), %xmm6
        subsd     %xmm8, %xmm9
        movsd     %xmm9, -40(%rsp)
        movsd     -48(%rsp), %xmm2
        movsd     -40(%rsp), %xmm3
        movsd     %xmm10, -48(%rsp)
        movsd     -48(%rsp), %xmm11
        movsd     1904+__datan_la_CoutTab(%rip), %xmm8
        subsd     -16(%rsp), %xmm11
        movsd     %xmm11, -40(%rsp)
        movsd     -48(%rsp), %xmm13
        movsd     -40(%rsp), %xmm12
        subsd     %xmm12, %xmm13
        movsd     %xmm13, -48(%rsp)
        movsd     -48(%rsp), %xmm14
        subsd     %xmm14, %xmm0
        movsd     1904+__datan_la_CoutTab(%rip), %xmm14
        movsd     %xmm0, -40(%rsp)
        movsd     -48(%rsp), %xmm5
        movsd     -40(%rsp), %xmm4
        mulsd     %xmm15, %xmm5
        mulsd     %xmm15, %xmm4
        movaps    %xmm5, %xmm1
        addsd     %xmm4, %xmm1
        movsd     %xmm1, -48(%rsp)
        movsd     -48(%rsp), %xmm0
        subsd     %xmm0, %xmm5
        addsd     %xmm4, %xmm5
        movsd     1928+__datan_la_CoutTab(%rip), %xmm4
        movsd     %xmm5, -40(%rsp)
        movsd     -48(%rsp), %xmm11
        movsd     -40(%rsp), %xmm1
        addsd     %xmm11, %xmm6
        movsd     %xmm6, -48(%rsp)
        movsd     -48(%rsp), %xmm7
        subsd     %xmm7, %xmm8
        movsd     %xmm8, -40(%rsp)
        movsd     -48(%rsp), %xmm10
        movsd     -40(%rsp), %xmm9
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -32(%rsp)
        movsd     -40(%rsp), %xmm12
        movsd     1928+__datan_la_CoutTab(%rip), %xmm10
        addsd     %xmm11, %xmm12
        movsd     %xmm12, -40(%rsp)
        movsd     -32(%rsp), %xmm13
        movsd     1904+__datan_la_CoutTab(%rip), %xmm11
        subsd     %xmm13, %xmm14
        movsd     %xmm14, -32(%rsp)
        movsd     -40(%rsp), %xmm0
        movsd     -32(%rsp), %xmm15
        addsd     %xmm15, %xmm0
        movsd     %xmm0, -32(%rsp)
        movsd     -48(%rsp), %xmm9
        mulsd     %xmm9, %xmm4
        movsd     -32(%rsp), %xmm0
        movsd     %xmm4, -48(%rsp)
        addsd     %xmm1, %xmm0
        movsd     -48(%rsp), %xmm5
        subsd     %xmm9, %xmm5
        movsd     %xmm5, -40(%rsp)
        movsd     -48(%rsp), %xmm7
        movsd     -40(%rsp), %xmm6
        subsd     %xmm6, %xmm7
        movsd     1904+__datan_la_CoutTab(%rip), %xmm6
        movsd     %xmm7, -48(%rsp)
        movsd     -48(%rsp), %xmm8
        subsd     %xmm8, %xmm9
        movsd     %xmm9, -40(%rsp)
        movsd     -48(%rsp), %xmm4
        divsd     %xmm4, %xmm11
        mulsd     %xmm11, %xmm10
        movsd     -40(%rsp), %xmm5
        movsd     %xmm10, -40(%rsp)
        addsd     %xmm0, %xmm5
        movsd     -40(%rsp), %xmm12
        subsd     %xmm11, %xmm12
        movsd     %xmm12, -32(%rsp)
        movsd     -40(%rsp), %xmm10
        movsd     -32(%rsp), %xmm13
        subsd     %xmm13, %xmm10
        movsd     %xmm10, -32(%rsp)
        movsd     -32(%rsp), %xmm14
        mulsd     %xmm14, %xmm4
        movsd     -32(%rsp), %xmm15
        subsd     %xmm4, %xmm6
        mulsd     %xmm15, %xmm5
        movsd     %xmm5, -40(%rsp)
        movsd     -40(%rsp), %xmm1
        subsd     %xmm1, %xmm6
        movsd     %xmm6, -40(%rsp)
        movsd     -40(%rsp), %xmm4
        movsd     -32(%rsp), %xmm5
        movsd     -40(%rsp), %xmm0
        movaps    %xmm5, %xmm7
        movsd     -32(%rsp), %xmm1
        mulsd     %xmm3, %xmm5
        addsd     1904+__datan_la_CoutTab(%rip), %xmm4
        mulsd     %xmm2, %xmm7
        mulsd     %xmm0, %xmm4
        mulsd     %xmm1, %xmm4
        mulsd     %xmm4, %xmm3
        mulsd     %xmm4, %xmm2
        addsd     %xmm3, %xmm5
        movsd     1872+__datan_la_CoutTab(%rip), %xmm6
        addsd     %xmm2, %xmm5
        movsd     %xmm5, -48(%rsp)
        movaps    %xmm7, %xmm2
        movsd     -48(%rsp), %xmm4
        addsd     %xmm4, %xmm2
        movsd     %xmm2, -48(%rsp)
        movsd     -48(%rsp), %xmm3
        movsd     (%rcx,%r8,8), %xmm2
        subsd     %xmm3, %xmm7
        addsd     %xmm4, %xmm7
        movsd     %xmm7, -40(%rsp)
        movsd     -48(%rsp), %xmm3
        movaps    %xmm3, %xmm5
        movaps    %xmm3, %xmm0
        mulsd     %xmm3, %xmm5
        addsd     %xmm2, %xmm0
        mulsd     %xmm5, %xmm6
        movsd     -40(%rsp), %xmm10
        movsd     %xmm0, -48(%rsp)
        movsd     -48(%rsp), %xmm1
        addsd     1864+__datan_la_CoutTab(%rip), %xmm6
        subsd     %xmm1, %xmm2
        mulsd     %xmm5, %xmm6
        addsd     %xmm3, %xmm2
        addsd     1856+__datan_la_CoutTab(%rip), %xmm6
        mulsd     %xmm5, %xmm6
        movsd     %xmm2, -40(%rsp)
        movsd     -48(%rsp), %xmm9
        movsd     -40(%rsp), %xmm8
        addsd     1848+__datan_la_CoutTab(%rip), %xmm6
        mulsd     %xmm5, %xmm6
        addsd     1840+__datan_la_CoutTab(%rip), %xmm6
        mulsd     %xmm5, %xmm6
        addsd     1832+__datan_la_CoutTab(%rip), %xmm6
        mulsd     %xmm5, %xmm6
        addsd     1824+__datan_la_CoutTab(%rip), %xmm6
        mulsd     %xmm5, %xmm6
        mulsd     %xmm3, %xmm6
        addsd     %xmm6, %xmm10
        addsd     8(%rcx,%r8,8), %xmm10
        addsd     %xmm8, %xmm10
        addsd     %xmm9, %xmm10
        movsd     %xmm10, -24(%rsp)
        movb      -17(%rsp), %r9b
        andb      $127, %r9b
        orb       %al, %r9b
        movb      %r9b, -17(%rsp)
        movq      -24(%rsp), %rax
        movq      %rax, (%rsi)
        jmp       .LBL_2_11

.LBL_2_5:

        movsd     1912+__datan_la_CoutTab(%rip), %xmm0
        shlb      $7, %al
        addsd     1920+__datan_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %dl
        andb      $127, %dl
        orb       %al, %dl
        movb      %dl, -17(%rsp)
        movq      -24(%rsp), %rax
        movq      %rax, (%rsi)
        jmp       .LBL_2_11

.LBL_2_6:

        comisd    1880+__datan_la_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movaps    %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        shlb      $7, %al
        movsd     1872+__datan_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1864+__datan_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1856+__datan_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1848+__datan_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1840+__datan_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1832+__datan_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        addsd     1824+__datan_la_CoutTab(%rip), %xmm2
        mulsd     %xmm1, %xmm2
        mulsd     %xmm0, %xmm2
        addsd     %xmm0, %xmm2
        movsd     %xmm2, -24(%rsp)
        movb      -17(%rsp), %dl
        andb      $127, %dl
        orb       %al, %dl
        movb      %dl, -17(%rsp)
        movq      -24(%rsp), %rax
        movq      %rax, (%rsi)
        jmp       .LBL_2_11

.LBL_2_8:

        movzwl    -10(%rsp), %edx
        testl     $32752, %edx
        je        .LBL_2_10


        movsd     1904+__datan_la_CoutTab(%rip), %xmm1
        shlb      $7, %al
        addsd     %xmm0, %xmm1
        movsd     %xmm1, -48(%rsp)
        movsd     -48(%rsp), %xmm0
        mulsd     -16(%rsp), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %dl
        andb      $127, %dl
        orb       %al, %dl
        movb      %dl, -17(%rsp)
        movq      -24(%rsp), %rax
        movq      %rax, (%rsi)
        jmp       .LBL_2_11

.LBL_2_10:

        mulsd     %xmm0, %xmm0
        shlb      $7, %al
        movsd     %xmm0, -48(%rsp)
        movsd     -48(%rsp), %xmm0
        addsd     -16(%rsp), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %dl
        andb      $127, %dl
        orb       %al, %dl
        movb      %dl, -17(%rsp)
        movq      -24(%rsp), %rax
        movq      %rax, (%rsi)

.LBL_2_11:

        xorl      %eax, %eax
        ret

.LBL_2_12:

        testl     $1048575, 4(%rdi)
        jne       .LBL_2_15


        cmpl      $0, (%rdi)
        jne       .LBL_2_15


        movsd     1912+__datan_la_CoutTab(%rip), %xmm0
        movb      7(%rdi), %al
        andb      $-128, %al
        addsd     1920+__datan_la_CoutTab(%rip), %xmm0
        movsd     %xmm0, -24(%rsp)
        movb      -17(%rsp), %dl
        andb      $127, %dl
        orb       %al, %dl
        movb      %dl, -17(%rsp)
        movq      -24(%rsp), %rcx
        movq      %rcx, (%rsi)
        jmp       .LBL_2_11

.LBL_2_15:

        movsd     (%rdi), %xmm0
        addsd     %xmm0, %xmm0
        movsd     %xmm0, (%rsi)
        jmp       .LBL_2_11
	.align    16,0x90

	.cfi_endproc

	.type	__svml_datan_cout_rare_internal,@function
	.size	__svml_datan_cout_rare_internal,.-__svml_datan_cout_rare_internal
..LN__svml_datan_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_datan_data_internal_avx512:
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
	.long	1125646336
	.long	0
	.long	1125646336
	.long	0
	.long	1125646336
	.long	0
	.long	1125646336
	.long	0
	.long	1125646336
	.long	0
	.long	1125646336
	.long	0
	.long	1125646336
	.long	0
	.long	1125646336
	.long	0
	.long	1075806208
	.long	0
	.long	1075806208
	.long	0
	.long	1075806208
	.long	0
	.long	1075806208
	.long	0
	.long	1075806208
	.long	0
	.long	1075806208
	.long	0
	.long	1075806208
	.long	0
	.long	1075806208
	.long	0
	.long	3220176896
	.long	0
	.long	3220176896
	.long	0
	.long	3220176896
	.long	0
	.long	3220176896
	.long	0
	.long	3220176896
	.long	0
	.long	3220176896
	.long	0
	.long	3220176896
	.long	0
	.long	3220176896
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
	.long	1206910976
	.long	0
	.long	1206910976
	.long	0
	.long	1206910976
	.long	0
	.long	1206910976
	.long	0
	.long	1206910976
	.long	0
	.long	1206910976
	.long	0
	.long	1206910976
	.long	0
	.long	1206910976
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	4180443357
	.long	1070553973
	.long	90291023
	.long	1071492199
	.long	2737217249
	.long	1071945615
	.long	1413754136
	.long	1072243195
	.long	1468297118
	.long	1072475260
	.long	3531732635
	.long	1072657163
	.long	744202399
	.long	1072747407
	.long	2464923204
	.long	1072805601
	.long	1436891685
	.long	1072853231
	.long	2037009832
	.long	1072892781
	.long	1826698067
	.long	1072926058
	.long	1803191648
	.long	1072954391
	.long	2205372832
	.long	1072978772
	.long	4234512805
	.long	1072999952
	.long	3932628503
	.long	1073018509
	.long	2501811453
	.long	1073034892
	.long	866379431
	.long	1073049455
	.long	1376865888
	.long	1073062480
	.long	3290094269
	.long	1073074195
	.long	354764887
	.long	1073084787
	.long	3332975497
	.long	1073094406
	.long	1141460092
	.long	1073103181
	.long	745761286
	.long	1073111216
	.long	1673304509
	.long	1073118600
	.long	983388243
	.long	1073125409
	.long	3895509104
	.long	1073131706
	.long	2128523669
	.long	1073137548
	.long	2075485693
	.long	1073142981
	.long	121855980
	.long	1073148047
	.long	4181733783
	.long	1073152780
	.long	2887813284
	.long	1073157214
	.long	0
	.long	0
	.long	1022865341
	.long	1013492590
	.long	573531618
	.long	1014639487
	.long	2280825944
	.long	1014120858
	.long	856972295
	.long	1015129638
	.long	986810987
	.long	1015077601
	.long	2062601149
	.long	1013974920
	.long	589036912
	.long	3164328156
	.long	1787331214
	.long	1016798022
	.long	2942272763
	.long	3164235441
	.long	2956702105
	.long	1016472908
	.long	3903328092
	.long	3162582135
	.long	3175026820
	.long	3158589859
	.long	787328196
	.long	1014621351
	.long	2317874517
	.long	3163795518
	.long	4071621134
	.long	1016673529
	.long	2492111345
	.long	3164172103
	.long	3606178875
	.long	3162371821
	.long	3365790232
	.long	1014547152
	.long	2710887773
	.long	1017086651
	.long	2755350986
	.long	3162706257
	.long	198095269
	.long	3162802133
	.long	2791076759
	.long	3164364640
	.long	4214434319
	.long	3162164074
	.long	773754012
	.long	3164190653
	.long	139561443
	.long	3164313657
	.long	2197796619
	.long	3164066219
	.long	3592486882
	.long	1016669082
	.long	1148791015
	.long	3163724934
	.long	386789398
	.long	3163117479
	.long	2518816264
	.long	3162291736
	.long	2545101323
	.long	3164592727
	.long	16
	.long	1125646336
	.long	16
	.long	1125646336
	.long	16
	.long	1125646336
	.long	16
	.long	1125646336
	.long	16
	.long	1125646336
	.long	16
	.long	1125646336
	.long	16
	.long	1125646336
	.long	16
	.long	1125646336
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	1413754136
	.long	1073291771
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	856972295
	.long	1016178214
	.long	4123328151
	.long	1068689849
	.long	4123328151
	.long	1068689849
	.long	4123328151
	.long	1068689849
	.long	4123328151
	.long	1068689849
	.long	4123328151
	.long	1068689849
	.long	4123328151
	.long	1068689849
	.long	4123328151
	.long	1068689849
	.long	4123328151
	.long	1068689849
	.long	3295121612
	.long	3216458327
	.long	3295121612
	.long	3216458327
	.long	3295121612
	.long	3216458327
	.long	3295121612
	.long	3216458327
	.long	3295121612
	.long	3216458327
	.long	3295121612
	.long	3216458327
	.long	3295121612
	.long	3216458327
	.long	3295121612
	.long	3216458327
	.long	4026078880
	.long	1069314495
	.long	4026078880
	.long	1069314495
	.long	4026078880
	.long	1069314495
	.long	4026078880
	.long	1069314495
	.long	4026078880
	.long	1069314495
	.long	4026078880
	.long	1069314495
	.long	4026078880
	.long	1069314495
	.long	4026078880
	.long	1069314495
	.long	2398029018
	.long	3217180964
	.long	2398029018
	.long	3217180964
	.long	2398029018
	.long	3217180964
	.long	2398029018
	.long	3217180964
	.long	2398029018
	.long	3217180964
	.long	2398029018
	.long	3217180964
	.long	2398029018
	.long	3217180964
	.long	2398029018
	.long	3217180964
	.long	2576905246
	.long	1070176665
	.long	2576905246
	.long	1070176665
	.long	2576905246
	.long	1070176665
	.long	2576905246
	.long	1070176665
	.long	2576905246
	.long	1070176665
	.long	2576905246
	.long	1070176665
	.long	2576905246
	.long	1070176665
	.long	2576905246
	.long	1070176665
	.long	1431655757
	.long	3218429269
	.long	1431655757
	.long	3218429269
	.long	1431655757
	.long	3218429269
	.long	1431655757
	.long	3218429269
	.long	1431655757
	.long	3218429269
	.long	1431655757
	.long	3218429269
	.long	1431655757
	.long	3218429269
	.long	1431655757
	.long	3218429269
	.type	__svml_datan_data_internal_avx512,@object
	.size	__svml_datan_data_internal_avx512,1536
	.align 32
__datan_la_CoutTab:
	.long	3892314112
	.long	1069799150
	.long	2332892550
	.long	1039715405
	.long	1342177280
	.long	1070305495
	.long	270726690
	.long	1041535749
	.long	939524096
	.long	1070817911
	.long	2253973841
	.long	3188654726
	.long	3221225472
	.long	1071277294
	.long	3853927037
	.long	1043226911
	.long	2818572288
	.long	1071767563
	.long	2677759107
	.long	1044314101
	.long	3355443200
	.long	1072103591
	.long	1636578514
	.long	3191094734
	.long	1476395008
	.long	1072475260
	.long	1864703685
	.long	3188646936
	.long	805306368
	.long	1072747407
	.long	192551812
	.long	3192726267
	.long	2013265920
	.long	1072892781
	.long	2240369452
	.long	1043768538
	.long	0
	.long	1072999953
	.long	3665168337
	.long	3192705970
	.long	402653184
	.long	1073084787
	.long	1227953434
	.long	3192313277
	.long	2013265920
	.long	1073142981
	.long	3853283127
	.long	1045277487
	.long	805306368
	.long	1073187261
	.long	1676192264
	.long	3192868861
	.long	134217728
	.long	1073217000
	.long	4290763938
	.long	1042034855
	.long	671088640
	.long	1073239386
	.long	994303084
	.long	3189643768
	.long	402653184
	.long	1073254338
	.long	1878067156
	.long	1042652475
	.long	1610612736
	.long	1073265562
	.long	670314820
	.long	1045138554
	.long	3221225472
	.long	1073273048
	.long	691126919
	.long	3189987794
	.long	3489660928
	.long	1073278664
	.long	1618990832
	.long	3188194509
	.long	1207959552
	.long	1073282409
	.long	2198872939
	.long	1044806069
	.long	3489660928
	.long	1073285217
	.long	2633982383
	.long	1042307894
	.long	939524096
	.long	1073287090
	.long	1059367786
	.long	3189114230
	.long	2281701376
	.long	1073288494
	.long	3158525533
	.long	1044484961
	.long	3221225472
	.long	1073289430
	.long	286581777
	.long	1044893263
	.long	4026531840
	.long	1073290132
	.long	2000245215
	.long	3191647611
	.long	134217728
	.long	1073290601
	.long	4205071590
	.long	1045035927
	.long	536870912
	.long	1073290952
	.long	2334392229
	.long	1043447393
	.long	805306368
	.long	1073291186
	.long	2281458177
	.long	3188885569
	.long	3087007744
	.long	1073291361
	.long	691611507
	.long	1044733832
	.long	3221225472
	.long	1073291478
	.long	1816229550
	.long	1044363390
	.long	2281701376
	.long	1073291566
	.long	1993843750
	.long	3189837440
	.long	134217728
	.long	1073291625
	.long	3654754496
	.long	1044970837
	.long	4026531840
	.long	1073291668
	.long	3224300229
	.long	3191935390
	.long	805306368
	.long	1073291698
	.long	2988777976
	.long	3188950659
	.long	536870912
	.long	1073291720
	.long	1030371341
	.long	1043402665
	.long	3221225472
	.long	1073291734
	.long	1524463765
	.long	1044361356
	.long	3087007744
	.long	1073291745
	.long	2754295320
	.long	1044731036
	.long	134217728
	.long	1073291753
	.long	3099629057
	.long	1044970710
	.long	2281701376
	.long	1073291758
	.long	962914160
	.long	3189838838
	.long	805306368
	.long	1073291762
	.long	3543908206
	.long	3188950786
	.long	4026531840
	.long	1073291764
	.long	1849909620
	.long	3191935434
	.long	3221225472
	.long	1073291766
	.long	1641333636
	.long	1044361352
	.long	536870912
	.long	1073291768
	.long	1373968792
	.long	1043402654
	.long	134217728
	.long	1073291769
	.long	2033191599
	.long	1044970710
	.long	3087007744
	.long	1073291769
	.long	4117947437
	.long	1044731035
	.long	805306368
	.long	1073291770
	.long	315378368
	.long	3188950787
	.long	2281701376
	.long	1073291770
	.long	2428571750
	.long	3189838838
	.long	3221225472
	.long	1073291770
	.long	1608007466
	.long	1044361352
	.long	4026531840
	.long	1073291770
	.long	1895711420
	.long	3191935434
	.long	134217728
	.long	1073291771
	.long	2031108713
	.long	1044970710
	.long	536870912
	.long	1073291771
	.long	1362518342
	.long	1043402654
	.long	805306368
	.long	1073291771
	.long	317461253
	.long	3188950787
	.long	939524096
	.long	1073291771
	.long	4117231784
	.long	1044731035
	.long	1073741824
	.long	1073291771
	.long	1607942376
	.long	1044361352
	.long	1207959552
	.long	1073291771
	.long	2428929577
	.long	3189838838
	.long	1207959552
	.long	1073291771
	.long	2031104645
	.long	1044970710
	.long	1342177280
	.long	1073291771
	.long	1895722602
	.long	3191935434
	.long	1342177280
	.long	1073291771
	.long	317465322
	.long	3188950787
	.long	1342177280
	.long	1073291771
	.long	1362515546
	.long	1043402654
	.long	1342177280
	.long	1073291771
	.long	1607942248
	.long	1044361352
	.long	1342177280
	.long	1073291771
	.long	4117231610
	.long	1044731035
	.long	1342177280
	.long	1073291771
	.long	2031104637
	.long	1044970710
	.long	1342177280
	.long	1073291771
	.long	1540251232
	.long	1045150466
	.long	1342177280
	.long	1073291771
	.long	2644671394
	.long	1045270303
	.long	1342177280
	.long	1073291771
	.long	2399244691
	.long	1045360181
	.long	1342177280
	.long	1073291771
	.long	803971124
	.long	1045420100
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192879152
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192849193
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192826724
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192811744
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192800509
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192793019
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192787402
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192783657
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192780848
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192778976
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192777572
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192776635
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192775933
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192775465
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192775114
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774880
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774704
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774587
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774500
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774441
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774397
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774368
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774346
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774331
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774320
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774313
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774308
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774304
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774301
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774299
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774298
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774297
	.long	1476395008
	.long	1073291771
	.long	3613709523
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	177735686
	.long	3192774296
	.long	1476395008
	.long	1073291771
	.long	3490996172
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	2754716064
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	2263862659
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1895722605
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1650295902
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1466225875
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1343512524
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1251477510
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1190120835
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1144103328
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1113424990
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1090416237
	.long	3192774295
	.long	1476395008
	.long	1073291771
	.long	1075077068
	.long	3192774295
	.long	1431655765
	.long	3218429269
	.long	2576978363
	.long	1070176665
	.long	2453154343
	.long	3217180964
	.long	4189149139
	.long	1069314502
	.long	1775019125
	.long	3216459198
	.long	273199057
	.long	1068739452
	.long	874748308
	.long	3215993277
	.long	0
	.long	1017118720
	.long	0
	.long	1069547520
	.long	0
	.long	1129316352
	.long	0
	.long	1072693248
	.long	1413754136
	.long	1073291771
	.long	856972295
	.long	1016178214
	.long	33554432
	.long	1101004800
	.type	__datan_la_CoutTab,@object
	.size	__datan_la_CoutTab,1936
	.align 8
.L_2il0floatpacket.14:
	.long	0x00000000,0x3ff00000
	.type	.L_2il0floatpacket.14,@object
	.size	.L_2il0floatpacket.14,8
