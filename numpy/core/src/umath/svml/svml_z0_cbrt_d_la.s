/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *   x=2^{3*k+j} * 1.b1 b2 ... b5 b6 ... b52
 *   Let r=(x*2^{-3k-j} - 1.b1 b2 ... b5 1)* rcp[b1 b2 ..b5],
 *   where rcp[b1 b2 .. b5]=1/(1.b1 b2 b3 b4 b5 1) in double precision
 *   cbrt(2^j * 1. b1 b2 .. b5 1) is approximated as T[j][b1..b5]+D[j][b1..b5]
 *   (T stores the high 53 bits, D stores the low order bits)
 *   Result=2^k*T+(2^k*T*r)*P+2^k*D
 *   where P=p1+p2*r+..+p8*r^7
 * 
 */


	.text
.L_2__routine_start___svml_cbrt8_z0_0:

	.align    16,0x90
	.globl __svml_cbrt8

__svml_cbrt8:


	.cfi_startproc
..L2:

        vgetmantpd $0, {sae}, %zmm0, %zmm14

/* GetExp(x) */
        vgetexppd {sae}, %zmm0, %zmm7
        vmovups   384+__svml_dcbrt_data_internal_avx512(%rip), %zmm8

/* exponent/3 */
        vmovups   512+__svml_dcbrt_data_internal_avx512(%rip), %zmm9
        vmovups   576+__svml_dcbrt_data_internal_avx512(%rip), %zmm10

/* Reduced argument: R = DblRcp*Mantissa - 1 */
        vmovups   704+__svml_dcbrt_data_internal_avx512(%rip), %zmm2

/* exponent%3 (to be used as index) */
        vmovups   640+__svml_dcbrt_data_internal_avx512(%rip), %zmm11

/* DblRcp ~ 1/Mantissa */
        vrcp14pd  %zmm14, %zmm13
        vaddpd    {rn-sae}, %zmm8, %zmm7, %zmm12
        vandpd    448+__svml_dcbrt_data_internal_avx512(%rip), %zmm0, %zmm6

/* round DblRcp to 3 fractional bits (RN mode, no Precision exception) */
        vrndscalepd $72, {sae}, %zmm13, %zmm15
        vfmsub231pd {rn-sae}, %zmm12, %zmm9, %zmm10

/* polynomial */
        vmovups   768+__svml_dcbrt_data_internal_avx512(%rip), %zmm0
        vmovups   896+__svml_dcbrt_data_internal_avx512(%rip), %zmm7
        vmovups   960+__svml_dcbrt_data_internal_avx512(%rip), %zmm9
        vfmsub231pd {rn-sae}, %zmm15, %zmm14, %zmm2
        vrndscalepd $9, {sae}, %zmm10, %zmm5

/* Table lookup */
        vmovups   128+__svml_dcbrt_data_internal_avx512(%rip), %zmm10
        vmovups   1024+__svml_dcbrt_data_internal_avx512(%rip), %zmm8
        vmovups   1216+__svml_dcbrt_data_internal_avx512(%rip), %zmm13
        vfmadd231pd {rn-sae}, %zmm2, %zmm7, %zmm9
        vfnmadd231pd {rn-sae}, %zmm5, %zmm11, %zmm12
        vmovups   1088+__svml_dcbrt_data_internal_avx512(%rip), %zmm11
        vmovups   1344+__svml_dcbrt_data_internal_avx512(%rip), %zmm14

/* Prepare table index */
        vpsrlq    $49, %zmm15, %zmm1

/* Table lookup: 2^(exponent%3) */
        vpermpd   __svml_dcbrt_data_internal_avx512(%rip), %zmm12, %zmm4
        vpermpd   64+__svml_dcbrt_data_internal_avx512(%rip), %zmm12, %zmm3
        vpermt2pd 192+__svml_dcbrt_data_internal_avx512(%rip), %zmm1, %zmm10
        vmovups   832+__svml_dcbrt_data_internal_avx512(%rip), %zmm1
        vfmadd231pd {rn-sae}, %zmm2, %zmm8, %zmm11
        vmovups   1280+__svml_dcbrt_data_internal_avx512(%rip), %zmm12
        vscalefpd {rn-sae}, %zmm5, %zmm10, %zmm15
        vfmadd231pd {rn-sae}, %zmm2, %zmm0, %zmm1
        vmovups   1152+__svml_dcbrt_data_internal_avx512(%rip), %zmm5
        vfmadd231pd {rn-sae}, %zmm2, %zmm12, %zmm14
        vmulpd    {rn-sae}, %zmm2, %zmm2, %zmm0
        vfmadd231pd {rn-sae}, %zmm2, %zmm5, %zmm13

/* Sh*R */
        vmulpd    {rn-sae}, %zmm2, %zmm4, %zmm2
        vfmadd213pd {rn-sae}, %zmm9, %zmm0, %zmm1
        vfmadd213pd {rn-sae}, %zmm11, %zmm0, %zmm1
        vfmadd213pd {rn-sae}, %zmm13, %zmm0, %zmm1
        vfmadd213pd {rn-sae}, %zmm14, %zmm0, %zmm1

/* Sl + (Sh*R)*Poly */
        vfmadd213pd {rn-sae}, %zmm3, %zmm1, %zmm2

/*
 * branch-free
 * scaled_Th*(Sh+Sl+Sh*R*Poly)
 */
        vaddpd    {rn-sae}, %zmm4, %zmm2, %zmm3
        vmulpd    {rn-sae}, %zmm15, %zmm3, %zmm4
        vorpd     %zmm6, %zmm4, %zmm0

/* no invcbrt in libm, so taking it out here */
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_cbrt8,@function
	.size	__svml_cbrt8,.-__svml_cbrt8
..LN__svml_cbrt8.0:

.L_2__routine_start___svml_dcbrt_cout_rare_internal_1:

	.align    16,0x90

__svml_dcbrt_cout_rare_internal:


	.cfi_startproc
..L5:

        movq      %rsi, %r8
        movzwl    6(%rdi), %r9d
        andl      $32752, %r9d
        shrl      $4, %r9d
        movb      7(%rdi), %sil
        movsd     (%rdi), %xmm1
        cmpl      $2047, %r9d
        je        .LBL_2_9


        ucomisd   432+__dcbrt_la__vmldCbrtTab(%rip), %xmm1
        jp        .LBL_2_3
        je        .LBL_2_8

.LBL_2_3:

        movb      %sil, %al
        lea       440+__dcbrt_la__vmldCbrtTab(%rip), %rdx
        andb      $-128, %al
        andb      $127, %sil
        shrb      $7, %al
        xorl      %edi, %edi
        movsd     %xmm1, -56(%rsp)
        movzbl    %al, %ecx
        movb      %sil, -49(%rsp)
        movsd     (%rdx,%rcx,8), %xmm5
        testl     %r9d, %r9d
        jne       .LBL_2_5


        movsd     -56(%rsp), %xmm0
        movl      $100, %edi
        mulsd     360+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        movsd     %xmm0, -56(%rsp)
        jmp       .LBL_2_6

.LBL_2_5:

        movsd     -56(%rsp), %xmm0

.LBL_2_6:

        movzwl    -50(%rsp), %esi
        movl      $1431655766, %eax
        andl      $32752, %esi
        lea       __dcbrt_la__vmldCbrtTab(%rip), %r11
        shrl      $4, %esi
        movsd     %xmm0, -40(%rsp)
        movsd     368+__dcbrt_la__vmldCbrtTab(%rip), %xmm14
        imull     %esi
        movl      $1431655766, %eax
        lea       (%rdx,%rdx,2), %ecx
        negl      %ecx
        addl      %esi, %ecx
        subl      %ecx, %esi
        addl      %ecx, %ecx
        addl      $-1023, %esi
        imull     %esi
        sarl      $31, %esi
        subl      %esi, %edx
        addl      $1023, %edx
        subl      %edi, %edx
        movzwl    -34(%rsp), %edi
        andl      $2047, %edx
        andl      $-32753, %edi
        addl      $16368, %edi
        movw      %di, -34(%rsp)
        movsd     -40(%rsp), %xmm11
        movaps    %xmm11, %xmm6
        mulsd     376+__dcbrt_la__vmldCbrtTab(%rip), %xmm6
        movsd     %xmm6, -32(%rsp)
        movsd     -32(%rsp), %xmm7
        movl      -36(%rsp), %r10d
        andl      $1048575, %r10d
        subsd     -40(%rsp), %xmm7
        movsd     %xmm7, -24(%rsp)
        movsd     -32(%rsp), %xmm9
        movsd     -24(%rsp), %xmm8
        shrl      $15, %r10d
        subsd     %xmm8, %xmm9
        movsd     %xmm9, -32(%rsp)
        movsd     -32(%rsp), %xmm10
        movsd     (%r11,%r10,8), %xmm4
        subsd     %xmm10, %xmm11
        movaps    %xmm4, %xmm12
        movaps    %xmm4, %xmm13
        mulsd     %xmm4, %xmm12
        movsd     %xmm11, -24(%rsp)
        movsd     -32(%rsp), %xmm2
        mulsd     %xmm12, %xmm2
        mulsd     %xmm2, %xmm13
        movsd     440+__dcbrt_la__vmldCbrtTab(%rip), %xmm6
        movsd     -24(%rsp), %xmm3
        subsd     %xmm13, %xmm6
        mulsd     %xmm12, %xmm3
        mulsd     %xmm6, %xmm14
        mulsd     %xmm3, %xmm4
        movsd     %xmm14, -32(%rsp)
        movsd     -32(%rsp), %xmm15
        xorps     .L_2il0floatpacket.81(%rip), %xmm4
        subsd     %xmm6, %xmm15
        movsd     %xmm15, -24(%rsp)
        movsd     -32(%rsp), %xmm1
        movsd     -24(%rsp), %xmm0
        movsd     256+__dcbrt_la__vmldCbrtTab(%rip), %xmm9
        subsd     %xmm0, %xmm1
        movsd     %xmm1, -32(%rsp)
        movsd     -32(%rsp), %xmm13
        movsd     352+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        subsd     %xmm13, %xmm6
        movsd     %xmm6, -24(%rsp)
        movsd     -32(%rsp), %xmm1
        movsd     -24(%rsp), %xmm7
        movaps    %xmm1, %xmm8
        movsd     256+__dcbrt_la__vmldCbrtTab(%rip), %xmm11
        addsd     %xmm7, %xmm4
        movsd     256+__dcbrt_la__vmldCbrtTab(%rip), %xmm7
        addsd     %xmm4, %xmm8
        mulsd     %xmm8, %xmm0
        movslq    %ecx, %rcx
        addsd     344+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        movq      440+__dcbrt_la__vmldCbrtTab(%rip), %r9
        movq      %r9, -48(%rsp)
        shrq      $48, %r9
        addsd     336+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        andl      $-32753, %r9d
        shll      $4, %edx
        addsd     328+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        orl       %edx, %r9d
        movw      %r9w, -42(%rsp)
        addsd     320+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        addsd     312+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        addsd     304+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        addsd     296+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        addsd     288+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        addsd     280+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        addsd     272+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm8, %xmm0
        addsd     %xmm0, %xmm9
        movsd     %xmm9, -32(%rsp)
        movsd     -32(%rsp), %xmm10
        movsd     368+__dcbrt_la__vmldCbrtTab(%rip), %xmm9
        subsd     %xmm10, %xmm11
        movsd     %xmm11, -24(%rsp)
        movsd     -32(%rsp), %xmm14
        movsd     -24(%rsp), %xmm12
        addsd     %xmm12, %xmm14
        movsd     %xmm14, -16(%rsp)
        movaps    %xmm2, %xmm14
        movsd     -24(%rsp), %xmm6
        addsd     %xmm0, %xmm6
        movsd     %xmm6, -24(%rsp)
        movsd     -16(%rsp), %xmm15
        subsd     %xmm15, %xmm7
        movsd     %xmm7, -16(%rsp)
        movsd     -24(%rsp), %xmm8
        movsd     -16(%rsp), %xmm0
        addsd     %xmm0, %xmm8
        movsd     %xmm8, -16(%rsp)
        movaps    %xmm1, %xmm8
        movsd     -32(%rsp), %xmm13
        mulsd     %xmm13, %xmm9
        movsd     -16(%rsp), %xmm0
        movsd     %xmm9, -32(%rsp)
        movsd     -32(%rsp), %xmm10
        subsd     %xmm13, %xmm10
        addsd     264+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        movsd     %xmm10, -24(%rsp)
        movsd     -32(%rsp), %xmm11
        movsd     -24(%rsp), %xmm6
        subsd     %xmm6, %xmm11
        movsd     %xmm11, -32(%rsp)
        movsd     -32(%rsp), %xmm12
        subsd     %xmm12, %xmm13
        movsd     %xmm13, -24(%rsp)
        movsd     -32(%rsp), %xmm7
        movsd     -24(%rsp), %xmm6
        mulsd     %xmm7, %xmm8
        addsd     %xmm0, %xmm6
        mulsd     %xmm4, %xmm7
        mulsd     %xmm6, %xmm4
        mulsd     %xmm6, %xmm1
        addsd     %xmm4, %xmm7
        movsd     368+__dcbrt_la__vmldCbrtTab(%rip), %xmm4
        addsd     %xmm1, %xmm7
        mulsd     %xmm8, %xmm4
        movsd     %xmm7, -32(%rsp)
        movsd     -32(%rsp), %xmm10
        movsd     %xmm4, -32(%rsp)
        movsd     -32(%rsp), %xmm0
        subsd     %xmm8, %xmm0
        movsd     %xmm0, -24(%rsp)
        movsd     -32(%rsp), %xmm1
        movsd     -24(%rsp), %xmm4
        subsd     %xmm4, %xmm1
        movsd     %xmm1, -32(%rsp)
        movsd     -32(%rsp), %xmm6
        subsd     %xmm6, %xmm8
        movsd     %xmm8, -24(%rsp)
        movsd     -32(%rsp), %xmm9
        movsd     -24(%rsp), %xmm7
        movaps    %xmm9, %xmm1
        mulsd     %xmm3, %xmm9
        addsd     %xmm7, %xmm10
        mulsd     %xmm2, %xmm1
        movaps    %xmm10, %xmm11
        movaps    %xmm1, %xmm12
        mulsd     %xmm3, %xmm10
        addsd     %xmm2, %xmm12
        mulsd     %xmm2, %xmm11
        addsd     %xmm9, %xmm10
        addsd     %xmm10, %xmm11
        movsd     %xmm11, -32(%rsp)
        movsd     -32(%rsp), %xmm0
        movsd     %xmm12, -32(%rsp)
        movsd     -32(%rsp), %xmm13
        subsd     %xmm13, %xmm14
        movsd     %xmm14, -24(%rsp)
        movsd     -32(%rsp), %xmm9
        movsd     -24(%rsp), %xmm15
        addsd     %xmm15, %xmm9
        movsd     %xmm9, -16(%rsp)
        movsd     -24(%rsp), %xmm10
        addsd     %xmm10, %xmm1
        movsd     %xmm1, -24(%rsp)
        movsd     -16(%rsp), %xmm4
        subsd     %xmm4, %xmm2
        movsd     368+__dcbrt_la__vmldCbrtTab(%rip), %xmm4
        movsd     %xmm2, -16(%rsp)
        movsd     -24(%rsp), %xmm1
        movsd     -16(%rsp), %xmm2
        addsd     %xmm2, %xmm1
        movsd     %xmm1, -16(%rsp)
        movsd     -32(%rsp), %xmm9
        mulsd     %xmm9, %xmm4
        movsd     -16(%rsp), %xmm11
        movsd     %xmm4, -32(%rsp)
        movsd     -32(%rsp), %xmm6
        subsd     %xmm9, %xmm6
        movsd     %xmm6, -24(%rsp)
        movsd     -32(%rsp), %xmm7
        movsd     -24(%rsp), %xmm2
        subsd     %xmm2, %xmm7
        movsd     %xmm7, -32(%rsp)
        movsd     -32(%rsp), %xmm8
        subsd     %xmm8, %xmm9
        movsd     %xmm9, -24(%rsp)
        movsd     -32(%rsp), %xmm12
        movsd     -24(%rsp), %xmm10
        addsd     %xmm0, %xmm10
        addsd     %xmm3, %xmm10
        movsd     392(%r11,%rcx,8), %xmm3
        movaps    %xmm3, %xmm0
        addsd     %xmm10, %xmm11
        mulsd     %xmm12, %xmm3
        mulsd     %xmm11, %xmm0
        movsd     384(%r11,%rcx,8), %xmm10
        addsd     %xmm3, %xmm0
        mulsd     %xmm10, %xmm11
        mulsd     %xmm10, %xmm12
        addsd     %xmm11, %xmm0
        movsd     %xmm0, -32(%rsp)
        movsd     -32(%rsp), %xmm3
        addsd     %xmm3, %xmm12
        mulsd     -48(%rsp), %xmm12
        mulsd     %xmm12, %xmm5
        movsd     %xmm5, (%r8)

.LBL_2_7:

        xorl      %eax, %eax
        ret

.LBL_2_8:

        movsd     440+__dcbrt_la__vmldCbrtTab(%rip), %xmm0
        mulsd     %xmm0, %xmm1
        movsd     %xmm1, (%r8)
        jmp       .LBL_2_7

.LBL_2_9:

        addsd     %xmm1, %xmm1
        movsd     %xmm1, (%r8)
        jmp       .LBL_2_7
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dcbrt_cout_rare_internal,@function
	.size	__svml_dcbrt_cout_rare_internal,.-__svml_dcbrt_cout_rare_internal
..LN__svml_dcbrt_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dcbrt_data_internal_avx512:
	.long	0
	.long	1072693248
	.long	4186796683
	.long	1072965794
	.long	2772266557
	.long	1073309182
	.long	0
	.long	0
	.long	0
	.long	3220176896
	.long	4186796683
	.long	3220449442
	.long	2772266557
	.long	3220792830
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1418634270
	.long	3162364962
	.long	2576690953
	.long	3164558313
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1418634270
	.long	1014881314
	.long	2576690953
	.long	1017074665
	.long	0
	.long	0
	.long	4186796683
	.long	1072965794
	.long	1554061055
	.long	1072914931
	.long	3992368458
	.long	1072871093
	.long	3714535808
	.long	1072832742
	.long	954824104
	.long	1072798779
	.long	3256858690
	.long	1072768393
	.long	3858344660
	.long	1072740974
	.long	1027250248
	.long	1072716050
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1418634270
	.long	3162364962
	.long	629721892
	.long	1016287007
	.long	1776620500
	.long	3163956186
	.long	648592220
	.long	1016269578
	.long	1295766103
	.long	3161896715
	.long	1348094586
	.long	3164476360
	.long	2407028709
	.long	1015925873
	.long	497428409
	.long	1014435402
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1127743488
	.long	0
	.long	1127743488
	.long	0
	.long	1127743488
	.long	0
	.long	1127743488
	.long	0
	.long	1127743488
	.long	0
	.long	1127743488
	.long	0
	.long	1127743488
	.long	0
	.long	1127743488
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	0
	.long	2147483648
	.long	1431655766
	.long	1070945621
	.long	1431655766
	.long	1070945621
	.long	1431655766
	.long	1070945621
	.long	1431655766
	.long	1070945621
	.long	1431655766
	.long	1070945621
	.long	1431655766
	.long	1070945621
	.long	1431655766
	.long	1070945621
	.long	1431655766
	.long	1070945621
	.long	0
	.long	1126170624
	.long	0
	.long	1126170624
	.long	0
	.long	1126170624
	.long	0
	.long	1126170624
	.long	0
	.long	1126170624
	.long	0
	.long	1126170624
	.long	0
	.long	1126170624
	.long	0
	.long	1126170624
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
	.long	0
	.long	1074266112
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
	.long	1792985698
	.long	3213372987
	.long	1792985698
	.long	3213372987
	.long	1792985698
	.long	3213372987
	.long	1792985698
	.long	3213372987
	.long	1792985698
	.long	3213372987
	.long	1792985698
	.long	3213372987
	.long	1792985698
	.long	3213372987
	.long	1792985698
	.long	3213372987
	.long	3135539317
	.long	1066129956
	.long	3135539317
	.long	1066129956
	.long	3135539317
	.long	1066129956
	.long	3135539317
	.long	1066129956
	.long	3135539317
	.long	1066129956
	.long	3135539317
	.long	1066129956
	.long	3135539317
	.long	1066129956
	.long	3135539317
	.long	1066129956
	.long	2087834975
	.long	3213899448
	.long	2087834975
	.long	3213899448
	.long	2087834975
	.long	3213899448
	.long	2087834975
	.long	3213899448
	.long	2087834975
	.long	3213899448
	.long	2087834975
	.long	3213899448
	.long	2087834975
	.long	3213899448
	.long	2087834975
	.long	3213899448
	.long	2476259604
	.long	1066628333
	.long	2476259604
	.long	1066628333
	.long	2476259604
	.long	1066628333
	.long	2476259604
	.long	1066628333
	.long	2476259604
	.long	1066628333
	.long	2476259604
	.long	1066628333
	.long	2476259604
	.long	1066628333
	.long	2476259604
	.long	1066628333
	.long	2012366478
	.long	3214412045
	.long	2012366478
	.long	3214412045
	.long	2012366478
	.long	3214412045
	.long	2012366478
	.long	3214412045
	.long	2012366478
	.long	3214412045
	.long	2012366478
	.long	3214412045
	.long	2012366478
	.long	3214412045
	.long	2012366478
	.long	3214412045
	.long	1104999785
	.long	1067378449
	.long	1104999785
	.long	1067378449
	.long	1104999785
	.long	1067378449
	.long	1104999785
	.long	1067378449
	.long	1104999785
	.long	1067378449
	.long	1104999785
	.long	1067378449
	.long	1104999785
	.long	1067378449
	.long	1104999785
	.long	1067378449
	.long	3534763582
	.long	3215266280
	.long	3534763582
	.long	3215266280
	.long	3534763582
	.long	3215266280
	.long	3534763582
	.long	3215266280
	.long	3534763582
	.long	3215266280
	.long	3534763582
	.long	3215266280
	.long	3534763582
	.long	3215266280
	.long	3534763582
	.long	3215266280
	.long	1007386161
	.long	1068473053
	.long	1007386161
	.long	1068473053
	.long	1007386161
	.long	1068473053
	.long	1007386161
	.long	1068473053
	.long	1007386161
	.long	1068473053
	.long	1007386161
	.long	1068473053
	.long	1007386161
	.long	1068473053
	.long	1007386161
	.long	1068473053
	.long	477218625
	.long	3216798151
	.long	477218625
	.long	3216798151
	.long	477218625
	.long	3216798151
	.long	477218625
	.long	3216798151
	.long	477218625
	.long	3216798151
	.long	477218625
	.long	3216798151
	.long	477218625
	.long	3216798151
	.long	477218625
	.long	3216798151
	.long	1431655767
	.long	1070945621
	.long	1431655767
	.long	1070945621
	.long	1431655767
	.long	1070945621
	.long	1431655767
	.long	1070945621
	.long	1431655767
	.long	1070945621
	.long	1431655767
	.long	1070945621
	.long	1431655767
	.long	1070945621
	.long	1431655767
	.long	1070945621
	.type	__svml_dcbrt_data_internal_avx512,@object
	.size	__svml_dcbrt_data_internal_avx512,1408
	.align 32
__dcbrt_la__vmldCbrtTab:
	.long	0
	.long	1072693248
	.long	0
	.long	1072668672
	.long	0
	.long	1072644096
	.long	0
	.long	1072627712
	.long	0
	.long	1072611328
	.long	0
	.long	1072586752
	.long	0
	.long	1072570368
	.long	0
	.long	1072553984
	.long	0
	.long	1072537600
	.long	0
	.long	1072521216
	.long	0
	.long	1072504832
	.long	0
	.long	1072488448
	.long	0
	.long	1072480256
	.long	0
	.long	1072463872
	.long	0
	.long	1072447488
	.long	0
	.long	1072439296
	.long	0
	.long	1072422912
	.long	0
	.long	1072414720
	.long	0
	.long	1072398336
	.long	0
	.long	1072390144
	.long	0
	.long	1072373760
	.long	0
	.long	1072365568
	.long	0
	.long	1072357376
	.long	0
	.long	1072340992
	.long	0
	.long	1072332800
	.long	0
	.long	1072324608
	.long	0
	.long	1072308224
	.long	0
	.long	1072300032
	.long	0
	.long	1072291840
	.long	0
	.long	1072283648
	.long	0
	.long	1072275456
	.long	0
	.long	1072267264
	.long	1431655765
	.long	1071994197
	.long	1431655765
	.long	1015371093
	.long	1908874354
	.long	1071761180
	.long	1007461464
	.long	1071618781
	.long	565592401
	.long	1071446176
	.long	241555088
	.long	1071319599
	.long	943963244
	.long	1071221150
	.long	2330668378
	.long	1071141453
	.long	2770428108
	.long	1071075039
	.long	3622256836
	.long	1071018464
	.long	1497196870
	.long	1070969433
	.long	280472551
	.long	1070926345
	.long	1585032765
	.long	1070888044
	.long	0
	.long	1387266048
	.long	33554432
	.long	1101004800
	.long	512
	.long	1117782016
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	4160749568
	.long	1072965794
	.long	2921479643
	.long	1043912488
	.long	2684354560
	.long	1073309182
	.long	4060791142
	.long	1045755320
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	0
	.long	3220176896
	.type	__dcbrt_la__vmldCbrtTab,@object
	.size	__dcbrt_la__vmldCbrtTab,456
	.space 8, 0x00 	
	.align 16
.L_2il0floatpacket.81:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.81,@object
	.size	.L_2il0floatpacket.81,16
