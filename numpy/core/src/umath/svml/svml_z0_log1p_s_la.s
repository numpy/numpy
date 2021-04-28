/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/


	.text
.L_2__routine_start___svml_log1pf16_z0_0:

	.align    16,0x90
	.globl __svml_log1pf16

__svml_log1pf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   1984+__svml_slog1p_data_internal(%rip), %zmm2

/* reduction: compute r,n */
        vmovups   2688+__svml_slog1p_data_internal(%rip), %zmm12
        vmovups   1088+__svml_slog1p_data_internal(%rip), %zmm4
        vmovaps   %zmm0, %zmm3

/* compute 1+x as high, low parts */
        vmaxps    {sae}, %zmm3, %zmm2, %zmm5
        vminps    {sae}, %zmm3, %zmm2, %zmm7
        vandnps   %zmm3, %zmm4, %zmm1
        vpternlogd $255, %zmm4, %zmm4, %zmm4
        vaddps    {rn-sae}, %zmm7, %zmm5, %zmm9
        vpsubd    %zmm12, %zmm9, %zmm10
        vsubps    {rn-sae}, %zmm9, %zmm5, %zmm6

/* check argument value ranges */
        vpaddd    2560+__svml_slog1p_data_internal(%rip), %zmm9, %zmm8
        vpsrad    $23, %zmm10, %zmm13
        vmovups   2304+__svml_slog1p_data_internal(%rip), %zmm9
        vpcmpd    $5, 2624+__svml_slog1p_data_internal(%rip), %zmm8, %k1
        vpslld    $23, %zmm13, %zmm14
        vaddps    {rn-sae}, %zmm7, %zmm6, %zmm15
        vcvtdq2ps {rn-sae}, %zmm13, %zmm0
        vpsubd    %zmm14, %zmm2, %zmm13
        vmovups   2496+__svml_slog1p_data_internal(%rip), %zmm7
        vmovups   2048+__svml_slog1p_data_internal(%rip), %zmm14
        vmulps    {rn-sae}, %zmm13, %zmm15, %zmm6
        vpandd    2752+__svml_slog1p_data_internal(%rip), %zmm10, %zmm11
        vpaddd    %zmm12, %zmm11, %zmm5
        vmovups   2240+__svml_slog1p_data_internal(%rip), %zmm10
        vmovups   2176+__svml_slog1p_data_internal(%rip), %zmm11
        vmovups   2112+__svml_slog1p_data_internal(%rip), %zmm12

/* polynomial evaluation */
        vsubps    {rn-sae}, %zmm2, %zmm5, %zmm2
        vaddps    {rn-sae}, %zmm6, %zmm2, %zmm15
        vmovups   2432+__svml_slog1p_data_internal(%rip), %zmm2
        vfmadd231ps {rn-sae}, %zmm15, %zmm7, %zmm2
        vpandnd   %zmm8, %zmm8, %zmm4{%k1}
        vmovups   2368+__svml_slog1p_data_internal(%rip), %zmm8

/* combine and get argument value range mask */
        vptestmd  %zmm4, %zmm4, %k0
        vfmadd213ps {rn-sae}, %zmm8, %zmm15, %zmm2
        kmovw     %k0, %edx
        vfmadd213ps {rn-sae}, %zmm9, %zmm15, %zmm2
        vfmadd213ps {rn-sae}, %zmm10, %zmm15, %zmm2
        vfmadd213ps {rn-sae}, %zmm11, %zmm15, %zmm2
        vfmadd213ps {rn-sae}, %zmm12, %zmm15, %zmm2
        vfmadd213ps {rn-sae}, %zmm14, %zmm15, %zmm2
        vmulps    {rn-sae}, %zmm15, %zmm2, %zmm4
        vfmadd213ps {rn-sae}, %zmm15, %zmm15, %zmm4

/* final reconstruction */
        vmovups   2816+__svml_slog1p_data_internal(%rip), %zmm15
        vfmadd213ps {rn-sae}, %zmm4, %zmm15, %zmm0
        vorps     %zmm1, %zmm0, %zmm0
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

        vmovups   %zmm3, 64(%rsp)
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

        call      __svml_slog1p_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log1pf16,@function
	.size	__svml_log1pf16,.-__svml_log1pf16
..LN__svml_log1pf16.0:

.L_2__routine_start___svml_slog1p_cout_rare_internal_1:

	.align    16,0x90

__svml_slog1p_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movss     .L_2il0floatpacket.90(%rip), %xmm1
        xorb      %r8b, %r8b
        movss     (%rdi), %xmm5
        addss     %xmm1, %xmm5
        movss     %xmm5, -20(%rsp)
        movzwl    -18(%rsp), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_15


        movss     %xmm5, -16(%rsp)
        xorl      %ecx, %ecx
        movzwl    -14(%rsp), %edx
        testl     $32640, %edx
        jne       .LBL_2_4


        mulss     .L_2il0floatpacket.75(%rip), %xmm5
        movb      $1, %r8b
        movss     %xmm5, -16(%rsp)
        movl      $-40, %ecx

.LBL_2_4:

        pxor      %xmm3, %xmm3
        comiss    %xmm3, %xmm5
        jbe       .LBL_2_10


        movaps    %xmm5, %xmm2
        subss     %xmm1, %xmm2
        movss     %xmm2, -20(%rsp)
        andb      $127, -17(%rsp)
        movss     -20(%rsp), %xmm0
        comiss    .L_2il0floatpacket.76(%rip), %xmm0
        jbe       .LBL_2_9


        movzwl    -14(%rsp), %edx
        pxor      %xmm6, %xmm6
        andl      $32640, %edx
        shrl      $7, %edx
        lea       -127(%rcx,%rdx), %ecx
        cvtsi2ss  %ecx, %xmm6
        cmpb      $1, %r8b
        je        .LBL_2_13


        movss     .L_2il0floatpacket.86(%rip), %xmm4
        movss     .L_2il0floatpacket.87(%rip), %xmm0
        mulss     %xmm6, %xmm4
        mulss     %xmm0, %xmm6

.LBL_2_8:

        movss     %xmm5, -20(%rsp)
        movaps    %xmm4, %xmm9
        movzwl    -18(%rsp), %edx
        lea       __slog1p_la_CoutTab(%rip), %r8
        andl      $-32641, %edx
        addl      $16256, %edx
        movw      %dx, -18(%rsp)
        movss     -20(%rsp), %xmm8
        movaps    %xmm8, %xmm2
        movss     .L_2il0floatpacket.89(%rip), %xmm7
        addss     .L_2il0floatpacket.88(%rip), %xmm2
        movss     %xmm2, -24(%rsp)
        movl      -24(%rsp), %ecx
        andl      $127, %ecx
        lea       (%rcx,%rcx,2), %edi
        movss     4(%r8,%rdi,4), %xmm5
        movss     (%r8,%rdi,4), %xmm0
        addss     %xmm5, %xmm9
        addss     8(%r8,%rdi,4), %xmm6
        movaps    %xmm9, %xmm3
        subss     %xmm4, %xmm3
        movss     %xmm3, -24(%rsp)
        movss     -24(%rsp), %xmm4
        subss     %xmm4, %xmm5
        movss     %xmm5, -24(%rsp)
        movss     -24(%rsp), %xmm10
        addss     %xmm6, %xmm10
        movaps    %xmm7, %xmm6
        addss     %xmm8, %xmm6
        movss     %xmm6, -24(%rsp)
        movss     -24(%rsp), %xmm12
        subss     %xmm7, %xmm12
        subss     %xmm12, %xmm8
        mulss     %xmm0, %xmm12
        subss     %xmm1, %xmm12
        mulss     %xmm8, %xmm0
        movaps    %xmm0, %xmm15
        movaps    %xmm12, %xmm2
        addss     %xmm10, %xmm15
        addss     %xmm9, %xmm12
        addss     %xmm0, %xmm2
        movaps    %xmm15, %xmm1
        movaps    %xmm12, %xmm13
        subss     %xmm10, %xmm1
        addss     %xmm15, %xmm13
        movss     %xmm1, -24(%rsp)
        movss     -24(%rsp), %xmm11
        subss     %xmm11, %xmm0
        movss     %xmm0, -24(%rsp)
        movss     -24(%rsp), %xmm0
        movss     %xmm13, (%rsi)
        subss     %xmm12, %xmm13
        movss     .L_2il0floatpacket.83(%rip), %xmm12
        mulss     %xmm2, %xmm12
        movss     %xmm13, -24(%rsp)
        movss     -24(%rsp), %xmm14
        addss     .L_2il0floatpacket.82(%rip), %xmm12
        subss     %xmm14, %xmm15
        mulss     %xmm2, %xmm12
        movss     %xmm15, -24(%rsp)
        movss     -24(%rsp), %xmm1
        addss     .L_2il0floatpacket.81(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.80(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.79(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.78(%rip), %xmm12
        mulss     %xmm2, %xmm12
        addss     .L_2il0floatpacket.77(%rip), %xmm12
        mulss     %xmm2, %xmm12
        mulss     %xmm2, %xmm12
        addss     %xmm12, %xmm0
        addss     %xmm0, %xmm1
        movss     %xmm1, -24(%rsp)
        movss     -24(%rsp), %xmm3
        addss     (%rsi), %xmm3
        movss     %xmm3, (%rsi)
        ret

.LBL_2_9:

        movss     .L_2il0floatpacket.83(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.82(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.81(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.80(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.79(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.78(%rip), %xmm0
        mulss     %xmm2, %xmm0
        addss     .L_2il0floatpacket.77(%rip), %xmm0
        mulss     %xmm2, %xmm0
        mulss     %xmm2, %xmm0
        addss     %xmm2, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_10:

        ucomiss   %xmm3, %xmm5
        jp        .LBL_2_11
        je        .LBL_2_14

.LBL_2_11:

        divss     %xmm3, %xmm3
        movss     %xmm3, (%rsi)
        movl      $1, %eax


        ret

.LBL_2_13:

        movss     .L_2il0floatpacket.85(%rip), %xmm0
        mulss     %xmm0, %xmm6
        movaps    %xmm6, %xmm4
        movaps    %xmm3, %xmm6
        jmp       .LBL_2_8

.LBL_2_14:

        movss     .L_2il0floatpacket.84(%rip), %xmm0
        movl      $2, %eax
        divss     %xmm3, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_15:

        movb      -17(%rsp), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_17

.LBL_2_16:

        mulss     %xmm5, %xmm5
        movss     %xmm5, (%rsi)
        ret

.LBL_2_17:

        testl     $8388607, -20(%rsp)
        jne       .LBL_2_16


        movl      $1, %eax
        pxor      %xmm1, %xmm1
        pxor      %xmm0, %xmm0
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_slog1p_cout_rare_internal,@function
	.size	__svml_slog1p_cout_rare_internal,.-__svml_slog1p_cout_rare_internal
..LN__svml_slog1p_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_slog1p_data_internal:
	.long	3266227256
	.long	3107766024
	.long	3266228276
	.long	3107776882
	.long	3266229284
	.long	3109949545
	.long	3266230292
	.long	3108055846
	.long	3266231292
	.long	3106351937
	.long	3266232276
	.long	3109092567
	.long	3266233260
	.long	3107948216
	.long	3266234236
	.long	3107170960
	.long	3266235204
	.long	3106817287
	.long	3266236164
	.long	3106942449
	.long	3266237116
	.long	3107600489
	.long	3266238060
	.long	3108844279
	.long	3266239004
	.long	3106531253
	.long	3266239932
	.long	3109100650
	.long	3266240860
	.long	3108213420
	.long	3266241780
	.long	3108112381
	.long	3266242692
	.long	3108845034
	.long	3266243604
	.long	3106263589
	.long	3266244500
	.long	3108802209
	.long	3266245396
	.long	3108116909
	.long	3266246284
	.long	3108445707
	.long	3266247164
	.long	3109831435
	.long	3266248044
	.long	3108121760
	.long	3266248916
	.long	3107552123
	.long	3266249780
	.long	3108162844
	.long	3266250644
	.long	3105799146
	.long	3266251492
	.long	3108888393
	.long	3266252340
	.long	3109079979
	.long	3266253188
	.long	3106411173
	.long	3266254020
	.long	3109307139
	.long	3266254852
	.long	3109415127
	.long	3266255684
	.long	3106770317
	.long	3266256500
	.long	3109795834
	.long	3266257324
	.long	3105942641
	.long	3266258132
	.long	3107826892
	.long	3266258940
	.long	3107092610
	.long	3266259740
	.long	3107966131
	.long	3266260540
	.long	3106284596
	.long	3266261332
	.long	3106273188
	.long	3266262116
	.long	3107962226
	.long	3266262900
	.long	3107187186
	.long	3266263676
	.long	3108171617
	.long	3266264452
	.long	3106749947
	.long	3266265220
	.long	3107144703
	.long	3266265980
	.long	3109383615
	.long	3266266740
	.long	3109299629
	.long	3266267500
	.long	3106919521
	.long	3266268252
	.long	3106463913
	.long	3266268996
	.long	3107958670
	.long	3266269740
	.long	3107234917
	.long	3266270476
	.long	3108511954
	.long	3266271212
	.long	3107620056
	.long	3266271940
	.long	3108777693
	.long	3266272668
	.long	3107814325
	.long	3266273388
	.long	3108947630
	.long	3266274108
	.long	3108006290
	.long	3266274820
	.long	3109207222
	.long	3266275532
	.long	3108378366
	.long	3266276236
	.long	3109735912
	.long	3266276940
	.long	3109107087
	.long	3266277644
	.long	3106513079
	.long	3266278340
	.long	3106169044
	.long	3266279028
	.long	3108095503
	.long	3266279716
	.long	3108118349
	.long	3266280404
	.long	3106257463
	.long	3266281084
	.long	3106726720
	.long	3266281756
	.long	3109545389
	.long	3266282436
	.long	3106343833
	.long	3266283100
	.long	3109723642
	.long	3266283772
	.long	3107120300
	.long	3266284436
	.long	3106940529
	.long	3266285092
	.long	3109202170
	.long	3266285748
	.long	3109728494
	.long	3266286404
	.long	3108536808
	.long	3266287052
	.long	3109838471
	.long	3266287700
	.long	3109455977
	.long	3266288348
	.long	3107405879
	.long	3266288988
	.long	3107898790
	.long	3266289628
	.long	3106756477
	.long	3266290260
	.long	3108189081
	.long	3266290892
	.long	3108017907
	.long	3266291524
	.long	3106258339
	.long	3266292148
	.long	3107119845
	.long	3266292772
	.long	3106423069
	.long	3266293388
	.long	3108377050
	.long	3266294004
	.long	3108802011
	.long	3266294620
	.long	3107712277
	.long	3266295228
	.long	3109316274
	.long	3266295836
	.long	3109433625
	.long	3266296444
	.long	3108078064
	.long	3266297044
	.long	3109457438
	.long	3266297644
	.long	3109390801
	.long	3266298244
	.long	3107891329
	.long	3266298836
	.long	3109166323
	.long	3266299428
	.long	3109034299
	.long	3266300020
	.long	3107507904
	.long	3266300604
	.long	3108793919
	.long	3266301188
	.long	3108710352
	.long	3266301772
	.long	3107269350
	.long	3266302348
	.long	3108677203
	.long	3266302924
	.long	3108751436
	.long	3266303500
	.long	3107503720
	.long	3266304068
	.long	3109139881
	.long	3266304636
	.long	3109476985
	.long	3266305204
	.long	3108526254
	.long	3266305772
	.long	3106298768
	.long	3266306332
	.long	3106999765
	.long	3266306892
	.long	3106445739
	.long	3266307444
	.long	3108841650
	.long	3266308004
	.long	3105809415
	.long	3266308548
	.long	3109942336
	.long	3266309100
	.long	3108667760
	.long	3266309652
	.long	3106190122
	.long	3266310196
	.long	3106713732
	.long	3266310740
	.long	3106054165
	.long	3266311276
	.long	3108415484
	.long	3266311812
	.long	3109613023
	.long	3266312348
	.long	3109656301
	.long	3266312884
	.long	3108554723
	.long	3266313420
	.long	3106317576
	.long	3266313948
	.long	3107148341
	.long	3266314476
	.long	3106861780
	.long	3266314996
	.long	3109661153
	.long	3266315524
	.long	3107166702
	.long	3266316044
	.long	3107775778
	.long	3266316564
	.long	3107302717
	.long	3266317076
	.long	3109950361
	.long	3266317596
	.long	3107338539
	.long	3266318108
	.long	3107864196
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
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
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
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
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	2063597568
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	1051372345
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	3204448310
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	998244352
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	3212836863
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	2055208960
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	4294967040
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
	.long	901758464
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
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	1051372180
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	3196061070
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	1045225872
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	3190336823
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	1041222418
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	3189430755
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	1041073389
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	16777216
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	25165824
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	1059760811
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	8388607
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	1060205080
	.long	2139095040
	.long	4286578688
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.long	1065353216
	.long	3212836864
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.long	0
	.long	2147483648
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.type	__svml_slog1p_data_internal,@object
	.size	__svml_slog1p_data_internal,3072
	.align 32
__slog1p_la_CoutTab:
	.long	1065353216
	.long	0
	.long	0
	.long	1065091072
	.long	1015087104
	.long	900509991
	.long	1064828928
	.long	1023541248
	.long	925811956
	.long	1064566784
	.long	1027915776
	.long	3084221144
	.long	1064304640
	.long	1032073216
	.long	3066991812
	.long	1064173568
	.long	1033195520
	.long	882149603
	.long	1063911424
	.long	1035468800
	.long	928189163
	.long	1063649280
	.long	1037783040
	.long	927501741
	.long	1063518208
	.long	1038958592
	.long	3076037756
	.long	1063256064
	.long	1040759808
	.long	904405630
	.long	1063124992
	.long	1041361920
	.long	3052231524
	.long	1062862848
	.long	1042581504
	.long	922094799
	.long	1062731776
	.long	1043201024
	.long	3070120623
	.long	1062469632
	.long	1044455424
	.long	3069864633
	.long	1062338560
	.long	1045091328
	.long	3063188516
	.long	1062207488
	.long	1045733376
	.long	3054902185
	.long	1061945344
	.long	1047035904
	.long	920635797
	.long	1061814272
	.long	1047697408
	.long	904920689
	.long	1061683200
	.long	1048365056
	.long	912483742
	.long	1061552128
	.long	1048807936
	.long	3052664405
	.long	1061421056
	.long	1049148416
	.long	912794238
	.long	1061158912
	.long	1049840384
	.long	889474359
	.long	1061027840
	.long	1050191872
	.long	3059868362
	.long	1060896768
	.long	1050546944
	.long	3059256525
	.long	1060765696
	.long	1050905600
	.long	912008988
	.long	1060634624
	.long	1051268352
	.long	912290698
	.long	1060503552
	.long	1051635200
	.long	3037211048
	.long	1060372480
	.long	1052005888
	.long	906226119
	.long	1060241408
	.long	1052380928
	.long	3052480305
	.long	1060110336
	.long	1052760064
	.long	3048768765
	.long	1059979264
	.long	1053143552
	.long	3049975450
	.long	1059848192
	.long	1053531392
	.long	894485718
	.long	1059717120
	.long	1053923840
	.long	897598623
	.long	1059586048
	.long	1054320896
	.long	907355277
	.long	1059586048
	.long	1054320896
	.long	907355277
	.long	1059454976
	.long	1054722816
	.long	881705073
	.long	1059323904
	.long	1055129600
	.long	3049723733
	.long	1059192832
	.long	1055541248
	.long	890353599
	.long	1059061760
	.long	1055958016
	.long	908173938
	.long	1059061760
	.long	1055958016
	.long	908173938
	.long	1058930688
	.long	1056380160
	.long	883644938
	.long	1058799616
	.long	1056807680
	.long	3052015799
	.long	1058668544
	.long	1057102592
	.long	884897284
	.long	1058668544
	.long	1057102592
	.long	884897284
	.long	1058537472
	.long	1057321920
	.long	3037632470
	.long	1058406400
	.long	1057544128
	.long	865017195
	.long	1058275328
	.long	1057769344
	.long	3042936546
	.long	1058275328
	.long	1057769344
	.long	3042936546
	.long	1058144256
	.long	1057997568
	.long	903344518
	.long	1058013184
	.long	1058228992
	.long	897862967
	.long	1058013184
	.long	1058228992
	.long	897862967
	.long	1057882112
	.long	1058463680
	.long	3047822280
	.long	1057882112
	.long	1058463680
	.long	3047822280
	.long	1057751040
	.long	1058701632
	.long	883793293
	.long	1057619968
	.long	1058943040
	.long	851667963
	.long	1057619968
	.long	1058943040
	.long	851667963
	.long	1057488896
	.long	1059187968
	.long	3000004036
	.long	1057488896
	.long	1059187968
	.long	3000004036
	.long	1057357824
	.long	1059436544
	.long	3047430717
	.long	1057357824
	.long	1059436544
	.long	3047430717
	.long	1057226752
	.long	1059688832
	.long	3043802308
	.long	1057226752
	.long	1059688832
	.long	3043802308
	.long	1057095680
	.long	1059944960
	.long	876113044
	.long	1057095680
	.long	1059944960
	.long	876113044
	.long	1056964608
	.long	1060205056
	.long	901758606
	.long	1060205056
	.long	901758606
	.long	1207959616
	.long	1174405120
	.long	1008730112
	.long	1400897536
	.long	0
	.long	1065353216
	.long	3204448256
	.long	1051372203
	.long	3196059648
	.long	1045220557
	.long	3190467243
	.long	1041387009
	.long	3187672480
	.type	__slog1p_la_CoutTab,@object
	.size	__slog1p_la_CoutTab,840
	.align 4
.L_2il0floatpacket.75:
	.long	0x53800000
	.type	.L_2il0floatpacket.75,@object
	.size	.L_2il0floatpacket.75,4
	.align 4
.L_2il0floatpacket.76:
	.long	0x3c200000
	.type	.L_2il0floatpacket.76,@object
	.size	.L_2il0floatpacket.76,4
	.align 4
.L_2il0floatpacket.77:
	.long	0xbf000000
	.type	.L_2il0floatpacket.77,@object
	.size	.L_2il0floatpacket.77,4
	.align 4
.L_2il0floatpacket.78:
	.long	0x3eaaaaab
	.type	.L_2il0floatpacket.78,@object
	.size	.L_2il0floatpacket.78,4
	.align 4
.L_2il0floatpacket.79:
	.long	0xbe800000
	.type	.L_2il0floatpacket.79,@object
	.size	.L_2il0floatpacket.79,4
	.align 4
.L_2il0floatpacket.80:
	.long	0x3e4ccccd
	.type	.L_2il0floatpacket.80,@object
	.size	.L_2il0floatpacket.80,4
	.align 4
.L_2il0floatpacket.81:
	.long	0xbe2aaaab
	.type	.L_2il0floatpacket.81,@object
	.size	.L_2il0floatpacket.81,4
	.align 4
.L_2il0floatpacket.82:
	.long	0x3e124e01
	.type	.L_2il0floatpacket.82,@object
	.size	.L_2il0floatpacket.82,4
	.align 4
.L_2il0floatpacket.83:
	.long	0xbe0005a0
	.type	.L_2il0floatpacket.83,@object
	.size	.L_2il0floatpacket.83,4
	.align 4
.L_2il0floatpacket.84:
	.long	0xbf800000
	.type	.L_2il0floatpacket.84,@object
	.size	.L_2il0floatpacket.84,4
	.align 4
.L_2il0floatpacket.85:
	.long	0x3f317218
	.type	.L_2il0floatpacket.85,@object
	.size	.L_2il0floatpacket.85,4
	.align 4
.L_2il0floatpacket.86:
	.long	0x3f317200
	.type	.L_2il0floatpacket.86,@object
	.size	.L_2il0floatpacket.86,4
	.align 4
.L_2il0floatpacket.87:
	.long	0x35bfbe8e
	.type	.L_2il0floatpacket.87,@object
	.size	.L_2il0floatpacket.87,4
	.align 4
.L_2il0floatpacket.88:
	.long	0x48000040
	.type	.L_2il0floatpacket.88,@object
	.size	.L_2il0floatpacket.88,4
	.align 4
.L_2il0floatpacket.89:
	.long	0x46000000
	.type	.L_2il0floatpacket.89,@object
	.size	.L_2il0floatpacket.89,4
	.align 4
.L_2il0floatpacket.90:
	.long	0x3f800000
	.type	.L_2il0floatpacket.90,@object
	.size	.L_2il0floatpacket.90,4
