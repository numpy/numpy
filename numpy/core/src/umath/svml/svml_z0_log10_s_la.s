/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 *  *  log10(x) = VGETEXP(x)*log10(2) + log10(VGETMANT(x))
 *  *       VGETEXP, VGETMANT will correctly treat special cases too (including denormals)
 *  *   mx = VGETMANT(x) is in [1,2) for all x>=0
 *  *   log10(mx) = -log10(RCP(mx)) + log10(1 +(mx*RCP(mx)-1))
 *  *      RCP(mx) is rounded to 4 fractional bits,
 *  *      and the table lookup for log(RCP(mx)) is based on a small permute instruction
 *  *
 *  *   LA, EP versions use interval interpolation (16 intervals)
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_log10f16_z0_0:

	.align    16,0x90
	.globl __svml_log10f16

__svml_log10f16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp

/* GetMant(x), normalized to [.75,1.5) for x>=0, NaN for x<0 */
        vgetmantps $11, {sae}, %zmm0, %zmm3
        vmovups   __svml_slog10_data_internal_avx512(%rip), %zmm1
        vgetexpps {sae}, %zmm0, %zmm5
        vmovups   320+__svml_slog10_data_internal_avx512(%rip), %zmm10
        vpsrld    $19, %zmm3, %zmm7
        vgetexpps {sae}, %zmm3, %zmm6
        vsubps    {rn-sae}, %zmm1, %zmm3, %zmm11
        vpermps   64+__svml_slog10_data_internal_avx512(%rip), %zmm7, %zmm1
        vpermps   128+__svml_slog10_data_internal_avx512(%rip), %zmm7, %zmm2
        vsubps    {rn-sae}, %zmm6, %zmm5, %zmm9
        vpermps   192+__svml_slog10_data_internal_avx512(%rip), %zmm7, %zmm4
        vpermps   256+__svml_slog10_data_internal_avx512(%rip), %zmm7, %zmm8

/* x<=0? */
        vfpclassps $94, %zmm0, %k0
        vfmadd213ps {rn-sae}, %zmm2, %zmm11, %zmm1
        vmulps    {rn-sae}, %zmm10, %zmm9, %zmm12
        vfmadd213ps {rn-sae}, %zmm4, %zmm11, %zmm1
        kmovw     %k0, %edx
        vfmadd213ps {rn-sae}, %zmm8, %zmm11, %zmm1
        vfmadd213ps {rn-sae}, %zmm12, %zmm11, %zmm1
        testl     %edx, %edx
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
        vmovups   %zmm1, 128(%rsp)
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
        vmovups   128(%rsp), %zmm1
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

        call      __svml_slog10_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log10f16,@function
	.size	__svml_log10f16,.-__svml_log10f16
..LN__svml_log10f16.0:

.L_2__routine_start___svml_slog10_cout_rare_internal_1:

	.align    16,0x90

__svml_slog10_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_12


        movss     (%rdi), %xmm2
        xorl      %ecx, %ecx
        movss     %xmm2, -16(%rsp)
        movzwl    -14(%rsp), %edx
        testl     $32640, %edx
        jne       .LBL_2_4


        mulss     .L_2il0floatpacket.77(%rip), %xmm2
        movl      $-40, %ecx
        movss     %xmm2, -16(%rsp)

.LBL_2_4:

        pxor      %xmm0, %xmm0
        comiss    %xmm0, %xmm2
        jbe       .LBL_2_8


        movaps    %xmm2, %xmm1
        subss     .L_2il0floatpacket.93(%rip), %xmm1
        movss     %xmm1, -20(%rsp)
        andb      $127, -17(%rsp)
        movss     -20(%rsp), %xmm0
        comiss    .L_2il0floatpacket.78(%rip), %xmm0
        jbe       .LBL_2_7


        movss     %xmm2, -20(%rsp)
        pxor      %xmm8, %xmm8
        movzwl    -18(%rsp), %edi
        lea       __slog10_la_CoutTab(%rip), %r10
        andl      $-32641, %edi
        addl      $16256, %edi
        movw      %di, -18(%rsp)
        movss     -20(%rsp), %xmm3
        movaps    %xmm3, %xmm0
        movss     .L_2il0floatpacket.80(%rip), %xmm2
        movaps    %xmm2, %xmm1
        addss     .L_2il0floatpacket.79(%rip), %xmm0
        addss     %xmm3, %xmm1
        movss     %xmm0, -24(%rsp)
        movl      -24(%rsp), %r8d
        movss     %xmm1, -24(%rsp)
        andl      $127, %r8d
        movss     -24(%rsp), %xmm9
        movss     .L_2il0floatpacket.89(%rip), %xmm6
        subss     %xmm2, %xmm9
        movzwl    -14(%rsp), %edx
        lea       (%r8,%r8,2), %r9d
        movss     (%r10,%r9,4), %xmm7
        andl      $32640, %edx
        shrl      $7, %edx
        subss     %xmm9, %xmm3
        mulss     %xmm7, %xmm9
        mulss     %xmm3, %xmm7
        subss     .L_2il0floatpacket.81(%rip), %xmm9
        movaps    %xmm9, %xmm4
        lea       -127(%rcx,%rdx), %ecx
        cvtsi2ss  %ecx, %xmm8
        addss     %xmm7, %xmm4
        mulss     %xmm4, %xmm6
        movss     .L_2il0floatpacket.90(%rip), %xmm10
        mulss     %xmm8, %xmm10
        addss     .L_2il0floatpacket.88(%rip), %xmm6
        addss     4(%r10,%r9,4), %xmm10
        mulss     %xmm4, %xmm6
        addss     %xmm9, %xmm10
        addss     .L_2il0floatpacket.87(%rip), %xmm6
        mulss     %xmm4, %xmm6
        movss     .L_2il0floatpacket.91(%rip), %xmm5
        mulss     %xmm5, %xmm8
        addss     .L_2il0floatpacket.86(%rip), %xmm6
        addss     8(%r10,%r9,4), %xmm8
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.85(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.84(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.83(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.82(%rip), %xmm6
        mulss     %xmm6, %xmm9
        mulss     %xmm7, %xmm6
        addss     %xmm6, %xmm8
        addss     %xmm7, %xmm8
        addss     %xmm8, %xmm9
        addss     %xmm9, %xmm10
        movss     %xmm10, (%rsi)
        ret

.LBL_2_7:

        movss     .L_2il0floatpacket.81(%rip), %xmm0
        mulss     %xmm0, %xmm1
        movss     .L_2il0floatpacket.89(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.88(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.87(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.86(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.85(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.84(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.83(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     .L_2il0floatpacket.82(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     %xmm1, %xmm2
        movss     %xmm2, (%rsi)
        ret

.LBL_2_8:

        ucomiss   %xmm0, %xmm2
        jp        .LBL_2_9
        je        .LBL_2_11

.LBL_2_9:

        divss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        movl      $1, %eax


        ret

.LBL_2_11:

        movss     .L_2il0floatpacket.92(%rip), %xmm1
        movl      $2, %eax
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret

.LBL_2_12:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_14

.LBL_2_13:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_14:

        testl     $8388607, (%rdi)
        jne       .LBL_2_13


        movl      $1, %eax
        pxor      %xmm1, %xmm1
        pxor      %xmm0, %xmm0
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_slog10_cout_rare_internal,@function
	.size	__svml_slog10_cout_rare_internal,.-__svml_slog10_cout_rare_internal
..LN__svml_slog10_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_slog10_data_internal_avx512:
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
	.long	3184111259
	.long	3181837556
	.long	3180047478
	.long	3177957936
	.long	3175655835
	.long	3173780127
	.long	3172237531
	.long	3170958066
	.long	3196205737
	.long	3194315592
	.long	3192585145
	.long	3191105685
	.long	3189833727
	.long	3188734475
	.long	3187779868
	.long	3186223182
	.long	1041488008
	.long	1041279100
	.long	1040946627
	.long	1040547595
	.long	1040048184
	.long	1039171801
	.long	1038306334
	.long	1037468467
	.long	1034710670
	.long	1037388369
	.long	1039339328
	.long	1040458589
	.long	1040937011
	.long	1041247376
	.long	1041427373
	.long	1041506296
	.long	3193854619
	.long	3193841271
	.long	3193799669
	.long	3193724950
	.long	3193617675
	.long	3193480957
	.long	3193318914
	.long	3193135864
	.long	3194544764
	.long	3194251079
	.long	3194067571
	.long	3193958490
	.long	3193898352
	.long	3193869008
	.long	3193857579
	.long	3193854984
	.long	1054759896
	.long	1054759749
	.long	1054758872
	.long	1054756529
	.long	1054752055
	.long	1054744934
	.long	1054734809
	.long	1054721467
	.long	1054738669
	.long	1054749415
	.long	1054755175
	.long	1054758032
	.long	1054759295
	.long	1054759760
	.long	1054759882
	.long	1054759897
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.long	1050288283
	.type	__svml_slog10_data_internal_avx512,@object
	.size	__svml_slog10_data_internal_avx512,384
	.align 32
__slog10_la_CoutTab:
	.long	1121868800
	.long	0
	.long	0
	.long	1121641104
	.long	1004535808
	.long	912917177
	.long	1121413408
	.long	1013055488
	.long	3065901602
	.long	1121185712
	.long	1017839616
	.long	3083361151
	.long	1120958016
	.long	1021673472
	.long	929297206
	.long	1120844168
	.long	1023524864
	.long	3077496589
	.long	1120616472
	.long	1025499136
	.long	3070500046
	.long	1120388776
	.long	1027506176
	.long	912271551
	.long	1120274928
	.long	1028521984
	.long	927716856
	.long	1120047232
	.long	1030586368
	.long	923757491
	.long	1119933384
	.long	1031634944
	.long	3056752848
	.long	1119705688
	.long	1032775680
	.long	917029265
	.long	1119591840
	.long	1033314304
	.long	3065085585
	.long	1119364144
	.long	1034403840
	.long	3064614024
	.long	1119250296
	.long	1034954752
	.long	921091539
	.long	1119136448
	.long	1035513856
	.long	3057436454
	.long	1118908752
	.long	1036644352
	.long	922468856
	.long	1118794904
	.long	1037219840
	.long	3049155845
	.long	1118681056
	.long	1037799424
	.long	904301451
	.long	1118567208
	.long	1038385152
	.long	908617625
	.long	1118453360
	.long	1038977024
	.long	905362229
	.long	1118225664
	.long	1040179200
	.long	3027570914
	.long	1118111816
	.long	1040488448
	.long	882280038
	.long	1117997968
	.long	1040796672
	.long	911375775
	.long	1117884120
	.long	1041108480
	.long	904500572
	.long	1117770272
	.long	1041423872
	.long	3057579304
	.long	1117656424
	.long	1041742336
	.long	3053334705
	.long	1117542576
	.long	1042064384
	.long	3053389931
	.long	1117428728
	.long	1042390016
	.long	3051561465
	.long	1117314880
	.long	1042719232
	.long	3011187895
	.long	1117201032
	.long	1043052544
	.long	3059907089
	.long	1117087184
	.long	1043389440
	.long	3057005374
	.long	1116973336
	.long	1043729920
	.long	911932638
	.long	1116859488
	.long	1044075008
	.long	892958461
	.long	1116859488
	.long	1044075008
	.long	892958461
	.long	1116745640
	.long	1044424192
	.long	3048660547
	.long	1116631792
	.long	1044777472
	.long	3049032043
	.long	1116517944
	.long	1045134848
	.long	906867152
	.long	1116404096
	.long	1045496832
	.long	911484894
	.long	1116404096
	.long	1045496832
	.long	911484894
	.long	1116290248
	.long	1045863424
	.long	912580963
	.long	1116176400
	.long	1046235136
	.long	3058440244
	.long	1116062552
	.long	1046610944
	.long	895945194
	.long	1116062552
	.long	1046610944
	.long	895945194
	.long	1115948704
	.long	1046991872
	.long	904357324
	.long	1115834856
	.long	1047377920
	.long	902293870
	.long	1115721008
	.long	1047769088
	.long	907149878
	.long	1115721008
	.long	1047769088
	.long	907149878
	.long	1115529456
	.long	1048165888
	.long	3052029263
	.long	1115301760
	.long	1048567808
	.long	3035959522
	.long	1115301760
	.long	1048567808
	.long	3035959522
	.long	1115074064
	.long	1048775680
	.long	892998645
	.long	1115074064
	.long	1048775680
	.long	892998645
	.long	1114846368
	.long	1048982400
	.long	881767775
	.long	1114618672
	.long	1049192064
	.long	893839142
	.long	1114618672
	.long	1049192064
	.long	893839142
	.long	1114390976
	.long	1049404800
	.long	896498651
	.long	1114390976
	.long	1049404800
	.long	896498651
	.long	1114163280
	.long	1049620736
	.long	3033695903
	.long	1114163280
	.long	1049620736
	.long	3033695903
	.long	1113935584
	.long	1049839872
	.long	3029986056
	.long	1113935584
	.long	1049839872
	.long	3029986056
	.long	1113707888
	.long	1050062336
	.long	884671939
	.long	1113707888
	.long	1050062336
	.long	884671939
	.long	1113480192
	.long	1050288256
	.long	894707678
	.long	1050279936
	.long	964848148
	.long	1207959616
	.long	1174405120
	.long	1002438656
	.long	1400897536
	.long	0
	.long	1065353216
	.long	1121868800
	.long	3212771328
	.long	3079888218
	.long	870463078
	.long	2957202361
	.long	749987585
	.long	2838272395
	.long	631921661
	.long	2720751022
	.type	__slog10_la_CoutTab,@object
	.size	__slog10_la_CoutTab,848
	.align 4
.L_2il0floatpacket.77:
	.long	0x53800000
	.type	.L_2il0floatpacket.77,@object
	.size	.L_2il0floatpacket.77,4
	.align 4
.L_2il0floatpacket.78:
	.long	0x3bc00000
	.type	.L_2il0floatpacket.78,@object
	.size	.L_2il0floatpacket.78,4
	.align 4
.L_2il0floatpacket.79:
	.long	0x48000040
	.type	.L_2il0floatpacket.79,@object
	.size	.L_2il0floatpacket.79,4
	.align 4
.L_2il0floatpacket.80:
	.long	0x46000000
	.type	.L_2il0floatpacket.80,@object
	.size	.L_2il0floatpacket.80,4
	.align 4
.L_2il0floatpacket.81:
	.long	0x42de5c00
	.type	.L_2il0floatpacket.81,@object
	.size	.L_2il0floatpacket.81,4
	.align 4
.L_2il0floatpacket.82:
	.long	0xbf7f0000
	.type	.L_2il0floatpacket.82,@object
	.size	.L_2il0floatpacket.82,4
	.align 4
.L_2il0floatpacket.83:
	.long	0xb7935d5a
	.type	.L_2il0floatpacket.83,@object
	.size	.L_2il0floatpacket.83,4
	.align 4
.L_2il0floatpacket.84:
	.long	0x33e23666
	.type	.L_2il0floatpacket.84,@object
	.size	.L_2il0floatpacket.84,4
	.align 4
.L_2il0floatpacket.85:
	.long	0xb04353b9
	.type	.L_2il0floatpacket.85,@object
	.size	.L_2il0floatpacket.85,4
	.align 4
.L_2il0floatpacket.86:
	.long	0x2cb3e701
	.type	.L_2il0floatpacket.86,@object
	.size	.L_2il0floatpacket.86,4
	.align 4
.L_2il0floatpacket.87:
	.long	0xa92c998b
	.type	.L_2il0floatpacket.87,@object
	.size	.L_2il0floatpacket.87,4
	.align 4
.L_2il0floatpacket.88:
	.long	0x25aa5bfd
	.type	.L_2il0floatpacket.88,@object
	.size	.L_2il0floatpacket.88,4
	.align 4
.L_2il0floatpacket.89:
	.long	0xa22b5dae
	.type	.L_2il0floatpacket.89,@object
	.size	.L_2il0floatpacket.89,4
	.align 4
.L_2il0floatpacket.90:
	.long	0x3e9a0000
	.type	.L_2il0floatpacket.90,@object
	.size	.L_2il0floatpacket.90,4
	.align 4
.L_2il0floatpacket.91:
	.long	0x39826a14
	.type	.L_2il0floatpacket.91,@object
	.size	.L_2il0floatpacket.91,4
	.align 4
.L_2il0floatpacket.92:
	.long	0xbf800000
	.type	.L_2il0floatpacket.92,@object
	.size	.L_2il0floatpacket.92,4
	.align 4
.L_2il0floatpacket.93:
	.long	0x3f800000
	.type	.L_2il0floatpacket.93,@object
	.size	.L_2il0floatpacket.93,4
