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
 *  *      and the table lookup for log2(RCP(mx)) is based on a small permute instruction
 *  *
 *  *   LA, EP versions use interval interpolation (16 intervals)
 *  *
 *  
 */


	.text
.L_2__routine_start___svml_log2f16_z0_0:

	.align    16,0x90
	.globl __svml_log2f16

__svml_log2f16:


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
        vmovups   __svml_slog2_data_internal_avx512(%rip), %zmm1
        vgetexpps {sae}, %zmm0, %zmm5

/* x<=0? */
        vfpclassps $94, %zmm0, %k0
        vsubps    {rn-sae}, %zmm1, %zmm3, %zmm9
        vpsrld    $19, %zmm3, %zmm7
        vgetexpps {sae}, %zmm3, %zmm6
        vpermps   64+__svml_slog2_data_internal_avx512(%rip), %zmm7, %zmm1
        vpermps   128+__svml_slog2_data_internal_avx512(%rip), %zmm7, %zmm2
        vpermps   192+__svml_slog2_data_internal_avx512(%rip), %zmm7, %zmm4
        vpermps   256+__svml_slog2_data_internal_avx512(%rip), %zmm7, %zmm8
        vsubps    {rn-sae}, %zmm6, %zmm5, %zmm10
        vfmadd213ps {rn-sae}, %zmm2, %zmm9, %zmm1
        kmovw     %k0, %edx
        vfmadd213ps {rn-sae}, %zmm4, %zmm9, %zmm1
        vfmadd213ps {rn-sae}, %zmm8, %zmm9, %zmm1
        vfmadd213ps {rn-sae}, %zmm10, %zmm9, %zmm1
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

        call      __svml_slog2_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_log2f16,@function
	.size	__svml_log2f16,.-__svml_log2f16
..LN__svml_log2f16.0:

.L_2__routine_start___svml_slog2_cout_rare_internal_1:

	.align    16,0x90

__svml_slog2_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    2(%rdi), %edx
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_13


        movss     (%rdi), %xmm2
        xorl      %ecx, %ecx
        pxor      %xmm1, %xmm1
        movss     %xmm2, -16(%rsp)
        ucomiss   %xmm1, %xmm2
        jp        .LBL_2_3
        je        .LBL_2_5

.LBL_2_3:

        movzwl    -14(%rsp), %edx
        testl     $32640, %edx
        jne       .LBL_2_5


        movss     .L_2il0floatpacket.76(%rip), %xmm0
        movl      $-27, %ecx
        mulss     %xmm0, %xmm2
        movss     %xmm2, -16(%rsp)

.LBL_2_5:

        comiss    %xmm1, %xmm2
        jbe       .LBL_2_9


        movaps    %xmm2, %xmm1
        subss     .L_2il0floatpacket.90(%rip), %xmm1
        movss     %xmm1, -20(%rsp)
        andb      $127, -17(%rsp)
        movss     -20(%rsp), %xmm0
        comiss    .L_2il0floatpacket.77(%rip), %xmm0
        jbe       .LBL_2_8


        movzwl    -14(%rsp), %edx
        pxor      %xmm8, %xmm8
        andl      $32640, %edx
        lea       __slog2_la_CoutTab(%rip), %r10
        shrl      $7, %edx
        movss     %xmm2, -20(%rsp)
        movss     .L_2il0floatpacket.79(%rip), %xmm2
        movaps    %xmm2, %xmm1
        movss     .L_2il0floatpacket.88(%rip), %xmm6
        lea       -127(%rcx,%rdx), %r9d
        movzwl    -18(%rsp), %ecx
        andl      $-32641, %ecx
        addl      $16256, %ecx
        movw      %cx, -18(%rsp)
        movss     -20(%rsp), %xmm3
        movaps    %xmm3, %xmm0
        addss     %xmm3, %xmm1
        addss     .L_2il0floatpacket.78(%rip), %xmm0
        cvtsi2ss  %r9d, %xmm8
        movss     %xmm0, -24(%rsp)
        movl      -24(%rsp), %edi
        movss     %xmm1, -24(%rsp)
        andl      $127, %edi
        movss     -24(%rsp), %xmm7
        subss     %xmm2, %xmm7
        lea       (%rdi,%rdi,2), %r8d
        movss     (%r10,%r8,4), %xmm5
        subss     %xmm7, %xmm3
        addss     4(%r10,%r8,4), %xmm8
        mulss     %xmm5, %xmm7
        mulss     %xmm3, %xmm5
        subss     .L_2il0floatpacket.80(%rip), %xmm7
        movaps    %xmm7, %xmm4
        addss     %xmm7, %xmm8
        addss     %xmm5, %xmm4
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.87(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.86(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.85(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.84(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.83(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.82(%rip), %xmm6
        mulss     %xmm4, %xmm6
        addss     .L_2il0floatpacket.81(%rip), %xmm6
        mulss     %xmm6, %xmm7
        mulss     %xmm5, %xmm6
        addss     8(%r10,%r8,4), %xmm6
        addss     %xmm5, %xmm6
        addss     %xmm6, %xmm7
        addss     %xmm7, %xmm8
        movss     %xmm8, (%rsi)
        ret

.LBL_2_8:

        movss     .L_2il0floatpacket.80(%rip), %xmm0
        mulss     %xmm0, %xmm1
        movss     .L_2il0floatpacket.88(%rip), %xmm2
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
        addss     .L_2il0floatpacket.81(%rip), %xmm2
        mulss     %xmm1, %xmm2
        addss     %xmm1, %xmm2
        movss     %xmm2, (%rsi)
        ret

.LBL_2_9:

        ucomiss   %xmm1, %xmm2
        jp        .LBL_2_10
        je        .LBL_2_12

.LBL_2_10:

        divss     %xmm1, %xmm1
        movss     %xmm1, (%rsi)
        movl      $1, %eax


        ret

.LBL_2_12:

        movss     .L_2il0floatpacket.89(%rip), %xmm0
        movl      $2, %eax
        divss     %xmm1, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_13:

        movb      3(%rdi), %dl
        andb      $-128, %dl
        cmpb      $-128, %dl
        je        .LBL_2_15

.LBL_2_14:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        ret

.LBL_2_15:

        testl     $8388607, (%rdi)
        jne       .LBL_2_14


        movl      $1, %eax
        pxor      %xmm1, %xmm1
        pxor      %xmm0, %xmm0
        divss     %xmm0, %xmm1
        movss     %xmm1, (%rsi)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_slog2_cout_rare_internal,@function
	.size	__svml_slog2_cout_rare_internal,.-__svml_slog2_cout_rare_internal
..LN__svml_slog2_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_slog2_data_internal_avx512:
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
	.long	3198647882
	.long	3196759613
	.long	3194486322
	.long	3192115638
	.long	3190203785
	.long	3188646043
	.long	3187058849
	.long	3184933705
	.long	3210235473
	.long	3208544417
	.long	3207107312
	.long	3205878647
	.long	3204822309
	.long	3203370544
	.long	3201784975
	.long	3200401796
	.long	1056280844
	.long	1055933857
	.long	1055381630
	.long	1054718852
	.long	1054004959
	.long	1053277138
	.long	1052558383
	.long	1051862550
	.long	1049572249
	.long	1051796030
	.long	1053416266
	.long	1054571017
	.long	1055365658
	.long	1055881163
	.long	1056180132
	.long	1056311220
	.long	3208161588
	.long	3208150502
	.long	3208115952
	.long	3208053899
	.long	3207964810
	.long	3207851268
	.long	3207716694
	.long	3207564674
	.long	3208734740
	.long	3208490840
	.long	3208338440
	.long	3208247850
	.long	3208197907
	.long	3208173537
	.long	3208164045
	.long	3208161890
	.long	1069066811
	.long	1069066688
	.long	1069065960
	.long	1069064014
	.long	1069060299
	.long	1069054385
	.long	1069045976
	.long	1069034896
	.long	1069049182
	.long	1069058106
	.long	1069062890
	.long	1069065263
	.long	1069066312
	.long	1069066697
	.long	1069066799
	.long	1069066811
	.type	__svml_slog2_data_internal_avx512,@object
	.size	__svml_slog2_data_internal_avx512,320
	.align 32
__slog2_la_CoutTab:
	.long	1136175680
	.long	0
	.long	0
	.long	1135986583
	.long	1018822656
	.long	930849160
	.long	1135809305
	.long	1026916352
	.long	941737263
	.long	1135632026
	.long	1032306688
	.long	936581683
	.long	1135466566
	.long	1035100160
	.long	929197062
	.long	1135301106
	.long	1037934592
	.long	897678483
	.long	1135135647
	.long	1040498688
	.long	3059980496
	.long	1134982005
	.long	1041852416
	.long	908010313
	.long	1134828364
	.long	1043226624
	.long	3073739761
	.long	1134686541
	.long	1044510720
	.long	918631281
	.long	1134538809
	.long	1045868544
	.long	3062817788
	.long	1134402896
	.long	1047134208
	.long	3064656237
	.long	1134266982
	.long	1048416256
	.long	3029590737
	.long	1134131069
	.long	1049145856
	.long	903671587
	.long	1134001065
	.long	1049775616
	.long	911388989
	.long	1133876970
	.long	1050384896
	.long	3069885983
	.long	1133752875
	.long	1051001344
	.long	3037530952
	.long	1133634689
	.long	1051596288
	.long	3069922038
	.long	1133516503
	.long	1052198400
	.long	3070222063
	.long	1133404227
	.long	1052776960
	.long	919559368
	.long	1133291951
	.long	1053363200
	.long	840060372
	.long	1133185584
	.long	1053924864
	.long	915603033
	.long	1133079217
	.long	1054493184
	.long	921334924
	.long	1132978759
	.long	1055036416
	.long	896601826
	.long	1132872392
	.long	1055618048
	.long	908913293
	.long	1132777843
	.long	1056141312
	.long	3065728751
	.long	1132677386
	.long	1056702976
	.long	909020429
	.long	1132582837
	.long	1057101312
	.long	3048020321
	.long	1132494198
	.long	1057354752
	.long	3038815896
	.long	1132337219
	.long	1057628160
	.long	3068137421
	.long	1132159940
	.long	1057887232
	.long	3069993595
	.long	1131994480
	.long	1058131456
	.long	3054354312
	.long	1131817202
	.long	1058395904
	.long	910223436
	.long	1131651742
	.long	1058645504
	.long	3046952660
	.long	1131486282
	.long	1058897664
	.long	3057670844
	.long	1131332641
	.long	1059133952
	.long	924929721
	.long	1131178999
	.long	1059373056
	.long	3068093797
	.long	1131025358
	.long	1059614208
	.long	3058851683
	.long	1130871717
	.long	1059857920
	.long	3069897752
	.long	1130729894
	.long	1060084736
	.long	924446297
	.long	1130576253
	.long	1060333312
	.long	903058075
	.long	1130434430
	.long	1060564992
	.long	3052757441
	.long	1130304426
	.long	1060779264
	.long	3045479197
	.long	1130162603
	.long	1061015040
	.long	924699798
	.long	1130032599
	.long	1061233664
	.long	3070937808
	.long	1129890776
	.long	1061473792
	.long	925912756
	.long	1129772591
	.long	1061676032
	.long	923952205
	.long	1129642586
	.long	1061900544
	.long	906547304
	.long	1129512582
	.long	1062127104
	.long	3050351427
	.long	1129394397
	.long	1062334976
	.long	3070601694
	.long	1129276211
	.long	1062544384
	.long	900519722
	.long	1129158025
	.long	1062755840
	.long	3055774932
	.long	1129039840
	.long	1062969088
	.long	3053661845
	.long	1128921654
	.long	1063184384
	.long	3073448373
	.long	1128815287
	.long	1063379456
	.long	907090876
	.long	1128697101
	.long	1063598336
	.long	881051555
	.long	1128590734
	.long	1063796992
	.long	898320955
	.long	1128484367
	.long	1063997440
	.long	3068804107
	.long	1128378000
	.long	1064199168
	.long	923531617
	.long	1128283452
	.long	1064380416
	.long	3070994608
	.long	1128177085
	.long	1064585472
	.long	901920533
	.long	1128082536
	.long	1064769536
	.long	3071653428
	.long	1127976169
	.long	1064977920
	.long	903017594
	.long	1127881621
	.long	1065164800
	.long	911713416
	.long	1127787072
	.long	1065353216
	.long	0
	.long	1065353216
	.long	0
	.long	1207959616
	.long	1174405120
	.long	1002438656
	.long	1291845632
	.long	0
	.long	1065353216
	.long	1136175680
	.long	3212771328
	.long	3065082383
	.long	841219731
	.long	2913632803
	.long	691870088
	.long	2765780188
	.long	545377693
	.long	2619180638
	.type	__slog2_la_CoutTab,@object
	.size	__slog2_la_CoutTab,848
	.align 4
.L_2il0floatpacket.76:
	.long	0x4d000000
	.type	.L_2il0floatpacket.76,@object
	.size	.L_2il0floatpacket.76,4
	.align 4
.L_2il0floatpacket.77:
	.long	0x3bc00000
	.type	.L_2il0floatpacket.77,@object
	.size	.L_2il0floatpacket.77,4
	.align 4
.L_2il0floatpacket.78:
	.long	0x48000040
	.type	.L_2il0floatpacket.78,@object
	.size	.L_2il0floatpacket.78,4
	.align 4
.L_2il0floatpacket.79:
	.long	0x46000000
	.type	.L_2il0floatpacket.79,@object
	.size	.L_2il0floatpacket.79,4
	.align 4
.L_2il0floatpacket.80:
	.long	0x43b8aa40
	.type	.L_2il0floatpacket.80,@object
	.size	.L_2il0floatpacket.80,4
	.align 4
.L_2il0floatpacket.81:
	.long	0xbf7f0000
	.type	.L_2il0floatpacket.81,@object
	.size	.L_2il0floatpacket.81,4
	.align 4
.L_2il0floatpacket.82:
	.long	0xb6b1720f
	.type	.L_2il0floatpacket.82,@object
	.size	.L_2il0floatpacket.82,4
	.align 4
.L_2il0floatpacket.83:
	.long	0x3223fe93
	.type	.L_2il0floatpacket.83,@object
	.size	.L_2il0floatpacket.83,4
	.align 4
.L_2il0floatpacket.84:
	.long	0xadaa8223
	.type	.L_2il0floatpacket.84,@object
	.size	.L_2il0floatpacket.84,4
	.align 4
.L_2il0floatpacket.85:
	.long	0x293d1988
	.type	.L_2il0floatpacket.85,@object
	.size	.L_2il0floatpacket.85,4
	.align 4
.L_2il0floatpacket.86:
	.long	0xa4da74dc
	.type	.L_2il0floatpacket.86,@object
	.size	.L_2il0floatpacket.86,4
	.align 4
.L_2il0floatpacket.87:
	.long	0x2081cd9d
	.type	.L_2il0floatpacket.87,@object
	.size	.L_2il0floatpacket.87,4
	.align 4
.L_2il0floatpacket.88:
	.long	0x9c1d865e
	.type	.L_2il0floatpacket.88,@object
	.size	.L_2il0floatpacket.88,4
	.align 4
.L_2il0floatpacket.89:
	.long	0xbf800000
	.type	.L_2il0floatpacket.89,@object
	.size	.L_2il0floatpacket.89,4
	.align 4
.L_2il0floatpacket.90:
	.long	0x3f800000
	.type	.L_2il0floatpacket.90,@object
	.size	.L_2il0floatpacket.90,4
