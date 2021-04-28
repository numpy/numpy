/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *   NOTE: Since the hyperbolic tangent function is odd
 *         (tanh(x) = -tanh(-x)), below algorithm deals with the absolute
 *         value of the argument |x|: tanh(x) = sign(x) * tanh(|x|)
 * 
 *   We use a table lookup method to compute tanh(|x|).
 *   The basic idea is to split the input range into a number of subintervals
 *   and to approximate tanh(.) with a polynomial on each of them.
 * 
 *   IEEE SPECIAL CONDITIONS:
 *   x = [+,-]0, r = [+,-]0
 *   x = +Inf,   r = +1
 *   x = -Inf,   r = -1
 *   x = QNaN,   r = QNaN
 *   x = SNaN,   r = QNaN
 * 
 * 
 *   ALGORITHM DETAILS
 *   We handle special values in a callout function, aside from main path
 *   computations. "Special" for this algorithm are:
 *   INF, NAN, |x| > HUGE_THRESHOLD
 * 
 * 
 *   Main path computations are organized as follows:
 *   Actually we split the interval [0, SATURATION_THRESHOLD)
 *   into a number of subintervals.  On each subinterval we approximate tanh(.)
 *   with a minimax polynomial of pre-defined degree. Polynomial coefficients
 *   are computed beforehand and stored in table. We also use
 * 
 *       y := |x| + B,
 * 
 *   here B depends on subinterval and is used to make argument
 *   closer to zero.
 *   We also add large fake interval [SATURATION_THRESHOLD, HUGE_THRESHOLD],
 *   where 1.0 + 0.0*y + 0.0*y^2 ... coefficients are stored - just to
 *   preserve main path computation logic but return 1.0 for all arguments.
 * 
 *   Hence reconstruction looks as follows:
 *   we extract proper polynomial and range reduction coefficients
 *        (Pj and B), corresponding to subinterval, to which |x| belongs,
 *        and return
 * 
 *       r := sign(x) * (P0 + P1 * y + ... + Pn * y^n)
 * 
 *   NOTE: we use multiprecision technique to multiply and sum the first
 *         K terms of the polynomial. So Pj, j = 0..K are stored in
 *         table each as a pair of target precision numbers (Pj and PLj) to
 *         achieve wider than target precision.
 * 
 * --
 * 
 */


	.text
.L_2__routine_start___svml_tanhf16_z0_0:

	.align    16,0x90
	.globl __svml_tanhf16

__svml_tanhf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovaps   %zmm0, %zmm1
        vmovups   __svml_stanh_data_internal(%rip), %zmm9
        vmovups   896+__svml_stanh_data_internal(%rip), %zmm11
        vmovups   768+__svml_stanh_data_internal(%rip), %zmm12
        vmovups   640+__svml_stanh_data_internal(%rip), %zmm13
        vmovups   512+__svml_stanh_data_internal(%rip), %zmm14
        vmovups   384+__svml_stanh_data_internal(%rip), %zmm15
        vpternlogd $255, %zmm2, %zmm2, %zmm2
        vandps    5696+__svml_stanh_data_internal(%rip), %zmm1, %zmm8
        vandps    5632+__svml_stanh_data_internal(%rip), %zmm1, %zmm0

/* Here huge arguments, INF and NaNs are filtered out to callout. */
        vpandd    1152+__svml_stanh_data_internal(%rip), %zmm1, %zmm3
        vpsubd    1216+__svml_stanh_data_internal(%rip), %zmm3, %zmm4
        vpcmpd    $2, 5824+__svml_stanh_data_internal(%rip), %zmm3, %k1

/*
 * * small table specific variables *
 * **********************************
 * -------------------- Constant loading -------------------
 */
        vpxord    %zmm5, %zmm5, %zmm5

/* if VMIN, VMAX is defined for I type */
        vpmaxsd   %zmm5, %zmm4, %zmm6
        vpminsd   1280+__svml_stanh_data_internal(%rip), %zmm6, %zmm7
        vpsrld    $21, %zmm7, %zmm10
        vmovups   1024+__svml_stanh_data_internal(%rip), %zmm4
        vpermt2ps 64+__svml_stanh_data_internal(%rip), %zmm10, %zmm9
        vpermt2ps 960+__svml_stanh_data_internal(%rip), %zmm10, %zmm11
        vpermt2ps 1088+__svml_stanh_data_internal(%rip), %zmm10, %zmm4
        vpermt2ps 832+__svml_stanh_data_internal(%rip), %zmm10, %zmm12
        vpermt2ps 704+__svml_stanh_data_internal(%rip), %zmm10, %zmm13
        vpermt2ps 576+__svml_stanh_data_internal(%rip), %zmm10, %zmm14
        vpermt2ps 448+__svml_stanh_data_internal(%rip), %zmm10, %zmm15
        vpandnd   %zmm3, %zmm3, %zmm2{%k1}
        vptestmd  %zmm2, %zmm2, %k0
        vmovups   128+__svml_stanh_data_internal(%rip), %zmm3
        vsubps    {rn-sae}, %zmm9, %zmm8, %zmm2
        kmovw     %k0, %edx
        vfmadd213ps {rn-sae}, %zmm11, %zmm2, %zmm4
        vpermt2ps 192+__svml_stanh_data_internal(%rip), %zmm10, %zmm3
        vfmadd213ps {rn-sae}, %zmm12, %zmm2, %zmm4
        vfmadd213ps {rn-sae}, %zmm13, %zmm2, %zmm4
        vfmadd213ps {rn-sae}, %zmm14, %zmm2, %zmm4
        vfmadd213ps {rn-sae}, %zmm15, %zmm2, %zmm4
        vfmadd213ps {rn-sae}, %zmm3, %zmm2, %zmm4
        vorps     %zmm0, %zmm4, %zmm0
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

        vmovups   %zmm1, 64(%rsp)
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

        call      __svml_stanh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_tanhf16,@function
	.size	__svml_tanhf16,.-__svml_tanhf16
..LN__svml_tanhf16.0:

.L_2__routine_start___svml_stanh_cout_rare_internal_1:

	.align    16,0x90

__svml_stanh_cout_rare_internal:


	.cfi_startproc
..L53:

        lea       __stanh_la__imlsTanhTab(%rip), %rdx
        movb      3(%rdi), %al
        andb      $-128, %al
        shrb      $7, %al
        movzbl    %al, %ecx
        movzwl    2(%rdi), %r8d
        andl      $32640, %r8d
        movl      (%rdx,%rcx,4), %eax
        cmpl      $32640, %r8d
        je        .LBL_2_4

.LBL_2_2:

        movl      %eax, (%rsi)

.LBL_2_3:

        xorl      %eax, %eax
        ret

.LBL_2_4:

        testl     $8388607, (%rdi)
        je        .LBL_2_2


        movss     (%rdi), %xmm0
        addss     %xmm0, %xmm0
        movss     %xmm0, (%rsi)
        jmp       .LBL_2_3
	.align    16,0x90

	.cfi_endproc

	.type	__svml_stanh_cout_rare_internal,@function
	.size	__svml_stanh_cout_rare_internal,.-__svml_stanh_cout_rare_internal
..LN__svml_stanh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_stanh_data_internal:
	.long	0
	.long	1030750208
	.long	1032847360
	.long	1034944512
	.long	1037041664
	.long	1039138816
	.long	1041235968
	.long	1043333120
	.long	1045430272
	.long	1047527424
	.long	1049624576
	.long	1051721728
	.long	1053818880
	.long	1055916032
	.long	1058013184
	.long	1060110336
	.long	1062207488
	.long	1064304640
	.long	1066401792
	.long	1068498944
	.long	1070596096
	.long	1072693248
	.long	1074790400
	.long	1076887552
	.long	1078984704
	.long	1081081856
	.long	1083179008
	.long	1085276160
	.long	1087373312
	.long	1089470464
	.long	1091567616
	.long	0
	.long	0
	.long	1030732233
	.long	1032831839
	.long	1034916201
	.long	1036994987
	.long	1039067209
	.long	1041174248
	.long	1043220868
	.long	1045245838
	.long	1047245614
	.long	1049383373
	.long	1051287907
	.long	1053115377
	.long	1054857013
	.long	1057129528
	.long	1058581488
	.long	1059832960
	.long	1060891676
	.long	1062153819
	.long	1063337043
	.long	1064100733
	.long	1064582223
	.long	1064984555
	.long	1065216645
	.long	1065302845
	.long	1065334668
	.long	1065349076
	.long	1065352656
	.long	1065353140
	.long	1065353206
	.long	1065353215
	.long	1065353216
	.long	0
	.long	2963361822
	.long	2971470750
	.long	2945658640
	.long	821708412
	.long	824483568
	.long	824941280
	.long	2984085072
	.long	2957298688
	.long	838449816
	.long	2966046080
	.long	2988320324
	.long	2989804564
	.long	842626356
	.long	3000013710
	.long	2972725824
	.long	3002017674
	.long	853753500
	.long	2987104448
	.long	3000350914
	.long	855535800
	.long	852410906
	.long	851608946
	.long	2988641656
	.long	2997011000
	.long	2989576736
	.long	3000884068
	.long	2999984336
	.long	840950056
	.long	2995215280
	.long	855269702
	.long	0
	.long	1065353216
	.long	1065295748
	.long	1065270545
	.long	1065229919
	.long	1065181343
	.long	1065124909
	.long	1065025765
	.long	1064867200
	.long	1064679597
	.long	1064464345
	.long	1064093083
	.long	1063517074
	.long	1062862743
	.long	1062146519
	.long	1060992371
	.long	1059386208
	.long	1057800167
	.long	1055660649
	.long	1051764737
	.long	1046959010
	.long	1041444634
	.long	1035462611
	.long	1026689093
	.long	1015337940
	.long	1002731447
	.long	990958554
	.long	973168670
	.long	948705851
	.long	924299482
	.long	899955662
	.long	864224966
	.long	0
	.long	2956213371
	.long	3178161821
	.long	3180268967
	.long	3182315389
	.long	3184339487
	.long	3186337805
	.long	3188474939
	.long	3190373619
	.long	3192189570
	.long	3193910865
	.long	3196176320
	.long	3197556682
	.long	3198679950
	.long	3199536798
	.long	3200331518
	.long	3200564882
	.long	3200049264
	.long	3199029518
	.long	3197040598
	.long	3192620804
	.long	3188208183
	.long	3182392393
	.long	3173916356
	.long	3162750726
	.long	3150176437
	.long	3138431708
	.long	3120650203
	.long	3096189170
	.long	3071783062
	.long	3047439278
	.long	3011707180
	.long	0
	.long	3198855845
	.long	3198879250
	.long	3198677023
	.long	3198476576
	.long	3198388151
	.long	3198245218
	.long	3197982711
	.long	3197594458
	.long	3197117197
	.long	3196587519
	.long	3195304371
	.long	3192667528
	.long	3189843074
	.long	3186330810
	.long	3177085101
	.long	1013669486
	.long	1032032579
	.long	1036132065
	.long	1038305199
	.long	1036774550
	.long	1033498413
	.long	1028927137
	.long	1021175553
	.long	1009568359
	.long	998361895
	.long	985691041
	.long	967585842
	.long	943363289
	.long	919210013
	.long	895139148
	.long	858471606
	.long	0
	.long	3077428921
	.long	3189516141
	.long	1008586543
	.long	1036101517
	.long	1033304453
	.long	1034073627
	.long	1036071831
	.long	1037235824
	.long	1039436298
	.long	1040631208
	.long	1041906362
	.long	1042793477
	.long	1043232976
	.long	1043086916
	.long	1042100375
	.long	1039444212
	.long	1034126600
	.long	1026638186
	.long	995501655
	.long	3165579977
	.long	3167654937
	.long	3165317828
	.long	3158960080
	.long	3148291549
	.long	3137354510
	.long	3124730373
	.long	3106670759
	.long	3082457650
	.long	3058305807
	.long	3034235241
	.long	2997581996
	.long	0
	.long	1040781545
	.long	1131811139
	.long	1097198812
	.long	3247503190
	.long	3230402941
	.long	3224086547
	.long	3212798938
	.long	1059790272
	.long	1053691997
	.long	1061317268
	.long	3134918084
	.long	1034173207
	.long	3176246152
	.long	3165561405
	.long	3174788493
	.long	3178015405
	.long	3178847213
	.long	3177176538
	.long	3171127099
	.long	3155996003
	.long	985352038
	.long	999682315
	.long	998398067
	.long	989522534
	.long	977926264
	.long	966355955
	.long	948911724
	.long	924561635
	.long	900244966
	.long	875993879
	.long	841254832
	.long	0
	.long	3155046246
	.long	1175181842
	.long	1138112751
	.long	3286309950
	.long	3267011817
	.long	3259619885
	.long	3246758786
	.long	1088248663
	.long	1078543936
	.long	1086795944
	.long	3205436942
	.long	1043392367
	.long	3198686087
	.long	3182586396
	.long	3174374999
	.long	3142320544
	.long	1008565243
	.long	1014115537
	.long	1016545052
	.long	1010017051
	.long	998649588
	.long	975680464
	.long	3124451591
	.long	3121544226
	.long	3112148751
	.long	3100159824
	.long	3082673659
	.long	3058641232
	.long	3034613169
	.long	3010665978
	.long	2975473412
	.long	0
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	2145386496
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	1027604480
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	65011712
	.long	0
	.long	0
	.long	36466923
	.long	1072693248
	.long	2365563284
	.long	3201306247
	.long	1829111721
	.long	3218428278
	.long	3823420055
	.long	3193165324
	.long	1098370908
	.long	1072693250
	.long	2493848101
	.long	3205978300
	.long	1742787824
	.long	3218423470
	.long	4013086462
	.long	3193609799
	.long	3667406935
	.long	1072693250
	.long	1352570026
	.long	3206273363
	.long	1445916117
	.long	3218422747
	.long	3835089556
	.long	3194065339
	.long	2420711447
	.long	1072693251
	.long	2169133006
	.long	3206574718
	.long	101347106
	.long	3218421982
	.long	713636389
	.long	3194411770
	.long	1709214929
	.long	1072693252
	.long	1710199222
	.long	3206758855
	.long	2216669086
	.long	3218421174
	.long	3814844772
	.long	3194836884
	.long	1591910603
	.long	1072693253
	.long	2011589771
	.long	3206962796
	.long	3721194918
	.long	3218420324
	.long	1295469166
	.long	3195182212
	.long	2130808979
	.long	1072693254
	.long	1222040805
	.long	3207187539
	.long	555871541
	.long	3218419433
	.long	401254514
	.long	3195493098
	.long	3390924472
	.long	1072693255
	.long	3112838711
	.long	3207434078
	.long	1558155560
	.long	3218418499
	.long	683885552
	.long	3195864282
	.long	1145294069
	.long	1072693257
	.long	1821670117
	.long	3207648695
	.long	2692181933
	.long	3218417523
	.long	2704036295
	.long	3196181879
	.long	4054831834
	.long	1072693258
	.long	1649153664
	.long	3207795248
	.long	4228604383
	.long	3218416505
	.long	4004758581
	.long	3196440654
	.long	3603542514
	.long	1072693260
	.long	1123737044
	.long	3207954183
	.long	2154630785
	.long	3218415446
	.long	2606327835
	.long	3196743208
	.long	4163276658
	.long	1072693262
	.long	2048619642
	.long	3208125991
	.long	1058893780
	.long	3218414345
	.long	250194901
	.long	3197094852
	.long	1518877548
	.long	1072693265
	.long	3379868633
	.long	3208311162
	.long	1246516380
	.long	3218413202
	.long	1621358934
	.long	3197304764
	.long	48033162
	.long	1072693268
	.long	957106177
	.long	3208510185
	.long	3034045498
	.long	3218412017
	.long	798414253
	.long	3197538466
	.long	4131323741
	.long	1072693270
	.long	2207647163
	.long	3208683052
	.long	2454449664
	.long	3218410791
	.long	1709648499
	.long	3197805932
	.long	972334142
	.long	1072693274
	.long	1279092327
	.long	3208797143
	.long	4141984674
	.long	3218409523
	.long	4279927728
	.long	3198110676
	.long	3547307714
	.long	1072693277
	.long	873763845
	.long	3208918886
	.long	4152286949
	.long	3218408214
	.long	947630201
	.long	3198400037
	.long	3779918761
	.long	1072693283
	.long	2452756690
	.long	3209115951
	.long	2900277560
	.long	3218406175
	.long	4114829322
	.long	3198840356
	.long	1550746642
	.long	1072693293
	.long	3179273584
	.long	3209408258
	.long	1697097694
	.long	3218403310
	.long	3189856208
	.long	3199299134
	.long	2567958063
	.long	1072693304
	.long	2177201355
	.long	3209713300
	.long	663593652
	.long	3218400281
	.long	1915839956
	.long	3199641526
	.long	3400799630
	.long	1072693317
	.long	2625933576
	.long	3209895292
	.long	3140727122
	.long	3218397088
	.long	2825210621
	.long	3200061154
	.long	661592278
	.long	1072693333
	.long	2751163500
	.long	3210096603
	.long	4055449010
	.long	3218393733
	.long	57286782
	.long	3200412116
	.long	3889851982
	.long	1072693350
	.long	2680384314
	.long	3210318153
	.long	2804057046
	.long	3218390217
	.long	1812582369
	.long	3200718155
	.long	1191734144
	.long	1072693371
	.long	867498228
	.long	3210560852
	.long	3250577733
	.long	3218386540
	.long	3326742410
	.long	3201083020
	.long	2188854007
	.long	1072693394
	.long	363145135
	.long	3210782655
	.long	840173644
	.long	3218382704
	.long	1735074843
	.long	3201408616
	.long	3657661222
	.long	1072693420
	.long	2204762872
	.long	3210926495
	.long	4072217384
	.long	3218378708
	.long	2965875284
	.long	3201662196
	.long	2413420961
	.long	1072693450
	.long	1162405510
	.long	3211082241
	.long	138720143
	.long	3218374556
	.long	454705634
	.long	3201958187
	.long	3899194868
	.long	1072693483
	.long	669033796
	.long	3211250324
	.long	2167069495
	.long	3218370246
	.long	1542529428
	.long	3202301612
	.long	710018948
	.long	1072693521
	.long	2883210692
	.long	3211431169
	.long	1973418570
	.long	3218365781
	.long	3156689511
	.long	3202524481
	.long	2656657906
	.long	1072693562
	.long	3498100032
	.long	3211625196
	.long	125419693
	.long	3218361162
	.long	1620715508
	.long	3202751895
	.long	2404819887
	.long	1072693608
	.long	4021715948
	.long	3211810552
	.long	1645154164
	.long	3218356389
	.long	703452016
	.long	3203011668
	.long	1243884135
	.long	1072693659
	.long	480742811
	.long	3211921363
	.long	3122063309
	.long	3218351464
	.long	223504399
	.long	3203307063
	.long	495862037
	.long	1072693715
	.long	300794004
	.long	3212039372
	.long	1300653287
	.long	3218346389
	.long	1176592120
	.long	3203610816
	.long	3274452673
	.long	1072693808
	.long	2613861807
	.long	3212229889
	.long	359514654
	.long	3218338503
	.long	2923503278
	.long	3204034910
	.long	3741678157
	.long	1072693954
	.long	3651935792
	.long	3212511436
	.long	2702507736
	.long	3218327463
	.long	2210624435
	.long	3204506468
	.long	1100251351
	.long	1072694127
	.long	3514265607
	.long	3212825173
	.long	957470414
	.long	3218315845
	.long	1305550937
	.long	3204833184
	.long	2996845965
	.long	1072694328
	.long	3142229264
	.long	3213004685
	.long	1883674585
	.long	3218303659
	.long	3326598907
	.long	3205231584
	.long	274977645
	.long	1072694562
	.long	3016319234
	.long	3213195820
	.long	1539002348
	.long	3218290918
	.long	3984366897
	.long	3205604614
	.long	1292806442
	.long	1072694830
	.long	3262520051
	.long	3213405058
	.long	2372653574
	.long	3218277634
	.long	2452462577
	.long	3205892032
	.long	1815353697
	.long	1072695136
	.long	3662183375
	.long	3213632998
	.long	250148789
	.long	3218263821
	.long	2158010733
	.long	3206232708
	.long	2150433494
	.long	1072695483
	.long	3919565056
	.long	3213880202
	.long	1540395291
	.long	3218249491
	.long	3182910565
	.long	3206589369
	.long	2808974545
	.long	1072695874
	.long	1965799910
	.long	3214016317
	.long	1545570484
	.long	3218234659
	.long	1413633121
	.long	3206823218
	.long	164723038
	.long	1072696313
	.long	81021250
	.long	3214159950
	.long	1583468454
	.long	3218219339
	.long	18571726
	.long	3207094384
	.long	3588317822
	.long	1072696801
	.long	4095098713
	.long	3214313941
	.long	298038316
	.long	3218203546
	.long	912947688
	.long	3207406859
	.long	1041448961
	.long	1072697344
	.long	4179586945
	.long	3214478495
	.long	2148027612
	.long	3218187294
	.long	3720520462
	.long	3207679413
	.long	1980273179
	.long	1072697943
	.long	1752543433
	.long	3214653794
	.long	124299728
	.long	3218170600
	.long	864061540
	.long	3207883361
	.long	2949417123
	.long	1072698602
	.long	4190073881
	.long	3214839998
	.long	1415544749
	.long	3218153478
	.long	3694676202
	.long	3208114560
	.long	420771937
	.long	1072699325
	.long	1282420231
	.long	3214985633
	.long	1533345938
	.long	3218135945
	.long	2892854837
	.long	3208375402
	.long	3632588569
	.long	1072700113
	.long	2712748874
	.long	3215089842
	.long	1386124246
	.long	3218118017
	.long	1634175019
	.long	3208733922
	.long	347483009
	.long	1072701423
	.long	1161341506
	.long	3215256321
	.long	1877668895
	.long	3218090451
	.long	4108664264
	.long	3209099007
	.long	3038559136
	.long	1072703435
	.long	2683048238
	.long	3215498732
	.long	1179454900
	.long	3218025113
	.long	1332126937
	.long	3209546908
	.long	3486307715
	.long	1072705767
	.long	2508275841
	.long	3215764027
	.long	359393231
	.long	3217946511
	.long	1961215910
	.long	3209890029
	.long	2252418763
	.long	1072708441
	.long	2827865151
	.long	3216017387
	.long	3509880367
	.long	3217865627
	.long	3222140085
	.long	3210213923
	.long	1486994624
	.long	1072711477
	.long	3160629487
	.long	3216172802
	.long	1178047112
	.long	3217782741
	.long	1901428593
	.long	3210596649
	.long	3746562216
	.long	1072714893
	.long	1033996001
	.long	3216339382
	.long	39866696
	.long	3217698130
	.long	2908573463
	.long	3210891953
	.long	2324761591
	.long	1072718707
	.long	2389401627
	.long	3216516876
	.long	3645939838
	.long	3217612071
	.long	2440236284
	.long	3211151116
	.long	1761407793
	.long	1072722932
	.long	2502005220
	.long	3216704962
	.long	1997398717
	.long	3217524842
	.long	2895026412
	.long	3211448596
	.long	4114502182
	.long	1072727579
	.long	2605061383
	.long	3216903247
	.long	58515002
	.long	3217436714
	.long	711731960
	.long	3211787230
	.long	3951984035
	.long	1072732658
	.long	3516063216
	.long	3217071220
	.long	3638757279
	.long	3217347953
	.long	3233370794
	.long	3211979005
	.long	658281779
	.long	1072738175
	.long	309554960
	.long	3217179845
	.long	1331214594
	.long	3217258823
	.long	2937534811
	.long	3212193451
	.long	2289001467
	.long	1072744131
	.long	2442043690
	.long	3217292793
	.long	2664085653
	.long	3217169576
	.long	3462081454
	.long	3212432263
	.long	3795084150
	.long	1072750527
	.long	1130374688
	.long	3217409754
	.long	3638257201
	.long	3217080459
	.long	3906637920
	.long	3212696488
	.long	4040275044
	.long	1072757360
	.long	3643517564
	.long	3217530392
	.long	4148962921
	.long	3216952251
	.long	1995056602
	.long	3212911950
	.long	1479928784
	.long	1072764624
	.long	2653868580
	.long	3217654355
	.long	1010106713
	.long	3216775940
	.long	3129357315
	.long	3213070764
	.long	2856812514
	.long	1072772308
	.long	1621093115
	.long	3217781272
	.long	1824031529
	.long	3216601249
	.long	2296680075
	.long	3213333328
	.long	2460843475
	.long	1072784536
	.long	3722925065
	.long	3217975665
	.long	3901534801
	.long	3216343697
	.long	741495589
	.long	3213734822
	.long	13033194
	.long	1072802249
	.long	1706090638
	.long	3218161084
	.long	4129724113
	.long	3216008854
	.long	2582831738
	.long	3214039094
	.long	3092949488
	.long	1072821329
	.long	4163404160
	.long	3218296789
	.long	673560195
	.long	3215391593
	.long	1852259695
	.long	3214295490
	.long	3663604487
	.long	1072841572
	.long	2783926416
	.long	3218433189
	.long	538028418
	.long	3214623601
	.long	955002681
	.long	3214577680
	.long	1457520314
	.long	1072862739
	.long	2977011911
	.long	3218568683
	.long	2456605794
	.long	3213048602
	.long	1782979473
	.long	3214883178
	.long	3136246603
	.long	1072884563
	.long	4172123069
	.long	3218701739
	.long	1041205303
	.long	1065062250
	.long	1494191018
	.long	3215071352
	.long	2030188257
	.long	1072906761
	.long	242335435
	.long	3218830924
	.long	3225388224
	.long	1066807264
	.long	2597486284
	.long	3215242091
	.long	1527970838
	.long	1072929036
	.long	2966426512
	.long	3218954923
	.long	1554080475
	.long	1067589039
	.long	1299968651
	.long	3215418462
	.long	3575385503
	.long	1072951087
	.long	2115725422
	.long	3219072567
	.long	1970202642
	.long	1068007486
	.long	633750547
	.long	3215597833
	.long	4170701031
	.long	1072972617
	.long	46586082
	.long	3219155579
	.long	1194653136
	.long	1068384025
	.long	4171340731
	.long	3215777352
	.long	3578160514
	.long	1072993337
	.long	3238411740
	.long	3219206599
	.long	3278559237
	.long	1068609006
	.long	2404931200
	.long	3215954013
	.long	533888921
	.long	1073012973
	.long	1507597629
	.long	3219253160
	.long	2386969249
	.long	1068756225
	.long	272720058
	.long	3216053653
	.long	54544651
	.long	1073031269
	.long	3020671348
	.long	3219294996
	.long	4122670807
	.long	1068883785
	.long	2536786852
	.long	3216134458
	.long	230026772
	.long	1073047994
	.long	2830531360
	.long	3219331923
	.long	2545616196
	.long	1068992498
	.long	3897096954
	.long	3216209170
	.long	867435464
	.long	1073062943
	.long	3935983781
	.long	3219363830
	.long	4280666630
	.long	1069083305
	.long	1614478429
	.long	3216276286
	.long	3991143559
	.long	1073075939
	.long	3165050417
	.long	3219390677
	.long	248866814
	.long	1069157251
	.long	2328429718
	.long	3216358398
	.long	1880129173
	.long	1073091258
	.long	4059723411
	.long	3219421162
	.long	3082848917
	.long	1069238148
	.long	1081358649
	.long	3216430553
	.long	1955557582
	.long	1073104040
	.long	2869422647
	.long	3219445318
	.long	1310544530
	.long	1069299023
	.long	434435025
	.long	3216450109
	.long	3947018234
	.long	1073107343
	.long	3414641036
	.long	3219451270
	.long	4069313179
	.long	1069313321
	.long	1392468754
	.long	3216410502
	.long	3271741504
	.long	1073101128
	.long	3884994071
	.long	3219440866
	.long	1456085694
	.long	1069290099
	.long	2332856790
	.long	3216307665
	.long	3950891192
	.long	1073085735
	.long	439037894
	.long	3219416288
	.long	1197951536
	.long	1069237765
	.long	4195788421
	.long	3216139842
	.long	3990997338
	.long	1073061787
	.long	550042602
	.long	3219379833
	.long	2832452545
	.long	1069163766
	.long	1200943255
	.long	3215832361
	.long	2073883731
	.long	1073030122
	.long	3042850267
	.long	3219333800
	.long	991641143
	.long	1069074535
	.long	1637244010
	.long	3215243222
	.long	3805163810
	.long	1072991715
	.long	3517445189
	.long	3219280382
	.long	2680864185
	.long	1068975465
	.long	1457843741
	.long	3214140932
	.long	1315080793
	.long	1072947617
	.long	3059804278
	.long	3219221594
	.long	2861308047
	.long	1068870963
	.long	3296491873
	.long	1064537111
	.long	3728462150
	.long	1072898893
	.long	3615137083
	.long	3219159232
	.long	3017963192
	.long	1068764532
	.long	3972434375
	.long	1067468619
	.long	1336398218
	.long	1072846587
	.long	1068664290
	.long	3219061390
	.long	122240345
	.long	1068658880
	.long	508009436
	.long	1068456917
	.long	3501538245
	.long	1072791681
	.long	968690691
	.long	3218931236
	.long	1514516445
	.long	1068556030
	.long	671541798
	.long	1069006173
	.long	3814409280
	.long	1072735081
	.long	1553551847
	.long	3218801852
	.long	2849431279
	.long	1068415930
	.long	285838780
	.long	1069554660
	.long	2881499585
	.long	1072661949
	.long	928028610
	.long	3218674977
	.long	8837506
	.long	1068229231
	.long	4283922105
	.long	1069842903
	.long	1813934616
	.long	1072546640
	.long	2296020303
	.long	3218551962
	.long	3757630126
	.long	1068054242
	.long	4184842874
	.long	1070138481
	.long	1612285858
	.long	1072432209
	.long	3568867548
	.long	3218433820
	.long	2489334631
	.long	1067891605
	.long	3119354956
	.long	1070586029
	.long	1945534618
	.long	1072265408
	.long	12375465
	.long	3218268030
	.long	869568690
	.long	1067671872
	.long	313517472
	.long	1070892138
	.long	1205077106
	.long	1072052753
	.long	2329101392
	.long	3218055718
	.long	31064032
	.long	1067390210
	.long	1106783211
	.long	1071186310
	.long	571472860
	.long	1071856508
	.long	2279775366
	.long	3217706561
	.long	3977902324
	.long	1066976012
	.long	1438560376
	.long	1071467394
	.long	433300635
	.long	1071678859
	.long	2662131044
	.long	3217407122
	.long	3461865003
	.long	1066639491
	.long	3761842524
	.long	1071687909
	.long	591758334
	.long	1071396321
	.long	1311878841
	.long	3217153539
	.long	3007781852
	.long	1066335723
	.long	1614590629
	.long	1071809819
	.long	1253814918
	.long	1071117476
	.long	1728609767
	.long	3216850667
	.long	2200561853
	.long	1065903347
	.long	3821226689
	.long	1071921115
	.long	2022982069
	.long	1070874479
	.long	2030156196
	.long	3216496942
	.long	874711265
	.long	1065560045
	.long	2003227996
	.long	1072021655
	.long	2808404217
	.long	1070664514
	.long	1372837647
	.long	3216204595
	.long	822053276
	.long	1065224094
	.long	3767175364
	.long	1072111660
	.long	3043371777
	.long	1070372670
	.long	1442419211
	.long	3215945892
	.long	298752438
	.long	1064796452
	.long	1111528881
	.long	1072191609
	.long	3513208196
	.long	1070065467
	.long	3837735739
	.long	3215552388
	.long	3701924119
	.long	1064460397
	.long	1230501085
	.long	1072262142
	.long	2161267832
	.long	1069804871
	.long	4188367704
	.long	3215231429
	.long	401190186
	.long	1064089052
	.long	3002339892
	.long	1072323996
	.long	1480019407
	.long	1069584807
	.long	1833655520
	.long	3214970435
	.long	2907956919
	.long	1063676311
	.long	875346000
	.long	1072377952
	.long	3150437403
	.long	1069251888
	.long	876861923
	.long	3214583482
	.long	1689748747
	.long	1063353511
	.long	2795554744
	.long	1072424793
	.long	3869705215
	.long	1068941581
	.long	1837883894
	.long	3214240854
	.long	2762317048
	.long	1062946513
	.long	2491972100
	.long	1072465284
	.long	4114823501
	.long	1068682289
	.long	2146865463
	.long	3213964103
	.long	800804261
	.long	1062552648
	.long	2634953449
	.long	1072500149
	.long	2952556276
	.long	1068433515
	.long	1872935290
	.long	3213596436
	.long	525130857
	.long	1062245296
	.long	3287041404
	.long	1072542855
	.long	164674845
	.long	1067923724
	.long	3273134342
	.long	3213089271
	.long	171708004
	.long	1061610314
	.long	3035032320
	.long	1072586748
	.long	3976243935
	.long	1067403539
	.long	3504708444
	.long	3212404491
	.long	3955947885
	.long	1060882840
	.long	513098494
	.long	1072618404
	.long	416924237
	.long	1066726877
	.long	1788945081
	.long	3211801737
	.long	1199639353
	.long	1060166859
	.long	2284134637
	.long	1072641010
	.long	754275327
	.long	1066136447
	.long	94803481
	.long	3211042671
	.long	3377507017
	.long	1059354147
	.long	1467291457
	.long	1072657015
	.long	2435597312
	.long	1065520743
	.long	2819017772
	.long	3210358609
	.long	1988617747
	.long	1058591581
	.long	633705514
	.long	1072668259
	.long	392269686
	.long	1064864433
	.long	2964449929
	.long	3209729770
	.long	3963893163
	.long	1057889872
	.long	2214892393
	.long	1072676103
	.long	2190738271
	.long	1064315837
	.long	3773826451
	.long	3208969045
	.long	3609404170
	.long	1057113308
	.long	3795216963
	.long	1072681541
	.long	3959981107
	.long	1063599490
	.long	1577139384
	.long	3208281591
	.long	1777963469
	.long	1056319886
	.long	2768813161
	.long	1072685290
	.long	452197850
	.long	1062981751
	.long	625862001
	.long	3207639777
	.long	1894515286
	.long	1055580811
	.long	809336726
	.long	1072687861
	.long	325412222
	.long	1062349073
	.long	3494772326
	.long	3206874482
	.long	3143092609
	.long	1054887561
	.long	1391578948
	.long	1072689615
	.long	370441451
	.long	1061660435
	.long	67458841
	.long	3206180214
	.long	1971759196
	.long	1054066692
	.long	57274217
	.long	1072690807
	.long	293665776
	.long	1061075750
	.long	1340185983
	.long	3205535605
	.long	2210177191
	.long	1053294335
	.long	1110907588
	.long	1072691613
	.long	3961986905
	.long	1060367146
	.long	3774614905
	.long	3204763416
	.long	3590429673
	.long	1052580827
	.long	2737507729
	.long	1072692156
	.long	370479370
	.long	1059701790
	.long	1033751386
	.long	3204059641
	.long	1162278823
	.long	1051830218
	.long	2341375458
	.long	1072692521
	.long	760364123
	.long	1059103172
	.long	910883556
	.long	3203420282
	.long	401067508
	.long	1051028170
	.long	3358303651
	.long	1072692765
	.long	3104773993
	.long	1058369139
	.long	2043715743
	.long	3202639413
	.long	4274377921
	.long	1050278646
	.long	3501940353
	.long	1072692984
	.long	2710387139
	.long	1057384557
	.long	2550611600
	.long	3201592258
	.long	2260324605
	.long	1049158690
	.long	3160989127
	.long	1072693133
	.long	1403521776
	.long	1056074537
	.long	760745859
	.long	3200238663
	.long	1163121055
	.long	1047679067
	.long	3509020169
	.long	1072693198
	.long	3352058101
	.long	1054743188
	.long	2289323607
	.long	3198722761
	.long	918272756
	.long	1046147840
	.long	4268817660
	.long	1072693226
	.long	1580550645
	.long	1053325591
	.long	1736251411
	.long	3197308470
	.long	1531106447
	.long	1044632576
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
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
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2146959360
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	2130706432
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	1022885888
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.long	69730304
	.type	__svml_stanh_data_internal,@object
	.size	__svml_stanh_data_internal,6016
	.align 4
__stanh_la__imlsTanhTab:
	.long	1065353216
	.long	3212836864
	.type	__stanh_la__imlsTanhTab,@object
	.size	__stanh_la__imlsTanhTab,8
