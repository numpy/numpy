/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/

/*
 * ALGORITHM DESCRIPTION:
 * 
 *     x=2^{3*k+j} * 1.b1 b2 ... b5 b6 ... b52
 *     Let r=(x*2^{-3k-j} - 1.b1 b2 ... b5 1)* rcp[b1 b2 ..b5],
 *     where rcp[b1 b2 .. b5]=1/(1.b1 b2 b3 b4 b5 1) in single precision
 *     cbrtf(2^j * 1. b1 b2 .. b5 1) is approximated as T[j][b1..b5]+D[j][b1..b5]
 *     (T stores the high 24 bits, D stores the low order bits)
 *     Result=2^k*T+(2^k*T*r)*P+2^k*D
 *      where P=p1+p2*r+..
 * 
 */


	.text
.L_2__routine_start___svml_cbrtf16_z0_0:

	.align    16,0x90
	.globl __svml_cbrtf16

__svml_cbrtf16:


	.cfi_startproc
..L2:

        vgetmantps $0, {sae}, %zmm0, %zmm8

/* GetExp(x) */
        vgetexpps {sae}, %zmm0, %zmm1
        vmovups   384+__svml_scbrt_data_internal_avx512(%rip), %zmm2

/* exponent/3 */
        vmovups   512+__svml_scbrt_data_internal_avx512(%rip), %zmm3
        vmovups   576+__svml_scbrt_data_internal_avx512(%rip), %zmm4
        vmovups   704+__svml_scbrt_data_internal_avx512(%rip), %zmm15

/* exponent%3 (to be used as index) */
        vmovups   640+__svml_scbrt_data_internal_avx512(%rip), %zmm5

/* polynomial */
        vmovups   768+__svml_scbrt_data_internal_avx512(%rip), %zmm11
        vmovups   896+__svml_scbrt_data_internal_avx512(%rip), %zmm14

/* Table lookup */
        vmovups   128+__svml_scbrt_data_internal_avx512(%rip), %zmm12

/* DblRcp ~ 1/Mantissa */
        vrcp14ps  %zmm8, %zmm7
        vaddps    {rn-sae}, %zmm2, %zmm1, %zmm6
        vandps    448+__svml_scbrt_data_internal_avx512(%rip), %zmm0, %zmm0

/* round DblRcp to 3 fractional bits (RN mode, no Precision exception) */
        vrndscaleps $88, {sae}, %zmm7, %zmm9
        vfmsub231ps {rn-sae}, %zmm6, %zmm3, %zmm4
        vmovups   832+__svml_scbrt_data_internal_avx512(%rip), %zmm7

/* Reduced argument: R = DblRcp*Mantissa - 1 */
        vfmsub231ps {rn-sae}, %zmm9, %zmm8, %zmm15
        vrndscaleps $9, {sae}, %zmm4, %zmm13

/* Prepare table index */
        vpsrld    $19, %zmm9, %zmm10
        vfmadd231ps {rn-sae}, %zmm15, %zmm11, %zmm7
        vfnmadd231ps {rn-sae}, %zmm13, %zmm5, %zmm6
        vpermt2ps 192+__svml_scbrt_data_internal_avx512(%rip), %zmm10, %zmm12
        vfmadd213ps {rn-sae}, %zmm14, %zmm15, %zmm7
        vscalefps {rn-sae}, %zmm13, %zmm12, %zmm2

/* Table lookup: 2^(exponent%3) */
        vpermps   __svml_scbrt_data_internal_avx512(%rip), %zmm6, %zmm1
        vpermps   64+__svml_scbrt_data_internal_avx512(%rip), %zmm6, %zmm6

/* Sh*R */
        vmulps    {rn-sae}, %zmm15, %zmm1, %zmm14

/* Sl + (Sh*R)*Poly */
        vfmadd213ps {rn-sae}, %zmm6, %zmm7, %zmm14

/*
 * branch-free
 * scaled_Th*(Sh+Sl+Sh*R*Poly)
 */
        vaddps    {rn-sae}, %zmm1, %zmm14, %zmm15
        vmulps    {rn-sae}, %zmm2, %zmm15, %zmm3
        vorps     %zmm0, %zmm3, %zmm0

/* no invcbrt in libm, so taking it out here */
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_cbrtf16,@function
	.size	__svml_cbrtf16,.-__svml_cbrtf16
..LN__svml_cbrtf16.0:

.L_2__routine_start___svml_scbrt_cout_rare_internal_1:

	.align    16,0x90

__svml_scbrt_cout_rare_internal:


	.cfi_startproc
..L5:

        movq      %rsi, %r9
        movl      $1065353216, -24(%rsp)
        movss     (%rdi), %xmm0
        movss     -24(%rsp), %xmm1
        mulss     %xmm0, %xmm1
        movss     %xmm1, -4(%rsp)
        movzwl    -2(%rsp), %eax
        andl      $32640, %eax
        shrl      $7, %eax
        cmpl      $255, %eax
        je        .LBL_2_9


        pxor      %xmm0, %xmm0
        ucomiss   %xmm0, %xmm1
        jp        .LBL_2_3
        je        .LBL_2_8

.LBL_2_3:

        testl     %eax, %eax
        jne       .LBL_2_5


        movl      $2122317824, -24(%rsp)
        movl      $713031680, -20(%rsp)
        jmp       .LBL_2_6

.LBL_2_5:

        movl      $1065353216, %eax
        movl      %eax, -24(%rsp)
        movl      %eax, -20(%rsp)

.LBL_2_6:

        movss     -24(%rsp), %xmm0
        lea       __scbrt_la_vscbrt_ha_cout_data(%rip), %rsi
        mulss     %xmm0, %xmm1
        movd      %xmm1, %ecx
        movss     %xmm1, -4(%rsp)
        movl      %ecx, %r10d
        movl      %ecx, %edi
        andl      $8388607, %r10d
        movl      %ecx, %r11d
        shrl      $23, %edi
        andl      $8257536, %r11d
        orl       $-1082130432, %r10d
        orl       $-1081999360, %r11d
        movl      %r10d, -16(%rsp)
        movl      %ecx, %edx
        movzbl    %dil, %r8d
        andl      $2147483647, %ecx
        movl      %r11d, -12(%rsp)
        andl      $-256, %edi
        movss     -16(%rsp), %xmm1
        addl      $2139095040, %ecx
        shrl      $16, %edx
        subss     -12(%rsp), %xmm1
        andl      $124, %edx
        lea       (%r8,%r8,4), %r10d
        mulss     (%rsi,%rdx), %xmm1
        lea       (%r10,%r10), %r11d
        movss     .L_2il0floatpacket.35(%rip), %xmm4
        lea       (%r11,%r11), %eax
        addl      %eax, %eax
        lea       (%r10,%r11,8), %r10d
        addl      %eax, %eax
        decl      %r8d
        mulss     %xmm1, %xmm4
        shll      $7, %r8d
        lea       (%r10,%rax,8), %r11d
        lea       (%r11,%rax,8), %r10d
        shrl      $12, %r10d
        addss     .L_2il0floatpacket.34(%rip), %xmm4
        mulss     %xmm1, %xmm4
        lea       85(%r10), %eax
        orl       %edi, %eax
        xorl      %edi, %edi
        cmpl      $-16777217, %ecx
        addss     .L_2il0floatpacket.33(%rip), %xmm4
        setg      %dil
        shll      $7, %r10d
        negl      %edi
        subl      %r10d, %r8d
        addl      %r10d, %r10d
        subl      %r10d, %r8d
        notl      %edi
        addl      %r8d, %edx
        andl      %edx, %edi
        shll      $23, %eax
        addl      %edi, %edi
        movl      %eax, -8(%rsp)
        movss     128(%rdi,%rsi), %xmm5
        movss     -8(%rsp), %xmm2
        mulss     %xmm1, %xmm4
        mulss     %xmm2, %xmm5
        addss     .L_2il0floatpacket.32(%rip), %xmm4
        mulss     %xmm5, %xmm1
        movss     132(%rsi,%rdi), %xmm3
        mulss     %xmm1, %xmm4
        mulss     %xmm2, %xmm3
        addss     %xmm3, %xmm4
        addss     %xmm4, %xmm5
        mulss     -20(%rsp), %xmm5
        movss     %xmm5, (%r9)

.LBL_2_7:

        xorl      %eax, %eax
        ret

.LBL_2_8:

        movss     %xmm1, (%r9)
        jmp       .LBL_2_7

.LBL_2_9:

        addss     %xmm0, %xmm0
        movss     %xmm0, (%r9)
        jmp       .LBL_2_7
	.align    16,0x90

	.cfi_endproc

	.type	__svml_scbrt_cout_rare_internal,@function
	.size	__svml_scbrt_cout_rare_internal,.-__svml_scbrt_cout_rare_internal
..LN__svml_scbrt_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_scbrt_data_internal_avx512:
	.long	1065353216
	.long	1067533592
	.long	1070280693
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	2999865775
	.long	849849800
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	1067533592
	.long	1067322155
	.long	1067126683
	.long	1066945178
	.long	1066775983
	.long	1066617708
	.long	1066469175
	.long	1066329382
	.long	1066197466
	.long	1066072682
	.long	1065954382
	.long	1065841998
	.long	1065735031
	.long	1065633040
	.long	1065535634
	.long	1065442463
	.long	1065353216
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	2999865775
	.long	849353281
	.long	2992093760
	.long	858369405
	.long	861891413
	.long	3001900484
	.long	2988845984
	.long	3009185201
	.long	3001209163
	.long	847824101
	.long	839380496
	.long	845124191
	.long	851391835
	.long	856440803
	.long	2989578734
	.long	852890174
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
	.long	0
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
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1249902592
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
	.long	1077936128
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
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	1031603580
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	3185812323
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.long	1051372202
	.type	__svml_scbrt_data_internal_avx512,@object
	.size	__svml_scbrt_data_internal_avx512,960
	.align 64
__scbrt_la_vscbrt_ha_cout_data:
	.long	3212578753
	.long	3212085645
	.long	3211621124
	.long	3211182772
	.long	3210768440
	.long	3210376206
	.long	3210004347
	.long	3209651317
	.long	3209315720
	.long	3208996296
	.long	3208691905
	.long	3208401508
	.long	3208124163
	.long	3207859009
	.long	3207605259
	.long	3207362194
	.long	3207129151
	.long	3206905525
	.long	3206690755
	.long	3206484326
	.long	3206285761
	.long	3206094618
	.long	3205910490
	.long	3205732998
	.long	3205561788
	.long	3205396533
	.long	3205236929
	.long	3205082689
	.long	3204933547
	.long	3204789256
	.long	3204649583
	.long	3204514308
	.long	1065396681
	.long	839340838
	.long	1065482291
	.long	867750258
	.long	1065566215
	.long	851786446
	.long	1065648532
	.long	853949398
	.long	1065729317
	.long	864938789
	.long	1065808640
	.long	864102364
	.long	1065886565
	.long	864209792
	.long	1065963152
	.long	865422805
	.long	1066038457
	.long	867593594
	.long	1066112533
	.long	854482593
	.long	1066185428
	.long	848298042
	.long	1066257188
	.long	860064854
	.long	1066327857
	.long	844792593
	.long	1066397474
	.long	870701309
	.long	1066466079
	.long	872023170
	.long	1066533708
	.long	860255342
	.long	1066600394
	.long	849966899
	.long	1066666169
	.long	863561479
	.long	1066731064
	.long	869115319
	.long	1066795108
	.long	871961375
	.long	1066858329
	.long	859537336
	.long	1066920751
	.long	871954398
	.long	1066982401
	.long	863817578
	.long	1067043301
	.long	861687921
	.long	1067103474
	.long	849594757
	.long	1067162941
	.long	816486846
	.long	1067221722
	.long	858183533
	.long	1067279837
	.long	864500406
	.long	1067337305
	.long	850523240
	.long	1067394143
	.long	808125243
	.long	1067450368
	.long	0
	.long	1067505996
	.long	861173761
	.long	1067588354
	.long	859000219
	.long	1067696217
	.long	823158129
	.long	1067801953
	.long	871826232
	.long	1067905666
	.long	871183196
	.long	1068007450
	.long	839030530
	.long	1068107390
	.long	867690638
	.long	1068205570
	.long	840440923
	.long	1068302063
	.long	868033274
	.long	1068396942
	.long	855856030
	.long	1068490271
	.long	865094453
	.long	1068582113
	.long	860418487
	.long	1068672525
	.long	866225006
	.long	1068761562
	.long	866458226
	.long	1068849275
	.long	865124659
	.long	1068935712
	.long	864837702
	.long	1069020919
	.long	811742505
	.long	1069104937
	.long	869432099
	.long	1069187809
	.long	864584201
	.long	1069269572
	.long	864183978
	.long	1069350263
	.long	844810573
	.long	1069429915
	.long	869245699
	.long	1069508563
	.long	859556409
	.long	1069586236
	.long	870675446
	.long	1069662966
	.long	814190139
	.long	1069738778
	.long	870686941
	.long	1069813702
	.long	861800510
	.long	1069887762
	.long	855649163
	.long	1069960982
	.long	869347119
	.long	1070033387
	.long	864252033
	.long	1070104998
	.long	867276215
	.long	1070175837
	.long	868189817
	.long	1070245925
	.long	849541095
	.long	1070349689
	.long	866633177
	.long	1070485588
	.long	843967686
	.long	1070618808
	.long	857522493
	.long	1070749478
	.long	862339487
	.long	1070877717
	.long	850054662
	.long	1071003634
	.long	864048556
	.long	1071127332
	.long	868027089
	.long	1071248907
	.long	848093931
	.long	1071368446
	.long	865355299
	.long	1071486034
	.long	848111485
	.long	1071601747
	.long	865557362
	.long	1071715659
	.long	870297525
	.long	1071827839
	.long	863416216
	.long	1071938350
	.long	869675693
	.long	1072047254
	.long	865888071
	.long	1072154608
	.long	825332584
	.long	1072260465
	.long	843309506
	.long	1072364876
	.long	870885636
	.long	1072467891
	.long	869119784
	.long	1072569555
	.long	865466648
	.long	1072669911
	.long	867459244
	.long	1072769001
	.long	861192764
	.long	1072866863
	.long	871247716
	.long	1072963536
	.long	864927982
	.long	1073059054
	.long	869195129
	.long	1073153452
	.long	864849564
	.long	1073246762
	.long	840005936
	.long	1073339014
	.long	852579258
	.long	1073430238
	.long	860852782
	.long	1073520462
	.long	869711141
	.long	1073609714
	.long	862506141
	.long	1073698019
	.long	837959274
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	3173551943
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	1031591658
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	3185806905
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
	.long	1051372203
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
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	8257536
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	3212967936
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	124
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	255
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	256
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	85
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
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
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	2155872256
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.long	4278190079
	.type	__scbrt_la_vscbrt_ha_cout_data,@object
	.size	__scbrt_la_vscbrt_ha_cout_data,1920
	.align 4
.L_2il0floatpacket.28:
	.long	0x007fffff
	.type	.L_2il0floatpacket.28,@object
	.size	.L_2il0floatpacket.28,4
	.align 4
.L_2il0floatpacket.29:
	.long	0x007e0000
	.type	.L_2il0floatpacket.29,@object
	.size	.L_2il0floatpacket.29,4
	.align 4
.L_2il0floatpacket.30:
	.long	0xbf800000
	.type	.L_2il0floatpacket.30,@object
	.size	.L_2il0floatpacket.30,4
	.align 4
.L_2il0floatpacket.31:
	.long	0xbf820000
	.type	.L_2il0floatpacket.31,@object
	.size	.L_2il0floatpacket.31,4
	.align 4
.L_2il0floatpacket.32:
	.long	0x3eaaaaab
	.type	.L_2il0floatpacket.32,@object
	.size	.L_2il0floatpacket.32,4
	.align 4
.L_2il0floatpacket.33:
	.long	0xbde38e39
	.type	.L_2il0floatpacket.33,@object
	.size	.L_2il0floatpacket.33,4
	.align 4
.L_2il0floatpacket.34:
	.long	0x3d7cd6ea
	.type	.L_2il0floatpacket.34,@object
	.size	.L_2il0floatpacket.34,4
	.align 4
.L_2il0floatpacket.35:
	.long	0xbd288f47
	.type	.L_2il0floatpacket.35,@object
	.size	.L_2il0floatpacket.35,4
