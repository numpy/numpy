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
.L_2__routine_start___svml_atanf16_z0_0:

	.align    16,0x90
	.globl __svml_atanf16

__svml_atanf16:


	.cfi_startproc
..L2:

        vandps    __svml_satan_data_internal_avx512(%rip), %zmm0, %zmm7
        vmovups   128+__svml_satan_data_internal_avx512(%rip), %zmm3
        vmovups   256+__svml_satan_data_internal_avx512(%rip), %zmm8

/* round to 2 bits after binary point */
        vreduceps $40, {sae}, %zmm7, %zmm5

/* saturate X range */
        vmovups   320+__svml_satan_data_internal_avx512(%rip), %zmm6
        vmovups   64+__svml_satan_data_internal_avx512(%rip), %zmm2
        vcmpps    $29, {sae}, %zmm3, %zmm7, %k1

/* table lookup sequence */
        vmovups   448+__svml_satan_data_internal_avx512(%rip), %zmm3
        vsubps    {rn-sae}, %zmm5, %zmm7, %zmm4
        vaddps    {rn-sae}, %zmm2, %zmm7, %zmm1
        vxorps    %zmm0, %zmm7, %zmm0
        vfmadd231ps {rn-sae}, %zmm7, %zmm4, %zmm8
        vmovups   896+__svml_satan_data_internal_avx512(%rip), %zmm4

/* if|X|>=MaxThreshold, set DiffX=-1 */
        vblendmps 192+__svml_satan_data_internal_avx512(%rip), %zmm5, %zmm9{%k1}
        vmovups   960+__svml_satan_data_internal_avx512(%rip), %zmm5

/* if|X|>=MaxThreshold, set Y=X */
        vminps    {sae}, %zmm7, %zmm6, %zmm8{%k1}

/* R+Rl = DiffX/Y */
        vgetmantps $0, {sae}, %zmm9, %zmm12
        vgetexpps {sae}, %zmm9, %zmm10
        vpermt2ps 512+__svml_satan_data_internal_avx512(%rip), %zmm1, %zmm3
        vgetmantps $0, {sae}, %zmm8, %zmm15
        vgetexpps {sae}, %zmm8, %zmm11
        vmovups   832+__svml_satan_data_internal_avx512(%rip), %zmm1

/* set table value to Pi/2 for large X */
        vblendmps 704+__svml_satan_data_internal_avx512(%rip), %zmm3, %zmm9{%k1}
        vrcp14ps  %zmm15, %zmm13
        vsubps    {rn-sae}, %zmm11, %zmm10, %zmm2
        vmulps    {rn-sae}, %zmm13, %zmm12, %zmm14
        vfnmadd213ps {rn-sae}, %zmm12, %zmm14, %zmm15
        vfmadd213ps {rn-sae}, %zmm14, %zmm13, %zmm15
        vscalefps {rn-sae}, %zmm2, %zmm15, %zmm7

/* polynomial evaluation */
        vmulps    {rn-sae}, %zmm7, %zmm7, %zmm8
        vmulps    {rn-sae}, %zmm7, %zmm8, %zmm6
        vfmadd231ps {rn-sae}, %zmm8, %zmm1, %zmm4
        vfmadd213ps {rn-sae}, %zmm5, %zmm4, %zmm8
        vfmadd213ps {rn-sae}, %zmm7, %zmm6, %zmm8
        vaddps    {rn-sae}, %zmm9, %zmm8, %zmm10
        vxorps    %zmm0, %zmm10, %zmm0

/* no invcbrt in libm, so taking it out here */
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_atanf16,@function
	.size	__svml_atanf16,.-__svml_atanf16
..LN__svml_atanf16.0:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_satan_data_internal_avx512:
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
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1241513984
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
	.long	1089994752
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
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
	.long	1333788672
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
	.long	1048239024
	.long	1055744824
	.long	1059372157
	.long	1061752795
	.long	1063609315
	.long	1065064543
	.long	1065786489
	.long	1066252045
	.long	1066633083
	.long	1066949484
	.long	1067215699
	.long	1067442363
	.long	1067637412
	.long	1067806856
	.long	1067955311
	.long	1068086373
	.long	1068202874
	.long	1068307075
	.long	1068400798
	.long	1068485529
	.long	1068562486
	.long	1068632682
	.long	1068696961
	.long	1068756035
	.long	1068810506
	.long	1068860887
	.long	1068907620
	.long	1068951084
	.long	1068991608
	.long	1069029480
	.long	1069064949
	.long	0
	.long	2975494116
	.long	833369962
	.long	835299256
	.long	2998648110
	.long	2995239174
	.long	3000492182
	.long	860207626
	.long	3008447516
	.long	3005590622
	.long	3000153675
	.long	860754741
	.long	859285590
	.long	844944488
	.long	2993069463
	.long	858157665
	.long	3006142000
	.long	3007693206
	.long	3009342234
	.long	847469400
	.long	3006114683
	.long	852829553
	.long	847325583
	.long	860305056
	.long	846145135
	.long	2997638646
	.long	855837703
	.long	2979047222
	.long	2995344192
	.long	854092798
	.long	3000498637
	.long	859965578
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	1070141403
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3007036718
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	3188697310
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	1045219554
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.long	3198855850
	.type	__svml_satan_data_internal_avx512,@object
	.size	__svml_satan_data_internal_avx512,1024
