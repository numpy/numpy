/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/


	.text
.L_2__routine_start___svml_acos8_z0_0:

	.align    16,0x90
	.globl __svml_acos8

__svml_acos8:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   __svml_dacos_data_internal(%rip), %zmm7
        vmovups   64+__svml_dacos_data_internal(%rip), %zmm8

/* S ~ 2*sqrt(Y) */
        vmovups   128+__svml_dacos_data_internal(%rip), %zmm11
        vmovups   384+__svml_dacos_data_internal(%rip), %zmm14
        vmovups   448+__svml_dacos_data_internal(%rip), %zmm15
        vmovups   512+__svml_dacos_data_internal(%rip), %zmm2
        vmovups   576+__svml_dacos_data_internal(%rip), %zmm1
        vmovups   256+__svml_dacos_data_internal(%rip), %zmm10
        vmovaps   %zmm0, %zmm6

/* x = -|arg| */
        vorpd     %zmm6, %zmm7, %zmm5
        vandpd    %zmm6, %zmm7, %zmm4

/* Y = 0.5 + 0.5*(-x) */
        vfmadd231pd {rn-sae}, %zmm5, %zmm8, %zmm8

/* x^2 */
        vmulpd    {rn-sae}, %zmm5, %zmm5, %zmm9
        vrsqrt14pd %zmm8, %zmm12
        vcmppd    $17, {sae}, %zmm11, %zmm8, %k2
        vcmppd    $17, {sae}, %zmm10, %zmm5, %k0
        vmovups   960+__svml_dacos_data_internal(%rip), %zmm10
        vmovups   1088+__svml_dacos_data_internal(%rip), %zmm11
        vminpd    {sae}, %zmm8, %zmm9, %zmm3
        vmovups   832+__svml_dacos_data_internal(%rip), %zmm9
        vxorpd    %zmm12, %zmm12, %zmm12{%k2}
        vaddpd    {rn-sae}, %zmm8, %zmm8, %zmm0
        vcmppd    $21, {sae}, %zmm8, %zmm3, %k1

/* X<X^2 iff X<0 */
        vcmppd    $17, {sae}, %zmm3, %zmm6, %k3
        vmulpd    {rn-sae}, %zmm12, %zmm12, %zmm13
        vmulpd    {rn-sae}, %zmm12, %zmm0, %zmm7
        vmovups   896+__svml_dacos_data_internal(%rip), %zmm12

/* polynomial */
        vmovups   704+__svml_dacos_data_internal(%rip), %zmm8
        vfmsub213pd {rn-sae}, %zmm14, %zmm13, %zmm0
        vmovups   640+__svml_dacos_data_internal(%rip), %zmm13
        vfmadd231pd {rn-sae}, %zmm3, %zmm9, %zmm12
        vmovups   1344+__svml_dacos_data_internal(%rip), %zmm9
        vfmadd231pd {rn-sae}, %zmm0, %zmm15, %zmm2
        vmovups   1216+__svml_dacos_data_internal(%rip), %zmm15
        vmulpd    {rn-sae}, %zmm0, %zmm7, %zmm14
        vfmadd213pd {rn-sae}, %zmm1, %zmm0, %zmm2
        vmovups   768+__svml_dacos_data_internal(%rip), %zmm1
        kmovw     %k1, %eax
        kmovw     %k3, %ecx
        kmovw     %k0, %edx
        vfmadd213pd {rn-sae}, %zmm13, %zmm0, %zmm2
        vfmadd231pd {rn-sae}, %zmm3, %zmm8, %zmm1
        vmovups   1280+__svml_dacos_data_internal(%rip), %zmm8
        vmulpd    {rn-sae}, %zmm3, %zmm3, %zmm0
        vfnmadd213pd {rn-sae}, %zmm7, %zmm14, %zmm2
        vmovups   1024+__svml_dacos_data_internal(%rip), %zmm7
        vfmadd231pd {rn-sae}, %zmm3, %zmm15, %zmm8
        vfmadd213pd {rn-sae}, %zmm12, %zmm0, %zmm1
        vblendmpd %zmm2, %zmm5, %zmm2{%k1}
        vfmadd231pd {rn-sae}, %zmm3, %zmm10, %zmm7
        vmovups   1152+__svml_dacos_data_internal(%rip), %zmm10
        vfmadd231pd {rn-sae}, %zmm3, %zmm11, %zmm10
        andl      %eax, %ecx
        vmovups   1408+__svml_dacos_data_internal(%rip), %zmm11
        kmovw     %ecx, %k2
        vfmadd213pd {rn-sae}, %zmm10, %zmm0, %zmm7
        vfmadd231pd {rn-sae}, %zmm3, %zmm9, %zmm11
        vmulpd    {rn-sae}, %zmm0, %zmm0, %zmm10
        vfmadd213pd {rn-sae}, %zmm7, %zmm10, %zmm1
        vfmadd213pd {rn-sae}, %zmm8, %zmm0, %zmm1
        vfmadd213pd {rn-sae}, %zmm11, %zmm0, %zmm1
        vmovups   1664+__svml_dacos_data_internal(%rip), %zmm0
        vmulpd    {rn-sae}, %zmm3, %zmm1, %zmm1
        vxorpd    %zmm4, %zmm2, %zmm3
        vxorpd    %zmm0, %zmm0, %zmm0{%k1}
        vfmadd213pd {rn-sae}, %zmm3, %zmm3, %zmm1
        vorpd     1536+__svml_dacos_data_internal(%rip), %zmm0, %zmm0{%k2}
        vaddpd    {rn-sae}, %zmm1, %zmm0, %zmm0
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

        vmovups   %zmm6, 64(%rsp)
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

        call      __svml_dacos_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_acos8,@function
	.size	__svml_acos8,.-__svml_acos8
..LN__svml_acos8.0:

.L_2__routine_start___svml_dacos_cout_rare_internal_1:

	.align    16,0x90

__svml_dacos_cout_rare_internal:


	.cfi_startproc
..L53:

        xorl      %eax, %eax
        movzwl    6(%rdi), %edx
        andl      $32752, %edx
        movb      7(%rdi), %cl
        movsd     (%rdi), %xmm1
        cmpl      $32752, %edx
        je        .LBL_2_20


        movsd     %xmm1, -32(%rsp)
        andb      $127, %cl
        movb      %cl, -25(%rsp)
        movsd     -32(%rsp), %xmm12
        movsd     4168+_vmldACosHATab(%rip), %xmm0
        comisd    %xmm12, %xmm0
        jbe       .LBL_2_14


        movsd     4176+_vmldACosHATab(%rip), %xmm1
        comisd    %xmm12, %xmm1
        jbe       .LBL_2_10


        comisd    4128+_vmldACosHATab(%rip), %xmm12
        jbe       .LBL_2_9


        movsd     4104+_vmldACosHATab(%rip), %xmm6
        movaps    %xmm12, %xmm8
        mulsd     %xmm12, %xmm6
        movaps    %xmm12, %xmm7
        movsd     %xmm6, -40(%rsp)
        movsd     -40(%rsp), %xmm13
        movsd     4104+_vmldACosHATab(%rip), %xmm5
        subsd     -32(%rsp), %xmm13
        movsd     %xmm13, -48(%rsp)
        movsd     -40(%rsp), %xmm15
        movsd     -48(%rsp), %xmm14
        subsd     %xmm14, %xmm15
        movaps    %xmm12, %xmm14
        movsd     %xmm15, -40(%rsp)
        movsd     -40(%rsp), %xmm6
        subsd     %xmm6, %xmm8
        movsd     %xmm8, -48(%rsp)
        movsd     -40(%rsp), %xmm9
        movaps    %xmm9, %xmm4
        addsd     %xmm9, %xmm7
        mulsd     %xmm9, %xmm4
        movsd     -48(%rsp), %xmm10
        movaps    %xmm4, %xmm11
        mulsd     %xmm10, %xmm7
        mulsd     %xmm4, %xmm5
        addsd     %xmm7, %xmm11
        movsd     4312+_vmldACosHATab(%rip), %xmm8
        mulsd     %xmm11, %xmm8
        movsd     %xmm5, -40(%rsp)
        movsd     -40(%rsp), %xmm1
        movsd     4104+_vmldACosHATab(%rip), %xmm5
        subsd     %xmm4, %xmm1
        addsd     4304+_vmldACosHATab(%rip), %xmm8
        mulsd     %xmm11, %xmm8
        movsd     %xmm1, -48(%rsp)
        movsd     -40(%rsp), %xmm0
        movsd     -48(%rsp), %xmm2
        addsd     4296+_vmldACosHATab(%rip), %xmm8
        subsd     %xmm2, %xmm0
        mulsd     %xmm11, %xmm8
        movsd     %xmm0, -40(%rsp)
        movsd     -40(%rsp), %xmm3
        addsd     4288+_vmldACosHATab(%rip), %xmm8
        subsd     %xmm3, %xmm4
        mulsd     %xmm11, %xmm8
        movsd     %xmm4, -48(%rsp)
        movsd     -40(%rsp), %xmm6
        mulsd     %xmm6, %xmm9
        addsd     4280+_vmldACosHATab(%rip), %xmm8
        mulsd     %xmm6, %xmm10
        mulsd     %xmm11, %xmm8
        mulsd     %xmm9, %xmm5
        addsd     4272+_vmldACosHATab(%rip), %xmm8
        mulsd     %xmm11, %xmm8
        movaps    %xmm9, %xmm0
        movsd     -48(%rsp), %xmm13
        movaps    %xmm6, %xmm4
        movsd     %xmm5, -40(%rsp)
        addsd     %xmm13, %xmm7
        addsd     4264+_vmldACosHATab(%rip), %xmm8
        mulsd     %xmm7, %xmm14
        mulsd     %xmm11, %xmm8
        addsd     %xmm14, %xmm10
        addsd     4256+_vmldACosHATab(%rip), %xmm8
        movsd     -40(%rsp), %xmm1
        mulsd     %xmm11, %xmm8
        subsd     %xmm9, %xmm1
        addsd     %xmm10, %xmm9
        addsd     4248+_vmldACosHATab(%rip), %xmm8
        mulsd     %xmm9, %xmm7
        mulsd     %xmm11, %xmm8
        movsd     %xmm1, -48(%rsp)
        movsd     -40(%rsp), %xmm2
        movsd     -48(%rsp), %xmm15
        subsd     %xmm15, %xmm2
        addsd     4240+_vmldACosHATab(%rip), %xmm8
        movsd     %xmm2, -40(%rsp)
        movsd     -40(%rsp), %xmm5
        mulsd     %xmm11, %xmm8
        subsd     %xmm5, %xmm0
        movsd     %xmm0, -48(%rsp)
        movsd     -40(%rsp), %xmm3
        movsd     -48(%rsp), %xmm5
        movaps    %xmm5, %xmm13
        addsd     4232+_vmldACosHATab(%rip), %xmm8
        mulsd     %xmm3, %xmm4
        addsd     %xmm10, %xmm13
        mulsd     %xmm11, %xmm8
        mulsd     %xmm13, %xmm6
        addsd     4224+_vmldACosHATab(%rip), %xmm8
        addsd     %xmm7, %xmm6
        mulsd     %xmm11, %xmm8
        movsd     4104+_vmldACosHATab(%rip), %xmm7
        movaps    %xmm4, %xmm13
        mulsd     %xmm4, %xmm7
        addsd     4216+_vmldACosHATab(%rip), %xmm8
        movsd     %xmm7, -40(%rsp)
        movsd     -40(%rsp), %xmm1
        movsd     4200+_vmldACosHATab(%rip), %xmm11
        subsd     %xmm4, %xmm1
        mulsd     %xmm9, %xmm11
        addsd     %xmm6, %xmm4
        movsd     %xmm1, -48(%rsp)
        movaps    %xmm12, %xmm9
        movsd     -40(%rsp), %xmm0
        movsd     -48(%rsp), %xmm2
        movsd     4192+_vmldACosHATab(%rip), %xmm1
        subsd     %xmm2, %xmm0
        mulsd     %xmm1, %xmm3
        mulsd     %xmm1, %xmm10
        mulsd     %xmm4, %xmm8
        addsd     %xmm3, %xmm9
        mulsd     %xmm1, %xmm5
        addsd     %xmm10, %xmm11
        movsd     %xmm0, -40(%rsp)
        addsd     %xmm11, %xmm8
        movsd     -40(%rsp), %xmm7
        addsd     %xmm5, %xmm8
        subsd     %xmm7, %xmm13
        movsd     %xmm13, -48(%rsp)
        movsd     -40(%rsp), %xmm0
        movsd     -48(%rsp), %xmm2
        movsd     %xmm9, -40(%rsp)
        addsd     %xmm2, %xmm6
        movsd     -40(%rsp), %xmm10
        movsd     %xmm3, -56(%rsp)
        subsd     %xmm10, %xmm12
        movsd     4208+_vmldACosHATab(%rip), %xmm4
        addsd     %xmm12, %xmm3
        mulsd     %xmm4, %xmm0
        mulsd     %xmm4, %xmm6
        movsd     %xmm3, -48(%rsp)
        movsd     -40(%rsp), %xmm3
        movaps    %xmm3, %xmm12
        movsd     -48(%rsp), %xmm7
        addsd     %xmm0, %xmm12
        addsd     %xmm7, %xmm8
        movsd     %xmm12, -40(%rsp)
        movsd     -40(%rsp), %xmm12
        subsd     %xmm12, %xmm3
        addsd     %xmm3, %xmm0
        movsd     %xmm0, -48(%rsp)
        movsd     -40(%rsp), %xmm3
        movsd     -48(%rsp), %xmm0
        movsd     (%rdi), %xmm1
        addsd     %xmm8, %xmm0
        comisd    4184+_vmldACosHATab(%rip), %xmm1
        addsd     %xmm0, %xmm6
        jbe       .LBL_2_7


        movsd     4136+_vmldACosHATab(%rip), %xmm2
        movaps    %xmm2, %xmm0
        subsd     %xmm3, %xmm0
        movsd     %xmm0, -40(%rsp)
        movsd     -40(%rsp), %xmm1
        movsd     4144+_vmldACosHATab(%rip), %xmm0
        subsd     %xmm1, %xmm2
        subsd     %xmm6, %xmm0
        subsd     %xmm3, %xmm2
        movsd     %xmm2, -48(%rsp)
        movsd     -40(%rsp), %xmm1
        movsd     -48(%rsp), %xmm3
        addsd     %xmm3, %xmm0
        jmp       .LBL_2_8

.LBL_2_7:

        movsd     4136+_vmldACosHATab(%rip), %xmm2
        movaps    %xmm3, %xmm0
        addsd     %xmm2, %xmm0
        movsd     %xmm0, -40(%rsp)
        movsd     -40(%rsp), %xmm1
        subsd     %xmm1, %xmm2
        addsd     %xmm2, %xmm3
        movsd     %xmm3, -48(%rsp)
        movsd     4144+_vmldACosHATab(%rip), %xmm3
        movsd     -40(%rsp), %xmm1
        addsd     %xmm3, %xmm6
        movsd     -48(%rsp), %xmm0
        addsd     %xmm6, %xmm0

.LBL_2_8:

        addsd     %xmm0, %xmm1
        movsd     %xmm1, (%rsi)
        ret

.LBL_2_9:

        movsd     4144+_vmldACosHATab(%rip), %xmm1
        addsd     %xmm0, %xmm12
        movsd     %xmm12, -40(%rsp)
        movsd     -40(%rsp), %xmm0
        mulsd     -32(%rsp), %xmm0
        movsd     %xmm0, -56(%rsp)
        movb      -49(%rsp), %dl
        movb      7(%rdi), %dil
        andb      $127, %dl
        andb      $-128, %dil
        orb       %dil, %dl
        movb      %dl, -49(%rsp)
        subsd     -56(%rsp), %xmm1
        addsd     4136+_vmldACosHATab(%rip), %xmm1
        movsd     %xmm1, (%rsi)
        ret

.LBL_2_10:

        movaps    %xmm0, %xmm8
        movzwl    4174+_vmldACosHATab(%rip), %r9d
        pxor      %xmm13, %xmm13
        andl      $-32753, %r9d
        subsd     %xmm12, %xmm8
        subsd     %xmm0, %xmm13
        mulsd     %xmm8, %xmm1
        movsd     %xmm1, -56(%rsp)
        movzwl    -50(%rsp), %ecx
        andl      $32752, %ecx
        shrl      $4, %ecx
        addl      $-1023, %ecx
        movl      %ecx, %r8d
        movl      %ecx, %edx
        negl      %r8d
        addl      $1023, %r8d
        andl      $2047, %r8d
        shll      $4, %r8d
        movsd     %xmm0, -32(%rsp)
        orl       %r8d, %r9d
        movw      %r9w, -26(%rsp)
        andl      $1, %edx
        movsd     -32(%rsp), %xmm4
        lea       _vmldACosHATab(%rip), %r8
        mulsd     %xmm4, %xmm1
        movl      %edx, %r10d
        movaps    %xmm1, %xmm15
        movsd     4112+_vmldACosHATab(%rip), %xmm6
        addsd     %xmm1, %xmm15
        jne       ..L54
        movaps    %xmm1, %xmm15
..L54:
        mulsd     %xmm15, %xmm6
        movaps    %xmm15, %xmm7
        movaps    %xmm6, %xmm9
        subl      %edx, %ecx
        movsd     4120+_vmldACosHATab(%rip), %xmm11
        subsd     %xmm15, %xmm9
        addsd     %xmm1, %xmm11
        movsd     %xmm9, -48(%rsp)
        movsd     -48(%rsp), %xmm10
        movsd     %xmm11, -24(%rsp)
        subsd     %xmm10, %xmm6
        movl      -24(%rsp), %r11d
        movaps    %xmm6, %xmm14
        shll      $8, %r10d
        andl      $511, %r11d
        addl      %r10d, %r11d
        subsd     %xmm6, %xmm7
        movsd     (%r8,%r11,8), %xmm5
        addsd     %xmm7, %xmm14
        mulsd     %xmm5, %xmm6
        movaps    %xmm5, %xmm12
        mulsd     %xmm5, %xmm12
        mulsd     %xmm12, %xmm14
        movsd     4512+_vmldACosHATab(%rip), %xmm4
        addsd     %xmm13, %xmm14
        mulsd     %xmm14, %xmm4
        shrl      $1, %ecx
        addsd     4504+_vmldACosHATab(%rip), %xmm4
        mulsd     %xmm14, %xmm4
        addl      $1023, %ecx
        andl      $2047, %ecx
        addsd     4496+_vmldACosHATab(%rip), %xmm4
        mulsd     %xmm14, %xmm4
        movzwl    4174+_vmldACosHATab(%rip), %r9d
        shll      $4, %ecx
        andl      $-32753, %r9d
        movsd     %xmm0, -16(%rsp)
        orl       %ecx, %r9d
        movw      %r9w, -10(%rsp)
        movsd     -16(%rsp), %xmm9
        mulsd     %xmm9, %xmm6
        addsd     4488+_vmldACosHATab(%rip), %xmm4
        mulsd     %xmm14, %xmm4
        movsd     4104+_vmldACosHATab(%rip), %xmm3
        mulsd     %xmm6, %xmm3
        addsd     4480+_vmldACosHATab(%rip), %xmm4
        mulsd     %xmm14, %xmm4
        movsd     %xmm3, -40(%rsp)
        movsd     -40(%rsp), %xmm1
        movsd     4352+_vmldACosHATab(%rip), %xmm11
        subsd     %xmm6, %xmm1
        addsd     4472+_vmldACosHATab(%rip), %xmm4
        addsd     4360+_vmldACosHATab(%rip), %xmm11
        mulsd     %xmm14, %xmm4
        mulsd     %xmm8, %xmm11
        addsd     4464+_vmldACosHATab(%rip), %xmm4
        mulsd     %xmm14, %xmm4
        mulsd     %xmm15, %xmm4
        movsd     %xmm1, -48(%rsp)
        addsd     %xmm4, %xmm7
        mulsd     %xmm5, %xmm7
        movsd     4456+_vmldACosHATab(%rip), %xmm5
        mulsd     %xmm8, %xmm5
        mulsd     %xmm9, %xmm7
        addsd     4448+_vmldACosHATab(%rip), %xmm5
        mulsd     %xmm8, %xmm5
        movsd     -40(%rsp), %xmm2
        movsd     -48(%rsp), %xmm0
        movsd     4104+_vmldACosHATab(%rip), %xmm4
        subsd     %xmm0, %xmm2
        mulsd     %xmm8, %xmm4
        addsd     4440+_vmldACosHATab(%rip), %xmm5
        mulsd     %xmm8, %xmm5
        movaps    %xmm8, %xmm0
        movsd     %xmm2, -40(%rsp)
        movsd     -40(%rsp), %xmm14
        movsd     4104+_vmldACosHATab(%rip), %xmm2
        subsd     %xmm14, %xmm6
        addsd     4432+_vmldACosHATab(%rip), %xmm5
        mulsd     %xmm8, %xmm5
        movsd     %xmm6, -48(%rsp)
        movsd     -40(%rsp), %xmm6
        movsd     -48(%rsp), %xmm10
        movsd     %xmm4, -40(%rsp)
        addsd     %xmm10, %xmm7
        addsd     4424+_vmldACosHATab(%rip), %xmm5
        mulsd     %xmm8, %xmm5
        movsd     -40(%rsp), %xmm3
        movsd     4336+_vmldACosHATab(%rip), %xmm12
        subsd     %xmm8, %xmm3
        addsd     4416+_vmldACosHATab(%rip), %xmm5
        addsd     4344+_vmldACosHATab(%rip), %xmm12
        mulsd     %xmm8, %xmm5
        addsd     %xmm11, %xmm12
        addsd     4408+_vmldACosHATab(%rip), %xmm5
        mulsd     %xmm8, %xmm12
        mulsd     %xmm8, %xmm5
        movsd     %xmm3, -48(%rsp)
        movsd     -40(%rsp), %xmm1
        movsd     -48(%rsp), %xmm15
        movsd     4320+_vmldACosHATab(%rip), %xmm13
        subsd     %xmm15, %xmm1
        addsd     4400+_vmldACosHATab(%rip), %xmm5
        addsd     4328+_vmldACosHATab(%rip), %xmm13
        mulsd     %xmm8, %xmm5
        addsd     %xmm12, %xmm13
        addsd     4392+_vmldACosHATab(%rip), %xmm5
        movsd     %xmm1, -40(%rsp)
        mulsd     %xmm8, %xmm5
        movsd     -40(%rsp), %xmm4
        subsd     %xmm4, %xmm0
        addsd     4384+_vmldACosHATab(%rip), %xmm5
        movsd     %xmm0, -48(%rsp)
        movsd     -40(%rsp), %xmm4
        movaps    %xmm4, %xmm14
        mulsd     %xmm4, %xmm14
        mulsd     %xmm8, %xmm5
        mulsd     %xmm14, %xmm2
        addsd     4376+_vmldACosHATab(%rip), %xmm5
        movsd     -48(%rsp), %xmm3
        movsd     %xmm2, -40(%rsp)
        movsd     -40(%rsp), %xmm9
        mulsd     %xmm8, %xmm5
        subsd     %xmm14, %xmm9
        movsd     %xmm9, -48(%rsp)
        movsd     -40(%rsp), %xmm11
        movsd     -48(%rsp), %xmm10
        movsd     4336+_vmldACosHATab(%rip), %xmm0
        subsd     %xmm10, %xmm11
        mulsd     %xmm4, %xmm0
        addsd     4368+_vmldACosHATab(%rip), %xmm5
        addsd     %xmm8, %xmm4
        mulsd     %xmm8, %xmm5
        mulsd     %xmm3, %xmm4
        mulsd     %xmm8, %xmm5
        movsd     %xmm11, -40(%rsp)
        movaps    %xmm0, %xmm1
        movsd     -40(%rsp), %xmm12
        mulsd     %xmm8, %xmm5
        subsd     %xmm12, %xmm14
        movsd     %xmm14, -48(%rsp)
        addsd     %xmm5, %xmm13
        movsd     -40(%rsp), %xmm9
        mulsd     4352+_vmldACosHATab(%rip), %xmm9
        mulsd     %xmm13, %xmm7
        addsd     %xmm9, %xmm1
        movsd     -48(%rsp), %xmm2
        movsd     %xmm1, -40(%rsp)
        addsd     %xmm2, %xmm4
        movsd     -40(%rsp), %xmm13
        movsd     %xmm9, -24(%rsp)
        subsd     %xmm13, %xmm0
        mulsd     4352+_vmldACosHATab(%rip), %xmm4
        addsd     %xmm0, %xmm9
        movsd     %xmm9, -48(%rsp)
        movsd     -40(%rsp), %xmm12
        movsd     4320+_vmldACosHATab(%rip), %xmm10
        movsd     -48(%rsp), %xmm1
        addsd     %xmm12, %xmm10
        movsd     %xmm10, -40(%rsp)
        movsd     -40(%rsp), %xmm15
        movsd     4320+_vmldACosHATab(%rip), %xmm11
        movsd     4104+_vmldACosHATab(%rip), %xmm9
        subsd     %xmm15, %xmm11
        movsd     4336+_vmldACosHATab(%rip), %xmm2
        addsd     %xmm11, %xmm12
        mulsd     %xmm3, %xmm2
        movsd     %xmm12, -48(%rsp)
        movsd     -40(%rsp), %xmm15
        mulsd     %xmm15, %xmm9
        movsd     -48(%rsp), %xmm0
        movsd     %xmm9, -40(%rsp)
        movsd     -40(%rsp), %xmm10
        movsd     4360+_vmldACosHATab(%rip), %xmm3
        subsd     %xmm15, %xmm10
        mulsd     %xmm8, %xmm3
        movsd     %xmm10, -48(%rsp)
        movsd     -40(%rsp), %xmm11
        movsd     -48(%rsp), %xmm13
        subsd     %xmm13, %xmm11
        addsd     4344+_vmldACosHATab(%rip), %xmm3
        movsd     %xmm11, -40(%rsp)
        movsd     -40(%rsp), %xmm14
        mulsd     %xmm8, %xmm3
        subsd     %xmm14, %xmm15
        movsd     %xmm15, -48(%rsp)
        movsd     -40(%rsp), %xmm10
        movsd     -48(%rsp), %xmm9
        addsd     %xmm9, %xmm4
        addsd     4328+_vmldACosHATab(%rip), %xmm3
        addsd     %xmm2, %xmm4
        addsd     %xmm5, %xmm3
        addsd     %xmm1, %xmm4
        addsd     %xmm0, %xmm4
        addsd     %xmm3, %xmm4
        mulsd     %xmm6, %xmm4
        mulsd     %xmm10, %xmm6
        addsd     %xmm7, %xmm4
        movsd     (%rdi), %xmm7
        comisd    4184+_vmldACosHATab(%rip), %xmm7
        ja        .LBL_2_13


        movsd     4152+_vmldACosHATab(%rip), %xmm2
        movaps    %xmm2, %xmm0
        movsd     4160+_vmldACosHATab(%rip), %xmm5
        subsd     %xmm6, %xmm0
        subsd     %xmm4, %xmm5
        movsd     %xmm0, -40(%rsp)
        movsd     -40(%rsp), %xmm1
        movsd     %xmm6, -56(%rsp)
        subsd     %xmm1, %xmm2
        subsd     %xmm6, %xmm2
        movsd     %xmm2, -48(%rsp)
        movsd     -40(%rsp), %xmm6
        movsd     -48(%rsp), %xmm3
        movaps    %xmm3, %xmm4
        addsd     %xmm5, %xmm4

.LBL_2_13:

        addsd     %xmm4, %xmm6
        movsd     %xmm6, (%rsi)
        ret

.LBL_2_14:

        ucomisd   %xmm0, %xmm1
        jp        .LBL_2_15
        je        .LBL_2_19

.LBL_2_15:

        xorps     .L_2il0floatpacket.197(%rip), %xmm0
        ucomisd   %xmm0, %xmm1
        jp        .LBL_2_16
        je        .LBL_2_18

.LBL_2_16:

        movl      $1, %eax
        pxor      %xmm1, %xmm1
        pxor      %xmm0, %xmm0
        divsd     %xmm0, %xmm1
        movsd     %xmm1, (%rsi)

.LBL_2_17:

        ret

.LBL_2_18:

        movsd     4152+_vmldACosHATab(%rip), %xmm0
        addsd     4160+_vmldACosHATab(%rip), %xmm0
        movsd     %xmm0, (%rsi)
        ret

.LBL_2_19:

        movq      4184+_vmldACosHATab(%rip), %rdx
        movq      %rdx, (%rsi)
        ret

.LBL_2_20:

        divsd     %xmm1, %xmm1
        movsd     %xmm1, (%rsi)
        testl     $1048575, 4(%rdi)
        jne       .LBL_2_17


        cmpl      $0, (%rdi)
        sete      %al
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_dacos_cout_rare_internal,@function
	.size	__svml_dacos_cout_rare_internal,.-__svml_dacos_cout_rare_internal
..LN__svml_dacos_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_dacos_data_internal:
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
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	1071644672
	.long	0
	.long	805306368
	.long	0
	.long	805306368
	.long	0
	.long	805306368
	.long	0
	.long	805306368
	.long	0
	.long	805306368
	.long	0
	.long	805306368
	.long	0
	.long	805306368
	.long	0
	.long	805306368
	.long	0
	.long	4294967040
	.long	0
	.long	4294967040
	.long	0
	.long	4294967040
	.long	0
	.long	4294967040
	.long	0
	.long	4294967040
	.long	0
	.long	4294967040
	.long	0
	.long	4294967040
	.long	0
	.long	4294967040
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
	.long	4227858432
	.long	4294967295
	.long	4227858432
	.long	4294967295
	.long	4227858432
	.long	4294967295
	.long	4227858432
	.long	4294967295
	.long	4227858432
	.long	4294967295
	.long	4227858432
	.long	4294967295
	.long	4227858432
	.long	4294967295
	.long	4227858432
	.long	4294967295
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	0
	.long	1073741824
	.long	2570790083
	.long	3213983744
	.long	2570790083
	.long	3213983744
	.long	2570790083
	.long	3213983744
	.long	2570790083
	.long	3213983744
	.long	2570790083
	.long	3213983744
	.long	2570790083
	.long	3213983744
	.long	2570790083
	.long	3213983744
	.long	2570790083
	.long	3213983744
	.long	1869665325
	.long	1067712512
	.long	1869665325
	.long	1067712512
	.long	1869665325
	.long	1067712512
	.long	1869665325
	.long	1067712512
	.long	1869665325
	.long	1067712512
	.long	1869665325
	.long	1067712512
	.long	1869665325
	.long	1067712512
	.long	1869665325
	.long	1067712512
	.long	4294966935
	.long	3216506879
	.long	4294966935
	.long	3216506879
	.long	4294966935
	.long	3216506879
	.long	4294966935
	.long	3216506879
	.long	4294966935
	.long	3216506879
	.long	4294966935
	.long	3216506879
	.long	4294966935
	.long	3216506879
	.long	4294966935
	.long	3216506879
	.long	4294967197
	.long	1070596095
	.long	4294967197
	.long	1070596095
	.long	4294967197
	.long	1070596095
	.long	4294967197
	.long	1070596095
	.long	4294967197
	.long	1070596095
	.long	4294967197
	.long	1070596095
	.long	4294967197
	.long	1070596095
	.long	4294967197
	.long	1070596095
	.long	3339630857
	.long	1067480352
	.long	3339630857
	.long	1067480352
	.long	3339630857
	.long	1067480352
	.long	3339630857
	.long	1067480352
	.long	3339630857
	.long	1067480352
	.long	3339630857
	.long	1067480352
	.long	3339630857
	.long	1067480352
	.long	3339630857
	.long	1067480352
	.long	4158370029
	.long	3213949719
	.long	4158370029
	.long	3213949719
	.long	4158370029
	.long	3213949719
	.long	4158370029
	.long	3213949719
	.long	4158370029
	.long	3213949719
	.long	4158370029
	.long	3213949719
	.long	4158370029
	.long	3213949719
	.long	4158370029
	.long	3213949719
	.long	3216784302
	.long	1066680132
	.long	3216784302
	.long	1066680132
	.long	3216784302
	.long	1066680132
	.long	3216784302
	.long	1066680132
	.long	3216784302
	.long	1066680132
	.long	3216784302
	.long	1066680132
	.long	3216784302
	.long	1066680132
	.long	3216784302
	.long	1066680132
	.long	2513723093
	.long	1064982579
	.long	2513723093
	.long	1064982579
	.long	2513723093
	.long	1064982579
	.long	2513723093
	.long	1064982579
	.long	2513723093
	.long	1064982579
	.long	2513723093
	.long	1064982579
	.long	2513723093
	.long	1064982579
	.long	2513723093
	.long	1064982579
	.long	721210070
	.long	1065941212
	.long	721210070
	.long	1065941212
	.long	721210070
	.long	1065941212
	.long	721210070
	.long	1065941212
	.long	721210070
	.long	1065941212
	.long	721210070
	.long	1065941212
	.long	721210070
	.long	1065941212
	.long	721210070
	.long	1065941212
	.long	3414736215
	.long	1066167739
	.long	3414736215
	.long	1066167739
	.long	3414736215
	.long	1066167739
	.long	3414736215
	.long	1066167739
	.long	3414736215
	.long	1066167739
	.long	3414736215
	.long	1066167739
	.long	3414736215
	.long	1066167739
	.long	3414736215
	.long	1066167739
	.long	4114132270
	.long	1066518236
	.long	4114132270
	.long	1066518236
	.long	4114132270
	.long	1066518236
	.long	4114132270
	.long	1066518236
	.long	4114132270
	.long	1066518236
	.long	4114132270
	.long	1066518236
	.long	4114132270
	.long	1066518236
	.long	4114132270
	.long	1066518236
	.long	3957258973
	.long	1066854556
	.long	3957258973
	.long	1066854556
	.long	3957258973
	.long	1066854556
	.long	3957258973
	.long	1066854556
	.long	3957258973
	.long	1066854556
	.long	3957258973
	.long	1066854556
	.long	3957258973
	.long	1066854556
	.long	3957258973
	.long	1066854556
	.long	3778730174
	.long	1067392114
	.long	3778730174
	.long	1067392114
	.long	3778730174
	.long	1067392114
	.long	3778730174
	.long	1067392114
	.long	3778730174
	.long	1067392114
	.long	3778730174
	.long	1067392114
	.long	3778730174
	.long	1067392114
	.long	3778730174
	.long	1067392114
	.long	3014936056
	.long	1067899757
	.long	3014936056
	.long	1067899757
	.long	3014936056
	.long	1067899757
	.long	3014936056
	.long	1067899757
	.long	3014936056
	.long	1067899757
	.long	3014936056
	.long	1067899757
	.long	3014936056
	.long	1067899757
	.long	3014936056
	.long	1067899757
	.long	859300062
	.long	1068708659
	.long	859300062
	.long	1068708659
	.long	859300062
	.long	1068708659
	.long	859300062
	.long	1068708659
	.long	859300062
	.long	1068708659
	.long	859300062
	.long	1068708659
	.long	859300062
	.long	1068708659
	.long	859300062
	.long	1068708659
	.long	1431655068
	.long	1069897045
	.long	1431655068
	.long	1069897045
	.long	1431655068
	.long	1069897045
	.long	1431655068
	.long	1069897045
	.long	1431655068
	.long	1069897045
	.long	1431655068
	.long	1069897045
	.long	1431655068
	.long	1069897045
	.long	1431655068
	.long	1069897045
	.long	856972295
	.long	1017226790
	.long	856972295
	.long	1017226790
	.long	856972295
	.long	1017226790
	.long	856972295
	.long	1017226790
	.long	856972295
	.long	1017226790
	.long	856972295
	.long	1017226790
	.long	856972295
	.long	1017226790
	.long	856972295
	.long	1017226790
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
	.long	1413754136
	.long	1074340347
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
	.long	0
	.long	4294705152
	.long	0
	.long	4294705152
	.long	0
	.long	4294705152
	.long	0
	.long	4294705152
	.long	0
	.long	4294705152
	.long	0
	.long	4294705152
	.long	0
	.long	4294705152
	.long	0
	.long	4294705152
	.long	4131758366
	.long	1067674714
	.long	4131758366
	.long	1067674714
	.long	4131758366
	.long	1067674714
	.long	4131758366
	.long	1067674714
	.long	4131758366
	.long	1067674714
	.long	4131758366
	.long	1067674714
	.long	4131758366
	.long	1067674714
	.long	4131758366
	.long	1067674714
	.long	3936260738
	.long	1066197319
	.long	3936260738
	.long	1066197319
	.long	3936260738
	.long	1066197319
	.long	3936260738
	.long	1066197319
	.long	3936260738
	.long	1066197319
	.long	3936260738
	.long	1066197319
	.long	3936260738
	.long	1066197319
	.long	3936260738
	.long	1066197319
	.long	354394453
	.long	1067472564
	.long	354394453
	.long	1067472564
	.long	354394453
	.long	1067472564
	.long	354394453
	.long	1067472564
	.long	354394453
	.long	1067472564
	.long	354394453
	.long	1067472564
	.long	354394453
	.long	1067472564
	.long	354394453
	.long	1067472564
	.long	709676628
	.long	1067895021
	.long	709676628
	.long	1067895021
	.long	709676628
	.long	1067895021
	.long	709676628
	.long	1067895021
	.long	709676628
	.long	1067895021
	.long	709676628
	.long	1067895021
	.long	709676628
	.long	1067895021
	.long	709676628
	.long	1067895021
	.long	3958922090
	.long	1068708761
	.long	3958922090
	.long	1068708761
	.long	3958922090
	.long	1068708761
	.long	3958922090
	.long	1068708761
	.long	3958922090
	.long	1068708761
	.long	3958922090
	.long	1068708761
	.long	3958922090
	.long	1068708761
	.long	3958922090
	.long	1068708761
	.long	2160605765
	.long	1069897044
	.long	2160605765
	.long	1069897044
	.long	2160605765
	.long	1069897044
	.long	2160605765
	.long	1069897044
	.long	2160605765
	.long	1069897044
	.long	2160605765
	.long	1069897044
	.long	2160605765
	.long	1069897044
	.long	2160605765
	.long	1069897044
	.long	0
	.long	2146435072
	.long	0
	.long	4293918720
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
	.long	1072693248
	.long	0
	.long	3220176896
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
	.long	0
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
	.type	__svml_dacos_data_internal,@object
	.size	__svml_dacos_data_internal,2496
	.align 32
_vmldACosHATab:
	.long	0
	.long	1072693248
	.long	0
	.long	1072689152
	.long	0
	.long	1072685056
	.long	0
	.long	1072680960
	.long	0
	.long	1072676864
	.long	0
	.long	1072672768
	.long	0
	.long	1072668672
	.long	0
	.long	1072665600
	.long	0
	.long	1072661504
	.long	0
	.long	1072657408
	.long	0
	.long	1072653312
	.long	0
	.long	1072649216
	.long	0
	.long	1072646144
	.long	0
	.long	1072642048
	.long	0
	.long	1072637952
	.long	0
	.long	1072634880
	.long	0
	.long	1072630784
	.long	0
	.long	1072626688
	.long	0
	.long	1072623616
	.long	0
	.long	1072619520
	.long	0
	.long	1072615424
	.long	0
	.long	1072612352
	.long	0
	.long	1072608256
	.long	0
	.long	1072605184
	.long	0
	.long	1072601088
	.long	0
	.long	1072598016
	.long	0
	.long	1072593920
	.long	0
	.long	1072590848
	.long	0
	.long	1072586752
	.long	0
	.long	1072583680
	.long	0
	.long	1072580608
	.long	0
	.long	1072576512
	.long	0
	.long	1072573440
	.long	0
	.long	1072570368
	.long	0
	.long	1072566272
	.long	0
	.long	1072563200
	.long	0
	.long	1072560128
	.long	0
	.long	1072556032
	.long	0
	.long	1072552960
	.long	0
	.long	1072549888
	.long	0
	.long	1072546816
	.long	0
	.long	1072542720
	.long	0
	.long	1072539648
	.long	0
	.long	1072536576
	.long	0
	.long	1072533504
	.long	0
	.long	1072530432
	.long	0
	.long	1072527360
	.long	0
	.long	1072523264
	.long	0
	.long	1072520192
	.long	0
	.long	1072517120
	.long	0
	.long	1072514048
	.long	0
	.long	1072510976
	.long	0
	.long	1072507904
	.long	0
	.long	1072504832
	.long	0
	.long	1072501760
	.long	0
	.long	1072498688
	.long	0
	.long	1072495616
	.long	0
	.long	1072492544
	.long	0
	.long	1072489472
	.long	0
	.long	1072486400
	.long	0
	.long	1072483328
	.long	0
	.long	1072480256
	.long	0
	.long	1072478208
	.long	0
	.long	1072475136
	.long	0
	.long	1072472064
	.long	0
	.long	1072468992
	.long	0
	.long	1072465920
	.long	0
	.long	1072462848
	.long	0
	.long	1072459776
	.long	0
	.long	1072457728
	.long	0
	.long	1072454656
	.long	0
	.long	1072451584
	.long	0
	.long	1072448512
	.long	0
	.long	1072446464
	.long	0
	.long	1072443392
	.long	0
	.long	1072440320
	.long	0
	.long	1072437248
	.long	0
	.long	1072435200
	.long	0
	.long	1072432128
	.long	0
	.long	1072429056
	.long	0
	.long	1072427008
	.long	0
	.long	1072423936
	.long	0
	.long	1072420864
	.long	0
	.long	1072418816
	.long	0
	.long	1072415744
	.long	0
	.long	1072412672
	.long	0
	.long	1072410624
	.long	0
	.long	1072407552
	.long	0
	.long	1072405504
	.long	0
	.long	1072402432
	.long	0
	.long	1072400384
	.long	0
	.long	1072397312
	.long	0
	.long	1072395264
	.long	0
	.long	1072392192
	.long	0
	.long	1072390144
	.long	0
	.long	1072387072
	.long	0
	.long	1072385024
	.long	0
	.long	1072381952
	.long	0
	.long	1072379904
	.long	0
	.long	1072376832
	.long	0
	.long	1072374784
	.long	0
	.long	1072371712
	.long	0
	.long	1072369664
	.long	0
	.long	1072366592
	.long	0
	.long	1072364544
	.long	0
	.long	1072362496
	.long	0
	.long	1072359424
	.long	0
	.long	1072357376
	.long	0
	.long	1072355328
	.long	0
	.long	1072352256
	.long	0
	.long	1072350208
	.long	0
	.long	1072347136
	.long	0
	.long	1072345088
	.long	0
	.long	1072343040
	.long	0
	.long	1072340992
	.long	0
	.long	1072337920
	.long	0
	.long	1072335872
	.long	0
	.long	1072333824
	.long	0
	.long	1072330752
	.long	0
	.long	1072328704
	.long	0
	.long	1072326656
	.long	0
	.long	1072324608
	.long	0
	.long	1072321536
	.long	0
	.long	1072319488
	.long	0
	.long	1072317440
	.long	0
	.long	1072315392
	.long	0
	.long	1072313344
	.long	0
	.long	1072310272
	.long	0
	.long	1072308224
	.long	0
	.long	1072306176
	.long	0
	.long	1072304128
	.long	0
	.long	1072302080
	.long	0
	.long	1072300032
	.long	0
	.long	1072296960
	.long	0
	.long	1072294912
	.long	0
	.long	1072292864
	.long	0
	.long	1072290816
	.long	0
	.long	1072288768
	.long	0
	.long	1072286720
	.long	0
	.long	1072284672
	.long	0
	.long	1072282624
	.long	0
	.long	1072280576
	.long	0
	.long	1072278528
	.long	0
	.long	1072275456
	.long	0
	.long	1072273408
	.long	0
	.long	1072271360
	.long	0
	.long	1072269312
	.long	0
	.long	1072267264
	.long	0
	.long	1072265216
	.long	0
	.long	1072263168
	.long	0
	.long	1072261120
	.long	0
	.long	1072259072
	.long	0
	.long	1072257024
	.long	0
	.long	1072254976
	.long	0
	.long	1072252928
	.long	0
	.long	1072250880
	.long	0
	.long	1072248832
	.long	0
	.long	1072246784
	.long	0
	.long	1072244736
	.long	0
	.long	1072243712
	.long	0
	.long	1072241664
	.long	0
	.long	1072239616
	.long	0
	.long	1072237568
	.long	0
	.long	1072235520
	.long	0
	.long	1072233472
	.long	0
	.long	1072231424
	.long	0
	.long	1072229376
	.long	0
	.long	1072227328
	.long	0
	.long	1072225280
	.long	0
	.long	1072223232
	.long	0
	.long	1072222208
	.long	0
	.long	1072220160
	.long	0
	.long	1072218112
	.long	0
	.long	1072216064
	.long	0
	.long	1072214016
	.long	0
	.long	1072211968
	.long	0
	.long	1072210944
	.long	0
	.long	1072208896
	.long	0
	.long	1072206848
	.long	0
	.long	1072204800
	.long	0
	.long	1072202752
	.long	0
	.long	1072201728
	.long	0
	.long	1072199680
	.long	0
	.long	1072197632
	.long	0
	.long	1072195584
	.long	0
	.long	1072193536
	.long	0
	.long	1072192512
	.long	0
	.long	1072190464
	.long	0
	.long	1072188416
	.long	0
	.long	1072186368
	.long	0
	.long	1072185344
	.long	0
	.long	1072183296
	.long	0
	.long	1072181248
	.long	0
	.long	1072179200
	.long	0
	.long	1072178176
	.long	0
	.long	1072176128
	.long	0
	.long	1072174080
	.long	0
	.long	1072173056
	.long	0
	.long	1072171008
	.long	0
	.long	1072168960
	.long	0
	.long	1072167936
	.long	0
	.long	1072165888
	.long	0
	.long	1072163840
	.long	0
	.long	1072161792
	.long	0
	.long	1072160768
	.long	0
	.long	1072158720
	.long	0
	.long	1072157696
	.long	0
	.long	1072155648
	.long	0
	.long	1072153600
	.long	0
	.long	1072152576
	.long	0
	.long	1072150528
	.long	0
	.long	1072148480
	.long	0
	.long	1072147456
	.long	0
	.long	1072145408
	.long	0
	.long	1072143360
	.long	0
	.long	1072142336
	.long	0
	.long	1072140288
	.long	0
	.long	1072139264
	.long	0
	.long	1072137216
	.long	0
	.long	1072135168
	.long	0
	.long	1072134144
	.long	0
	.long	1072132096
	.long	0
	.long	1072131072
	.long	0
	.long	1072129024
	.long	0
	.long	1072128000
	.long	0
	.long	1072125952
	.long	0
	.long	1072124928
	.long	0
	.long	1072122880
	.long	0
	.long	1072120832
	.long	0
	.long	1072119808
	.long	0
	.long	1072117760
	.long	0
	.long	1072116736
	.long	0
	.long	1072114688
	.long	0
	.long	1072113664
	.long	0
	.long	1072111616
	.long	0
	.long	1072110592
	.long	0
	.long	1072108544
	.long	0
	.long	1072107520
	.long	0
	.long	1072105472
	.long	0
	.long	1072104448
	.long	0
	.long	1072102400
	.long	0
	.long	1072101376
	.long	0
	.long	1072099328
	.long	0
	.long	1072098304
	.long	0
	.long	1072096256
	.long	0
	.long	1072095232
	.long	0
	.long	1072094208
	.long	0
	.long	1072092160
	.long	0
	.long	1072091136
	.long	0
	.long	1072089088
	.long	0
	.long	1072088064
	.long	0
	.long	1072086016
	.long	0
	.long	1072084992
	.long	0
	.long	1072082944
	.long	0
	.long	1072081920
	.long	0
	.long	1072080896
	.long	0
	.long	1072078848
	.long	0
	.long	1072075776
	.long	0
	.long	1072073728
	.long	0
	.long	1072070656
	.long	0
	.long	1072067584
	.long	0
	.long	1072064512
	.long	0
	.long	1072061440
	.long	0
	.long	1072059392
	.long	0
	.long	1072056320
	.long	0
	.long	1072053248
	.long	0
	.long	1072051200
	.long	0
	.long	1072048128
	.long	0
	.long	1072045056
	.long	0
	.long	1072043008
	.long	0
	.long	1072039936
	.long	0
	.long	1072037888
	.long	0
	.long	1072034816
	.long	0
	.long	1072031744
	.long	0
	.long	1072029696
	.long	0
	.long	1072026624
	.long	0
	.long	1072024576
	.long	0
	.long	1072021504
	.long	0
	.long	1072019456
	.long	0
	.long	1072016384
	.long	0
	.long	1072014336
	.long	0
	.long	1072011264
	.long	0
	.long	1072009216
	.long	0
	.long	1072006144
	.long	0
	.long	1072004096
	.long	0
	.long	1072002048
	.long	0
	.long	1071998976
	.long	0
	.long	1071996928
	.long	0
	.long	1071993856
	.long	0
	.long	1071991808
	.long	0
	.long	1071989760
	.long	0
	.long	1071986688
	.long	0
	.long	1071984640
	.long	0
	.long	1071982592
	.long	0
	.long	1071979520
	.long	0
	.long	1071977472
	.long	0
	.long	1071975424
	.long	0
	.long	1071972352
	.long	0
	.long	1071970304
	.long	0
	.long	1071968256
	.long	0
	.long	1071966208
	.long	0
	.long	1071964160
	.long	0
	.long	1071961088
	.long	0
	.long	1071959040
	.long	0
	.long	1071956992
	.long	0
	.long	1071954944
	.long	0
	.long	1071952896
	.long	0
	.long	1071949824
	.long	0
	.long	1071947776
	.long	0
	.long	1071945728
	.long	0
	.long	1071943680
	.long	0
	.long	1071941632
	.long	0
	.long	1071939584
	.long	0
	.long	1071937536
	.long	0
	.long	1071935488
	.long	0
	.long	1071933440
	.long	0
	.long	1071930368
	.long	0
	.long	1071928320
	.long	0
	.long	1071926272
	.long	0
	.long	1071924224
	.long	0
	.long	1071922176
	.long	0
	.long	1071920128
	.long	0
	.long	1071918080
	.long	0
	.long	1071916032
	.long	0
	.long	1071913984
	.long	0
	.long	1071911936
	.long	0
	.long	1071909888
	.long	0
	.long	1071907840
	.long	0
	.long	1071905792
	.long	0
	.long	1071903744
	.long	0
	.long	1071901696
	.long	0
	.long	1071900672
	.long	0
	.long	1071898624
	.long	0
	.long	1071896576
	.long	0
	.long	1071894528
	.long	0
	.long	1071892480
	.long	0
	.long	1071890432
	.long	0
	.long	1071888384
	.long	0
	.long	1071886336
	.long	0
	.long	1071884288
	.long	0
	.long	1071883264
	.long	0
	.long	1071881216
	.long	0
	.long	1071879168
	.long	0
	.long	1071877120
	.long	0
	.long	1071875072
	.long	0
	.long	1071873024
	.long	0
	.long	1071872000
	.long	0
	.long	1071869952
	.long	0
	.long	1071867904
	.long	0
	.long	1071865856
	.long	0
	.long	1071864832
	.long	0
	.long	1071862784
	.long	0
	.long	1071860736
	.long	0
	.long	1071858688
	.long	0
	.long	1071856640
	.long	0
	.long	1071855616
	.long	0
	.long	1071853568
	.long	0
	.long	1071851520
	.long	0
	.long	1071850496
	.long	0
	.long	1071848448
	.long	0
	.long	1071846400
	.long	0
	.long	1071844352
	.long	0
	.long	1071843328
	.long	0
	.long	1071841280
	.long	0
	.long	1071839232
	.long	0
	.long	1071838208
	.long	0
	.long	1071836160
	.long	0
	.long	1071834112
	.long	0
	.long	1071833088
	.long	0
	.long	1071831040
	.long	0
	.long	1071830016
	.long	0
	.long	1071827968
	.long	0
	.long	1071825920
	.long	0
	.long	1071824896
	.long	0
	.long	1071822848
	.long	0
	.long	1071821824
	.long	0
	.long	1071819776
	.long	0
	.long	1071817728
	.long	0
	.long	1071816704
	.long	0
	.long	1071814656
	.long	0
	.long	1071813632
	.long	0
	.long	1071811584
	.long	0
	.long	1071810560
	.long	0
	.long	1071808512
	.long	0
	.long	1071806464
	.long	0
	.long	1071805440
	.long	0
	.long	1071803392
	.long	0
	.long	1071802368
	.long	0
	.long	1071800320
	.long	0
	.long	1071799296
	.long	0
	.long	1071797248
	.long	0
	.long	1071796224
	.long	0
	.long	1071794176
	.long	0
	.long	1071793152
	.long	0
	.long	1071791104
	.long	0
	.long	1071790080
	.long	0
	.long	1071788032
	.long	0
	.long	1071787008
	.long	0
	.long	1071784960
	.long	0
	.long	1071783936
	.long	0
	.long	1071782912
	.long	0
	.long	1071780864
	.long	0
	.long	1071779840
	.long	0
	.long	1071777792
	.long	0
	.long	1071776768
	.long	0
	.long	1071774720
	.long	0
	.long	1071773696
	.long	0
	.long	1071772672
	.long	0
	.long	1071770624
	.long	0
	.long	1071769600
	.long	0
	.long	1071767552
	.long	0
	.long	1071766528
	.long	0
	.long	1071765504
	.long	0
	.long	1071763456
	.long	0
	.long	1071762432
	.long	0
	.long	1071760384
	.long	0
	.long	1071759360
	.long	0
	.long	1071758336
	.long	0
	.long	1071756288
	.long	0
	.long	1071755264
	.long	0
	.long	1071754240
	.long	0
	.long	1071752192
	.long	0
	.long	1071751168
	.long	0
	.long	1071750144
	.long	0
	.long	1071748096
	.long	0
	.long	1071747072
	.long	0
	.long	1071746048
	.long	0
	.long	1071744000
	.long	0
	.long	1071742976
	.long	0
	.long	1071741952
	.long	0
	.long	1071739904
	.long	0
	.long	1071738880
	.long	0
	.long	1071737856
	.long	0
	.long	1071736832
	.long	0
	.long	1071734784
	.long	0
	.long	1071733760
	.long	0
	.long	1071732736
	.long	0
	.long	1071730688
	.long	0
	.long	1071729664
	.long	0
	.long	1071728640
	.long	0
	.long	1071727616
	.long	0
	.long	1071725568
	.long	0
	.long	1071724544
	.long	0
	.long	1071723520
	.long	0
	.long	1071722496
	.long	0
	.long	1071720448
	.long	0
	.long	1071719424
	.long	0
	.long	1071718400
	.long	0
	.long	1071717376
	.long	0
	.long	1071715328
	.long	0
	.long	1071714304
	.long	0
	.long	1071713280
	.long	0
	.long	1071712256
	.long	0
	.long	1071711232
	.long	0
	.long	1071709184
	.long	0
	.long	1071708160
	.long	0
	.long	1071707136
	.long	0
	.long	1071706112
	.long	0
	.long	1071705088
	.long	0
	.long	1071704064
	.long	0
	.long	1071702016
	.long	0
	.long	1071700992
	.long	0
	.long	1071699968
	.long	0
	.long	1071698944
	.long	0
	.long	1071697920
	.long	0
	.long	1071696896
	.long	0
	.long	1071694848
	.long	0
	.long	1071693824
	.long	0
	.long	1071692800
	.long	0
	.long	1071691776
	.long	0
	.long	1071690752
	.long	0
	.long	1071689728
	.long	0
	.long	1071688704
	.long	0
	.long	1071686656
	.long	0
	.long	1071685632
	.long	0
	.long	1071684608
	.long	0
	.long	1071683584
	.long	0
	.long	1071682560
	.long	0
	.long	1071681536
	.long	0
	.long	1071680512
	.long	0
	.long	1071679488
	.long	0
	.long	1071677440
	.long	0
	.long	1071676416
	.long	0
	.long	1071675392
	.long	0
	.long	1071674368
	.long	0
	.long	1071673344
	.long	0
	.long	1071672320
	.long	0
	.long	1071671296
	.long	0
	.long	1071670272
	.long	0
	.long	1071669248
	.long	0
	.long	1071668224
	.long	0
	.long	1071667200
	.long	0
	.long	1071666176
	.long	0
	.long	1071665152
	.long	0
	.long	1071663104
	.long	0
	.long	1071662080
	.long	0
	.long	1071661056
	.long	0
	.long	1071660032
	.long	0
	.long	1071659008
	.long	0
	.long	1071657984
	.long	0
	.long	1071656960
	.long	0
	.long	1071655936
	.long	0
	.long	1071654912
	.long	0
	.long	1071653888
	.long	0
	.long	1071652864
	.long	0
	.long	1071651840
	.long	0
	.long	1071650816
	.long	0
	.long	1071649792
	.long	0
	.long	1071648768
	.long	0
	.long	1071647744
	.long	0
	.long	1071646720
	.long	0
	.long	1071645696
	.long	0
	.long	1071644672
	.long	0
	.long	1101004800
	.long	1073741824
	.long	1095761920
	.long	256
	.long	1118830592
	.long	0
	.long	1017118720
	.long	1413754136
	.long	1073291771
	.long	856972295
	.long	1016178214
	.long	1413754136
	.long	1074340347
	.long	856972295
	.long	1017226790
	.long	0
	.long	1072693248
	.long	0
	.long	1071644672
	.long	0
	.long	0
	.long	1476395008
	.long	1069897045
	.long	1768958041
	.long	3189069141
	.long	805306368
	.long	1068708659
	.long	3580333578
	.long	1040816593
	.long	3067382784
	.long	1067899757
	.long	3397590151
	.long	1067392113
	.long	2939529726
	.long	1066854585
	.long	1423429166
	.long	1066517752
	.long	1775218934
	.long	1066178574
	.long	1185392460
	.long	1065859647
	.long	289998670
	.long	1065577550
	.long	3179807072
	.long	1065648121
	.long	3781007284
	.long	1061576176
	.long	2482106687
	.long	1067019199
	.long	763519713
	.long	3214591591
	.long	3695107454
	.long	1067530646
	.long	0
	.long	1073741824
	.long	1124791109
	.long	1006764147
	.long	1476395008
	.long	1069897045
	.long	1953913876
	.long	3189069141
	.long	805306368
	.long	1067660083
	.long	165110192
	.long	1039768033
	.long	3067304082
	.long	1065802605
	.long	3404727379
	.long	1064246385
	.long	2737480376
	.long	1062660281
	.long	933797922
	.long	1061274873
	.long	1475716730
	.long	1059887095
	.long	1511619763
	.long	1058519827
	.long	556024211
	.long	1057187555
	.long	3482101045
	.long	1056217350
	.long	1174622859
	.long	1050762633
	.long	899668651
	.long	1055506366
	.long	1081094694
	.long	3202035365
	.long	2559814773
	.long	1053906576
	.long	0
	.long	3219128320
	.long	0
	.long	1071120384
	.long	0
	.long	3218341888
	.long	0
	.long	1070694400
	.long	0
	.long	3218046976
	.long	0
	.long	1070391296
	.long	0
	.long	3217739776
	.type	_vmldACosHATab,@object
	.size	_vmldACosHATab,4520
	.space 88, 0x00 	
	.align 16
.L_2il0floatpacket.197:
	.long	0x00000000,0x80000000,0x00000000,0x00000000
	.type	.L_2il0floatpacket.197,@object
	.size	.L_2il0floatpacket.197,16
