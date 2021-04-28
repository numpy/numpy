/*******************************************
* Copyright (C) 2021 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*******************************************/


	.text
.L_2__routine_start___svml_coshf16_z0_0:

	.align    16,0x90
	.globl __svml_coshf16

__svml_coshf16:


	.cfi_startproc
..L2:

        pushq     %rbp
	.cfi_def_cfa_offset 16
        movq      %rsp, %rbp
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16
        andq      $-64, %rsp
        subq      $192, %rsp
        vmovups   1024+__svml_scosh_data_internal(%rip), %zmm4
        vmovups   384+__svml_scosh_data_internal(%rip), %zmm6

/*
 * ............... Load argument ............................
 * dM = x/log(2) + RShifter
 */
        vmovups   768+__svml_scosh_data_internal(%rip), %zmm10
        vmovups   896+__svml_scosh_data_internal(%rip), %zmm7
        vmovups   960+__svml_scosh_data_internal(%rip), %zmm9

/* ... */
        vmovups   704+__svml_scosh_data_internal(%rip), %zmm2

/* x^2 */
        vmovups   640+__svml_scosh_data_internal(%rip), %zmm3

/* ............... G1,G2 2^N,2^(-N) ........... */
        vmovups   __svml_scosh_data_internal(%rip), %zmm12
        vmovups   256+__svml_scosh_data_internal(%rip), %zmm13

/*
 * -------------------- Implementation  -------------------
 * ............... Abs argument ............................
 */
        vandnps   %zmm0, %zmm4, %zmm1

/* ...............Check for overflow\underflow ............. */
        vpternlogd $255, %zmm5, %zmm5, %zmm5
        vfmadd213ps {rn-sae}, %zmm6, %zmm1, %zmm10
        vpcmpd    $1, 512+__svml_scosh_data_internal(%rip), %zmm1, %k1

/* iM now is an EXP(2^N) */
        vpslld    $18, %zmm10, %zmm11

/*
 * ................... R ...................................
 * sN = sM - RShifter
 */
        vsubps    {rn-sae}, %zmm6, %zmm10, %zmm8
        vpermt2ps 64+__svml_scosh_data_internal(%rip), %zmm10, %zmm12
        vpermt2ps 320+__svml_scosh_data_internal(%rip), %zmm10, %zmm13
        vpandnd   %zmm1, %zmm1, %zmm5{%k1}

/* sR = sX - sN*Log2_hi */
        vfnmadd231ps {rn-sae}, %zmm7, %zmm8, %zmm1
        vptestmd  %zmm5, %zmm5, %k0

/* sR = (sX - sN*Log2_hi) - sN*Log2_lo */
        vfnmadd231ps {rn-sae}, %zmm9, %zmm8, %zmm1
        kmovw     %k0, %edx
        vmulps    {rn-sae}, %zmm1, %zmm1, %zmm4
        vmulps    {rn-sae}, %zmm4, %zmm2, %zmm2

/* sSinh_r = r + r*(r^2*(a3)) */
        vfmadd213ps {rn-sae}, %zmm1, %zmm1, %zmm2

/* sOut = r^2*(a2) */
        vmulps    {rn-sae}, %zmm4, %zmm3, %zmm1
        vpandd    1216+__svml_scosh_data_internal(%rip), %zmm11, %zmm14
        vpaddd    %zmm14, %zmm12, %zmm15
        vpsubd    %zmm14, %zmm13, %zmm10

/* sG2 = 2^N*Th + 2^(-N)*T_h */
        vaddps    {rn-sae}, %zmm10, %zmm15, %zmm5

/* sG1 = 2^N*Th - 2^(-N)*T_h */
        vsubps    {rn-sae}, %zmm10, %zmm15, %zmm6

/* res = sG1*(r + r*(r^2*(a3))) + sG2*(1+r^2*(a2)) */
        vfmadd213ps {rn-sae}, %zmm5, %zmm5, %zmm1
        vfmadd213ps {rn-sae}, %zmm1, %zmm2, %zmm6
        testl     %edx, %edx
        jne       .LBL_1_3

.LBL_1_2:


/* no invcbrt in libm, so taking it out here */
        vmovaps   %zmm6, %zmm0
        movq      %rbp, %rsp
        popq      %rbp
	.cfi_def_cfa 7, 8
	.cfi_restore 6
        ret
	.cfi_def_cfa 6, 16
	.cfi_offset 6, -16

.LBL_1_3:

        vmovups   %zmm0, 64(%rsp)
        vmovups   %zmm6, 128(%rsp)
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
        vmovups   128(%rsp), %zmm6
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

        call      __svml_scosh_cout_rare_internal
        jmp       .LBL_1_8
	.align    16,0x90

	.cfi_endproc

	.type	__svml_coshf16,@function
	.size	__svml_coshf16,.-__svml_coshf16
..LN__svml_coshf16.0:

.L_2__routine_start___svml_scosh_cout_rare_internal_1:

	.align    16,0x90

__svml_scosh_cout_rare_internal:


	.cfi_startproc
..L53:

        movq      %rsi, %r8
        movzwl    2(%rdi), %edx
        xorl      %eax, %eax
        andl      $32640, %edx
        cmpl      $32640, %edx
        je        .LBL_2_12


        pxor      %xmm0, %xmm0
        cvtss2sd  (%rdi), %xmm0
        movsd     %xmm0, -8(%rsp)
        andb      $127, -1(%rsp)
        movzwl    -2(%rsp), %edx
        andl      $32752, %edx
        cmpl      $15504, %edx
        jle       .LBL_2_10


        movsd     -8(%rsp), %xmm0
        movsd     1096+__scosh_la_CoutTab(%rip), %xmm1
        comisd    %xmm0, %xmm1
        jbe       .LBL_2_9


        movq      1128+__scosh_la_CoutTab(%rip), %rdx
        movq      %rdx, -8(%rsp)
        comisd    1144+__scosh_la_CoutTab(%rip), %xmm0
        jb        .LBL_2_8


        movsd     1040+__scosh_la_CoutTab(%rip), %xmm1
        lea       __scosh_la_CoutTab(%rip), %r9
        mulsd     %xmm0, %xmm1
        addsd     1048+__scosh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movsd     1088+__scosh_la_CoutTab(%rip), %xmm1
        movl      -40(%rsp), %edx
        movl      %edx, %esi
        andl      $63, %esi
        subsd     1048+__scosh_la_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       (%rsi,%rsi), %ecx
        movsd     -32(%rsp), %xmm3
        lea       1(%rsi,%rsi), %edi
        mulsd     1104+__scosh_la_CoutTab(%rip), %xmm3
        movsd     -32(%rsp), %xmm4
        subsd     %xmm3, %xmm0
        mulsd     1112+__scosh_la_CoutTab(%rip), %xmm4
        shrl      $6, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm0, %xmm1
        addl      $1022, %edx
        andl      $2047, %edx
        addsd     1080+__scosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1072+__scosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1064+__scosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        addsd     1056+__scosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm1
        mulsd     %xmm0, %xmm1
        addsd     %xmm0, %xmm1
        movsd     (%r9,%rcx,8), %xmm0
        mulsd     %xmm0, %xmm1
        addsd     (%r9,%rdi,8), %xmm1
        addsd     %xmm0, %xmm1
        cmpl      $2046, %edx
        ja        .LBL_2_7


        movq      1128+__scosh_la_CoutTab(%rip), %rcx
        shrq      $48, %rcx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm1, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)
        ret

.LBL_2_7:

        decl      %edx
        andl      $2047, %edx
        movzwl    -2(%rsp), %ecx
        shll      $4, %edx
        andl      $-32753, %ecx
        orl       %edx, %ecx
        movw      %cx, -2(%rsp)
        movsd     -8(%rsp), %xmm0
        mulsd     %xmm0, %xmm1
        mulsd     1024+__scosh_la_CoutTab(%rip), %xmm1
        cvtsd2ss  %xmm1, %xmm1
        movss     %xmm1, (%r8)
        ret

.LBL_2_8:

        movsd     1040+__scosh_la_CoutTab(%rip), %xmm1
        lea       __scosh_la_CoutTab(%rip), %rcx
        movzwl    -2(%rsp), %esi
        andl      $-32753, %esi
        movsd     1080+__scosh_la_CoutTab(%rip), %xmm14
        mulsd     %xmm0, %xmm1
        addsd     1048+__scosh_la_CoutTab(%rip), %xmm1
        movsd     %xmm1, -40(%rsp)
        movsd     -40(%rsp), %xmm2
        movl      -40(%rsp), %r10d
        movl      %r10d, %r9d
        shrl      $6, %r9d
        subsd     1048+__scosh_la_CoutTab(%rip), %xmm2
        movsd     %xmm2, -32(%rsp)
        lea       1023(%r9), %edi
        movsd     -32(%rsp), %xmm3
        addl      $1022, %r9d
        mulsd     1104+__scosh_la_CoutTab(%rip), %xmm3
        andl      $63, %r10d
        movsd     -32(%rsp), %xmm4
        lea       (%r10,%r10), %edx
        mulsd     1112+__scosh_la_CoutTab(%rip), %xmm4
        subsd     %xmm3, %xmm0
        andl      $2047, %r9d
        negl      %edi
        movsd     (%rcx,%rdx,8), %xmm15
        negl      %edx
        shll      $4, %r9d
        addl      $-4, %edi
        orl       %r9d, %esi
        andl      $2047, %edi
        movw      %si, -2(%rsp)
        andl      $-32753, %esi
        shll      $4, %edi
        lea       1(%r10,%r10), %r11d
        movsd     -8(%rsp), %xmm6
        orl       %edi, %esi
        movw      %si, -2(%rsp)
        lea       128(%rdx), %esi
        addl      $129, %edx
        subsd     %xmm4, %xmm0
        mulsd     %xmm6, %xmm15
        movaps    %xmm0, %xmm5
        movaps    %xmm15, %xmm8
        mulsd     %xmm0, %xmm5
        movaps    %xmm15, %xmm10
        movsd     (%rcx,%r11,8), %xmm2
        mulsd     %xmm6, %xmm2
        mulsd     %xmm5, %xmm14
        movsd     -8(%rsp), %xmm7
        movaps    %xmm2, %xmm12
        movsd     (%rcx,%rdx,8), %xmm13
        mulsd     %xmm7, %xmm13
        addsd     1064+__scosh_la_CoutTab(%rip), %xmm14
        movsd     1088+__scosh_la_CoutTab(%rip), %xmm1
        subsd     %xmm13, %xmm12
        mulsd     %xmm5, %xmm1
        mulsd     %xmm5, %xmm14
        mulsd     %xmm0, %xmm12
        addsd     1072+__scosh_la_CoutTab(%rip), %xmm1
        mulsd     %xmm0, %xmm14
        addsd     %xmm12, %xmm2
        mulsd     %xmm5, %xmm1
        addsd     %xmm13, %xmm2
        addsd     1056+__scosh_la_CoutTab(%rip), %xmm1
        movsd     (%rcx,%rsi,8), %xmm11
        mulsd     %xmm7, %xmm11
        mulsd     %xmm5, %xmm1
        addsd     %xmm11, %xmm8
        subsd     %xmm11, %xmm15
        movsd     %xmm8, -24(%rsp)
        movsd     -24(%rsp), %xmm9
        mulsd     %xmm15, %xmm14
        subsd     %xmm9, %xmm10
        mulsd     %xmm15, %xmm0
        addsd     %xmm11, %xmm10
        addsd     %xmm14, %xmm2
        movsd     %xmm10, -16(%rsp)
        addsd     %xmm0, %xmm2
        movsd     -24(%rsp), %xmm3
        mulsd     %xmm3, %xmm1
        movsd     -16(%rsp), %xmm6
        addsd     %xmm1, %xmm2
        addsd     %xmm6, %xmm2
        movsd     %xmm2, -24(%rsp)
        movsd     -24(%rsp), %xmm0
        addsd     %xmm0, %xmm3
        cvtsd2ss  %xmm3, %xmm3
        movss     %xmm3, (%r8)
        ret

.LBL_2_9:

        movsd     1120+__scosh_la_CoutTab(%rip), %xmm0
        movl      $3, %eax
        mulsd     %xmm0, %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)
        ret

.LBL_2_10:

        movsd     1136+__scosh_la_CoutTab(%rip), %xmm0
        addsd     -8(%rsp), %xmm0
        cvtsd2ss  %xmm0, %xmm0
        movss     %xmm0, (%r8)


        ret

.LBL_2_12:

        movss     (%rdi), %xmm0
        mulss     %xmm0, %xmm0
        movss     %xmm0, (%r8)
        ret
	.align    16,0x90

	.cfi_endproc

	.type	__svml_scosh_cout_rare_internal,@function
	.size	__svml_scosh_cout_rare_internal,.-__svml_scosh_cout_rare_internal
..LN__svml_scosh_cout_rare_internal.1:

	.section .rodata, "a"
	.align 64
	.align 64
__svml_scosh_data_internal:
	.long	1056964608
	.long	1057148295
	.long	1057336003
	.long	1057527823
	.long	1057723842
	.long	1057924154
	.long	1058128851
	.long	1058338032
	.long	1058551792
	.long	1058770234
	.long	1058993458
	.long	1059221571
	.long	1059454679
	.long	1059692891
	.long	1059936319
	.long	1060185078
	.long	1060439283
	.long	1060699055
	.long	1060964516
	.long	1061235789
	.long	1061513002
	.long	1061796286
	.long	1062085772
	.long	1062381598
	.long	1062683901
	.long	1062992824
	.long	1063308511
	.long	1063631111
	.long	1063960775
	.long	1064297658
	.long	1064641917
	.long	1064993715
	.long	0
	.long	2999887785
	.long	852465809
	.long	3003046475
	.long	2984291233
	.long	3001644133
	.long	854021668
	.long	2997748242
	.long	849550193
	.long	2995541347
	.long	851518274
	.long	809701978
	.long	2997656926
	.long	2996185864
	.long	2980965110
	.long	3002882728
	.long	844097402
	.long	848217591
	.long	2999013352
	.long	2992006718
	.long	831170615
	.long	3002278818
	.long	833158180
	.long	3000769962
	.long	2991891850
	.long	2999994908
	.long	2979965785
	.long	2982419430
	.long	2982221534
	.long	2999469642
	.long	833168438
	.long	2987538264
	.long	1056964608
	.long	1056605107
	.long	1056253309
	.long	1055909050
	.long	1055572167
	.long	1055242503
	.long	1054919903
	.long	1054604216
	.long	1054295293
	.long	1053992990
	.long	1053697164
	.long	1053407678
	.long	1053124394
	.long	1052847181
	.long	1052575908
	.long	1052310447
	.long	1052050675
	.long	1051796470
	.long	1051547711
	.long	1051304283
	.long	1051066071
	.long	1050832963
	.long	1050604850
	.long	1050381626
	.long	1050163184
	.long	1049949424
	.long	1049740243
	.long	1049535546
	.long	1049335234
	.long	1049139215
	.long	1048947395
	.long	1048759687
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	1220542464
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
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
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1056964879
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1042983629
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	849703008
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	1060204544
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
	.long	939916788
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
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	31
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
	.long	1118743630
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
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1042983511
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1026206322
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	1007228001
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	985049251
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.type	__svml_scosh_data_internal,@object
	.size	__svml_scosh_data_internal,1920
	.align 32
__scosh_la_CoutTab:
	.long	0
	.long	1072693248
	.long	0
	.long	0
	.long	1048019041
	.long	1072704666
	.long	1398474845
	.long	3161559171
	.long	3541402996
	.long	1072716208
	.long	2759177317
	.long	1015903202
	.long	410360776
	.long	1072727877
	.long	1269990655
	.long	1013024446
	.long	1828292879
	.long	1072739672
	.long	1255956747
	.long	1016636974
	.long	852742562
	.long	1072751596
	.long	667253587
	.long	1010842135
	.long	3490863953
	.long	1072763649
	.long	960797498
	.long	3163997456
	.long	2930322912
	.long	1072775834
	.long	2599499422
	.long	3163762623
	.long	1014845819
	.long	1072788152
	.long	3117910646
	.long	3162607681
	.long	3949972341
	.long	1072800603
	.long	2068408548
	.long	1015962444
	.long	828946858
	.long	1072813191
	.long	10642492
	.long	1016988014
	.long	2288159958
	.long	1072825915
	.long	2169144469
	.long	1015924597
	.long	1853186616
	.long	1072838778
	.long	3066496371
	.long	1016705150
	.long	1709341917
	.long	1072851781
	.long	2571168217
	.long	1015201075
	.long	4112506593
	.long	1072864925
	.long	2947355221
	.long	1015419624
	.long	2799960843
	.long	1072878213
	.long	1423655381
	.long	1016070727
	.long	171030293
	.long	1072891646
	.long	3526460132
	.long	1015477354
	.long	2992903935
	.long	1072905224
	.long	2218154406
	.long	1016276769
	.long	926591435
	.long	1072918951
	.long	3208833762
	.long	3163962090
	.long	887463927
	.long	1072932827
	.long	3596744163
	.long	3161842742
	.long	1276261410
	.long	1072946854
	.long	300981948
	.long	1015732745
	.long	569847338
	.long	1072961034
	.long	472945272
	.long	3160339305
	.long	1617004845
	.long	1072975368
	.long	82804944
	.long	1011391354
	.long	3049340112
	.long	1072989858
	.long	3062915824
	.long	1014219171
	.long	3577096743
	.long	1073004506
	.long	2951496418
	.long	1014842263
	.long	1990012071
	.long	1073019314
	.long	3529070563
	.long	3163861769
	.long	1453150082
	.long	1073034283
	.long	498154669
	.long	3162536638
	.long	917841882
	.long	1073049415
	.long	18715565
	.long	1016707884
	.long	3712504873
	.long	1073064711
	.long	88491949
	.long	1016476236
	.long	363667784
	.long	1073080175
	.long	813753950
	.long	1016833785
	.long	2956612997
	.long	1073095806
	.long	2118169751
	.long	3163784129
	.long	2186617381
	.long	1073111608
	.long	2270764084
	.long	3164321289
	.long	1719614413
	.long	1073127582
	.long	330458198
	.long	3164331316
	.long	1013258799
	.long	1073143730
	.long	1748797611
	.long	3161177658
	.long	3907805044
	.long	1073160053
	.long	2257091225
	.long	3162598983
	.long	1447192521
	.long	1073176555
	.long	1462857171
	.long	3163563097
	.long	1944781191
	.long	1073193236
	.long	3993278767
	.long	3162772855
	.long	919555682
	.long	1073210099
	.long	3121969534
	.long	1013996802
	.long	2571947539
	.long	1073227145
	.long	3558159064
	.long	3164425245
	.long	2604962541
	.long	1073244377
	.long	2614425274
	.long	3164587768
	.long	1110089947
	.long	1073261797
	.long	1451641639
	.long	1016523249
	.long	2568320822
	.long	1073279406
	.long	2732824428
	.long	1015401491
	.long	2966275557
	.long	1073297207
	.long	2176155324
	.long	3160891335
	.long	2682146384
	.long	1073315202
	.long	2082178513
	.long	3164411995
	.long	2191782032
	.long	1073333393
	.long	2960257726
	.long	1014791238
	.long	2069751141
	.long	1073351782
	.long	1562170675
	.long	3163773257
	.long	2990417245
	.long	1073370371
	.long	3683467745
	.long	3164417902
	.long	1434058175
	.long	1073389163
	.long	251133233
	.long	1016134345
	.long	2572866477
	.long	1073408159
	.long	878562433
	.long	1016570317
	.long	3092190715
	.long	1073427362
	.long	814012168
	.long	3160571998
	.long	4076559943
	.long	1073446774
	.long	2119478331
	.long	3161806927
	.long	2420883922
	.long	1073466398
	.long	2049810052
	.long	1015168464
	.long	3716502172
	.long	1073486235
	.long	2303740125
	.long	1015091301
	.long	777507147
	.long	1073506289
	.long	4282924205
	.long	1016236109
	.long	3706687593
	.long	1073526560
	.long	3521726939
	.long	1014301643
	.long	1242007932
	.long	1073547053
	.long	1132034716
	.long	3164388407
	.long	3707479175
	.long	1073567768
	.long	3613079303
	.long	1015213314
	.long	64696965
	.long	1073588710
	.long	1768797490
	.long	1016865536
	.long	863738719
	.long	1073609879
	.long	1326992220
	.long	3163661773
	.long	3884662774
	.long	1073631278
	.long	2158611599
	.long	1015258761
	.long	2728693978
	.long	1073652911
	.long	396109971
	.long	3164511267
	.long	3999357479
	.long	1073674779
	.long	2258941616
	.long	1016973300
	.long	1533953344
	.long	1073696886
	.long	769171851
	.long	1016714209
	.long	2174652632
	.long	1073719233
	.long	4087714590
	.long	1015498835
	.long	0
	.long	1073741824
	.long	0
	.long	0
	.long	1697350398
	.long	1079448903
	.long	0
	.long	1127743488
	.long	0
	.long	1071644672
	.long	1431652600
	.long	1069897045
	.long	1431670732
	.long	1067799893
	.long	984555731
	.long	1065423122
	.long	472530941
	.long	1062650218
	.long	2684354560
	.long	1079401119
	.long	4277796864
	.long	1065758274
	.long	3164486458
	.long	1025308570
	.long	4294967295
	.long	2146435071
	.long	0
	.long	0
	.long	0
	.long	1072693248
	.long	3875694624
	.long	1077247184
	.type	__scosh_la_CoutTab,@object
	.size	__scosh_la_CoutTab,1152
