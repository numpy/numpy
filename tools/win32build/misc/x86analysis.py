#! /usr/bin/env python
# Last Change: Sat Mar 28 02:00 AM 2009 J

# Try to identify instruction set used in binary (x86 only). This works by
# checking the assembly for instructions specific to sse, etc... Obviously,
# this won't work all the times (for example, if some instructions are used
# only after proper detection of the running CPU, this will give false alarm).
from __future__ import division, print_function

import sys
import re
import os
import subprocess
import popen2
import optparse

I486_SET = ["cmpxchg", "xadd", "bswap", "invd", "wbinvd", "invlpg"]
I586_SET = ["rdmsr", "wrmsr", "rdtsc", "cmpxch8B", "rsm"]
PPRO_SET = ["cmovcc", "fcmovcc", "fcomi", "fcomip", "fucomi", "fucomip", "rdpmc", "ud2"]
MMX_SET = ["emms", "movd", "movq", "packsswb", "packssdw", "packuswb", "paddb",
        "paddw", "paddd", "paddsb", "paddsw", "paddusb", "paddusw", "pand",
        "pandn", "pcmpeqb", "pcmpeqw", "pcmpeqd", "pcmpgtb", "pcmpgtw",
        "pcmpgtd", "pmaddwd", "pmulhw", "pmullw", "por", "psllw", "pslld",
        "psllq", "psraw", "psrad", "psrlw", "psrld", "psrlq", "psubb", "psubw",
        "psubd", "psubsb", "psubsw", "psubusb", "psubusw", "punpckhbw",
        "punpckhwd", "punpckhdq", "punpcklbw", "punpcklwd", "punpckldq",
        "pxor"]
SSE_SET = ["addps",  "addss",  "andnps", "andps", "cmpps", "cmpss", "comiss",
        "cvtpi2ps", "cvtps2pi", "cvtsi2ss", "cvtss2si", "cvttps2pi",
        "cvttss2si", "divps", "divss", "fxrstor", "fxsave", "ldmxcsr", "maxps",
        "maxss", "minps", "minss", "movaps", "movhlps", "movhps", "movlhps",
        "movlps", "movmskps", "movss", "movups", "mulps", "mulss", "orps",
        "pavgb", "pavgw", "psadbw", "rcpps", "rcpss", "rsqrtps", "rsqrtss",
        "shufps", "sqrtps", "sqrtss", "stmxcsr", "subps", "subss", "ucomiss",
        "unpckhps", "unpcklps", "xorps", "pextrw", "pinsrw", "pmaxsw",
        "pmaxub", "pminsw", "pminub", "pmovmskb", "pmulhuw", "pshufw",
        "maskmovq", "movntps", "movntq", "prefetch", "sfence"]

SSE2_SET = ["addpd", "addsd", "andnpd", "andpd", "clflush", "cmppd", "cmpsd",
        "comisd", "cvtdq2pd", "cvtdq2ps", "cvtpd2pi", "cvtpd2pq", "cvtpd2ps",
        "cvtpi2pd", "cvtps2dq", "cvtps2pd", "cvtsd2si", "cvtsd2ss", "cvtsi2sd",
        "cvtss2sd", "cvttpd2pi", "cvttpd2dq", "cvttps2dq", "cvttsd2si",
        "divpd", "divsd", "lfence", "maskmovdqu", "maxpd", "maxsd", "mfence",
        "minpd", "minsd", "movapd", "movd", "movdq2q", "movdqa", "movdqu",
        "movhpd", "movlpd", "movmskpd", "movntdq", "movnti", "movntpd", "movq",
        "movq2dq", "movsd", "movupd", "mulpd", "mulsd", "orpd", "packsswb",
        "packssdw", "packuswb", "paddb", "paddw", "paddd", "paddq", "paddq",
        "paddsb", "paddsw", "paddusb", "paddusw", "pand", "pandn", "pause",
        "pavgb", "pavgw", "pcmpeqb", "pcmpeqw", "pcmpeqd", "pcmpgtb",
        "pcmpgtw", "pcmpgtd", "pextrw", "pinsrw", "pmaddwd", "pmaxsw",
        "pmaxub", "pminsw", "pminub", "pmovmskb", "pmulhw", "pmulhuw",
        "pmullw", "pmuludq", "pmuludq", "por", "psadbw", "pshufd", "pshufhw",
        "pshuflw", "pslldq", "psllw", "pslld", "psllq", "psraw", "psrad",
        "psrldq", "psrlw", "psrld", "psrlq", "psubb", "psubw", "psubd",
        "psubq", "psubq", "psubsb", "psubsw", "psubusb", "psubusw", "psubsb",
        "punpckhbw", "punpckhwd", "punpckhdq", "punpckhqdq", "punpcklbw",
        "punpcklwd", "punpckldq", "punpcklqdq", "pxor", "shufpd", "sqrtpd",
        "sqrtsd", "subpd", "subsd", "ucomisd", "unpckhpd", "unpcklpd", "xorpd"]

SSE3_SET = [ "addsubpd", "addsubps", "haddpd", "haddps", "hsubpd", "hsubps",
        "lddqu", "movddup", "movshdup", "movsldup", "fisttp"]

def get_vendor_string():
    """Return the vendor string reading cpuinfo."""
    try:
        a = open('/proc/cpuinfo').readlines()
        b = re.compile('^vendor_id.*')
        c = [i for i in a if b.match(i)]
    except IOError:
        raise ValueError("Could not read cpuinfo")


    int = re.compile("GenuineIntel")
    amd = re.compile("AuthenticAMD")
    cyr = re.compile("CyrixInstead")
    tra = re.compile("GenuineTMx86")
    if int.search(c[0]):
        return "intel"
    elif amd.search(c[0]):
        return "amd"
    elif cyr.search(c[0]):
        return "cyrix"
    elif tra.search(c[0]):
        return "tra"
    else:
        raise ValueError("Unknown vendor")

def disassemble(filename):
    """From a filename, returns a list of all asm instructions."""
    cmd = "i586-mingw32msvc-objdump -d %s " % filename
    o, i = popen2.popen2(cmd)
    def floupi(line):
        line1 = line.split('\t')
        if len(line1) > 2:
            line2 = line1[2]
        else:
            line2 = line1[0]
        line3 = line2.split(' ')
        if len(line3) > 1:
            inst = line3[0]
        else:
            inst = line3[0]
        return inst
    inst = [floupi(i) for i in o]
    return inst

def has_set(seq, asm_set):
    a = dict([(i, 0) for i in asm_set])
    for i in asm_set:
        a[i] = seq.count(i)
    return a

def has_sse(seq):
    return has_set(seq, SSE_SET)

def has_sse2(seq):
    return has_set(seq, SSE2_SET)

def has_sse3(seq):
    return has_set(seq, SSE3_SET)

def has_mmx(seq):
    return has_set(seq, MMX_SET)

def has_ppro(seq):
    return has_set(seq, PPRO_SET)

def cntset(seq):
    cnt = 0
    for i in seq.values():
        cnt += i
    return cnt

def main():
    #parser = optparse.OptionParser()
    #parser.add_option("-f", "--filename
    args = sys.argv[1:]
    filename = args[0]
    analyse(filename)

def analyse(filename):
    print(get_vendor_string())
    print("Getting instructions...")
    inst = disassemble(filename)
    print("Counting instructions...")
    sse = has_sse(inst)
    sse2 = has_sse2(inst)
    sse3 = has_sse3(inst)
    #mmx = has_mmx(inst)
    #ppro = has_ppro(inst)
    #print sse
    #print sse2
    #print sse3
    print("SSE3 inst %d" % cntset(sse3))
    print("SSE2 inst %d" % cntset(sse2))
    print("SSE inst %d" % cntset(sse))
    print("Analysed %d instructions" % len(inst))

if __name__ == '__main__':
    main()
    #filename = "/usr/lib/sse2/libatlas.a"
    ##filename = "/usr/lib/sse2/libcblas.a"
