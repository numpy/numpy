/**
 * This header implements `npyv_divisor_*` intrinsics used for computing the parameters
 * of fast integer division, while division intrinsics `npyv_divc_*` are defined in
 * {extension}/arithmetic.h.
 */
#ifndef NPY_SIMD
    #error "Not a standalone header, use simd/simd.h instead"
#endif
#ifndef _NPY_SIMD_INTDIV_H
#define _NPY_SIMD_INTDIV_H
/**********************************************************************************
 ** Integer division
 **********************************************************************************
 * Almost all architecture (except Power10) doesn't support integer vector division,
 * also the cost of scalar division in architectures like x86 is too high it can take
 * 30 to 40 cycles on modern chips and up to 100 on old ones.
 *
 * Therefore we are using division by multiplying with precomputed reciprocal technique,
 * the method that been used in this implementation is based on T. Granlund and P. L. Montgomery
 * “Division by invariant integers using multiplication(see [Figure 4.1, 5.1]
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.2556)
 *
 * It shows a good impact for all architectures especially on X86,
 * however computing divisor parameters is kind of expensive so this implementation
 * should only works when divisor is a scalar and used multiple of times.
 *
 * The division process is separated into two intrinsics for each data type
 *
 *  1- npyv_{dtype}x3 npyv_divisor_{dtype} ({dtype} divisor);
 *     For computing the divisor parameters (multiplier + shifters + sign of divisor(signed only))
 *
 *  2- npyv_{dtype} npyv_divisor_{dtype} (npyv_{dtype} dividend, npyv_{dtype}x3 divisor_parms);
 *     For performing the final division.
 *
 ** For example:
 *    int vstep = npyv_nlanes_s32;                // number of lanes
 *    int x     = 0x6e70;
 *    npyv_s32x3 divisor = npyv_divisor_s32(x);   // init divisor params
 *    for (; len >= vstep; src += vstep, dst += vstep, len -= vstep) {
 *        npyv_s32 a = npyv_load_s32(*src);       // load s32 vector from memory
 *                 a = npyv_divc_s32(a, divisor); // divide all elements by x
 *        npyv_store_s32(dst, a);                 // store s32 vector into memory
 *    }
 *
 ** NOTES:
 *  - For 64-bit division on Aarch64 and IBM/Power, we fall-back to the scalar division
 *    since emulating multiply-high is expensive and both architectures have very fast dividers.
 *
 ***************************************************************
 ** Figure 4.1: Unsigned division by run–time invariant divisor
 ***************************************************************
 * Initialization (given uword d with 1 ≤ d < 2^N):
 *    int l   = ceil(log2(d));
 *    uword m = 2^N * (2^l− d) / d + 1;
 *    int sh1 = min(l, 1);
 *    int sh2 = max(l − 1, 0);
 *
 * For q = FLOOR(a/d), all uword:
 *    uword t1 = MULUH(m, a);
 *    q = SRL(t1 + SRL(a − t1, sh1), sh2);
 *
 ************************************************************************************
 ** Figure 5.1: Signed division by run–time invariant divisor, rounded towards zero
 ************************************************************************************
 * Initialization (given constant sword d with d !=0):
 *    int l       = max(ceil(log2(abs(d))), 1);
 *    udword m0   = 1 + (2^(N+l-1)) / abs(d);
 *    sword  m    = m0 − 2^N;
 *    sword dsign = XSIGN(d);
 *    int sh      = l − 1;
 *
 * For q = TRUNC(a/d), all sword:
 *    sword q0 = a + MULSH(m, a);
 *          q0 = SRA(q0, sh) − XSIGN(a);
 *    q = EOR(q0, dsign) − dsign;
 */
/**
 * bit-scan reverse for non-zeros. returns the index of the highest set bit.
 * equivalent to floor(log2(a))
 */
#ifdef _MSC_VER
    #include <intrin.h> // _BitScanReverse
#endif
NPY_FINLINE unsigned npyv__bitscan_revnz_u32(npy_uint32 a)
{
    assert(a > 0); // due to use __builtin_clz
    unsigned r;
#if defined(NPY_HAVE_SSE2) && defined(_MSC_VER)
    unsigned long rl;
    (void)_BitScanReverse(&rl, (unsigned long)a);
    r = (unsigned)rl;

#elif defined(NPY_HAVE_SSE2) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)) \
    &&  (defined(NPY_CPU_X86) || defined(NPY_CPU_AMD64))
    __asm__("bsr %1, %0" : "=r" (r) : "r"(a));
#elif defined(__GNUC__) || defined(__clang__)
    r = 31 - __builtin_clz(a); // performs on arm -> clz, ppc -> cntlzw
#else
    r = 0;
    while (a >>= 1) {
        r++;
    }
#endif
    return r;
}
NPY_FINLINE unsigned npyv__bitscan_revnz_u64(npy_uint64 a)
{
    assert(a > 0); // due to use __builtin_clzll
#if defined(_M_AMD64) && defined(_MSC_VER)
    unsigned long rl;
    (void)_BitScanReverse64(&rl, a);
    return (unsigned)rl;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
    npy_uint64 r;
    __asm__("bsrq %1, %0" : "=r"(r) : "r"(a));
    return (unsigned)r;
#elif defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(a);
#else
    npy_uint64 a_hi = a >> 32;
    if (a_hi == 0) {
        return npyv__bitscan_revnz_u32((npy_uint32)a);
    }
    return 32 + npyv__bitscan_revnz_u32((npy_uint32)a_hi);
#endif
}
/**
 * Divides 128-bit unsigned integer by a 64-bit when the lower
 * 64-bit of the dividend is zero.
 *
 * This function is needed to calculate the multiplier of 64-bit integer division
 * see npyv_divisor_u64/npyv_divisor_s64.
 */
NPY_FINLINE npy_uint64 npyv__divh128_u64(npy_uint64 high, npy_uint64 divisor)
{
    assert(divisor > 1);
    npy_uint64 quotient;
#if defined(_M_X64) && defined(_MSC_VER) && _MSC_VER >= 1920 && !defined(__clang__)
    npy_uint64 remainder;
    quotient = _udiv128(high, 0, divisor, &remainder);
    (void)remainder;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
    __asm__("divq %[d]" : "=a"(quotient) : [d] "r"(divisor), "a"(0), "d"(high));
#elif defined(__SIZEOF_INT128__)
    quotient = (npy_uint64)((((__uint128_t)high) << 64) / divisor);
#else
    /**
     * Minified version based on Donald Knuth’s Algorithm D (Division of nonnegative integers),
     * and Generic implementation in Hacker’s Delight.
     *
     * See https://skanthak.homepage.t-online.de/division.html
     * with respect to the license of the Hacker's Delight book
     * (https://web.archive.org/web/20190408122508/http://www.hackersdelight.org/permissions.htm)
     */
    // shift amount for normalize
    unsigned ldz = 63 - npyv__bitscan_revnz_u64(divisor);
    // normalize divisor
    divisor <<= ldz;
    high    <<= ldz;
    // break divisor up into two 32-bit digits
    npy_uint32 divisor_hi  = divisor >> 32;
    npy_uint32 divisor_lo  = divisor & 0xFFFFFFFF;
    // compute high quotient digit
    npy_uint64 quotient_hi = high / divisor_hi;
    npy_uint64 remainder   = high - divisor_hi * quotient_hi;
    npy_uint64 base32      = 1ULL << 32;
    while (quotient_hi >= base32 || quotient_hi*divisor_lo > base32*remainder) {
        --quotient_hi;
        remainder += divisor_hi;
        if (remainder >= base32) {
            break;
        }
    }
    // compute dividend digit pairs
    npy_uint64 dividend_pairs = base32*high - divisor*quotient_hi;
    // compute second quotient digit for lower zeros
    npy_uint32 quotient_lo = (npy_uint32)(dividend_pairs / divisor_hi);
    quotient = base32*quotient_hi + quotient_lo;
#endif
    return quotient;
}
// Initializing divisor parameters for unsigned 8-bit division
NPY_FINLINE npyv_u8x3 npyv_divisor_u8(npy_uint8 d)
{
    unsigned l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // for potential divide by zero, On x86 GCC inserts `ud2` instruction
        // instead of letting the HW/CPU trap it which leads to illegal instruction exception.
        // 'volatile' should suppress this behavior and allow us to raise HW/CPU
        // arithmetic exception.
        m = sh1 = sh2 = 1 / ((npy_uint8 volatile *)&d)[0];
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;
    default:
        l   = npyv__bitscan_revnz_u32(d - 1) + 1;  // ceil(log2(d))
        l2  = (npy_uint8)(1 << l);                 // 2^l, overflow to 0 if l = 8
        m   = ((npy_uint16)((l2 - d) << 8)) / d + 1; // multiplier
        sh1 = 1;  sh2 = l - 1;                     // shift counts
    }
    npyv_u8x3 divisor;
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[0] = npyv_setall_u16(m);
    divisor.val[1] = npyv_set_u8(sh1);
    divisor.val[2] = npyv_set_u8(sh2);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[0] = npyv_setall_u8(m);
    divisor.val[1] = npyv_setall_u8(sh1);
    divisor.val[2] = npyv_setall_u8(sh2);
#elif defined(NPY_HAVE_NEON)
    divisor.val[0] = npyv_setall_u8(m);
    divisor.val[1] = npyv_reinterpret_u8_s8(npyv_setall_s8(-sh1));
    divisor.val[2] = npyv_reinterpret_u8_s8(npyv_setall_s8(-sh2));
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}
// Initializing divisor parameters for signed 8-bit division
NPY_FINLINE npyv_s16x3 npyv_divisor_s16(npy_int16 d);
NPY_FINLINE npyv_s8x3 npyv_divisor_s8(npy_int8 d)
{
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    npyv_s16x3 p = npyv_divisor_s16(d);
    npyv_s8x3 r;
    r.val[0] = npyv_reinterpret_s8_s16(p.val[0]);
    r.val[1] = npyv_reinterpret_s8_s16(p.val[1]);
    r.val[2] = npyv_reinterpret_s8_s16(p.val[2]);
    return r;
#else
    int d1 = abs(d);
    int sh, m;
    if (d1 > 1) {
        sh = (int)npyv__bitscan_revnz_u32(d1-1); // ceil(log2(abs(d))) - 1
        m = (1 << (8 + sh)) / d1 + 1;            // multiplier
    }
    else if (d1 == 1) {
        sh = 0; m = 1;
    }
    else {
        // raise arithmetic exception for d == 0
        sh = m = 1 / ((npy_int8 volatile *)&d)[0]; // LCOV_EXCL_LINE
    }
    npyv_s8x3 divisor;
    divisor.val[0] = npyv_setall_s8(m);
    divisor.val[2] = npyv_setall_s8(d < 0 ? -1 : 0);
    #if defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
        divisor.val[1] = npyv_setall_s8(sh);
    #elif defined(NPY_HAVE_NEON)
        divisor.val[1] = npyv_setall_s8(-sh);
    #else
        #error "please initialize the shifting operand for the new architecture"
    #endif
    return divisor;
#endif
}
// Initializing divisor parameters for unsigned 16-bit division
NPY_FINLINE npyv_u16x3 npyv_divisor_u16(npy_uint16 d)
{
    unsigned l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // raise arithmetic exception for d == 0
        m = sh1 = sh2 = 1 / ((npy_uint16 volatile *)&d)[0];
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;
    default:
        l   = npyv__bitscan_revnz_u32(d - 1) + 1; // ceil(log2(d))
        l2  = (npy_uint16)(1 << l);               // 2^l, overflow to 0 if l = 16
        m   = ((l2 - d) << 16) / d + 1;           // multiplier
        sh1 = 1;  sh2 = l - 1;                    // shift counts
    }
    npyv_u16x3 divisor;
    divisor.val[0] = npyv_setall_u16(m);
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_u16(sh1);
    divisor.val[2] = npyv_set_u16(sh2);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[1] = npyv_setall_u16(sh1);
    divisor.val[2] = npyv_setall_u16(sh2);
#elif defined(NPY_HAVE_NEON)
    divisor.val[1] = npyv_reinterpret_u16_s16(npyv_setall_s16(-sh1));
    divisor.val[2] = npyv_reinterpret_u16_s16(npyv_setall_s16(-sh2));
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}
// Initializing divisor parameters for signed 16-bit division
NPY_FINLINE npyv_s16x3 npyv_divisor_s16(npy_int16 d)
{
    int d1 = abs(d);
    int sh, m;
    if (d1 > 1) {
        sh = (int)npyv__bitscan_revnz_u32(d1 - 1); // ceil(log2(abs(d))) - 1
        m = (1 << (16 + sh)) / d1 + 1;             // multiplier
    }
    else if (d1 == 1) {
        sh = 0; m = 1;
    }
    else {
        // raise arithmetic exception for d == 0
        sh = m = 1 / ((npy_int16 volatile *)&d)[0]; // LCOV_EXCL_LINE
    }
    npyv_s16x3 divisor;
    divisor.val[0] = npyv_setall_s16(m);
    divisor.val[2] = npyv_setall_s16(d < 0 ? -1 : 0); // sign of divisor
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_s16(sh);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[1] = npyv_setall_s16(sh);
#elif defined(NPY_HAVE_NEON)
    divisor.val[1] = npyv_setall_s16(-sh);
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}
// Initializing divisor parameters for unsigned 32-bit division
NPY_FINLINE npyv_u32x3 npyv_divisor_u32(npy_uint32 d)
{
    npy_uint32 l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // raise arithmetic exception for d == 0
        m = sh1 = sh2 = 1 / ((npy_uint32 volatile *)&d)[0]; // LCOV_EXCL_LINE
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;
    default:
        l   = npyv__bitscan_revnz_u32(d - 1) + 1;     // ceil(log2(d))
        l2  = (npy_uint32)(1ULL << l);                // 2^l, overflow to 0 if l = 32
        m   = ((npy_uint64)(l2 - d) << 32) / d + 1;   // multiplier
        sh1 = 1;  sh2 = l - 1;                        // shift counts
    }
    npyv_u32x3 divisor;
    divisor.val[0] = npyv_setall_u32(m);
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_u32(sh1);
    divisor.val[2] = npyv_set_u32(sh2);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[1] = npyv_setall_u32(sh1);
    divisor.val[2] = npyv_setall_u32(sh2);
#elif defined(NPY_HAVE_NEON)
    divisor.val[1] = npyv_reinterpret_u32_s32(npyv_setall_s32(-sh1));
    divisor.val[2] = npyv_reinterpret_u32_s32(npyv_setall_s32(-sh2));
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}
// Initializing divisor parameters for signed 32-bit division
NPY_FINLINE npyv_s32x3 npyv_divisor_s32(npy_int32 d)
{
    npy_int32 d1 = abs(d);
    npy_int32 sh, m;
    // Handel abs overflow
    if ((npy_uint32)d == 0x80000000U) {
        m = 0x80000001;
        sh = 30;
    }
    else if (d1 > 1) {
        sh = npyv__bitscan_revnz_u32(d1 - 1); // ceil(log2(abs(d))) - 1
        m =  (1ULL << (32 + sh)) / d1 + 1;    // multiplier
    }
    else if (d1 == 1) {
        sh = 0; m = 1;
    }
    else {
        // raise arithmetic exception for d == 0
        sh = m = 1 / ((npy_int32 volatile *)&d)[0]; // LCOV_EXCL_LINE
    }
    npyv_s32x3 divisor;
    divisor.val[0] = npyv_setall_s32(m);
    divisor.val[2] = npyv_setall_s32(d < 0 ? -1 : 0); // sign of divisor
#ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_s32(sh);
#elif defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX)
    divisor.val[1] = npyv_setall_s32(sh);
#elif defined(NPY_HAVE_NEON)
    divisor.val[1] = npyv_setall_s32(-sh);
#else
    #error "please initialize the shifting operand for the new architecture"
#endif
    return divisor;
}
// Initializing divisor parameters for unsigned 64-bit division
NPY_FINLINE npyv_u64x3 npyv_divisor_u64(npy_uint64 d)
{
    npyv_u64x3 divisor;
#if defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX) || defined(NPY_HAVE_NEON)
    divisor.val[0] = npyv_setall_u64(d);
#else
    npy_uint64 l, l2, sh1, sh2, m;
    switch (d) {
    case 0: // LCOV_EXCL_LINE
        // raise arithmetic exception for d == 0
        m = sh1 = sh2 = 1 / ((npy_uint64 volatile *)&d)[0]; // LCOV_EXCL_LINE
        break;
    case 1:
        m = 1; sh1 = sh2 = 0;
        break;
    case 2:
        m = 1; sh1 = 1; sh2 = 0;
        break;
    default:
        l = npyv__bitscan_revnz_u64(d - 1) + 1;      // ceil(log2(d))
        l2 = l < 64 ? 1ULL << l : 0;                 // 2^l
        m = npyv__divh128_u64(l2 - d, d) + 1;        // multiplier
        sh1 = 1;  sh2 = l - 1;                       // shift counts
    }
    divisor.val[0] = npyv_setall_u64(m);
    #ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
        divisor.val[1] = npyv_set_u64(sh1);
        divisor.val[2] = npyv_set_u64(sh2);
    #else
        #error "please initialize the shifting operand for the new architecture"
    #endif
#endif
    return divisor;
}
// Initializing divisor parameters for signed 64-bit division
NPY_FINLINE npyv_s64x3 npyv_divisor_s64(npy_int64 d)
{
    npyv_s64x3 divisor;
#if defined(NPY_HAVE_VSX2) || defined(NPY_HAVE_VX) || defined(NPY_HAVE_NEON)
    divisor.val[0] = npyv_setall_s64(d);
    divisor.val[1] = npyv_cvt_s64_b64(
        npyv_cmpeq_s64(npyv_setall_s64(-1), divisor.val[0])
    );
#else
    npy_int64 d1 = llabs(d);
    npy_int64 sh, m;
    // Handel abs overflow
    if ((npy_uint64)d == 0x8000000000000000ULL) {
        m = 0x8000000000000001LL;
        sh = 62;
    }
    else if (d1 > 1) {
        sh = npyv__bitscan_revnz_u64(d1 - 1);       // ceil(log2(abs(d))) - 1
        m  = npyv__divh128_u64(1ULL << sh, d1) + 1; // multiplier
    }
    else if (d1 == 1) {
        sh = 0; m = 1;
    }
    else {
        // raise arithmetic exception for d == 0
        sh = m = 1 / ((npy_int64 volatile *)&d)[0]; // LCOV_EXCL_LINE
    }
    divisor.val[0] = npyv_setall_s64(m);
    divisor.val[2] = npyv_setall_s64(d < 0 ? -1 : 0);  // sign of divisor
    #ifdef NPY_HAVE_SSE2 // SSE/AVX2/AVX512
    divisor.val[1] = npyv_set_s64(sh);
    #else
        #error "please initialize the shifting operand for the new architecture"
    #endif
#endif
    return divisor;
}

#endif // _NPY_SIMD_INTDIV_H
