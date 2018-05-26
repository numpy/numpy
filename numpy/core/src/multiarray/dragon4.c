/*
 * Copyright (c) 2014 Ryan Juckett
 * http://www.ryanjuckett.com/
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 */

/*
 * This file contains a modified version of Ryan Juckett's Dragon4
 * implementation, which has been ported from C++ to C and which has
 * modifications specific to printing floats in numpy.
 */

#include "dragon4.h"
#include <numpy/npy_common.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <assert.h>

#if 0
#define DEBUG_ASSERT(stmnt) assert(stmnt)
#else
#define DEBUG_ASSERT(stmnt) do {} while(0)
#endif

/*
 *  Get the log base 2 of a 32-bit unsigned integer.
 *  http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup
 */
static npy_uint32
LogBase2_32(npy_uint32 val)
{
    static const npy_uint8 logTable[256] =
    {
        0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
    };

    npy_uint32 temp;

    temp = val >> 24;
    if (temp) {
        return 24 + logTable[temp];
    }

    temp = val >> 16;
    if (temp) {
        return 16 + logTable[temp];
    }

    temp = val >> 8;
    if (temp) {
        return 8 + logTable[temp];
    }

    return logTable[val];
}

static npy_uint32
LogBase2_64(npy_uint64 val)
{
    npy_uint64 temp;

    temp = val >> 32;
    if (temp) {
        return 32 + LogBase2_32((npy_uint32)temp);
    }

    return LogBase2_32((npy_uint32)val);
}


/*
 * Maximum number of 32 bit blocks needed in high precision arithmetic to print
 * out 128 bit IEEE floating point values. 1023 chosen to be large enough for
 * 128 bit floats, and BigInt is exactly 4kb (nice for page/cache?)
 */
#define c_BigInt_MaxBlocks  1023

/*
 * This structure stores a high precision unsigned integer. It uses a buffer of
 * 32 bit integer blocks along with a length. The lowest bits of the integer
 * are stored at the start of the buffer and the length is set to the minimum
 * value that contains the integer. Thus, there are never any zero blocks at
 * the end of the buffer.
 */
typedef struct BigInt {
    npy_uint32 length;
    npy_uint32 blocks[c_BigInt_MaxBlocks];
} BigInt;

/* Copy integer */
static void
BigInt_Copy(BigInt *dst, const BigInt *src)
{
    npy_uint32 length = src->length;
    npy_uint32 * dstp = dst->blocks;
    const npy_uint32 *srcp;
    for (srcp = src->blocks; srcp != src->blocks + length; ++dstp, ++srcp) {
        *dstp = *srcp;
    }
    dst->length = length;
}

/* Basic type accessors */
static void
BigInt_Set_uint64(BigInt *i, npy_uint64 val)
{
    if (val > 0xFFFFFFFF) {
        i->blocks[0] = val & 0xFFFFFFFF;
        i->blocks[1] = (val >> 32) & 0xFFFFFFFF;
        i->length = 2;
    }
    else if (val != 0) {
        i->blocks[0] = val & 0xFFFFFFFF;
        i->length = 1;
    }
    else {
        i->length = 0;
    }
}

static void
BigInt_Set_uint32(BigInt *i, npy_uint32 val)
{
    if (val != 0) {
        i->blocks[0] = val;
        i->length = (val != 0);
    }
    else {
        i->length = 0;
    }
}

/*
 * Returns 0 if (lhs = rhs), negative if (lhs < rhs), positive if (lhs > rhs)
 */
static npy_int32
BigInt_Compare(const BigInt *lhs, const BigInt *rhs)
{
    int i;

    /* A bigger length implies a bigger number. */
    npy_int32 lengthDiff = lhs->length - rhs->length;
    if (lengthDiff != 0) {
        return lengthDiff;
    }

    /* Compare blocks one by one from high to low. */
    for (i = lhs->length - 1; i >= 0; --i) {
        if (lhs->blocks[i] == rhs->blocks[i]) {
            continue;
        }
        else if (lhs->blocks[i] > rhs->blocks[i]) {
            return 1;
        }
        else {
            return -1;
        }
    }

    /* no blocks differed */
    return 0;
}

/* result = lhs + rhs */
static void
BigInt_Add(BigInt *result, const BigInt *lhs, const BigInt *rhs)
{
    /* determine which operand has the smaller length */
    const BigInt *large, *small;
    npy_uint64 carry = 0;
    const npy_uint32 *largeCur, *smallCur, *largeEnd, *smallEnd;
    npy_uint32 *resultCur;

    if (lhs->length < rhs->length) {
        small = lhs;
        large = rhs;
    }
    else {
        small = rhs;
        large = lhs;
    }

    /* The output will be at least as long as the largest input */
    result->length = large->length;

    /* Add each block and add carry the overflow to the next block */
    largeCur  = large->blocks;
    largeEnd  = largeCur + large->length;
    smallCur  = small->blocks;
    smallEnd  = smallCur + small->length;
    resultCur = result->blocks;
    while (smallCur != smallEnd) {
        npy_uint64 sum = carry + (npy_uint64)(*largeCur) +
                                 (npy_uint64)(*smallCur);
        carry = sum >> 32;
        *resultCur = sum & 0xFFFFFFFF;
        ++largeCur;
        ++smallCur;
        ++resultCur;
    }

    /* Add the carry to any blocks that only exist in the large operand */
    while (largeCur != largeEnd) {
        npy_uint64 sum = carry + (npy_uint64)(*largeCur);
        carry = sum >> 32;
        (*resultCur) = sum & 0xFFFFFFFF;
        ++largeCur;
        ++resultCur;
    }

    /* If there's still a carry, append a new block */
    if (carry != 0) {
        DEBUG_ASSERT(carry == 1);
        DEBUG_ASSERT((npy_uint32)(resultCur - result->blocks) ==
               large->length && (large->length < c_BigInt_MaxBlocks));
        *resultCur = 1;
        result->length = large->length + 1;
    }
    else {
        result->length = large->length;
    }
}

/*
 * result = lhs * rhs
 */
static void
BigInt_Multiply(BigInt *result, const BigInt *lhs, const BigInt *rhs)
{
    const BigInt *large;
    const BigInt *small;
    npy_uint32 maxResultLen;
    npy_uint32 *cur, *end, *resultStart;
    const npy_uint32 *smallCur;

    DEBUG_ASSERT(result != lhs && result != rhs);

    /* determine which operand has the smaller length */
    if (lhs->length < rhs->length) {
        small = lhs;
        large = rhs;
    }
    else {
        small = rhs;
        large = lhs;
    }

    /* set the maximum possible result length */
    maxResultLen = large->length + small->length;
    DEBUG_ASSERT(maxResultLen <= c_BigInt_MaxBlocks);

    /* clear the result data */
    for (cur = result->blocks, end = cur + maxResultLen; cur != end; ++cur) {
        *cur = 0;
    }

    /* perform standard long multiplication for each small block */
    resultStart = result->blocks;
    for (smallCur = small->blocks;
            smallCur != small->blocks + small->length;
            ++smallCur, ++resultStart) {
        /*
         * if non-zero, multiply against all the large blocks and add into the
         * result
         */
        const npy_uint32 multiplier = *smallCur;
        if (multiplier != 0) {
            const npy_uint32 *largeCur = large->blocks;
            npy_uint32 *resultCur = resultStart;
            npy_uint64 carry = 0;
            do {
                npy_uint64 product = (*resultCur) +
                                     (*largeCur)*(npy_uint64)multiplier + carry;
                carry = product >> 32;
                *resultCur = product & 0xFFFFFFFF;
                ++largeCur;
                ++resultCur;
            } while(largeCur != large->blocks + large->length);

            DEBUG_ASSERT(resultCur < result->blocks + maxResultLen);
            *resultCur = (npy_uint32)(carry & 0xFFFFFFFF);
        }
    }

    /* check if the terminating block has no set bits */
    if (maxResultLen > 0 && result->blocks[maxResultLen - 1] == 0) {
        result->length = maxResultLen-1;
    }
    else {
        result->length = maxResultLen;
    }
}

/* result = lhs * rhs */
static void
BigInt_Multiply_int(BigInt *result, const BigInt *lhs, npy_uint32 rhs)
{
    /* perform long multiplication */
    npy_uint32 carry = 0;
    npy_uint32 *resultCur = result->blocks;
    const npy_uint32 *pLhsCur = lhs->blocks;
    const npy_uint32 *pLhsEnd = lhs->blocks + lhs->length;
    for ( ; pLhsCur != pLhsEnd; ++pLhsCur, ++resultCur) {
        npy_uint64 product = (npy_uint64)(*pLhsCur) * rhs + carry;
        *resultCur = (npy_uint32)(product & 0xFFFFFFFF);
        carry = product >> 32;
    }

    /* if there is a remaining carry, grow the array */
    if (carry != 0) {
        /* grow the array */
        DEBUG_ASSERT(lhs->length + 1 <= c_BigInt_MaxBlocks);
        *resultCur = (npy_uint32)carry;
        result->length = lhs->length + 1;
    }
    else {
        result->length = lhs->length;
    }
}

/* result = in * 2 */
static void
BigInt_Multiply2(BigInt *result, const BigInt *in)
{
    /* shift all the blocks by one */
    npy_uint32 carry = 0;

    npy_uint32 *resultCur = result->blocks;
    const npy_uint32 *pLhsCur = in->blocks;
    const npy_uint32 *pLhsEnd = in->blocks + in->length;
    for ( ; pLhsCur != pLhsEnd; ++pLhsCur, ++resultCur) {
        npy_uint32 cur = *pLhsCur;
        *resultCur = (cur << 1) | carry;
        carry = cur >> 31;
    }

    if (carry != 0) {
        /* grow the array */
        DEBUG_ASSERT(in->length + 1 <= c_BigInt_MaxBlocks);
        *resultCur = carry;
        result->length = in->length + 1;
    }
    else {
        result->length = in->length;
    }
}

/* result = result * 2 */
static void
BigInt_Multiply2_inplace(BigInt *result)
{
    /* shift all the blocks by one */
    npy_uint32 carry = 0;

    npy_uint32 *cur = result->blocks;
    npy_uint32 *end = result->blocks + result->length;
    for ( ; cur != end; ++cur) {
        npy_uint32 tmpcur = *cur;
        *cur = (tmpcur << 1) | carry;
        carry = tmpcur >> 31;
    }

    if (carry != 0) {
        /* grow the array */
        DEBUG_ASSERT(result->length + 1 <= c_BigInt_MaxBlocks);
        *cur = carry;
        ++result->length;
    }
}

/* result = result * 10 */
static void
BigInt_Multiply10(BigInt *result)
{
    /* multiply all the blocks */
    npy_uint64 carry = 0;

    npy_uint32 *cur = result->blocks;
    npy_uint32 *end = result->blocks + result->length;
    for ( ; cur != end; ++cur) {
        npy_uint64 product = (npy_uint64)(*cur) * 10ull + carry;
        (*cur) = (npy_uint32)(product & 0xFFFFFFFF);
        carry = product >> 32;
    }

    if (carry != 0) {
        /* grow the array */
        DEBUG_ASSERT(result->length + 1 <= c_BigInt_MaxBlocks);
        *cur = (npy_uint32)carry;
        ++result->length;
    }
}

static npy_uint32 g_PowerOf10_U32[] =
{
    1,          /* 10 ^ 0 */
    10,         /* 10 ^ 1 */
    100,        /* 10 ^ 2 */
    1000,       /* 10 ^ 3 */
    10000,      /* 10 ^ 4 */
    100000,     /* 10 ^ 5 */
    1000000,    /* 10 ^ 6 */
    10000000,   /* 10 ^ 7 */
};

/*
 * Note: This has a lot of wasted space in the big integer structures of the
 *       early table entries. It wouldn't be terribly hard to make the multiply
 *       function work on integer pointers with an array length instead of
 *       the BigInt struct which would allow us to store a minimal amount of
 *       data here.
 */
static BigInt g_PowerOf10_Big[] =
{
    /* 10 ^ 8 */
    { 1, { 100000000 } },
    /* 10 ^ 16 */
    { 2, { 0x6fc10000, 0x002386f2 } },
    /* 10 ^ 32 */
    { 4, { 0x00000000, 0x85acef81, 0x2d6d415b, 0x000004ee, } },
    /* 10 ^ 64 */
    { 7, { 0x00000000, 0x00000000, 0xbf6a1f01, 0x6e38ed64, 0xdaa797ed,
           0xe93ff9f4, 0x00184f03, } },
    /* 10 ^ 128 */
    { 14, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x2e953e01,
            0x03df9909, 0x0f1538fd, 0x2374e42f, 0xd3cff5ec, 0xc404dc08,
            0xbccdb0da, 0xa6337f19, 0xe91f2603, 0x0000024e, } },
    /* 10 ^ 256 */
    { 27, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x982e7c01, 0xbed3875b,
            0xd8d99f72, 0x12152f87, 0x6bde50c6, 0xcf4a6e70, 0xd595d80f,
            0x26b2716e, 0xadc666b0, 0x1d153624, 0x3c42d35a, 0x63ff540e,
            0xcc5573c0, 0x65f9ef17, 0x55bc28f2, 0x80dcc7f7, 0xf46eeddc,
            0x5fdcefce, 0x000553f7, } },
    /* 10 ^ 512 */
    { 54, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0xfc6cf801, 0x77f27267, 0x8f9546dc, 0x5d96976f,
            0xb83a8a97, 0xc31e1ad9, 0x46c40513, 0x94e65747, 0xc88976c1,
            0x4475b579, 0x28f8733b, 0xaa1da1bf, 0x703ed321, 0x1e25cfea,
            0xb21a2f22, 0xbc51fb2e, 0x96e14f5d, 0xbfa3edac, 0x329c57ae,
            0xe7fc7153, 0xc3fc0695, 0x85a91924, 0xf95f635e, 0xb2908ee0,
            0x93abade4, 0x1366732a, 0x9449775c, 0x69be5b0e, 0x7343afac,
            0xb099bc81, 0x45a71d46, 0xa2699748, 0x8cb07303, 0x8a0b1f13,
            0x8cab8a97, 0xc1d238d9, 0x633415d4, 0x0000001c, } },
    /* 10 ^ 1024 */
    { 107, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x2919f001, 0xf55b2b72, 0x6e7c215b,
             0x1ec29f86, 0x991c4e87, 0x15c51a88, 0x140ac535, 0x4c7d1e1a,
             0xcc2cd819, 0x0ed1440e, 0x896634ee, 0x7de16cfb, 0x1e43f61f,
             0x9fce837d, 0x231d2b9c, 0x233e55c7, 0x65dc60d7, 0xf451218b,
             0x1c5cd134, 0xc9635986, 0x922bbb9f, 0xa7e89431, 0x9f9f2a07,
             0x62be695a, 0x8e1042c4, 0x045b7a74, 0x1abe1de3, 0x8ad822a5,
             0xba34c411, 0xd814b505, 0xbf3fdeb3, 0x8fc51a16, 0xb1b896bc,
             0xf56deeec, 0x31fb6bfd, 0xb6f4654b, 0x101a3616, 0x6b7595fb,
             0xdc1a47fe, 0x80d98089, 0x80bda5a5, 0x9a202882, 0x31eb0f66,
             0xfc8f1f90, 0x976a3310, 0xe26a7b7e, 0xdf68368a, 0x3ce3a0b8,
             0x8e4262ce, 0x75a351a2, 0x6cb0b6c9, 0x44597583, 0x31b5653f,
             0xc356e38a, 0x35faaba6, 0x0190fba0, 0x9fc4ed52, 0x88bc491b,
             0x1640114a, 0x005b8041, 0xf4f3235e, 0x1e8d4649, 0x36a8de06,
             0x73c55349, 0xa7e6bd2a, 0xc1a6970c, 0x47187094, 0xd2db49ef,
             0x926c3f5b, 0xae6209d4, 0x2d433949, 0x34f4a3c6, 0xd4305d94,
             0xd9d61a05, 0x00000325, } },
    /* 10 ^ 2048 */
    { 213, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x1333e001,
             0xe3096865, 0xb27d4d3f, 0x49e28dcf, 0xec2e4721, 0xee87e354,
             0xb6067584, 0x368b8abb, 0xa5e5a191, 0x2ed56d55, 0xfd827773,
             0xea50d142, 0x51b78db2, 0x98342c9e, 0xc850dabc, 0x866ed6f1,
             0x19342c12, 0x92794987, 0xd2f869c2, 0x66912e4a, 0x71c7fd8f,
             0x57a7842d, 0x235552eb, 0xfb7fedcc, 0xf3861ce0, 0x38209ce1,
             0x9713b449, 0x34c10134, 0x8c6c54de, 0xa7a8289c, 0x2dbb6643,
             0xe3cb64f3, 0x8074ff01, 0xe3892ee9, 0x10c17f94, 0xa8f16f92,
             0xa8281ed6, 0x967abbb3, 0x5a151440, 0x9952fbed, 0x13b41e44,
             0xafe609c3, 0xa2bca416, 0xf111821f, 0xfb1264b4, 0x91bac974,
             0xd6c7d6ab, 0x8e48ff35, 0x4419bd43, 0xc4a65665, 0x685e5510,
             0x33554c36, 0xab498697, 0x0dbd21fe, 0x3cfe491d, 0x982da466,
             0xcbea4ca7, 0x9e110c7b, 0x79c56b8a, 0x5fc5a047, 0x84d80e2e,
             0x1aa9f444, 0x730f203c, 0x6a57b1ab, 0xd752f7a6, 0x87a7dc62,
             0x944545ff, 0x40660460, 0x77c1a42f, 0xc9ac375d, 0xe866d7ef,
             0x744695f0, 0x81428c85, 0xa1fc6b96, 0xd7917c7b, 0x7bf03c19,
             0x5b33eb41, 0x5715f791, 0x8f6cae5f, 0xdb0708fd, 0xb125ac8e,
             0x785ce6b7, 0x56c6815b, 0x6f46eadb, 0x4eeebeee, 0x195355d8,
             0xa244de3c, 0x9d7389c0, 0x53761abd, 0xcf99d019, 0xde9ec24b,
             0x0d76ce39, 0x70beb181, 0x2e55ecee, 0xd5f86079, 0xf56d9d4b,
             0xfb8886fb, 0x13ef5a83, 0x408f43c5, 0x3f3389a4, 0xfad37943,
             0x58ccf45c, 0xf82df846, 0x415c7f3e, 0x2915e818, 0x8b3d5cf4,
             0x6a445f27, 0xf8dbb57a, 0xca8f0070, 0x8ad803ec, 0xb2e87c34,
             0x038f9245, 0xbedd8a6c, 0xc7c9dee0, 0x0eac7d56, 0x2ad3fa14,
             0xe0de0840, 0xf775677c, 0xf1bd0ad5, 0x92be221e, 0x87fa1fb9,
             0xce9d04a4, 0xd2c36fa9, 0x3f6f7024, 0xb028af62, 0x907855ee,
             0xd83e49d6, 0x4efac5dc, 0xe7151aab, 0x77cd8c6b, 0x0a753b7d,
             0x0af908b4, 0x8c983623, 0xe50f3027, 0x94222771, 0x1d08e2d6,
             0xf7e928e6, 0xf2ee5ca6, 0x1b61b93c, 0x11eb962b, 0x9648b21c,
             0xce2bcba1, 0x34f77154, 0x7bbebe30, 0xe526a319, 0x8ce329ac,
             0xde4a74d2, 0xb5dc53d5, 0x0009e8b3, } },
    /* 10 ^ 4096 */
    { 426, { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
             0x00000000, 0x00000000, 0x00000000, 0x2a67c001, 0xd4724e8d,
             0x8efe7ae7, 0xf89a1e90, 0xef084117, 0x54e05154, 0x13b1bb51,
             0x506be829, 0xfb29b172, 0xe599574e, 0xf0da6146, 0x806c0ed3,
             0xb86ae5be, 0x45155e93, 0xc0591cc2, 0x7e1e7c34, 0x7c4823da,
             0x1d1f4cce, 0x9b8ba1e8, 0xd6bfdf75, 0xe341be10, 0xc2dfae78,
             0x016b67b2, 0x0f237f1a, 0x3dbeabcd, 0xaf6a2574, 0xcab3e6d7,
             0x142e0e80, 0x61959127, 0x2c234811, 0x87009701, 0xcb4bf982,
             0xf8169c84, 0x88052f8c, 0x68dde6d4, 0xbc131761, 0xff0b0905,
             0x54ab9c41, 0x7613b224, 0x1a1c304e, 0x3bfe167b, 0x441c2d47,
             0x4f6cea9c, 0x78f06181, 0xeb659fb8, 0x30c7ae41, 0x947e0d0e,
             0xa1ebcad7, 0xd97d9556, 0x2130504d, 0x1a8309cb, 0xf2acd507,
             0x3f8ec72a, 0xfd82373a, 0x95a842bc, 0x280f4d32, 0xf3618ac0,
             0x811a4f04, 0x6dc3a5b4, 0xd3967a1b, 0x15b8c898, 0xdcfe388f,
             0x454eb2a0, 0x8738b909, 0x10c4e996, 0x2bd9cc11, 0x3297cd0c,
             0x655fec30, 0xae0725b1, 0xf4090ee8, 0x037d19ee, 0x398c6fed,
             0x3b9af26b, 0xc994a450, 0xb5341743, 0x75a697b2, 0xac50b9c1,
             0x3ccb5b92, 0xffe06205, 0xa8329761, 0xdfea5242, 0xeb83cadb,
             0xe79dadf7, 0x3c20ee69, 0x1e0a6817, 0x7021b97a, 0x743074fa,
             0x176ca776, 0x77fb8af6, 0xeca19beb, 0x92baf1de, 0xaf63b712,
             0xde35c88b, 0xa4eb8f8c, 0xe137d5e9, 0x40b464a0, 0x87d1cde8,
             0x42923bbd, 0xcd8f62ff, 0x2e2690f3, 0x095edc16, 0x59c89f1b,
             0x1fa8fd5d, 0x5138753d, 0x390a2b29, 0x80152f18, 0x2dd8d925,
             0xf984d83e, 0x7a872e74, 0xc19e1faf, 0xed4d542d, 0xecf9b5d0,
             0x9462ea75, 0xc53c0adf, 0x0caea134, 0x37a2d439, 0xc8fa2e8a,
             0x2181327e, 0x6e7bb827, 0x2d240820, 0x50be10e0, 0x5893d4b8,
             0xab312bb9, 0x1f2b2322, 0x440b3f25, 0xbf627ede, 0x72dac789,
             0xb608b895, 0x78787e2a, 0x86deb3f0, 0x6fee7aab, 0xbb9373f4,
             0x27ecf57b, 0xf7d8b57e, 0xfca26a9f, 0x3d04e8d2, 0xc9df13cb,
             0x3172826a, 0xcd9e8d7c, 0xa8fcd8e0, 0xb2c39497, 0x307641d9,
             0x1cc939c1, 0x2608c4cf, 0xb6d1c7bf, 0x3d326a7e, 0xeeaf19e6,
             0x8e13e25f, 0xee63302b, 0x2dfe6d97, 0x25971d58, 0xe41d3cc4,
             0x0a80627c, 0xab8db59a, 0x9eea37c8, 0xe90afb77, 0x90ca19cf,
             0x9ee3352c, 0x3613c850, 0xfe78d682, 0x788f6e50, 0x5b060904,
             0xb71bd1a4, 0x3fecb534, 0xb32c450c, 0x20c33857, 0xa6e9cfda,
             0x0239f4ce, 0x48497187, 0xa19adb95, 0xb492ed8a, 0x95aca6a8,
             0x4dcd6cd9, 0xcf1b2350, 0xfbe8b12a, 0x1a67778c, 0x38eb3acc,
             0xc32da383, 0xfb126ab1, 0xa03f40a8, 0xed5bf546, 0xe9ce4724,
             0x4c4a74fd, 0x73a130d8, 0xd9960e2d, 0xa2ebd6c1, 0x94ab6feb,
             0x6f233b7c, 0x49126080, 0x8e7b9a73, 0x4b8c9091, 0xd298f999,
             0x35e836b5, 0xa96ddeff, 0x96119b31, 0x6b0dd9bc, 0xc6cc3f8d,
             0x282566fb, 0x72b882e7, 0xd6769f3b, 0xa674343d, 0x00fc509b,
             0xdcbf7789, 0xd6266a3f, 0xae9641fd, 0x4e89541b, 0x11953407,
             0x53400d03, 0x8e0dd75a, 0xe5b53345, 0x108f19ad, 0x108b89bc,
             0x41a4c954, 0xe03b2b63, 0x437b3d7f, 0x97aced8e, 0xcbd66670,
             0x2c5508c2, 0x650ebc69, 0x5c4f2ef0, 0x904ff6bf, 0x9985a2df,
             0x9faddd9e, 0x5ed8d239, 0x25585832, 0xe3e51cb9, 0x0ff4f1d4,
             0x56c02d9a, 0x8c4ef804, 0xc1a08a13, 0x13fd01c8, 0xe6d27671,
             0xa7c234f4, 0x9d0176cc, 0xd0d73df2, 0x4d8bfa89, 0x544f10cd,
             0x2b17e0b2, 0xb70a5c7d, 0xfd86fe49, 0xdf373f41, 0x214495bb,
             0x84e857fd, 0x00d313d5, 0x0496fcbe, 0xa4ba4744, 0xe8cac982,
             0xaec29e6e, 0x87ec7038, 0x7000a519, 0xaeee333b, 0xff66e42c,
             0x8afd6b25, 0x03b4f63b, 0xbd7991dc, 0x5ab8d9c7, 0x2ed4684e,
             0x48741a6c, 0xaf06940d, 0x2fdc6349, 0xb03d7ecd, 0xe974996f,
             0xac7867f9, 0x52ec8721, 0xbcdd9d4a, 0x8edd2d00, 0x3557de06,
             0x41c759f8, 0x3956d4b9, 0xa75409f2, 0x123cd8a1, 0xb6100fab,
             0x3e7b21e2, 0x2e8d623b, 0x92959da2, 0xbca35f77, 0x200c03a5,
             0x35fcb457, 0x1bb6c6e4, 0xf74eb928, 0x3d5d0b54, 0x87cc1d21,
             0x4964046f, 0x18ae4240, 0xd868b275, 0x8bd2b496, 0x1c5563f4,
             0xc234d8f5, 0xf868e970, 0xf9151fff, 0xae7be4a2, 0x271133ee,
             0xbb0fd922, 0x25254932, 0xa60a9fc0, 0x104bcd64, 0x30290145,
             0x00000062, } },
};

/* result = 10^exponent */
static void
BigInt_Pow10(BigInt *result, npy_uint32 exponent)
{
    /* create two temporary values to reduce large integer copy operations */
    BigInt temp1;
    BigInt temp2;
    BigInt *curTemp = &temp1;
    BigInt *pNextTemp = &temp2;
    npy_uint32 smallExponent;
    npy_uint32 tableIdx = 0;

    /* make sure the exponent is within the bounds of the lookup table data */
    DEBUG_ASSERT(exponent < 8192);

    /*
     * initialize the result by looking up a 32-bit power of 10 corresponding to
     * the first 3 bits
     */
    smallExponent = exponent & 0x7;
    BigInt_Set_uint32(curTemp, g_PowerOf10_U32[smallExponent]);

    /* remove the low bits that we used for the 32-bit lookup table */
    exponent >>= 3;

    /* while there are remaining bits in the exponent to be processed */
    while (exponent != 0) {
        /* if the current bit is set, multiply by this power of 10 */
        if (exponent & 1) {
            BigInt *pSwap;

            /* multiply into the next temporary */
            BigInt_Multiply(pNextTemp, curTemp, &g_PowerOf10_Big[tableIdx]);

            /* swap to the next temporary */
            pSwap = curTemp;
            curTemp = pNextTemp;
            pNextTemp = pSwap;
        }

        /* advance to the next bit */
        ++tableIdx;
        exponent >>= 1;
    }

    /* output the result */
    BigInt_Copy(result, curTemp);
}

/* result = in * 10^exponent */
static void
BigInt_MultiplyPow10(BigInt *result, const BigInt *in, npy_uint32 exponent)
{

    /* create two temporary values to reduce large integer copy operations */
    BigInt temp1;
    BigInt temp2;
    BigInt *curTemp = &temp1;
    BigInt *pNextTemp = &temp2;
    npy_uint32 smallExponent;
    npy_uint32 tableIdx = 0;

    /* make sure the exponent is within the bounds of the lookup table data */
    DEBUG_ASSERT(exponent < 8192);

    /*
     * initialize the result by looking up a 32-bit power of 10 corresponding to
     * the first 3 bits
     */
    smallExponent = exponent & 0x7;
    if (smallExponent != 0) {
        BigInt_Multiply_int(curTemp, in, g_PowerOf10_U32[smallExponent]);
    }
    else {
        BigInt_Copy(curTemp, in);
    }

    /* remove the low bits that we used for the 32-bit lookup table */
    exponent >>= 3;

    /* while there are remaining bits in the exponent to be processed */
    while (exponent != 0) {
        /* if the current bit is set, multiply by this power of 10 */
        if (exponent & 1) {
            BigInt *pSwap;

            /* multiply into the next temporary */
            BigInt_Multiply(pNextTemp, curTemp, &g_PowerOf10_Big[tableIdx]);

            // swap to the next temporary
            pSwap = curTemp;
            curTemp = pNextTemp;
            pNextTemp = pSwap;
        }

        /* advance to the next bit */
        ++tableIdx;
        exponent >>= 1;
    }

    /* output the result */
    BigInt_Copy(result, curTemp);
}

/* result = 2^exponent */
static inline void
BigInt_Pow2(BigInt *result, npy_uint32 exponent)
{
    npy_uint32 bitIdx;
    npy_uint32 blockIdx = exponent / 32;
    npy_uint32 i;

    DEBUG_ASSERT(blockIdx < c_BigInt_MaxBlocks);

    for (i = 0; i <= blockIdx; ++i) {
        result->blocks[i] = 0;
    }

    result->length = blockIdx + 1;

    bitIdx = (exponent % 32);
    result->blocks[blockIdx] |= (1 << bitIdx);
}

/*
 * This function will divide two large numbers under the assumption that the
 * result is within the range [0,10) and the input numbers have been shifted
 * to satisfy:
 * - The highest block of the divisor is greater than or equal to 8 such that
 *   there is enough precision to make an accurate first guess at the quotient.
 * - The highest block of the divisor is less than the maximum value on an
 *   unsigned 32-bit integer such that we can safely increment without overflow.
 * - The dividend does not contain more blocks than the divisor such that we
 *   can estimate the quotient by dividing the equivalently placed high blocks.
 *
 * quotient  = floor(dividend / divisor)
 * remainder = dividend - quotient*divisor
 *
 * dividend is updated to be the remainder and the quotient is returned.
 */
static npy_uint32
BigInt_DivideWithRemainder_MaxQuotient9(BigInt *dividend, const BigInt *divisor)
{
    npy_uint32 length, quotient;
    const npy_uint32 *finalDivisorBlock;
    npy_uint32 *finalDividendBlock;

    /*
     * Check that the divisor has been correctly shifted into range and that it
     * is not smaller than the dividend in length.
     */
    DEBUG_ASSERT(!divisor->length == 0 &&
                divisor->blocks[divisor->length-1] >= 8 &&
                divisor->blocks[divisor->length-1] < 0xFFFFFFFF &&
                dividend->length <= divisor->length);

    /*
     * If the dividend is smaller than the divisor, the quotient is zero and the
     * divisor is already the remainder.
     */
    length = divisor->length;
    if (dividend->length < divisor->length) {
        return 0;
    }

    finalDivisorBlock = divisor->blocks + length - 1;
    finalDividendBlock = dividend->blocks + length - 1;

    /*
     * Compute an estimated quotient based on the high block value. This will
     * either match the actual quotient or undershoot by one.
     */
    quotient = *finalDividendBlock / (*finalDivisorBlock + 1);
    DEBUG_ASSERT(quotient <= 9);

    /* Divide out the estimated quotient */
    if (quotient != 0) {
        /* dividend = dividend - divisor*quotient */
        const npy_uint32 *divisorCur = divisor->blocks;
        npy_uint32 *dividendCur = dividend->blocks;

        npy_uint64 borrow = 0;
        npy_uint64 carry = 0;
        do {
            npy_uint64 difference, product;

            product = (npy_uint64)*divisorCur * (npy_uint64)quotient + carry;
            carry = product >> 32;

            difference = (npy_uint64)*dividendCur
                       - (product & 0xFFFFFFFF) - borrow;
            borrow = (difference >> 32) & 1;

            *dividendCur = difference & 0xFFFFFFFF;

            ++divisorCur;
            ++dividendCur;
        } while(divisorCur <= finalDivisorBlock);

        /* remove all leading zero blocks from dividend */
        while (length > 0 && dividend->blocks[length - 1] == 0) {
            --length;
        }

        dividend->length = length;
    }

    /*
     * If the dividend is still larger than the divisor, we overshot our
     * estimate quotient. To correct, we increment the quotient and subtract one
     * more divisor from the dividend.
     */
    if (BigInt_Compare(dividend, divisor) >= 0) {
        /* dividend = dividend - divisor */
        const npy_uint32 *divisorCur = divisor->blocks;
        npy_uint32 *dividendCur = dividend->blocks;
        npy_uint64 borrow = 0;

        ++quotient;

        do {
            npy_uint64 difference = (npy_uint64)*dividendCur
                                  - (npy_uint64)*divisorCur - borrow;
            borrow = (difference >> 32) & 1;

            *dividendCur = difference & 0xFFFFFFFF;

            ++divisorCur;
            ++dividendCur;
        } while(divisorCur <= finalDivisorBlock);

        /* remove all leading zero blocks from dividend */
        while (length > 0 && dividend->blocks[length - 1] == 0) {
            --length;
        }

        dividend->length = length;
    }

    return quotient;
}

/* result = result << shift */
static void
BigInt_ShiftLeft(BigInt *result, npy_uint32 shift)
{
    npy_uint32 shiftBlocks = shift / 32;
    npy_uint32 shiftBits = shift % 32;

    /* process blocks high to low so that we can safely process in place */
    const npy_uint32 *pInBlocks = result->blocks;
    npy_int32 inLength = result->length;
    npy_uint32 *pInCur, *pOutCur;

    DEBUG_ASSERT(inLength + shiftBlocks < c_BigInt_MaxBlocks);
    DEBUG_ASSERT(shift != 0);

    /* check if the shift is block aligned */
    if (shiftBits == 0) {
        npy_uint32 i;

        /* copy blocks from high to low */
        for (pInCur = result->blocks + result->length,
                 pOutCur = pInCur + shiftBlocks;
                 pInCur >= pInBlocks;
                 --pInCur, --pOutCur) {
            *pOutCur = *pInCur;
        }

        /* zero the remaining low blocks */
        for (i  = 0; i < shiftBlocks; ++i) {
            result->blocks[i] = 0;
        }

        result->length += shiftBlocks;
    }
    /* else we need to shift partial blocks */
    else {
        npy_uint32 i;
        npy_int32 inBlockIdx = inLength - 1;
        npy_uint32 outBlockIdx = inLength + shiftBlocks;

        /* output the initial blocks */
        const npy_uint32 lowBitsShift = (32 - shiftBits);
        npy_uint32 highBits = 0;
        npy_uint32 block = result->blocks[inBlockIdx];
        npy_uint32 lowBits = block >> lowBitsShift;

        /* set the length to hold the shifted blocks */
        DEBUG_ASSERT(outBlockIdx < c_BigInt_MaxBlocks);
        result->length = outBlockIdx + 1;

        while (inBlockIdx > 0) {
            result->blocks[outBlockIdx] = highBits | lowBits;
            highBits = block << shiftBits;

            --inBlockIdx;
            --outBlockIdx;

            block = result->blocks[inBlockIdx];
            lowBits = block >> lowBitsShift;
        }

        /* output the final blocks */
        DEBUG_ASSERT(outBlockIdx == shiftBlocks + 1);
        result->blocks[outBlockIdx] = highBits | lowBits;
        result->blocks[outBlockIdx-1] = block << shiftBits;

        /* zero the remaining low blocks */
        for (i = 0; i < shiftBlocks; ++i) {
            result->blocks[i] = 0;
        }

        /* check if the terminating block has no set bits */
        if (result->blocks[result->length - 1] == 0) {
            --result->length;
        }
    }
}


/*
 * This is an implementation the Dragon4 algorithm to convert a binary number in
 * floating point format to a decimal number in string format. The function
 * returns the number of digits written to the output buffer and the output is
 * not NUL terminated.
 *
 * The floating point input value is (mantissa * 2^exponent).
 *
 * See the following papers for more information on the algorithm:
 *  "How to Print Floating-Point Numbers Accurately"
 *    Steele and White
 *    http://kurtstephens.com/files/p372-steele.pdf
 *  "Printing Floating-Point Numbers Quickly and Accurately"
 *    Burger and Dybvig
 *    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.4656
 *
 * This implementation is essentially a port of the "Figure 3" Scheme code from
 * Burger and Dybvig, but with the following additional differences:
 *   1. Instead of finding the highest k such that high < B**k, we search
 *      for the one where v < B**k. This has a downside that if a power
 *      of 10 exists between v and high, we will output a 9 instead of a 1 as
 *      first digit, violating the "no-carry" guarantee of the paper. This is
 *      accounted for in a new post-processing loop which implements a carry
 *      operation. The upside is one less BigInt multiplication.
 *   2. The approximate value of k found is offset by a different amount
 *      (0.69), in order to hit the "fast" branch more often. This is
 *      extensively described on Ryan Juckett's website.
 *   3. The fixed precision mode is much simpler than proposed in the paper.
 *      It simply outputs digits by repeatedly dividing by 10. The new "carry"
 *      loop at the end rounds this output nicely.
 *  There is also some new code to account for details of the BigInt
 *  implementation, which are not present in the paper since it does not specify
 *  details of the integer calculations.
 *
 * There is some more documentation of these changes on Ryan Juckett's website
 * at http://www.ryanjuckett.com/programming/printing-floating-point-numbers/
 *
 * Ryan Juckett's implementation did not implement "IEEE unbiased rounding",
 * except in the last digit. This has been added back, following the Burger &
 * Dybvig code, using the isEven variable.
 *
 * Arguments:
 *   * mantissa - value significand
 *   * exponent - value exponent in base 2
 *   * mantissaBit - index of the highest set mantissa bit
 *   * hasUnequalMargins - is the high margin twice as large as the low margin
 *   * cutoffMode - how to interpret cutoffNumber: fractional or total digits?
 *   * cutoffNumber - cut off printing after this many digits. -1 for no cutoff
 *   * pOutBuffer - buffer to output into
 *   * bufferSize - maximum characters that can be printed to pOutBuffer
 *   * pOutExponent - the base 10 exponent of the first digit
 */
static npy_uint32
Dragon4(const npy_uint64 mantissa, const npy_int32 exponent,
        const npy_uint32 mantissaBit, const npy_bool hasUnequalMargins,
        const DigitMode digitMode, const CutoffMode cutoffMode,
        npy_int32 cutoffNumber, char *pOutBuffer,
        npy_uint32 bufferSize, npy_int32 *pOutExponent)
{
    char *curDigit = pOutBuffer;

    /*
     * We compute values in integer format by rescaling as
     *   mantissa = scaledValue / scale
     *   marginLow = scaledMarginLow / scale
     *   marginHigh = scaledMarginHigh / scale
     * Here, marginLow and marginHigh represent 1/2 of the distance to the next
     * floating point value above/below the mantissa.
     *
     * scaledMarginHigh is a pointer so that it can point to scaledMarginLow in
     * the case they must be equal to each other, otherwise it will point to
     * optionalMarginHigh.
     */
    BigInt scale;
    BigInt scaledValue;
    BigInt scaledMarginLow;
    BigInt *scaledMarginHigh;
    BigInt optionalMarginHigh;

    const npy_float64 log10_2 = 0.30102999566398119521373889472449;
    npy_int32 digitExponent, cutoffExponent, hiBlock;
    npy_uint32 outputDigit;    /* current digit being output */
    npy_uint32 outputLen;
    npy_bool isEven = (mantissa % 2) == 0;
    npy_int32 cmp;

    /* values used to determine how to round */
    npy_bool low, high, roundDown;

    DEBUG_ASSERT(bufferSize > 0);

    /* if the mantissa is zero, the value is zero regardless of the exponent */
    if (mantissa == 0) {
        *curDigit = '0';
        *pOutExponent = 0;
        return 1;
    }

    if (hasUnequalMargins) {
        /* if we have no fractional component */
        if (exponent > 0) {
            /*
             * 1) Expand the input value by multiplying out the mantissa and
             *    exponent. This represents the input value in its whole number
             *    representation.
             * 2) Apply an additional scale of 2 such that later comparisons
             *    against the margin values are simplified.
             * 3) Set the margin value to the lowest mantissa bit's scale.
             */

            /* scaledValue      = 2 * 2 * mantissa*2^exponent */
            BigInt_Set_uint64(&scaledValue, mantissa);
            BigInt_ShiftLeft(&scaledValue, exponent + 2);

            /* scale            = 2 * 2 * 1 */
            BigInt_Set_uint32(&scale,  4);

            /* scaledMarginLow  = 2 * 2^(exponent-1) */
            BigInt_Pow2(&scaledMarginLow, exponent);

            /* scaledMarginHigh = 2 * 2 * 2^(exponent-1) */
            BigInt_Pow2(&optionalMarginHigh, exponent + 1);
        }
        /* else we have a fractional exponent */
        else {
            /*
             * In order to track the mantissa data as an integer, we store it as
             * is with a large scale
             */

            /* scaledValue      = 2 * 2 * mantissa */
            BigInt_Set_uint64(&scaledValue, mantissa);
            BigInt_ShiftLeft(&scaledValue, 2);

            /* scale            = 2 * 2 * 2^(-exponent) */
            BigInt_Pow2(&scale, -exponent + 2);

            /* scaledMarginLow  = 2 * 2^(-1) */
            BigInt_Set_uint32(&scaledMarginLow, 1);

            /* scaledMarginHigh = 2 * 2 * 2^(-1) */
            BigInt_Set_uint32(&optionalMarginHigh, 2);
        }

        /* the high and low margins are different */
        scaledMarginHigh = &optionalMarginHigh;
    }
    else {
        /* if we have no fractional component */
        if (exponent > 0) {
            /* scaledValue     = 2 * mantissa*2^exponent */
            BigInt_Set_uint64(&scaledValue, mantissa);
            BigInt_ShiftLeft(&scaledValue, exponent + 1);

            /* scale           = 2 * 1 */
            BigInt_Set_uint32(&scale, 2);

            /* scaledMarginLow = 2 * 2^(exponent-1) */
            BigInt_Pow2(&scaledMarginLow, exponent);
        }
        /* else we have a fractional exponent */
        else {
            /*
             * In order to track the mantissa data as an integer, we store it as
             * is with a large scale
             */

            /* scaledValue     = 2 * mantissa */
            BigInt_Set_uint64(&scaledValue, mantissa);
            BigInt_ShiftLeft(&scaledValue, 1);

            /* scale           = 2 * 2^(-exponent) */
            BigInt_Pow2(&scale, -exponent + 1);

            /* scaledMarginLow = 2 * 2^(-1) */
            BigInt_Set_uint32(&scaledMarginLow, 1);
        }

        /* the high and low margins are equal */
        scaledMarginHigh = &scaledMarginLow;
    }

    /*
     * Compute an estimate for digitExponent that will be correct or undershoot
     * by one.  This optimization is based on the paper "Printing Floating-Point
     * Numbers Quickly and Accurately" by Burger and Dybvig
     * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.4656
     * We perform an additional subtraction of 0.69 to increase the frequency of
     * a failed estimate because that lets us take a faster branch in the code.
     * 0.69 is chosen because 0.69 + log10(2) is less than one by a reasonable
     * epsilon that will account for any floating point error.
     *
     * We want to set digitExponent to floor(log10(v)) + 1
     *  v = mantissa*2^exponent
     *  log2(v) = log2(mantissa) + exponent;
     *  log10(v) = log2(v) * log10(2)
     *  floor(log2(v)) = mantissaBit + exponent;
     *  log10(v) - log10(2) < (mantissaBit + exponent) * log10(2) <= log10(v)
     *  log10(v) < (mantissaBit + exponent) * log10(2) + log10(2)
     *                                                 <= log10(v) + log10(2)
     *  floor(log10(v)) < ceil((mantissaBit + exponent) * log10(2))
     *                                                 <= floor(log10(v)) + 1
     */
    digitExponent = (npy_int32)(
       ceil((npy_float64)((npy_int32)mantissaBit + exponent) * log10_2 - 0.69));

    /*
     * if the digit exponent is smaller than the smallest desired digit for
     * fractional cutoff, pull the digit back into legal range at which point we
     * will round to the appropriate value.  Note that while our value for
     * digitExponent is still an estimate, this is safe because it only
     * increases the number. This will either correct digitExponent to an
     * accurate value or it will clamp it above the accurate value.
     */
    if (cutoffNumber >= 0 && cutoffMode == CutoffMode_FractionLength &&
            digitExponent <= -cutoffNumber) {
        digitExponent = -cutoffNumber + 1;
    }


    /* Divide value by 10^digitExponent. */
    if (digitExponent > 0) {
        /* A positive exponent creates a division so we multiply the scale. */
        BigInt temp;
        BigInt_MultiplyPow10(&temp, &scale, digitExponent);
        BigInt_Copy(&scale, &temp);
    }
    else if (digitExponent < 0) {
        /*
         * A negative exponent creates a multiplication so we multiply up the
         * scaledValue, scaledMarginLow and scaledMarginHigh.
         */
        BigInt pow10, temp;
        BigInt_Pow10(&pow10, -digitExponent);

        BigInt_Multiply(&temp, &scaledValue, &pow10);
        BigInt_Copy(&scaledValue, &temp);

        BigInt_Multiply(&temp, &scaledMarginLow, &pow10);
        BigInt_Copy(&scaledMarginLow, &temp);

        if (scaledMarginHigh != &scaledMarginLow) {
            BigInt_Multiply2(scaledMarginHigh, &scaledMarginLow);
        }
    }

    /* If (value >= 1), our estimate for digitExponent was too low */
    if (BigInt_Compare(&scaledValue, &scale) >= 0) {
        /*
         * The exponent estimate was incorrect.
         * Increment the exponent and don't perform the premultiply needed
         * for the first loop iteration.
         */
        digitExponent = digitExponent + 1;
    }
    else {
        /*
         * The exponent estimate was correct.
         * Multiply larger by the output base to prepare for the first loop
         * iteration.
         */
        BigInt_Multiply10(&scaledValue);
        BigInt_Multiply10(&scaledMarginLow);
        if (scaledMarginHigh != &scaledMarginLow) {
            BigInt_Multiply2(scaledMarginHigh, &scaledMarginLow);
        }
    }

    /*
     * Compute the cutoff exponent (the exponent of the final digit to print).
     * Default to the maximum size of the output buffer.
     */
    cutoffExponent = digitExponent - bufferSize;
    if (cutoffNumber >= 0) {
        npy_int32 desiredCutoffExponent;

        if (cutoffMode == CutoffMode_TotalLength) {
            desiredCutoffExponent = digitExponent - cutoffNumber;
            if (desiredCutoffExponent > cutoffExponent) {
                cutoffExponent = desiredCutoffExponent;
            }
        }
        /* Otherwise it's CutoffMode_FractionLength. Print cutoffNumber digits
         * past the decimal point or until we reach the buffer size
         */
        else {
            desiredCutoffExponent = -cutoffNumber;
            if (desiredCutoffExponent > cutoffExponent) {
                cutoffExponent = desiredCutoffExponent;
            }
        }
    }

    /* Output the exponent of the first digit we will print */
    *pOutExponent = digitExponent-1;

    /*
     * In preparation for calling BigInt_DivideWithRemainder_MaxQuotient9(), we
     * need to scale up our values such that the highest block of the
     * denominator is greater than or equal to 8. We also need to guarantee that
     * the numerator can never have a length greater than the denominator after
     * each loop iteration.  This requires the highest block of the denominator
     * to be less than or equal to 429496729 which is the highest number that
     * can be multiplied by 10 without overflowing to a new block.
     */
    DEBUG_ASSERT(scale.length > 0);
    hiBlock = scale.blocks[scale.length - 1];
    if (hiBlock < 8 || hiBlock > 429496729) {
        npy_uint32 hiBlockLog2, shift;

        /*
         * Perform a bit shift on all values to get the highest block of the
         * denominator into the range [8,429496729]. We are more likely to make
         * accurate quotient estimations in
         * BigInt_DivideWithRemainder_MaxQuotient9() with higher denominator
         * values so we shift the denominator to place the highest bit at index
         * 27 of the highest block.  This is safe because (2^28 - 1) = 268435455
         * which is less than 429496729. This means that all values with a
         * highest bit at index 27 are within range.
         */
        hiBlockLog2 = LogBase2_32(hiBlock);
        DEBUG_ASSERT(hiBlockLog2 < 3 || hiBlockLog2 > 27);
        shift = (32 + 27 - hiBlockLog2) % 32;

        BigInt_ShiftLeft(&scale, shift);
        BigInt_ShiftLeft(&scaledValue, shift);
        BigInt_ShiftLeft(&scaledMarginLow, shift);
        if (scaledMarginHigh != &scaledMarginLow) {
            BigInt_Multiply2(scaledMarginHigh, &scaledMarginLow);
        }
    }

    if (digitMode == DigitMode_Unique) {
        /*
         * For the unique cutoff mode, we will try to print until we have
         * reached a level of precision that uniquely distinguishes this value
         * from its neighbors. If we run out of space in the output buffer, we
         * terminate early.
         */
        for (;;) {
            BigInt scaledValueHigh;

            digitExponent = digitExponent-1;

            /* divide out the scale to extract the digit */
            outputDigit =
                BigInt_DivideWithRemainder_MaxQuotient9(&scaledValue, &scale);
            DEBUG_ASSERT(outputDigit < 10);

            /* update the high end of the value */
            BigInt_Add(&scaledValueHigh, &scaledValue, scaledMarginHigh);

            /*
             * stop looping if we are far enough away from our neighboring
             * values or if we have reached the cutoff digit
             */
            cmp = BigInt_Compare(&scaledValue, &scaledMarginLow);
            low = isEven ? (cmp <= 0) : (cmp < 0);
            cmp = BigInt_Compare(&scaledValueHigh, &scale);
            high = isEven ? (cmp >= 0) : (cmp > 0);
            if (low | high | (digitExponent == cutoffExponent))
                break;

            /* store the output digit */
            *curDigit = (char)('0' + outputDigit);
            ++curDigit;

            /* multiply larger by the output base */
            BigInt_Multiply10(&scaledValue);
            BigInt_Multiply10(&scaledMarginLow);
            if (scaledMarginHigh != &scaledMarginLow) {
                BigInt_Multiply2(scaledMarginHigh, &scaledMarginLow);
            }
        }
    }
    else {
        /*
         * For exact digit mode, we will try to print until we
         * have exhausted all precision (i.e. all remaining digits are zeros) or
         * until we reach the desired cutoff digit.
         */
        low = NPY_FALSE;
        high = NPY_FALSE;

        for (;;) {
            digitExponent = digitExponent-1;

            /* divide out the scale to extract the digit */
            outputDigit =
                BigInt_DivideWithRemainder_MaxQuotient9(&scaledValue, &scale);
            DEBUG_ASSERT(outputDigit < 10);

            if ((scaledValue.length == 0) | (digitExponent == cutoffExponent)) {
                break;
            }

            /* store the output digit */
            *curDigit = (char)('0' + outputDigit);
            ++curDigit;

            /* multiply larger by the output base */
            BigInt_Multiply10(&scaledValue);
        }
    }

    /* default to rounding down the final digit if value got too close to 0 */
    roundDown = low;

    /* if it is legal to round up and down */
    if (low == high) {
        npy_int32 compare;

        /*
         * round to the closest digit by comparing value with 0.5. To do this we
         * need to convert the inequality to large integer values.
         *  compare( value, 0.5 )
         *  compare( scale * value, scale * 0.5 )
         *  compare( 2 * scale * value, scale )
         */
        BigInt_Multiply2_inplace(&scaledValue);
        compare = BigInt_Compare(&scaledValue, &scale);
        roundDown = compare < 0;

        /*
         * if we are directly in the middle, round towards the even digit (i.e.
         * IEEE rounding rules)
         */
        if (compare == 0) {
            roundDown = (outputDigit & 1) == 0;
        }
    }

    /* print the rounded digit */
    if (roundDown) {
        *curDigit = (char)('0' + outputDigit);
        ++curDigit;
    }
    else {
        /* handle rounding up */
        if (outputDigit == 9) {
            /* find the first non-nine prior digit */
            for (;;) {
                /* if we are at the first digit */
                if (curDigit == pOutBuffer) {
                    /* output 1 at the next highest exponent */
                    *curDigit = '1';
                    ++curDigit;
                    *pOutExponent += 1;
                    break;
                }

                --curDigit;
                if (*curDigit != '9') {
                    /* increment the digit */
                    *curDigit += 1;
                    ++curDigit;
                    break;
                }
            }
        }
        else {
            /* values in the range [0,8] can perform a simple round up */
            *curDigit = (char)('0' + outputDigit + 1);
            ++curDigit;
        }
    }

    /* return the number of digits output */
    outputLen = (npy_uint32)(curDigit - pOutBuffer);
    DEBUG_ASSERT(outputLen <= bufferSize);
    return outputLen;
}


/*
 * Helper union to decompose a 16-bit IEEE float.
 * sign:      1 bit
 * exponent:  5 bits
 * mantissa: 10 bits
 */
typedef union FloatUnion16
{
    npy_uint16 integer;
} FloatUnion16;

npy_bool   IsNegative_F16(FloatUnion16 *v) { return (v->integer >> 15) != 0; }
npy_uint32 GetExponent_F16(FloatUnion16 *v) { return (v->integer >> 10) & 0x1F;}
npy_uint32 GetMantissa_F16(FloatUnion16 *v) { return v->integer & 0x3FF; }


/*
 * Helper union to decompose a 32-bit IEEE float.
 * sign:      1 bit
 * exponent:  8 bits
 * mantissa: 23 bits
 */
typedef union FloatUnion32
{
    npy_float32 floatingPoint;
    npy_uint32 integer;
} FloatUnion32;

npy_bool   IsNegative_F32(FloatUnion32 *v) { return (v->integer >> 31) != 0; }
npy_uint32 GetExponent_F32(FloatUnion32 *v) { return (v->integer >> 23) & 0xFF;}
npy_uint32 GetMantissa_F32(FloatUnion32 *v) { return v->integer & 0x7FFFFF; }

/*
 * Helper union to decompose a 64-bit IEEE float.
 * sign:      1 bit
 * exponent: 11 bits
 * mantissa: 52 bits
 */
typedef union FloatUnion64
{
    npy_float64 floatingPoint;
    npy_uint64 integer;
} FloatUnion64;
npy_bool   IsNegative_F64(FloatUnion64 *v) { return (v->integer >> 63) != 0; }
npy_uint32 GetExponent_F64(FloatUnion64 *v) { return (v->integer >> 52) & 0x7FF; }
npy_uint64 GetMantissa_F64(FloatUnion64 *v) { return v->integer & 0xFFFFFFFFFFFFFull; }

/*
 * Helper unions and datatype to decompose a 80-bit IEEE float
 * sign:      1 bit,  second u64
 * exponent: 15 bits, second u64
 * intbit     1 bit,  first u64
 * mantissa: 63 bits, first u64
 */

/*
 * Since systems have different types of long doubles, and may not necessarily
 * have a 128-byte format we can use to pass values around, here we create
 * our own 128-bit storage type for convenience.
 */
typedef struct FloatVal128 {
    npy_uint64 integer[2];
} FloatVal128;
npy_bool   IsNegative_F128(FloatVal128 *v) {
    return ((v->integer[1] >> 15) & 0x1) != 0;
}
npy_uint32 GetExponent_F128(FloatVal128 *v) { return v->integer[1] & 0x7FFF; }
npy_uint64 GetMantissa_F128(FloatVal128 *v) {
    return v->integer[0] &  0x7FFFFFFFFFFFFFFFull;
}

/*
 * then for each different definition of long double, we create a union to
 * unpack the float data safely. We can then copy these integers to a
 * FloatVal128.
 */
#ifdef NPY_FLOAT128
typedef union FloatUnion128
{
    npy_float128 floatingPoint;
    struct {
        npy_uint64 a;
        npy_uint16 b;
    } integer;
} FloatUnion128;
#endif

#ifdef NPY_FLOAT96
typedef union FloatUnion96
{
    npy_float96 floatingPoint;
    struct {
        npy_uint64 a;
        npy_uint32 b;
    } integer;
} FloatUnion96;
#endif

#ifdef NPY_FLOAT80
typedef union FloatUnion80
{
    npy_float80 floatingPoint;
    struct {
        npy_uint64 a;
        npy_uint16 b;
    } integer;
} FloatUnion80;
#endif


/*
 * The main changes above this point, relative to Ryan Juckett's code, are:
 *  1. fixed overflow problems when mantissa was 64 bits (in float128 types),
 *     by replacing multiplication by 2 or 4 by BigInt_ShiftLeft calls.
 *  2. Increased c_BigInt_MaxBlocks
 *  3. Added more entries to the g_PowerOf10_Big table
 *  4. Added unbiased rounding calculation with isEven
 *
 * Below this point, the FormatPositional and FormatScientific functions have
 * been more significantly rewritten. The Dragon4_PrintFloat16 and
 * Dragon4_PrintFloat128 functions are new, and were adapted from the 64 and 32
 * bit versions. The python interfacing functions (in the header) are new.
 */


/*
 * Outputs the positive number with positional notation: ddddd.dddd
 * The output is always NUL terminated and the output length (not including the
 * NUL) is returned.
 * Arguments:
 *    buffer - buffer to output into
 *    bufferSize - maximum characters that can be printed to buffer
 *    mantissa - value significand
 *    exponent - value exponent in base 2
 *    signbit - value of the sign position. Should be '+', '-' or ''
 *    mantissaBit - index of the highest set mantissa bit
 *    hasUnequalMargins - is the high margin twice as large as the low margin
 *    precision - Negative prints as many digits as are needed for a unique
 *                number. Positive specifies the maximum number of significant
 *                digits to print past the decimal point.
 *    trim_mode - how to treat trailing 0s and '.'. See TrimMode comments.
 *    digits_left - pad characters to left of decimal point. -1 for no padding
 *    digits_right - pad characters to right of decimal point. -1 for no padding
 *                   padding adds whitespace until there are the specified
 *                   number characters to sides of decimal point. Applies after
 *                   trim_mode characters were removed. If digits_right is
 *                   positive and the decimal point was trimmed, decimal point
 *                   will be replaced by a whitespace character.
 */
static npy_uint32
FormatPositional(char *buffer, npy_uint32 bufferSize, npy_uint64 mantissa,
                 npy_int32 exponent, char signbit, npy_uint32 mantissaBit,
                 npy_bool hasUnequalMargins, DigitMode digit_mode,
                 CutoffMode cutoff_mode, npy_int32 precision,
                 TrimMode trim_mode, npy_int32 digits_left,
                 npy_int32 digits_right)
{
    npy_int32 printExponent;
    npy_int32 numDigits, numWholeDigits=0, has_sign=0;

    npy_int32 maxPrintLen = (npy_int32)bufferSize - 1, pos = 0;

    /* track the # of digits past the decimal point that have been printed */
    npy_int32 numFractionDigits = 0, desiredFractionalDigits;

    DEBUG_ASSERT(bufferSize > 0);

    if (digit_mode != DigitMode_Unique) {
        DEBUG_ASSERT(precision >= 0);
    }

    if (signbit == '+' && pos < maxPrintLen) {
        buffer[pos++] = '+';
        has_sign = 1;
    }
    else if (signbit == '-' && pos < maxPrintLen) {
        buffer[pos++] = '-';
        has_sign = 1;
    }

    numDigits = Dragon4(mantissa, exponent, mantissaBit, hasUnequalMargins,
                        digit_mode, cutoff_mode, precision, buffer + has_sign,
                        maxPrintLen - has_sign, &printExponent);

    DEBUG_ASSERT(numDigits > 0);
    DEBUG_ASSERT(numDigits <= bufferSize);

    /* if output has a whole number */
    if (printExponent >= 0) {
        /* leave the whole number at the start of the buffer */
        numWholeDigits = printExponent+1;
        if (numDigits <= numWholeDigits) {
            npy_int32 count = numWholeDigits - numDigits;
            pos += numDigits;

            /* don't overflow the buffer */
            if (pos + count > maxPrintLen) {
                count = maxPrintLen - pos;
            }

            /* add trailing zeros up to the decimal point */
            numDigits += count;
            for ( ; count > 0; count--) {
                buffer[pos++] = '0';
            }
        }
        /* insert the decimal point prior to the fraction */
        else if (numDigits > numWholeDigits) {
            npy_int32 maxFractionDigits;

            numFractionDigits = numDigits - numWholeDigits;
            maxFractionDigits = maxPrintLen - numWholeDigits - 1 - pos;
            if (numFractionDigits > maxFractionDigits) {
                numFractionDigits = maxFractionDigits;
            }

            memmove(buffer + pos + numWholeDigits + 1,
                    buffer + pos + numWholeDigits, numFractionDigits);
            pos += numWholeDigits;
            buffer[pos] = '.';
            numDigits = numWholeDigits + 1 + numFractionDigits;
            pos += 1 + numFractionDigits;
        }
    }
    else {
        /* shift out the fraction to make room for the leading zeros */
        npy_int32 numFractionZeros = 0;
        if (pos + 2 < maxPrintLen) {
            npy_int32 maxFractionZeros, digitsStartIdx, maxFractionDigits, i;

            maxFractionZeros = maxPrintLen - 2 - pos;
            numFractionZeros = -(printExponent + 1);
            if (numFractionZeros > maxFractionZeros) {
                numFractionZeros = maxFractionZeros;
            }

            digitsStartIdx = 2 + numFractionZeros;

            /*
             * shift the significant digits right such that there is room for
             * leading zeros
             */
            numFractionDigits = numDigits;
            maxFractionDigits = maxPrintLen - digitsStartIdx - pos;
            if (numFractionDigits > maxFractionDigits) {
                numFractionDigits = maxFractionDigits;
            }

            memmove(buffer + pos + digitsStartIdx, buffer + pos,
                    numFractionDigits);

            /* insert the leading zeros */
            for (i = 2; i < digitsStartIdx; ++i) {
                buffer[pos + i] = '0';
            }

            /* update the counts */
            numFractionDigits += numFractionZeros;
            numDigits = numFractionDigits;
        }

        /* add the decimal point */
        if (pos + 1 < maxPrintLen) {
            buffer[pos+1] = '.';
        }

        /* add the initial zero */
        if (pos < maxPrintLen) {
            buffer[pos] = '0';
            numDigits += 1;
        }
        numWholeDigits = 1;
        pos += 2 + numFractionDigits;
    }

    /* always add decimal point, except for DprZeros mode */
    if (trim_mode != TrimMode_DptZeros && numFractionDigits == 0 &&
            pos < maxPrintLen){
        buffer[pos++] = '.';
    }

    desiredFractionalDigits = precision;
    if (cutoff_mode == CutoffMode_TotalLength && precision >= 0) {
        desiredFractionalDigits = precision - numWholeDigits;
    }

    if (trim_mode == TrimMode_LeaveOneZero) {
        /* if we didn't print any fractional digits, add a trailing 0 */
        if (numFractionDigits == 0 && pos < maxPrintLen) {
            buffer[pos++] = '0';
            numFractionDigits++;
        }
    }
    else if (trim_mode == TrimMode_None &&
             digit_mode != DigitMode_Unique &&
             desiredFractionalDigits > numFractionDigits &&
             pos < maxPrintLen) {
        /* add trailing zeros up to precision length */
        /* compute the number of trailing zeros needed */
        npy_int32 count = desiredFractionalDigits - numFractionDigits;
        if (pos + count > maxPrintLen) {
            count = maxPrintLen - pos;
        }
        numFractionDigits += count;

        for ( ; count > 0; count--) {
            buffer[pos++] = '0';
        }
    }
    /* else, for trim_mode Zeros or DptZeros, there is nothing more to add */

    /*
     * when rounding, we may still end up with trailing zeros. Remove them
     * depending on trim settings.
     */
    if (precision >= 0 && trim_mode != TrimMode_None && numFractionDigits > 0){
        while (buffer[pos-1] == '0') {
            pos--;
            numFractionDigits--;
        }
        if (trim_mode == TrimMode_LeaveOneZero && buffer[pos-1] == '.') {
            buffer[pos++] = '0';
            numFractionDigits++;
        }
    }

    /* add any whitespace padding to right side */
    if (digits_right >= numFractionDigits) {
        npy_int32 count = digits_right - numFractionDigits;

        /* in trim_mode DptZeros, if right padding, add a space for the . */
        if (trim_mode == TrimMode_DptZeros && numFractionDigits == 0
                && pos < maxPrintLen) {
            buffer[pos++] = ' ';
        }

        if (pos + count > maxPrintLen) {
            count = maxPrintLen - pos;
        }

        for ( ; count > 0; count--) {
            buffer[pos++] = ' ';
        }
    }
    /* add any whitespace padding to left side */
    if (digits_left > numWholeDigits + has_sign) {
        npy_int32 shift = digits_left - (numWholeDigits + has_sign);
        npy_int32 count = pos;

        if (count + shift > maxPrintLen){
            count = maxPrintLen - shift;
        }

        if (count > 0) {
            memmove(buffer + shift, buffer, count);
        }
        pos = shift + count;
        for ( ; shift > 0; shift--) {
            buffer[shift - 1] = ' ';
        }
    }

    /* terminate the buffer */
    DEBUG_ASSERT(pos <= maxPrintLen);
    buffer[pos] = '\0';

    return pos;
}

/*
 * Outputs the positive number with scientific notation: d.dddde[sign]ddd
 * The output is always NUL terminated and the output length (not including the
 * NUL) is returned.
 * Arguments:
 *    buffer - buffer to output into
 *    bufferSize - maximum characters that can be printed to buffer
 *    mantissa - value significand
 *    exponent - value exponent in base 2
 *    signbit - value of the sign position. Should be '+', '-' or ''
 *    mantissaBit - index of the highest set mantissa bit
 *    hasUnequalMargins - is the high margin twice as large as the low margin
 *    precision - Negative prints as many digits as are needed for a unique
 *                number. Positive specifies the maximum number of significant
 *                digits to print past the decimal point.
 *    trim_mode - how to treat trailing 0s and '.'. See TrimMode comments.
 *    digits_left - pad characters to left of decimal point. -1 for no padding
 *    exp_digits - pads exponent with zeros until it has this many digits
 */
static npy_uint32
FormatScientific (char *buffer, npy_uint32 bufferSize, npy_uint64 mantissa,
                  npy_int32 exponent, char signbit, npy_uint32 mantissaBit,
                  npy_bool hasUnequalMargins, DigitMode digit_mode,
                  npy_int32 precision, TrimMode trim_mode,
                  npy_int32 digits_left, npy_int32 exp_digits)
{
    npy_int32 printExponent;
    npy_int32 numDigits;
    char *pCurOut;
    npy_int32 numFractionDigits;
    npy_int32 leftchars;

    if (digit_mode != DigitMode_Unique) {
        DEBUG_ASSERT(precision >= 0);
    }


    DEBUG_ASSERT(bufferSize > 0);

    pCurOut = buffer;

    /* add any whitespace padding to left side */
    leftchars = 1 + (signbit == '-' || signbit == '+');
    if (digits_left > leftchars) {
        int i;
        for (i = 0; i < digits_left - leftchars && bufferSize > 1; i++){
            *pCurOut = ' ';
            pCurOut++;
            --bufferSize;
        }
    }

    if (signbit == '+' && bufferSize > 1) {
        *pCurOut = '+';
        pCurOut++;
        --bufferSize;
    }
    else if (signbit == '-'  && bufferSize > 1) {
        *pCurOut = '-';
        pCurOut++;
        --bufferSize;
    }

    numDigits = Dragon4(mantissa, exponent, mantissaBit, hasUnequalMargins,
                        digit_mode, CutoffMode_TotalLength, precision + 1,
                        pCurOut, bufferSize, &printExponent);

    DEBUG_ASSERT(numDigits > 0);
    DEBUG_ASSERT(numDigits <= bufferSize);

    /* keep the whole number as the first digit */
    if (bufferSize > 1) {
        pCurOut += 1;
        bufferSize -= 1;
    }

    /* insert the decimal point prior to the fractional number */
    numFractionDigits = numDigits-1;
    if (numFractionDigits > 0 && bufferSize > 1) {
        npy_int32 maxFractionDigits = (npy_int32)bufferSize - 2;

        if (numFractionDigits > maxFractionDigits) {
            numFractionDigits =  maxFractionDigits;
        }

        memmove(pCurOut + 1, pCurOut, numFractionDigits);
        pCurOut[0] = '.';
        pCurOut += (1 + numFractionDigits);
        bufferSize -= (1 + numFractionDigits);
    }

    /* always add decimal point, except for DprZeros mode */
    if (trim_mode != TrimMode_DptZeros && numFractionDigits == 0 &&
            bufferSize > 1){
        *pCurOut = '.';
        ++pCurOut;
        --bufferSize;
    }

    if (trim_mode == TrimMode_LeaveOneZero) {
        /* if we didn't print any fractional digits, add the 0 */
        if (numFractionDigits == 0 && bufferSize > 1) {
            *pCurOut = '0';
            ++pCurOut;
            --bufferSize;
            ++numFractionDigits;
        }
    }
    else if (trim_mode == TrimMode_None &&
            digit_mode != DigitMode_Unique) {
        /* add trailing zeros up to precision length */
        if (precision > (npy_int32)numFractionDigits) {
            char *pEnd;
            /* compute the number of trailing zeros needed */
            npy_int32 numZeros = (precision - numFractionDigits);

            if (numZeros > (npy_int32)bufferSize - 1) {
                numZeros = (npy_int32)bufferSize - 1;
            }

            for (pEnd = pCurOut + numZeros; pCurOut < pEnd; ++pCurOut) {
                *pCurOut = '0';
                ++numFractionDigits;
            }
        }
    }
    /* else, for trim_mode Zeros or DptZeros, there is nothing more to add */

    /*
     * when rounding, we may still end up with trailing zeros. Remove them
     * depending on trim settings.
     */
    if (precision >= 0 && trim_mode != TrimMode_None && numFractionDigits > 0){
        --pCurOut;
        while (*pCurOut == '0') {
            --pCurOut;
            ++bufferSize;
            --numFractionDigits;
        }
        if (trim_mode == TrimMode_LeaveOneZero && *pCurOut == '.') {
            ++pCurOut;
            *pCurOut = '0';
            --bufferSize;
            ++numFractionDigits;
        }
        ++pCurOut;
    }

    /* print the exponent into a local buffer and copy into output buffer */
    if (bufferSize > 1) {
        char exponentBuffer[7];
        npy_int32 digits[5];
        npy_int32 i, exp_size, count;

        if (exp_digits > 5) {
            exp_digits = 5;
        }
        if (exp_digits < 0) {
            exp_digits = 2;
        }

        exponentBuffer[0] = 'e';
        if (printExponent >= 0) {
            exponentBuffer[1] = '+';
        }
        else {
            exponentBuffer[1] = '-';
            printExponent = -printExponent;
        }

        DEBUG_ASSERT(printExponent < 100000);

        /* get exp digits */
        for (i = 0; i < 5; i++){
            digits[i] = printExponent % 10;
            printExponent /= 10;
        }
        /* count back over leading zeros */
        for (i = 5; i > exp_digits && digits[i-1] == 0; i--) {
        }
        exp_size = i;
        /* write remaining digits to tmp buf */
        for (i = exp_size; i > 0; i--){
            exponentBuffer[2 + (exp_size-i)] = (char)('0' + digits[i-1]);
        }

        /* copy the exponent buffer into the output */
        count = exp_size + 2;
        if (count > (npy_int32)bufferSize - 1) {
            count = (npy_int32)bufferSize - 1;
        }
        memcpy(pCurOut, exponentBuffer, count);
        pCurOut += count;
        bufferSize -= count;
    }


    DEBUG_ASSERT(bufferSize > 0);
    pCurOut[0] = '\0';

    return pCurOut - buffer;
}

/*
 * Print a hexadecimal value with a given width.
 * The output string is always NUL terminated and the string length (not
 * including the NUL) is returned.
 */
/*  Unused for now
static npy_uint32
PrintHex(char * buffer, npy_uint32 bufferSize, npy_uint64 value,
         npy_uint32 width)
{
    const char digits[] = "0123456789abcdef";
    char *pCurOut;

    DEBUG_ASSERT(bufferSize > 0);

    npy_uint32 maxPrintLen = bufferSize-1;
    if (width > maxPrintLen) {
        width = maxPrintLen;
    }

    pCurOut = buffer;
    while (width > 0) {
        --width;

        npy_uint8 digit = (npy_uint8)((value >> 4ull*(npy_uint64)width) & 0xF);
        *pCurOut = digits[digit];

        ++pCurOut;
    }

    *pCurOut = '\0';
    return pCurOut - buffer;
}
*/

/*
 * Print special case values for infinities and NaNs.
 * The output string is always NUL terminated and the string length (not
 * including the NUL) is returned.
 */
static npy_uint32
PrintInfNan(char *buffer, npy_uint32 bufferSize, npy_uint64 mantissa,
            npy_uint32 mantissaHexWidth, char signbit)
{
    npy_uint32 maxPrintLen = bufferSize-1;
    npy_uint32 pos = 0;

    DEBUG_ASSERT(bufferSize > 0);

    /* Check for infinity */
    if (mantissa == 0) {
        npy_uint32 printLen;

        /* only print sign for inf values (though nan can have a sign set) */
        if (signbit == '+') {
            if (pos < maxPrintLen-1){
                buffer[pos++] = '+';
            }
        }
        else if (signbit == '-') {
            if (pos < maxPrintLen-1){
                buffer[pos++] = '-';
            }
        }

        /* copy and make sure the buffer is terminated */
        printLen = (3 < maxPrintLen - pos) ? 3 : maxPrintLen - pos;
        memcpy(buffer + pos, "inf", printLen);
        buffer[pos + printLen] = '\0';
        return pos + printLen;
    }
    else {
        /* copy and make sure the buffer is terminated */
        npy_uint32 printLen = (3 < maxPrintLen - pos) ? 3 : maxPrintLen - pos;
        memcpy(buffer + pos, "nan", printLen);
        buffer[pos + printLen] = '\0';

        /*
         * // XXX: Should we change this for numpy?
         * // append HEX value
         * if (maxPrintLen > 3) {
         *     printLen += PrintHex(buffer+3, bufferSize-3, mantissa,
         *                          mantissaHexWidth);
         * }
         */

        return pos + printLen;
    }
}

/*
 * These functions print a floating-point number as a decimal string.
 * The output string is always NUL terminated and the string length (not
 * including the NUL) is returned.
 *
 * Arguments are:
 *   buffer - buffer to output into
 *   bufferSize - maximum characters that can be printed to buffer
 *   value - value significand
 *   scientific - boolean controlling whether scientific notation is used
 *   precision - If positive, specifies the number of decimals to show after
 *               decimal point. If negative, sufficient digits to uniquely
 *               specify the float will be output.
 *   trim_mode - how to treat trailing zeros and decimal point. See TrimMode.
 *   digits_right - pad the result with '' on the right past the decimal point
 *   digits_left - pad the result with '' on the right past the decimal point
 *   exp_digits - Only affects scientific output. If positive, pads the
 *                exponent with 0s until there are this many digits. If
 *                negative, only use sufficient digits.
 */
static npy_uint32
Dragon4_PrintFloat16(char *buffer, npy_uint32 bufferSize, npy_uint16 value,
                     npy_bool scientific, DigitMode digit_mode,
                     CutoffMode cutoff_mode, npy_int32 precision,
                     npy_bool sign, TrimMode trim_mode, npy_int32 digits_left,
                     npy_int32 digits_right, npy_int32 exp_digits)
{
    FloatUnion16 floatUnion;
    npy_uint32 floatExponent, floatMantissa;

    npy_uint32 mantissa;
    npy_int32 exponent;
    npy_uint32 mantissaBit;
    npy_bool hasUnequalMargins;
    char signbit = '\0';

    if (bufferSize == 0) {
        return 0;
    }

    if (bufferSize == 1) {
        buffer[0] = '\0';
        return 0;
    }

    /* deconstruct the floating point value */
    floatUnion.integer = value;
    floatExponent = GetExponent_F16(&floatUnion);
    floatMantissa = GetMantissa_F16(&floatUnion);

    /* output the sign */
    if (IsNegative_F16(&floatUnion)) {
        signbit = '-';
    }
    else if (sign) {
        signbit = '+';
    }

    /* if this is a special value */
    if (floatExponent == 0x1F) {
        return PrintInfNan(buffer, bufferSize, floatMantissa, 3, signbit);
    }
    /* else this is a number */

    /* factor the value into its parts */
    if (floatExponent != 0) {
        /*
         * normalized
         * The floating point equation is:
         *  value = (1 + mantissa/2^10) * 2 ^ (exponent-15)
         * We convert the integer equation by factoring a 2^10 out of the
         * exponent
         *  value = (1 + mantissa/2^10) * 2^10 * 2 ^ (exponent-15-10)
         *  value = (2^10 + mantissa) * 2 ^ (exponent-15-10)
         * Because of the implied 1 in front of the mantissa we have 10 bits of
         * precision.
         *   m = (2^10 + mantissa)
         *   e = (exponent-15-10)
         */
        mantissa            = (1UL << 10) | floatMantissa;
        exponent            = floatExponent - 15 - 10;
        mantissaBit         = 10;
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);
    }
    else {
        /*
         * denormalized
         * The floating point equation is:
         *  value = (mantissa/2^10) * 2 ^ (1-15)
         * We convert the integer equation by factoring a 2^23 out of the
         * exponent
         *  value = (mantissa/2^10) * 2^10 * 2 ^ (1-15-10)
         *  value = mantissa * 2 ^ (1-15-10)
         * We have up to 10 bits of precision.
         *   m = (mantissa)
         *   e = (1-15-10)
         */
        mantissa           = floatMantissa;
        exponent           = 1 - 15 - 10;
        mantissaBit        = LogBase2_32(mantissa);
        hasUnequalMargins  = NPY_FALSE;
    }

    /* format the value */
    if (scientific) {
        return FormatScientific(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                precision, trim_mode, digits_left, exp_digits);
    }
    else {
        return FormatPositional(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                cutoff_mode, precision, trim_mode,
                                digits_left, digits_right);
    }
}

static npy_uint32
Dragon4_PrintFloat32(char *buffer, npy_uint32 bufferSize, npy_float32 value,
                     npy_bool scientific, DigitMode digit_mode,
                     CutoffMode cutoff_mode, npy_int32 precision,
                     npy_bool sign, TrimMode trim_mode, npy_int32 digits_left,
                     npy_int32 digits_right, npy_int32 exp_digits)
{
    FloatUnion32 floatUnion;
    npy_uint32 floatExponent, floatMantissa;

    npy_uint32 mantissa;
    npy_int32 exponent;
    npy_uint32 mantissaBit;
    npy_bool hasUnequalMargins;
    char signbit = '\0';

    if (bufferSize == 0) {
        return 0;
    }

    if (bufferSize == 1) {
        buffer[0] = '\0';
        return 0;
    }

    /* deconstruct the floating point value */
    floatUnion.floatingPoint = value;
    floatExponent = GetExponent_F32(&floatUnion);
    floatMantissa = GetMantissa_F32(&floatUnion);

    /* output the sign */
    if (IsNegative_F32(&floatUnion)) {
        signbit = '-';
    }
    else if (sign) {
        signbit = '+';
    }

    /* if this is a special value */
    if (floatExponent == 0xFF) {
        return PrintInfNan(buffer, bufferSize, floatMantissa, 6, signbit);
    }
    /* else this is a number */

    /* factor the value into its parts */
    if (floatExponent != 0) {
        /*
         * normalized
         * The floating point equation is:
         *  value = (1 + mantissa/2^23) * 2 ^ (exponent-127)
         * We convert the integer equation by factoring a 2^23 out of the
         * exponent
         *  value = (1 + mantissa/2^23) * 2^23 * 2 ^ (exponent-127-23)
         *  value = (2^23 + mantissa) * 2 ^ (exponent-127-23)
         * Because of the implied 1 in front of the mantissa we have 24 bits of
         * precision.
         *   m = (2^23 + mantissa)
         *   e = (exponent-127-23)
         */
        mantissa            = (1UL << 23) | floatMantissa;
        exponent            = floatExponent - 127 - 23;
        mantissaBit         = 23;
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);
    }
    else {
        /*
         * denormalized
         * The floating point equation is:
         *  value = (mantissa/2^23) * 2 ^ (1-127)
         * We convert the integer equation by factoring a 2^23 out of the
         * exponent
         *  value = (mantissa/2^23) * 2^23 * 2 ^ (1-127-23)
         *  value = mantissa * 2 ^ (1-127-23)
         * We have up to 23 bits of precision.
         *   m = (mantissa)
         *   e = (1-127-23)
         */
        mantissa           = floatMantissa;
        exponent           = 1 - 127 - 23;
        mantissaBit        = LogBase2_32(mantissa);
        hasUnequalMargins  = NPY_FALSE;
    }

    /* format the value */
    if (scientific) {
        return FormatScientific(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                precision, trim_mode, digits_left, exp_digits);
    }
    else {
        return FormatPositional(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                cutoff_mode, precision, trim_mode,
                                digits_left, digits_right);
    }
}

static npy_uint32
Dragon4_PrintFloat64(char *buffer, npy_uint32 bufferSize, npy_float64 value,
                     npy_bool scientific, DigitMode digit_mode,
                     CutoffMode cutoff_mode, npy_int32 precision,
                     npy_bool sign, TrimMode trim_mode, npy_int32 digits_left,
                     npy_int32 digits_right, npy_int32 exp_digits)
{
    FloatUnion64 floatUnion;
    npy_uint32 floatExponent;
    npy_uint64 floatMantissa;

    npy_uint64 mantissa;
    npy_int32 exponent;
    npy_uint32 mantissaBit;
    npy_bool hasUnequalMargins;
    char signbit = '\0';

    if (bufferSize == 0) {
        return 0;
    }

    if (bufferSize == 1) {
        buffer[0] = '\0';
        return 0;
    }

    /* deconstruct the floating point value */
    floatUnion.floatingPoint = value;
    floatExponent = GetExponent_F64(&floatUnion);
    floatMantissa = GetMantissa_F64(&floatUnion);

    /* output the sign */
    if (IsNegative_F64(&floatUnion)) {
        signbit = '-';
    }
    else if (sign) {
        signbit = '+';
    }

    /* if this is a special value */
    if (floatExponent == 0x7FF) {
        return PrintInfNan(buffer, bufferSize, floatMantissa, 13, signbit);
    }
    /* else this is a number */

    /* factor the value into its parts */
    if (floatExponent != 0) {
        /*
         * normal
         * The floating point equation is:
         *  value = (1 + mantissa/2^52) * 2 ^ (exponent-1023)
         * We convert the integer equation by factoring a 2^52 out of the
         * exponent
         *  value = (1 + mantissa/2^52) * 2^52 * 2 ^ (exponent-1023-52)
         *  value = (2^52 + mantissa) * 2 ^ (exponent-1023-52)
         * Because of the implied 1 in front of the mantissa we have 53 bits of
         * precision.
         *   m = (2^52 + mantissa)
         *   e = (exponent-1023+1-53)
         */
        mantissa            = (1ull << 52) | floatMantissa;
        exponent            = floatExponent - 1023 - 52;
        mantissaBit         = 52;
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);
    }
    else {
        /*
         * subnormal
         * The floating point equation is:
         *  value = (mantissa/2^52) * 2 ^ (1-1023)
         * We convert the integer equation by factoring a 2^52 out of the
         * exponent
         *  value = (mantissa/2^52) * 2^52 * 2 ^ (1-1023-52)
         *  value = mantissa * 2 ^ (1-1023-52)
         * We have up to 52 bits of precision.
         *   m = (mantissa)
         *   e = (1-1023-52)
         */
        mantissa            = floatMantissa;
        exponent            = 1 - 1023 - 52;
        mantissaBit         = LogBase2_64(mantissa);
        hasUnequalMargins   = NPY_FALSE;
    }

    /* format the value */
    if (scientific) {
        return FormatScientific(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                precision, trim_mode, digits_left, exp_digits);
    }
    else {
        return FormatPositional(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                cutoff_mode, precision, trim_mode,
                                digits_left, digits_right);
    }
}

static npy_uint32
Dragon4_PrintFloat128(char *buffer, npy_uint32 bufferSize, FloatVal128 value,
                      npy_bool scientific, DigitMode digit_mode,
                      CutoffMode cutoff_mode, npy_int32 precision,
                      npy_bool sign, TrimMode trim_mode, npy_int32 digits_left,
                      npy_int32 digits_right, npy_int32 exp_digits)
{
    npy_uint32 floatExponent;
    npy_uint64 floatMantissa;

    npy_uint64 mantissa;
    npy_int32 exponent;
    npy_uint32 mantissaBit;
    npy_bool hasUnequalMargins;
    char signbit = '\0';

    if (bufferSize == 0) {
        return 0;
    }

    if (bufferSize == 1) {
        buffer[0] = '\0';
        return 0;
    }

    /* deconstruct the floating point value */
    floatExponent = GetExponent_F128(&value);
    floatMantissa = GetMantissa_F128(&value);

    /* output the sign */
    if (IsNegative_F128(&value)) {
        signbit = '-';
    }
    else if (sign) {
        signbit = '+';
    }

    /* if this is a special value */
    if (floatExponent == 0x7FFF) {
        return PrintInfNan(buffer, bufferSize, floatMantissa, 16, signbit);
    }
    /* else this is a number */

    /* factor the value into its parts */
    if (floatExponent != 0) {
        /*
         * normal
         * The floating point equation is:
         *  value = (1 + mantissa/2^63) * 2 ^ (exponent-16383)
         * We convert the integer equation by factoring a 2^63 out of the
         * exponent
         *  value = (1 + mantissa/2^63) * 2^63 * 2 ^ (exponent-16383-63)
         *  value = (2^63 + mantissa) * 2 ^ (exponent-16383-63)
         * Because of the implied 1 in front of the mantissa we have 64 bits of
         * precision.
         *   m = (2^63 + mantissa)
         *   e = (exponent-16383+1-64)
         */
        mantissa            = (1ull << 63) | floatMantissa;
        exponent            = floatExponent - 16383 - 63;
        mantissaBit         = 63;
        hasUnequalMargins   = (floatExponent != 1) && (floatMantissa == 0);
    }
    else {
        /*
         * subnormal
         * The floating point equation is:
         *  value = (mantissa/2^63) * 2 ^ (1-16383)
         * We convert the integer equation by factoring a 2^52 out of the
         * exponent
         *  value = (mantissa/2^63) * 2^52 * 2 ^ (1-16383-63)
         *  value = mantissa * 2 ^ (1-16383-63)
         * We have up to 63 bits of precision.
         *   m = (mantissa)
         *   e = (1-16383-63)
         */
        mantissa            = floatMantissa;
        exponent            = 1 - 16383 - 63;
        mantissaBit         = LogBase2_64(mantissa);
        hasUnequalMargins   = NPY_FALSE;
    }

    /* format the value */
    if (scientific) {
        return FormatScientific(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                precision, trim_mode, digits_left, exp_digits);
    }
    else {
        return FormatPositional(buffer, bufferSize, mantissa, exponent, signbit,
                                mantissaBit, hasUnequalMargins, digit_mode,
                                cutoff_mode, precision, trim_mode,
                                digits_left, digits_right);
    }
}

PyObject *
Dragon4_Positional_AnySize(void *val, size_t size, DigitMode digit_mode,
                           CutoffMode cutoff_mode, int precision, int sign,
                           TrimMode trim, int pad_left, int pad_right)
{
    /*
     * Use a very large buffer in case anyone tries to output a large numberG.
     * 16384 should be enough to uniquely print any float128, which goes up
     * to about 10^4932 */
    static char repr[16384];
    FloatVal128 val128;
#ifdef NPY_FLOAT80
    FloatUnion80 buf80;;
#endif
#ifdef NPY_FLOAT96
    FloatUnion96 buf96;
#endif
#ifdef NPY_FLOAT128
    FloatUnion128 buf128;
#endif

    switch (size) {
        case 2:
            Dragon4_PrintFloat16(repr, sizeof(repr), *(npy_float16*)val,
                                 0, digit_mode, cutoff_mode, precision,
                                 sign, trim, pad_left, pad_right, -1);
            break;
        case 4:
            Dragon4_PrintFloat32(repr, sizeof(repr), *(npy_float32*)val,
                                 0, digit_mode, cutoff_mode, precision,
                                 sign, trim, pad_left, pad_right, -1);
            break;
        case 8:
            Dragon4_PrintFloat64(repr, sizeof(repr), *(npy_float64*)val,
                                 0, digit_mode, cutoff_mode, precision,
                                 sign, trim, pad_left, pad_right, -1);
            break;
#ifdef NPY_FLOAT80
        case 10:
            buf80.floatingPoint = *(npy_float80*)val;
            val128.integer[0] = buf80.integer.a;
            val128.integer[1] = buf80.integer.b;
            Dragon4_PrintFloat128(repr, sizeof(repr), val128,
                                  0, digit_mode, cutoff_mode, precision,
                                  sign, trim, pad_left, pad_right, -1);
            break;
#endif
#ifdef NPY_FLOAT96
        case 12:
            buf96.floatingPoint = *(npy_float96*)val;
            val128.integer[0] = buf96.integer.a;
            val128.integer[1] = buf96.integer.b;
            Dragon4_PrintFloat128(repr, sizeof(repr), val128,
                                  0, digit_mode, cutoff_mode, precision,
                                  sign, trim, pad_left, pad_right, -1);
            break;
#endif
#ifdef NPY_FLOAT128
        case 16:
            buf128.floatingPoint = *(npy_float128*)val;
            val128.integer[0] = buf128.integer.a;
            val128.integer[1] = buf128.integer.b;
            Dragon4_PrintFloat128(repr, sizeof(repr), val128,
                                  0, digit_mode, cutoff_mode, precision,
                                  sign, trim, pad_left, pad_right, -1);
            break;
#endif
        default:
            PyErr_Format(PyExc_ValueError, "unexpected itemsize %zu", size);
            return NULL;
    }

    return PyUString_FromString(repr);
}

PyObject *
Dragon4_Positional(PyObject *obj, DigitMode digit_mode, CutoffMode cutoff_mode,
                   int precision, int sign, TrimMode trim, int pad_left,
                   int pad_right)
{
    double val;

    if (PyArray_IsScalar(obj, Half)) {
        npy_half x = ((PyHalfScalarObject *)obj)->obval;
        return Dragon4_Positional_AnySize(&x, sizeof(npy_half),
                                          digit_mode, cutoff_mode, precision,
                                          sign, trim, pad_left, pad_right);
    }
    else if (PyArray_IsScalar(obj, Float)) {
        npy_float x = ((PyFloatScalarObject *)obj)->obval;
        return Dragon4_Positional_AnySize(&x, sizeof(npy_float),
                                          digit_mode, cutoff_mode, precision,
                                          sign, trim, pad_left, pad_right);
    }
    else if (PyArray_IsScalar(obj, Double)) {
        npy_double x = ((PyDoubleScalarObject *)obj)->obval;
        return Dragon4_Positional_AnySize(&x, sizeof(npy_double),
                                          digit_mode, cutoff_mode, precision,
                                          sign, trim, pad_left, pad_right);
    }
    else if (PyArray_IsScalar(obj, LongDouble)) {
        npy_longdouble x = ((PyLongDoubleScalarObject *)obj)->obval;
        return Dragon4_Positional_AnySize(&x, sizeof(npy_longdouble),
                                          digit_mode, cutoff_mode, precision,
                                          sign, trim, pad_left, pad_right);
    }

    val = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Dragon4_Positional_AnySize(&val, sizeof(double),
                                      digit_mode, cutoff_mode, precision,
                                      sign, trim, pad_left, pad_right);
}

PyObject *
Dragon4_Scientific_AnySize(void *val, size_t size, DigitMode digit_mode,
                           int precision, int sign, TrimMode trim,
                           int pad_left, int exp_digits)
{
    /* use a very large buffer in case anyone tries to output a large precision */
    static char repr[4096];
    FloatVal128 val128;
#ifdef NPY_FLOAT80
    FloatUnion80 buf80;;
#endif
#ifdef NPY_FLOAT96
    FloatUnion96 buf96;
#endif
#ifdef NPY_FLOAT128
    FloatUnion128 buf128;
#endif

    /* dummy, is ignored in scientific mode */
    CutoffMode cutoff_mode = CutoffMode_TotalLength;

    switch (size) {
        case 2:
            Dragon4_PrintFloat16(repr, sizeof(repr), *(npy_float16*)val,
                                 1, digit_mode, cutoff_mode, precision, sign,
                                 trim, pad_left, -1, exp_digits);
            break;
        case 4:
            Dragon4_PrintFloat32(repr, sizeof(repr), *(npy_float32*)val,
                                 1, digit_mode, cutoff_mode, precision, sign,
                                 trim, pad_left, -1, exp_digits);
            break;
        case 8:
            Dragon4_PrintFloat64(repr, sizeof(repr), *(npy_float64*)val,
                                 1, digit_mode, cutoff_mode, precision, sign,
                                 trim, pad_left, -1, exp_digits);
            break;
#ifdef NPY_FLOAT80
        case 10:
            buf80.floatingPoint = *(npy_float80*)val;
            val128.integer[0] = buf80.integer.a;
            val128.integer[1] = buf80.integer.b;
            Dragon4_PrintFloat128(repr, sizeof(repr), val128,
                                  1, digit_mode, cutoff_mode, precision, sign,
                                  trim, pad_left, -1, exp_digits);
            break;
#endif
#ifdef NPY_FLOAT96
        case 12:
            buf96.floatingPoint = *(npy_float96*)val;
            val128.integer[0] = buf96.integer.a;
            val128.integer[1] = buf96.integer.b;
            Dragon4_PrintFloat128(repr, sizeof(repr), val128,
                                  1, digit_mode, cutoff_mode, precision, sign,
                                  trim, pad_left, -1, exp_digits);
            break;
#endif
#ifdef NPY_FLOAT128
        case 16:
            buf128.floatingPoint = *(npy_float128*)val;
            val128.integer[0] = buf128.integer.a;
            val128.integer[1] = buf128.integer.b;
            Dragon4_PrintFloat128(repr, sizeof(repr), val128,
                                  1, digit_mode, cutoff_mode, precision, sign,
                                  trim, pad_left, -1, exp_digits);
            break;
#endif
        default:
            PyErr_Format(PyExc_ValueError, "unexpected itemsize %zu", size);
            return NULL;
    }

    return PyUString_FromString(repr);
}

PyObject *
Dragon4_Scientific(PyObject *obj, DigitMode digit_mode, int precision,
                   int sign, TrimMode trim, int pad_left, int exp_digits)
{
    double val;

    if (PyArray_IsScalar(obj, Half)) {
        npy_half x = ((PyHalfScalarObject *)obj)->obval;
        return Dragon4_Scientific_AnySize(&x, sizeof(npy_half),
                                          digit_mode, precision,
                                          sign, trim, pad_left, exp_digits);
    }
    else if (PyArray_IsScalar(obj, Float)) {
        npy_float x = ((PyFloatScalarObject *)obj)->obval;
        return Dragon4_Scientific_AnySize(&x, sizeof(npy_float),
                                          digit_mode, precision,
                                          sign, trim, pad_left, exp_digits);
    }
    else if (PyArray_IsScalar(obj, Double)) {
        npy_double x = ((PyDoubleScalarObject *)obj)->obval;
        return Dragon4_Scientific_AnySize(&x, sizeof(npy_double),
                                          digit_mode, precision,
                                          sign, trim, pad_left, exp_digits);
    }
    else if (PyArray_IsScalar(obj, LongDouble)) {
        npy_longdouble x = ((PyLongDoubleScalarObject *)obj)->obval;
        return Dragon4_Scientific_AnySize(&x, sizeof(npy_longdouble),
                                          digit_mode, precision,
                                          sign, trim, pad_left, exp_digits);
    }

    val = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Dragon4_Scientific_AnySize(&val, sizeof(double),
                                      digit_mode, precision,
                                      sign, trim, pad_left, exp_digits);
}
