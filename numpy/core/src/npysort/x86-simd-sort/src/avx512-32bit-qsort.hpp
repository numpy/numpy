/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * Copyright (C) 2021 Serge Sans Paille
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 *          Serge Sans Paille <serge.guelton@telecom-bretagne.eu>
 * ****************************************************************/
#ifndef __AVX512_QSORT_32BIT__
#define __AVX512_QSORT_32BIT__

#include "avx512-common-qsort.h"

/*
 * Constants used in sorting 16 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
#define NETWORK_32BIT_1 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1
#define NETWORK_32BIT_2 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
#define NETWORK_32BIT_3 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_32BIT_4 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2
#define NETWORK_32BIT_5 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define NETWORK_32BIT_6 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4
#define NETWORK_32BIT_7 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8

template <>
struct vector<int32_t> {
    using type_t = int32_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT32;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT32;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi32(type_max());
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask16(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_NLT);
    }
    template <int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static zmm_t merge(ymm_t y1, ymm_t y2)
    {
        zmm_t z1 = _mm512_castsi256_si512(y1);
        return _mm512_inserti32x8(z1, y2, 1);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epi32(x, y);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epi32(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epi32(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epi32(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi32(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_epi32(x, y);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_epi32(x, y);
    }
};
template <>
struct vector<uint32_t> {
    using type_t = uint32_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT32;
    }
    static type_t type_min()
    {
        return 0;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi32(type_max());
    } // TODO: this should broadcast bits as is?

    template <int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static zmm_t merge(ymm_t y1, ymm_t y2)
    {
        zmm_t z1 = _mm512_castsi256_si512(y1);
        return _mm512_inserti32x8(z1, y2, 1);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask16(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu32_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epu32(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epu32(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epu32(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epu32(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi32(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_epu32(x, y);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_epu32(x, y);
    }
};
template <>
struct vector<float> {
    using type_t = float;
    using zmm_t = __m512;
    using ymm_t = __m256;
    using opmask_t = __mmask16;
    static const uint8_t numlanes = 16;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITYF;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITYF;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_ps(type_max());
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask16(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
    }
    template <int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_ps(index, base, scale);
    }
    static zmm_t merge(ymm_t y1, ymm_t y2)
    {
        zmm_t z1 = _mm512_castsi512_ps(
                _mm512_castsi256_si512(_mm256_castps_si256(y1)));
        return _mm512_insertf32x8(z1, y2, 1);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_ps(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_ps(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_ps(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_ps(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_ps(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_ps(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_ps(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_ps(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_ps(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_ps(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_ps(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_ps(zmm, zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_ps(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y)
    {
        return _mm256_max_ps(x, y);
    }
    static ymm_t min(ymm_t x, ymm_t y)
    {
        return _mm256_min_ps(x, y);
    }
};

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
NPY_FINLINE zmm_t sort_zmm_32bit(zmm_t zmm)
{
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(0, 1, 2, 3)>(zmm),
            0xCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_3), zmm),
            0xF0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm),
            0xFF00);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_6), zmm),
            0xF0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
NPY_FINLINE zmm_t bitonic_merge_zmm_32bit(zmm_t zmm)
{
    // 1) half_cleaner[16]: compare 1-9, 2-10, 3-11 etc ..
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_7), zmm),
            0xFF00);
    // 2) half_cleaner[8]: compare 1-5, 2-6, 3-7 etc ..
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_6), zmm),
            0xF0F0);
    // 3) half_cleaner[4]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCC);
    // 3) half_cleaner[1]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAA);
    return zmm;
}

// Assumes zmm1 and zmm2 are sorted and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
NPY_FINLINE void bitonic_merge_two_zmm_32bit(zmm_t *zmm1, zmm_t *zmm2)
{
    // 1) First step of a merging network: coex of zmm1 and zmm2 reversed
    *zmm2 = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), *zmm2);
    zmm_t zmm3 = vtype::min(*zmm1, *zmm2);
    zmm_t zmm4 = vtype::max(*zmm1, *zmm2);
    // 2) Recursive half cleaner for each
    *zmm1 = bitonic_merge_zmm_32bit<vtype>(zmm3);
    *zmm2 = bitonic_merge_zmm_32bit<vtype>(zmm4);
}

// Assumes [zmm0, zmm1] and [zmm2, zmm3] are sorted and performs a recursive
// half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
NPY_FINLINE void bitonic_merge_four_zmm_32bit(zmm_t *zmm)
{
    zmm_t zmm2r = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm[2]);
    zmm_t zmm3r = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm[3]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm3r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm2r);
    zmm_t zmm_t3 = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5),
                                      vtype::max(zmm[1], zmm2r));
    zmm_t zmm_t4 = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5),
                                      vtype::max(zmm[0], zmm3r));
    zmm_t zmm0 = vtype::min(zmm_t1, zmm_t2);
    zmm_t zmm1 = vtype::max(zmm_t1, zmm_t2);
    zmm_t zmm2 = vtype::min(zmm_t3, zmm_t4);
    zmm_t zmm3 = vtype::max(zmm_t3, zmm_t4);
    zmm[0] = bitonic_merge_zmm_32bit<vtype>(zmm0);
    zmm[1] = bitonic_merge_zmm_32bit<vtype>(zmm1);
    zmm[2] = bitonic_merge_zmm_32bit<vtype>(zmm2);
    zmm[3] = bitonic_merge_zmm_32bit<vtype>(zmm3);
}

template <typename vtype, typename zmm_t = typename vtype::zmm_t>
NPY_FINLINE void bitonic_merge_eight_zmm_32bit(zmm_t *zmm)
{
    zmm_t zmm4r = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm[4]);
    zmm_t zmm5r = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm[5]);
    zmm_t zmm6r = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm[6]);
    zmm_t zmm7r = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5), zmm[7]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm7r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm6r);
    zmm_t zmm_t3 = vtype::min(zmm[2], zmm5r);
    zmm_t zmm_t4 = vtype::min(zmm[3], zmm4r);
    zmm_t zmm_t5 = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5),
                                      vtype::max(zmm[3], zmm4r));
    zmm_t zmm_t6 = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5),
                                      vtype::max(zmm[2], zmm5r));
    zmm_t zmm_t7 = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5),
                                      vtype::max(zmm[1], zmm6r));
    zmm_t zmm_t8 = vtype::permutexvar(_mm512_set_epi32(NETWORK_32BIT_5),
                                      vtype::max(zmm[0], zmm7r));
    COEX<vtype>(zmm_t1, zmm_t3);
    COEX<vtype>(zmm_t2, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t7);
    COEX<vtype>(zmm_t6, zmm_t8);
    COEX<vtype>(zmm_t1, zmm_t2);
    COEX<vtype>(zmm_t3, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t6);
    COEX<vtype>(zmm_t7, zmm_t8);
    zmm[0] = bitonic_merge_zmm_32bit<vtype>(zmm_t1);
    zmm[1] = bitonic_merge_zmm_32bit<vtype>(zmm_t2);
    zmm[2] = bitonic_merge_zmm_32bit<vtype>(zmm_t3);
    zmm[3] = bitonic_merge_zmm_32bit<vtype>(zmm_t4);
    zmm[4] = bitonic_merge_zmm_32bit<vtype>(zmm_t5);
    zmm[5] = bitonic_merge_zmm_32bit<vtype>(zmm_t6);
    zmm[6] = bitonic_merge_zmm_32bit<vtype>(zmm_t7);
    zmm[7] = bitonic_merge_zmm_32bit<vtype>(zmm_t8);
}

template <typename vtype, typename type_t>
NPY_FINLINE void sort_16_32bit(type_t *arr, int32_t N)
{
    typename vtype::opmask_t load_mask = (0x0001 << N) - 0x0001;
    typename vtype::zmm_t zmm
            = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr);
    vtype::mask_storeu(arr, load_mask, sort_zmm_32bit<vtype>(zmm));
}

template <typename vtype, typename type_t>
NPY_FINLINE void sort_32_32bit(type_t *arr, int32_t N)
{
    if (N <= 16) {
        sort_16_32bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    zmm_t zmm1 = vtype::loadu(arr);
    typename vtype::opmask_t load_mask = (0x0001 << (N - 16)) - 0x0001;
    zmm_t zmm2 = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr + 16);
    zmm1 = sort_zmm_32bit<vtype>(zmm1);
    zmm2 = sort_zmm_32bit<vtype>(zmm2);
    bitonic_merge_two_zmm_32bit<vtype>(&zmm1, &zmm2);
    vtype::storeu(arr, zmm1);
    vtype::mask_storeu(arr + 16, load_mask, zmm2);
}

template <typename vtype, typename type_t>
NPY_FINLINE void sort_64_32bit(type_t *arr, int32_t N)
{
    if (N <= 32) {
        sort_32_32bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    using opmask_t = typename vtype::opmask_t;
    zmm_t zmm[4];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 16);
    opmask_t load_mask1 = 0xFFFF, load_mask2 = 0xFFFF;
    uint64_t combined_mask = (0x1ull << (N - 32)) - 0x1ull;
    load_mask1 &= combined_mask & 0xFFFF;
    load_mask2 &= (combined_mask >> 16) & 0xFFFF;
    zmm[2] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 32);
    zmm[3] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 48);
    zmm[0] = sort_zmm_32bit<vtype>(zmm[0]);
    zmm[1] = sort_zmm_32bit<vtype>(zmm[1]);
    zmm[2] = sort_zmm_32bit<vtype>(zmm[2]);
    zmm[3] = sort_zmm_32bit<vtype>(zmm[3]);
    bitonic_merge_two_zmm_32bit<vtype>(&zmm[0], &zmm[1]);
    bitonic_merge_two_zmm_32bit<vtype>(&zmm[2], &zmm[3]);
    bitonic_merge_four_zmm_32bit<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 16, zmm[1]);
    vtype::mask_storeu(arr + 32, load_mask1, zmm[2]);
    vtype::mask_storeu(arr + 48, load_mask2, zmm[3]);
}

template <typename vtype, typename type_t>
NPY_FINLINE void sort_128_32bit(type_t *arr, int32_t N)
{
    if (N <= 64) {
        sort_64_32bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    using opmask_t = typename vtype::opmask_t;
    zmm_t zmm[8];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 16);
    zmm[2] = vtype::loadu(arr + 32);
    zmm[3] = vtype::loadu(arr + 48);
    zmm[0] = sort_zmm_32bit<vtype>(zmm[0]);
    zmm[1] = sort_zmm_32bit<vtype>(zmm[1]);
    zmm[2] = sort_zmm_32bit<vtype>(zmm[2]);
    zmm[3] = sort_zmm_32bit<vtype>(zmm[3]);
    opmask_t load_mask1 = 0xFFFF, load_mask2 = 0xFFFF;
    opmask_t load_mask3 = 0xFFFF, load_mask4 = 0xFFFF;
    if (N != 128) {
        uint64_t combined_mask = (0x1ull << (N - 64)) - 0x1ull;
        load_mask1 &= combined_mask & 0xFFFF;
        load_mask2 &= (combined_mask >> 16) & 0xFFFF;
        load_mask3 &= (combined_mask >> 32) & 0xFFFF;
        load_mask4 &= (combined_mask >> 48) & 0xFFFF;
    }
    zmm[4] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 64);
    zmm[5] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 80);
    zmm[6] = vtype::mask_loadu(vtype::zmm_max(), load_mask3, arr + 96);
    zmm[7] = vtype::mask_loadu(vtype::zmm_max(), load_mask4, arr + 112);
    zmm[4] = sort_zmm_32bit<vtype>(zmm[4]);
    zmm[5] = sort_zmm_32bit<vtype>(zmm[5]);
    zmm[6] = sort_zmm_32bit<vtype>(zmm[6]);
    zmm[7] = sort_zmm_32bit<vtype>(zmm[7]);
    bitonic_merge_two_zmm_32bit<vtype>(&zmm[0], &zmm[1]);
    bitonic_merge_two_zmm_32bit<vtype>(&zmm[2], &zmm[3]);
    bitonic_merge_two_zmm_32bit<vtype>(&zmm[4], &zmm[5]);
    bitonic_merge_two_zmm_32bit<vtype>(&zmm[6], &zmm[7]);
    bitonic_merge_four_zmm_32bit<vtype>(zmm);
    bitonic_merge_four_zmm_32bit<vtype>(zmm + 4);
    bitonic_merge_eight_zmm_32bit<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 16, zmm[1]);
    vtype::storeu(arr + 32, zmm[2]);
    vtype::storeu(arr + 48, zmm[3]);
    vtype::mask_storeu(arr + 64, load_mask1, zmm[4]);
    vtype::mask_storeu(arr + 80, load_mask2, zmm[5]);
    vtype::mask_storeu(arr + 96, load_mask3, zmm[6]);
    vtype::mask_storeu(arr + 112, load_mask4, zmm[7]);
}

template <typename vtype, typename type_t>
NPY_FINLINE type_t
get_pivot_32bit(type_t *arr, const int64_t left, const int64_t right)
{
    // median of 16
    int64_t size = (right - left) / 16;
    using zmm_t = typename vtype::zmm_t;
    using ymm_t = typename vtype::ymm_t;
    __m512i rand_index1 = _mm512_set_epi64(left + size,
                                           left + 2 * size,
                                           left + 3 * size,
                                           left + 4 * size,
                                           left + 5 * size,
                                           left + 6 * size,
                                           left + 7 * size,
                                           left + 8 * size);
    __m512i rand_index2 = _mm512_set_epi64(left + 9 * size,
                                           left + 10 * size,
                                           left + 11 * size,
                                           left + 12 * size,
                                           left + 13 * size,
                                           left + 14 * size,
                                           left + 15 * size,
                                           left + 16 * size);
    ymm_t rand_vec1
            = vtype::template i64gather<sizeof(type_t)>(rand_index1, arr);
    ymm_t rand_vec2
            = vtype::template i64gather<sizeof(type_t)>(rand_index2, arr);
    zmm_t rand_vec = vtype::merge(rand_vec1, rand_vec2);
    zmm_t sort = sort_zmm_32bit<vtype>(rand_vec);
    // pivot will never be a nan, since there are no nan's!
    return ((type_t *)&sort)[8];
}

template <typename vtype, typename type_t>
static void
qsort_32bit_(type_t *arr, int64_t left, int64_t right, int64_t max_iters)
{
    /*
     * Resort to std::sort if quicksort isnt making any progress
     */
    if (max_iters <= 0) {
        std::sort(arr + left, arr + right + 1);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if (right + 1 - left <= 128) {
        sort_128_32bit<vtype>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_32bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512<vtype>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        qsort_32bit_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qsort_32bit_<vtype>(arr, pivot_index, right, max_iters - 1);
}

NPY_FINLINE int64_t replace_nan_with_inf(float *arr, int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask16 loadmask = 0xFFFF;
    while (arrsize > 0) {
        if (arrsize < 16) { loadmask = (0x0001 << arrsize) - 0x0001; }
        __m512 in_zmm = _mm512_maskz_loadu_ps(loadmask, arr);
        __mmask16 nanmask = _mm512_cmp_ps_mask(in_zmm, in_zmm, _CMP_NEQ_UQ);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        _mm512_mask_storeu_ps(arr, nanmask, ZMM_MAX_FLOAT);
        arr += 16;
        arrsize -= 16;
    }
    return nan_count;
}

NPY_FINLINE void
replace_inf_with_nan(float *arr, int64_t arrsize, int64_t nan_count)
{
    for (int64_t ii = arrsize - 1; nan_count > 0; --ii) {
        arr[ii] = std::nanf("1");
        nan_count -= 1;
    }
}

template <>
void avx512_qsort<int32_t>(int32_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_32bit_<vector<int32_t>, int32_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx512_qsort<uint32_t>(uint32_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_32bit_<vector<uint32_t>, uint32_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx512_qsort<float>(float *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_32bit_<vector<float>, float>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}

#endif //__AVX512_QSORT_32BIT__
