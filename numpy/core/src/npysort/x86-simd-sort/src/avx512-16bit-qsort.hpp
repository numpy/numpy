/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef __AVX512_QSORT_16BIT__
#define __AVX512_QSORT_16BIT__

#include "avx512-common-qsort.h"

/*
 * Constants used in sorting 32 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ZMM register: 31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
#define NETWORK_16BIT_1 \
    24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, \
            11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_16BIT_2 \
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, \
            3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define NETWORK_16BIT_3 \
    27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, 11, 10, 9, \
            8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4
#define NETWORK_16BIT_4 \
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
#define NETWORK_16BIT_5 \
    23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24, 7, 6, 5, \
            4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
#define NETWORK_16BIT_6 \
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 31, 30, 29, 28, 27, \
            26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16

template <>
struct vector<int16_t> {
    using type_t = int16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT16;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT16;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi16_mask(x, y, _MM_CMPINT_NLT);
    }
    //template <int scale>
    //static zmm_t i64gather(__m512i index, void const *base)
    //{
    //    return _mm512_i64gather_epi64(index, base, scale);
    //}
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epi16(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        // AVX512_VBMI2
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        // AVX512BW
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epi16(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_max = (type_t)_mm512_reduce_max_epi32(lo);
        type_t hi_max = (type_t)_mm512_reduce_max_epi32(hi);
        return std::max(lo_max, hi_max);
    }
    static type_t reducemin(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_min = (type_t)_mm512_reduce_min_epi32(lo);
        type_t hi_min = (type_t)_mm512_reduce_min_epi32(hi);
        return std::min(lo_min, hi_min);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};
template <>
struct vector<uint16_t> {
    using type_t = uint16_t;
    using zmm_t = __m512i;
    using ymm_t = __m256i;
    using opmask_t = __mmask32;
    static const uint8_t numlanes = 32;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT16;
    }
    static type_t type_min()
    {
        return 0;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi16(type_max());
    } // TODO: this should broadcast bits as is?

    //template<int scale>
    //static zmm_t i64gather(__m512i index, void const *base)
    //{
    //    return _mm512_i64gather_epi64(index, base, scale);
    //}
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask32(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu16_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epu16(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi16(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi16(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi16(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi16(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epu16(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi16(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_max = (type_t)_mm512_reduce_max_epi32(lo);
        type_t hi_max = (type_t)_mm512_reduce_max_epi32(hi);
        return std::max(lo_max, hi_max);
    }
    static type_t reducemin(zmm_t v)
    {
        zmm_t lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 0));
        zmm_t hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1));
        type_t lo_min = (type_t)_mm512_reduce_min_epi32(lo);
        type_t hi_min = (type_t)_mm512_reduce_min_epi32(hi);
        return std::min(lo_min, hi_min);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi16(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        zmm = _mm512_shufflehi_epi16(zmm, (_MM_PERM_ENUM)mask);
        return _mm512_shufflelo_epi16(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline zmm_t sort_zmm_16bit(zmm_t zmm)
{
    // Level 1
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 2
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(0, 1, 2, 3)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 3
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_1), zmm),
            0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 4
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_2), zmm),
            0xFF00FF00);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_3), zmm),
            0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    // Level 5
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_4), zmm),
            0xFFFF0000);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_5), zmm),
            0xFF00FF00);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_3), zmm),
            0xF0F0F0F0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline zmm_t bitonic_merge_zmm_16bit(zmm_t zmm)
{
    // 1) half_cleaner[32]: compare 1-17, 2-18, 3-19 etc ..
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_6), zmm),
            0xFFFF0000);
    // 2) half_cleaner[16]: compare 1-9, 2-10, 3-11 etc ..
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_5), zmm),
            0xFF00FF00);
    // 3) half_cleaner[8]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_3), zmm),
            0xF0F0F0F0);
    // 3) half_cleaner[4]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
            0xCCCCCCCC);
    // 3) half_cleaner[2]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
            0xAAAAAAAA);
    return zmm;
}

// Assumes zmm1 and zmm2 are sorted and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void bitonic_merge_two_zmm_16bit(zmm_t &zmm1, zmm_t &zmm2)
{
    // 1) First step of a merging network: coex of zmm1 and zmm2 reversed
    zmm2 = vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_4), zmm2);
    zmm_t zmm3 = vtype::min(zmm1, zmm2);
    zmm_t zmm4 = vtype::max(zmm1, zmm2);
    // 2) Recursive half cleaner for each
    zmm1 = bitonic_merge_zmm_16bit<vtype>(zmm3);
    zmm2 = bitonic_merge_zmm_16bit<vtype>(zmm4);
}

// Assumes [zmm0, zmm1] and [zmm2, zmm3] are sorted and performs a recursive
// half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void bitonic_merge_four_zmm_16bit(zmm_t *zmm)
{
    zmm_t zmm2r = vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_4), zmm[2]);
    zmm_t zmm3r = vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_4), zmm[3]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm3r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm2r);
    zmm_t zmm_t3 = vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_4),
                                      vtype::max(zmm[1], zmm2r));
    zmm_t zmm_t4 = vtype::permutexvar(_mm512_set_epi16(NETWORK_16BIT_4),
                                      vtype::max(zmm[0], zmm3r));
    zmm_t zmm0 = vtype::min(zmm_t1, zmm_t2);
    zmm_t zmm1 = vtype::max(zmm_t1, zmm_t2);
    zmm_t zmm2 = vtype::min(zmm_t3, zmm_t4);
    zmm_t zmm3 = vtype::max(zmm_t3, zmm_t4);
    zmm[0] = bitonic_merge_zmm_16bit<vtype>(zmm0);
    zmm[1] = bitonic_merge_zmm_16bit<vtype>(zmm1);
    zmm[2] = bitonic_merge_zmm_16bit<vtype>(zmm2);
    zmm[3] = bitonic_merge_zmm_16bit<vtype>(zmm3);
}

template <typename vtype, typename type_t>
static inline void sort_32_16bit(type_t *arr, int32_t N)
{
    typename vtype::opmask_t load_mask = ((0x1ull << N) - 0x1ull) & 0xFFFFFFFF;
    typename vtype::zmm_t zmm
            = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr);
    vtype::mask_storeu(arr, load_mask, sort_zmm_16bit<vtype>(zmm));
}

template <typename vtype, typename type_t>
static inline void sort_64_16bit(type_t *arr, int32_t N)
{
    if (N <= 32) {
        sort_32_16bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    typename vtype::opmask_t load_mask
            = ((0x1ull << (N - 32)) - 0x1ull) & 0xFFFFFFFF;
    zmm_t zmm1 = vtype::loadu(arr);
    zmm_t zmm2 = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr + 32);
    zmm1 = sort_zmm_16bit<vtype>(zmm1);
    zmm2 = sort_zmm_16bit<vtype>(zmm2);
    bitonic_merge_two_zmm_16bit<vtype>(zmm1, zmm2);
    vtype::storeu(arr, zmm1);
    vtype::mask_storeu(arr + 32, load_mask, zmm2);
}

template <typename vtype, typename type_t>
static inline void sort_128_16bit(type_t *arr, int32_t N)
{
    if (N <= 64) {
        sort_64_16bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    using opmask_t = typename vtype::opmask_t;
    zmm_t zmm[4];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 32);
    opmask_t load_mask1 = 0xFFFFFFFF, load_mask2 = 0xFFFFFFFF;
    if (N != 128) {
        uint64_t combined_mask = (0x1ull << (N - 64)) - 0x1ull;
        load_mask1 = combined_mask & 0xFFFFFFFF;
        load_mask2 = (combined_mask >> 32) & 0xFFFFFFFF;
    }
    zmm[2] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 64);
    zmm[3] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 96);
    zmm[0] = sort_zmm_16bit<vtype>(zmm[0]);
    zmm[1] = sort_zmm_16bit<vtype>(zmm[1]);
    zmm[2] = sort_zmm_16bit<vtype>(zmm[2]);
    zmm[3] = sort_zmm_16bit<vtype>(zmm[3]);
    bitonic_merge_two_zmm_16bit<vtype>(zmm[0], zmm[1]);
    bitonic_merge_two_zmm_16bit<vtype>(zmm[2], zmm[3]);
    bitonic_merge_four_zmm_16bit<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 32, zmm[1]);
    vtype::mask_storeu(arr + 64, load_mask1, zmm[2]);
    vtype::mask_storeu(arr + 96, load_mask2, zmm[3]);
}

template <typename vtype, typename type_t>
static inline type_t
get_pivot_16bit(type_t *arr, const int64_t left, const int64_t right)
{
    // median of 32
    int64_t size = (right - left) / 32;
    __m512i rand_vec = _mm512_set_epi16(arr[left],
                                        arr[left + size],
                                        arr[left + 2 * size],
                                        arr[left + 3 * size],
                                        arr[left + 4 * size],
                                        arr[left + 5 * size],
                                        arr[left + 6 * size],
                                        arr[left + 7 * size],
                                        arr[left + 8 * size],
                                        arr[left + 9 * size],
                                        arr[left + 10 * size],
                                        arr[left + 11 * size],
                                        arr[left + 12 * size],
                                        arr[left + 13 * size],
                                        arr[left + 14 * size],
                                        arr[left + 15 * size],
                                        arr[left + 16 * size],
                                        arr[left + 17 * size],
                                        arr[left + 18 * size],
                                        arr[left + 19 * size],
                                        arr[left + 20 * size],
                                        arr[left + 21 * size],
                                        arr[left + 22 * size],
                                        arr[left + 23 * size],
                                        arr[left + 24 * size],
                                        arr[left + 25 * size],
                                        arr[left + 26 * size],
                                        arr[left + 27 * size],
                                        arr[left + 28 * size],
                                        arr[left + 29 * size],
                                        arr[left + 30 * size],
                                        arr[left + 31 * size]);
    __m512i sort = sort_zmm_16bit<vtype>(rand_vec);
    return ((type_t *)&sort)[16];
}

template <typename vtype, typename type_t>
static inline void
qsort_16bit_(type_t *arr, int64_t left, int64_t right, int64_t max_iters)
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
        sort_128_16bit<vtype>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_16bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512<vtype>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        qsort_16bit_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qsort_16bit_<vtype>(arr, pivot_index, right, max_iters - 1);
}

template <>
void avx512_qsort(int16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_16bit_<vector<int16_t>, int16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx512_qsort(uint16_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_16bit_<vector<uint16_t>, uint16_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}
#endif // __AVX512_QSORT_16BIT__
