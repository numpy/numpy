/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 * ****************************************************************/

#ifndef __AVX512_QSORT_64BIT__
#define __AVX512_QSORT_64BIT__

#include "avx512-common-qsort.h"

/*
 * Constants used in sorting 8 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
// ZMM                  7, 6, 5, 4, 3, 2, 1, 0
#define NETWORK_64BIT_1 4, 5, 6, 7, 0, 1, 2, 3
#define NETWORK_64BIT_2 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK_64BIT_3 5, 4, 7, 6, 1, 0, 3, 2
#define NETWORK_64BIT_4 3, 2, 1, 0, 7, 6, 5, 4
static const __m512i rev_index = _mm512_set_epi64(NETWORK_64BIT_2);

template <>
struct vector<int64_t> {
    using type_t = int64_t;
    using zmm_t = __m512i;
    using ymm_t = __m512i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_INT64;
    }
    static type_t type_min()
    {
        return X86_SIMD_SORT_MIN_INT64;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi64(type_max());
    } // TODO: this should broadcast bits as is?

    static zmm_t set(type_t v1,
                     type_t v2,
                     type_t v3,
                     type_t v4,
                     type_t v5,
                     type_t v6,
                     type_t v7,
                     type_t v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi64_mask(x, y, _MM_CMPINT_NLT);
    }
    template <int scale>
    static zmm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi64(index, base, scale);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epi64(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi64(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi64(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi64(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi64(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epi64(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi64(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epi64(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epi64(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi64(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        __m512d temp = _mm512_castsi512_pd(zmm);
        return _mm512_castpd_si512(
                _mm512_shuffle_pd(temp, temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};
template <>
struct vector<uint64_t> {
    using type_t = uint64_t;
    using zmm_t = __m512i;
    using ymm_t = __m512i;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_MAX_UINT64;
    }
    static type_t type_min()
    {
        return 0;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_epi64(type_max());
    }

    static zmm_t set(type_t v1,
                     type_t v2,
                     type_t v3,
                     type_t v4,
                     type_t v5,
                     type_t v6,
                     type_t v7,
                     type_t v8)
    {
        return _mm512_set_epi64(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    template <int scale>
    static zmm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi64(index, base, scale);
    }
    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu64_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_si512(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_epu64(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi64(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_epi64(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_epi64(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi64(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_epu64(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi64(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_epu64(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_epu64(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_epi64(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        __m512d temp = _mm512_castsi512_pd(zmm);
        return _mm512_castpd_si512(
                _mm512_shuffle_pd(temp, temp, (_MM_PERM_ENUM)mask));
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }
};
template <>
struct vector<double> {
    using type_t = double;
    using zmm_t = __m512d;
    using ymm_t = __m512d;
    using opmask_t = __mmask8;
    static const uint8_t numlanes = 8;

    static type_t type_max()
    {
        return X86_SIMD_SORT_INFINITY;
    }
    static type_t type_min()
    {
        return -X86_SIMD_SORT_INFINITY;
    }
    static zmm_t zmm_max()
    {
        return _mm512_set1_pd(type_max());
    }

    static zmm_t set(type_t v1,
                     type_t v2,
                     type_t v3,
                     type_t v4,
                     type_t v5,
                     type_t v6,
                     type_t v7,
                     type_t v8)
    {
        return _mm512_set_pd(v1, v2, v3, v4, v5, v6, v7, v8);
    }

    static opmask_t knot_opmask(opmask_t x)
    {
        return _knot_mask8(x);
    }
    static opmask_t ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_GE_OQ);
    }
    template <int scale>
    static zmm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_pd(index, base, scale);
    }
    static zmm_t loadu(void const *mem)
    {
        return _mm512_loadu_pd(mem);
    }
    static zmm_t max(zmm_t x, zmm_t y)
    {
        return _mm512_max_pd(x, y);
    }
    static void mask_compressstoreu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_pd(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, opmask_t mask, void const *mem)
    {
        return _mm512_mask_loadu_pd(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, opmask_t mask, zmm_t y)
    {
        return _mm512_mask_mov_pd(x, mask, y);
    }
    static void mask_storeu(void *mem, opmask_t mask, zmm_t x)
    {
        return _mm512_mask_storeu_pd(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y)
    {
        return _mm512_min_pd(x, y);
    }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_pd(idx, zmm);
    }
    static type_t reducemax(zmm_t v)
    {
        return _mm512_reduce_max_pd(v);
    }
    static type_t reducemin(zmm_t v)
    {
        return _mm512_reduce_min_pd(v);
    }
    static zmm_t set1(type_t v)
    {
        return _mm512_set1_pd(v);
    }
    template <uint8_t mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_pd(zmm, zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_pd(mem, x);
    }
};

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline zmm_t sort_zmm_64bit(zmm_t zmm)
{
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_1), zmm),
            0xCC);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    zmm = cmp_merge<vtype>(zmm, vtype::permutexvar(rev_index, zmm), 0xF0);
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_3), zmm),
            0xCC);
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline zmm_t bitonic_merge_zmm_64bit(zmm_t zmm)
{

    // 1) half_cleaner[8]: compare 0-4, 1-5, 2-6, 3-7
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_4), zmm),
            0xF0);
    // 2) half_cleaner[4]
    zmm = cmp_merge<vtype>(
            zmm,
            vtype::permutexvar(_mm512_set_epi64(NETWORK_64BIT_3), zmm),
            0xCC);
    // 3) half_cleaner[1]
    zmm = cmp_merge<vtype>(
            zmm, vtype::template shuffle<SHUFFLE_MASK(1, 1, 1, 1)>(zmm), 0xAA);
    return zmm;
}

// Assumes zmm1 and zmm2 are sorted and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void bitonic_merge_two_zmm_64bit(zmm_t &zmm1, zmm_t &zmm2)
{
    // 1) First step of a merging network: coex of zmm1 and zmm2 reversed
    zmm2 = vtype::permutexvar(rev_index, zmm2);
    zmm_t zmm3 = vtype::min(zmm1, zmm2);
    zmm_t zmm4 = vtype::max(zmm1, zmm2);
    // 2) Recursive half cleaner for each
    zmm1 = bitonic_merge_zmm_64bit<vtype>(zmm3);
    zmm2 = bitonic_merge_zmm_64bit<vtype>(zmm4);
}

// Assumes [zmm0, zmm1] and [zmm2, zmm3] are sorted and performs a recursive
// half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void bitonic_merge_four_zmm_64bit(zmm_t *zmm)
{
    // 1) First step of a merging network
    zmm_t zmm2r = vtype::permutexvar(rev_index, zmm[2]);
    zmm_t zmm3r = vtype::permutexvar(rev_index, zmm[3]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm3r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm2r);
    // 2) Recursive half clearer: 16
    zmm_t zmm_t3 = vtype::permutexvar(rev_index, vtype::max(zmm[1], zmm2r));
    zmm_t zmm_t4 = vtype::permutexvar(rev_index, vtype::max(zmm[0], zmm3r));
    zmm_t zmm0 = vtype::min(zmm_t1, zmm_t2);
    zmm_t zmm1 = vtype::max(zmm_t1, zmm_t2);
    zmm_t zmm2 = vtype::min(zmm_t3, zmm_t4);
    zmm_t zmm3 = vtype::max(zmm_t3, zmm_t4);
    zmm[0] = bitonic_merge_zmm_64bit<vtype>(zmm0);
    zmm[1] = bitonic_merge_zmm_64bit<vtype>(zmm1);
    zmm[2] = bitonic_merge_zmm_64bit<vtype>(zmm2);
    zmm[3] = bitonic_merge_zmm_64bit<vtype>(zmm3);
}

template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void bitonic_merge_eight_zmm_64bit(zmm_t *zmm)
{
    zmm_t zmm4r = vtype::permutexvar(rev_index, zmm[4]);
    zmm_t zmm5r = vtype::permutexvar(rev_index, zmm[5]);
    zmm_t zmm6r = vtype::permutexvar(rev_index, zmm[6]);
    zmm_t zmm7r = vtype::permutexvar(rev_index, zmm[7]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm7r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm6r);
    zmm_t zmm_t3 = vtype::min(zmm[2], zmm5r);
    zmm_t zmm_t4 = vtype::min(zmm[3], zmm4r);
    zmm_t zmm_t5 = vtype::permutexvar(rev_index, vtype::max(zmm[3], zmm4r));
    zmm_t zmm_t6 = vtype::permutexvar(rev_index, vtype::max(zmm[2], zmm5r));
    zmm_t zmm_t7 = vtype::permutexvar(rev_index, vtype::max(zmm[1], zmm6r));
    zmm_t zmm_t8 = vtype::permutexvar(rev_index, vtype::max(zmm[0], zmm7r));
    COEX<vtype>(zmm_t1, zmm_t3);
    COEX<vtype>(zmm_t2, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t7);
    COEX<vtype>(zmm_t6, zmm_t8);
    COEX<vtype>(zmm_t1, zmm_t2);
    COEX<vtype>(zmm_t3, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t6);
    COEX<vtype>(zmm_t7, zmm_t8);
    zmm[0] = bitonic_merge_zmm_64bit<vtype>(zmm_t1);
    zmm[1] = bitonic_merge_zmm_64bit<vtype>(zmm_t2);
    zmm[2] = bitonic_merge_zmm_64bit<vtype>(zmm_t3);
    zmm[3] = bitonic_merge_zmm_64bit<vtype>(zmm_t4);
    zmm[4] = bitonic_merge_zmm_64bit<vtype>(zmm_t5);
    zmm[5] = bitonic_merge_zmm_64bit<vtype>(zmm_t6);
    zmm[6] = bitonic_merge_zmm_64bit<vtype>(zmm_t7);
    zmm[7] = bitonic_merge_zmm_64bit<vtype>(zmm_t8);
}

template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void bitonic_merge_sixteen_zmm_64bit(zmm_t *zmm)
{
    zmm_t zmm8r = vtype::permutexvar(rev_index, zmm[8]);
    zmm_t zmm9r = vtype::permutexvar(rev_index, zmm[9]);
    zmm_t zmm10r = vtype::permutexvar(rev_index, zmm[10]);
    zmm_t zmm11r = vtype::permutexvar(rev_index, zmm[11]);
    zmm_t zmm12r = vtype::permutexvar(rev_index, zmm[12]);
    zmm_t zmm13r = vtype::permutexvar(rev_index, zmm[13]);
    zmm_t zmm14r = vtype::permutexvar(rev_index, zmm[14]);
    zmm_t zmm15r = vtype::permutexvar(rev_index, zmm[15]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm15r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm14r);
    zmm_t zmm_t3 = vtype::min(zmm[2], zmm13r);
    zmm_t zmm_t4 = vtype::min(zmm[3], zmm12r);
    zmm_t zmm_t5 = vtype::min(zmm[4], zmm11r);
    zmm_t zmm_t6 = vtype::min(zmm[5], zmm10r);
    zmm_t zmm_t7 = vtype::min(zmm[6], zmm9r);
    zmm_t zmm_t8 = vtype::min(zmm[7], zmm8r);
    zmm_t zmm_t9 = vtype::permutexvar(rev_index, vtype::max(zmm[7], zmm8r));
    zmm_t zmm_t10 = vtype::permutexvar(rev_index, vtype::max(zmm[6], zmm9r));
    zmm_t zmm_t11 = vtype::permutexvar(rev_index, vtype::max(zmm[5], zmm10r));
    zmm_t zmm_t12 = vtype::permutexvar(rev_index, vtype::max(zmm[4], zmm11r));
    zmm_t zmm_t13 = vtype::permutexvar(rev_index, vtype::max(zmm[3], zmm12r));
    zmm_t zmm_t14 = vtype::permutexvar(rev_index, vtype::max(zmm[2], zmm13r));
    zmm_t zmm_t15 = vtype::permutexvar(rev_index, vtype::max(zmm[1], zmm14r));
    zmm_t zmm_t16 = vtype::permutexvar(rev_index, vtype::max(zmm[0], zmm15r));
    // Recusive half clear 16 zmm regs
    COEX<vtype>(zmm_t1, zmm_t5);
    COEX<vtype>(zmm_t2, zmm_t6);
    COEX<vtype>(zmm_t3, zmm_t7);
    COEX<vtype>(zmm_t4, zmm_t8);
    COEX<vtype>(zmm_t9, zmm_t13);
    COEX<vtype>(zmm_t10, zmm_t14);
    COEX<vtype>(zmm_t11, zmm_t15);
    COEX<vtype>(zmm_t12, zmm_t16);
    //
    COEX<vtype>(zmm_t1, zmm_t3);
    COEX<vtype>(zmm_t2, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t7);
    COEX<vtype>(zmm_t6, zmm_t8);
    COEX<vtype>(zmm_t9, zmm_t11);
    COEX<vtype>(zmm_t10, zmm_t12);
    COEX<vtype>(zmm_t13, zmm_t15);
    COEX<vtype>(zmm_t14, zmm_t16);
    //
    COEX<vtype>(zmm_t1, zmm_t2);
    COEX<vtype>(zmm_t3, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t6);
    COEX<vtype>(zmm_t7, zmm_t8);
    COEX<vtype>(zmm_t9, zmm_t10);
    COEX<vtype>(zmm_t11, zmm_t12);
    COEX<vtype>(zmm_t13, zmm_t14);
    COEX<vtype>(zmm_t15, zmm_t16);
    //
    zmm[0] = bitonic_merge_zmm_64bit<vtype>(zmm_t1);
    zmm[1] = bitonic_merge_zmm_64bit<vtype>(zmm_t2);
    zmm[2] = bitonic_merge_zmm_64bit<vtype>(zmm_t3);
    zmm[3] = bitonic_merge_zmm_64bit<vtype>(zmm_t4);
    zmm[4] = bitonic_merge_zmm_64bit<vtype>(zmm_t5);
    zmm[5] = bitonic_merge_zmm_64bit<vtype>(zmm_t6);
    zmm[6] = bitonic_merge_zmm_64bit<vtype>(zmm_t7);
    zmm[7] = bitonic_merge_zmm_64bit<vtype>(zmm_t8);
    zmm[8] = bitonic_merge_zmm_64bit<vtype>(zmm_t9);
    zmm[9] = bitonic_merge_zmm_64bit<vtype>(zmm_t10);
    zmm[10] = bitonic_merge_zmm_64bit<vtype>(zmm_t11);
    zmm[11] = bitonic_merge_zmm_64bit<vtype>(zmm_t12);
    zmm[12] = bitonic_merge_zmm_64bit<vtype>(zmm_t13);
    zmm[13] = bitonic_merge_zmm_64bit<vtype>(zmm_t14);
    zmm[14] = bitonic_merge_zmm_64bit<vtype>(zmm_t15);
    zmm[15] = bitonic_merge_zmm_64bit<vtype>(zmm_t16);
}

template <typename vtype, typename type_t>
static inline void sort_8_64bit(type_t *arr, int32_t N)
{
    typename vtype::opmask_t load_mask = (0x01 << N) - 0x01;
    typename vtype::zmm_t zmm
            = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr);
    vtype::mask_storeu(arr, load_mask, sort_zmm_64bit<vtype>(zmm));
}

template <typename vtype, typename type_t>
static inline void sort_16_64bit(type_t *arr, int32_t N)
{
    if (N <= 8) {
        sort_8_64bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    zmm_t zmm1 = vtype::loadu(arr);
    typename vtype::opmask_t load_mask = (0x01 << (N - 8)) - 0x01;
    zmm_t zmm2 = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr + 8);
    zmm1 = sort_zmm_64bit<vtype>(zmm1);
    zmm2 = sort_zmm_64bit<vtype>(zmm2);
    bitonic_merge_two_zmm_64bit<vtype>(zmm1, zmm2);
    vtype::storeu(arr, zmm1);
    vtype::mask_storeu(arr + 8, load_mask, zmm2);
}

template <typename vtype, typename type_t>
static inline void sort_32_64bit(type_t *arr, int32_t N)
{
    if (N <= 16) {
        sort_16_64bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    using opmask_t = typename vtype::opmask_t;
    zmm_t zmm[4];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 8);
    opmask_t load_mask1 = 0xFF, load_mask2 = 0xFF;
    uint64_t combined_mask = (0x1ull << (N - 16)) - 0x1ull;
    load_mask1 = (combined_mask)&0xFF;
    load_mask2 = (combined_mask >> 8) & 0xFF;
    zmm[2] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 16);
    zmm[3] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 24);
    zmm[0] = sort_zmm_64bit<vtype>(zmm[0]);
    zmm[1] = sort_zmm_64bit<vtype>(zmm[1]);
    zmm[2] = sort_zmm_64bit<vtype>(zmm[2]);
    zmm[3] = sort_zmm_64bit<vtype>(zmm[3]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[0], zmm[1]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[2], zmm[3]);
    bitonic_merge_four_zmm_64bit<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 8, zmm[1]);
    vtype::mask_storeu(arr + 16, load_mask1, zmm[2]);
    vtype::mask_storeu(arr + 24, load_mask2, zmm[3]);
}

template <typename vtype, typename type_t>
static inline void sort_64_64bit(type_t *arr, int32_t N)
{
    if (N <= 32) {
        sort_32_64bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    using opmask_t = typename vtype::opmask_t;
    zmm_t zmm[8];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 8);
    zmm[2] = vtype::loadu(arr + 16);
    zmm[3] = vtype::loadu(arr + 24);
    zmm[0] = sort_zmm_64bit<vtype>(zmm[0]);
    zmm[1] = sort_zmm_64bit<vtype>(zmm[1]);
    zmm[2] = sort_zmm_64bit<vtype>(zmm[2]);
    zmm[3] = sort_zmm_64bit<vtype>(zmm[3]);
    opmask_t load_mask1 = 0xFF, load_mask2 = 0xFF;
    opmask_t load_mask3 = 0xFF, load_mask4 = 0xFF;
    // N-32 >= 1
    uint64_t combined_mask = (0x1ull << (N - 32)) - 0x1ull;
    load_mask1 = (combined_mask)&0xFF;
    load_mask2 = (combined_mask >> 8) & 0xFF;
    load_mask3 = (combined_mask >> 16) & 0xFF;
    load_mask4 = (combined_mask >> 24) & 0xFF;
    zmm[4] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 32);
    zmm[5] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 40);
    zmm[6] = vtype::mask_loadu(vtype::zmm_max(), load_mask3, arr + 48);
    zmm[7] = vtype::mask_loadu(vtype::zmm_max(), load_mask4, arr + 56);
    zmm[4] = sort_zmm_64bit<vtype>(zmm[4]);
    zmm[5] = sort_zmm_64bit<vtype>(zmm[5]);
    zmm[6] = sort_zmm_64bit<vtype>(zmm[6]);
    zmm[7] = sort_zmm_64bit<vtype>(zmm[7]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[0], zmm[1]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[2], zmm[3]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[4], zmm[5]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[6], zmm[7]);
    bitonic_merge_four_zmm_64bit<vtype>(zmm);
    bitonic_merge_four_zmm_64bit<vtype>(zmm + 4);
    bitonic_merge_eight_zmm_64bit<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 8, zmm[1]);
    vtype::storeu(arr + 16, zmm[2]);
    vtype::storeu(arr + 24, zmm[3]);
    vtype::mask_storeu(arr + 32, load_mask1, zmm[4]);
    vtype::mask_storeu(arr + 40, load_mask2, zmm[5]);
    vtype::mask_storeu(arr + 48, load_mask3, zmm[6]);
    vtype::mask_storeu(arr + 56, load_mask4, zmm[7]);
}

template <typename vtype, typename type_t>
static inline void sort_128_64bit(type_t *arr, int32_t N)
{
    if (N <= 64) {
        sort_64_64bit<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    using opmask_t = typename vtype::opmask_t;
    zmm_t zmm[16];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 8);
    zmm[2] = vtype::loadu(arr + 16);
    zmm[3] = vtype::loadu(arr + 24);
    zmm[4] = vtype::loadu(arr + 32);
    zmm[5] = vtype::loadu(arr + 40);
    zmm[6] = vtype::loadu(arr + 48);
    zmm[7] = vtype::loadu(arr + 56);
    zmm[0] = sort_zmm_64bit<vtype>(zmm[0]);
    zmm[1] = sort_zmm_64bit<vtype>(zmm[1]);
    zmm[2] = sort_zmm_64bit<vtype>(zmm[2]);
    zmm[3] = sort_zmm_64bit<vtype>(zmm[3]);
    zmm[4] = sort_zmm_64bit<vtype>(zmm[4]);
    zmm[5] = sort_zmm_64bit<vtype>(zmm[5]);
    zmm[6] = sort_zmm_64bit<vtype>(zmm[6]);
    zmm[7] = sort_zmm_64bit<vtype>(zmm[7]);
    opmask_t load_mask1 = 0xFF, load_mask2 = 0xFF;
    opmask_t load_mask3 = 0xFF, load_mask4 = 0xFF;
    opmask_t load_mask5 = 0xFF, load_mask6 = 0xFF;
    opmask_t load_mask7 = 0xFF, load_mask8 = 0xFF;
    if (N != 128) {
        uint64_t combined_mask = (0x1ull << (N - 64)) - 0x1ull;
        load_mask1 = (combined_mask)&0xFF;
        load_mask2 = (combined_mask >> 8) & 0xFF;
        load_mask3 = (combined_mask >> 16) & 0xFF;
        load_mask4 = (combined_mask >> 24) & 0xFF;
        load_mask5 = (combined_mask >> 32) & 0xFF;
        load_mask6 = (combined_mask >> 40) & 0xFF;
        load_mask7 = (combined_mask >> 48) & 0xFF;
        load_mask8 = (combined_mask >> 56) & 0xFF;
    }
    zmm[8] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 64);
    zmm[9] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 72);
    zmm[10] = vtype::mask_loadu(vtype::zmm_max(), load_mask3, arr + 80);
    zmm[11] = vtype::mask_loadu(vtype::zmm_max(), load_mask4, arr + 88);
    zmm[12] = vtype::mask_loadu(vtype::zmm_max(), load_mask5, arr + 96);
    zmm[13] = vtype::mask_loadu(vtype::zmm_max(), load_mask6, arr + 104);
    zmm[14] = vtype::mask_loadu(vtype::zmm_max(), load_mask7, arr + 112);
    zmm[15] = vtype::mask_loadu(vtype::zmm_max(), load_mask8, arr + 120);
    zmm[8] = sort_zmm_64bit<vtype>(zmm[8]);
    zmm[9] = sort_zmm_64bit<vtype>(zmm[9]);
    zmm[10] = sort_zmm_64bit<vtype>(zmm[10]);
    zmm[11] = sort_zmm_64bit<vtype>(zmm[11]);
    zmm[12] = sort_zmm_64bit<vtype>(zmm[12]);
    zmm[13] = sort_zmm_64bit<vtype>(zmm[13]);
    zmm[14] = sort_zmm_64bit<vtype>(zmm[14]);
    zmm[15] = sort_zmm_64bit<vtype>(zmm[15]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[0], zmm[1]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[2], zmm[3]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[4], zmm[5]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[6], zmm[7]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[8], zmm[9]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[10], zmm[11]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[12], zmm[13]);
    bitonic_merge_two_zmm_64bit<vtype>(zmm[14], zmm[15]);
    bitonic_merge_four_zmm_64bit<vtype>(zmm);
    bitonic_merge_four_zmm_64bit<vtype>(zmm + 4);
    bitonic_merge_four_zmm_64bit<vtype>(zmm + 8);
    bitonic_merge_four_zmm_64bit<vtype>(zmm + 12);
    bitonic_merge_eight_zmm_64bit<vtype>(zmm);
    bitonic_merge_eight_zmm_64bit<vtype>(zmm + 8);
    bitonic_merge_sixteen_zmm_64bit<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 8, zmm[1]);
    vtype::storeu(arr + 16, zmm[2]);
    vtype::storeu(arr + 24, zmm[3]);
    vtype::storeu(arr + 32, zmm[4]);
    vtype::storeu(arr + 40, zmm[5]);
    vtype::storeu(arr + 48, zmm[6]);
    vtype::storeu(arr + 56, zmm[7]);
    vtype::mask_storeu(arr + 64, load_mask1, zmm[8]);
    vtype::mask_storeu(arr + 72, load_mask2, zmm[9]);
    vtype::mask_storeu(arr + 80, load_mask3, zmm[10]);
    vtype::mask_storeu(arr + 88, load_mask4, zmm[11]);
    vtype::mask_storeu(arr + 96, load_mask5, zmm[12]);
    vtype::mask_storeu(arr + 104, load_mask6, zmm[13]);
    vtype::mask_storeu(arr + 112, load_mask7, zmm[14]);
    vtype::mask_storeu(arr + 120, load_mask8, zmm[15]);
}

template <typename vtype, typename type_t>
static inline type_t
get_pivot_64bit(type_t *arr, const int64_t left, const int64_t right)
{
    // median of 8
    int64_t size = (right - left) / 8;
    using zmm_t = typename vtype::zmm_t;
    __m512i rand_index = _mm512_set_epi64(left + size,
                                          left + 2 * size,
                                          left + 3 * size,
                                          left + 4 * size,
                                          left + 5 * size,
                                          left + 6 * size,
                                          left + 7 * size,
                                          left + 8 * size);
    zmm_t rand_vec = vtype::template i64gather<sizeof(type_t)>(rand_index, arr);
    // pivot will never be a nan, since there are no nan's!
    zmm_t sort = sort_zmm_64bit<vtype>(rand_vec);
    return ((type_t *)&sort)[4];
}

template <typename vtype, typename type_t>
static inline void
qsort_64bit_(type_t *arr, int64_t left, int64_t right, int64_t max_iters)
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
        sort_128_64bit<vtype>(arr + left, (int32_t)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot_64bit<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    int64_t pivot_index = partition_avx512<vtype>(
            arr, left, right + 1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        qsort_64bit_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qsort_64bit_<vtype>(arr, pivot_index, right, max_iters - 1);
}

static inline int64_t replace_nan_with_inf(double *arr, int64_t arrsize)
{
    int64_t nan_count = 0;
    __mmask8 loadmask = 0xFF;
    while (arrsize > 0) {
        if (arrsize < 8) { loadmask = (0x01 << arrsize) - 0x01; }
        __m512d in_zmm = _mm512_maskz_loadu_pd(loadmask, arr);
        __mmask8 nanmask = _mm512_cmp_pd_mask(in_zmm, in_zmm, _CMP_NEQ_UQ);
        nan_count += _mm_popcnt_u32((int32_t)nanmask);
        _mm512_mask_storeu_pd(arr, nanmask, ZMM_MAX_DOUBLE);
        arr += 8;
        arrsize -= 8;
    }
    return nan_count;
}

static inline void
replace_inf_with_nan(double *arr, int64_t arrsize, int64_t nan_count)
{
    for (int64_t ii = arrsize - 1; nan_count > 0; --ii) {
        arr[ii] = std::nan("1");
        nan_count -= 1;
    }
}

template <>
void avx512_qsort<int64_t>(int64_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_64bit_<vector<int64_t>, int64_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx512_qsort<uint64_t>(uint64_t *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        qsort_64bit_<vector<uint64_t>, uint64_t>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
    }
}

template <>
void avx512_qsort<double>(double *arr, int64_t arrsize)
{
    if (arrsize > 1) {
        int64_t nan_count = replace_nan_with_inf(arr, arrsize);
        qsort_64bit_<vector<double>, double>(
                arr, 0, arrsize - 1, 2 * (int64_t)log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}
#endif // __AVX512_QSORT_64BIT__
