/*******************************************************************
 * Copyright (C) 2022 Intel Corporation
 * Copyright (C) 2021 Serge Sans Paille
 * SPDX-License-Identifier: BSD-3-Clause
 * Authors: Raghuveer Devulapalli <raghuveer.devulapalli@intel.com>
 *          Serge Sans Paille <serge.guelton@telecom-bretagne.eu>
 * ****************************************************************/

#ifndef __AVX512_QSORT_COMMON__
#define __AVX512_QSORT_COMMON__

/*
 * Quicksort using AVX-512. The ideas and code are based on these two research
 * papers [1] and [2]. On a high level, the idea is to vectorize quicksort
 * partitioning using AVX-512 compressstore instructions. If the array size is
 * < 128, then use Bitonic sorting network implemented on 512-bit registers.
 * The precise network definitions depend on the dtype and are defined in
 * separate files: avx512-16bit-qsort.hpp, avx512-32bit-qsort.hpp and
 * avx512-64bit-qsort.hpp. Article [4] is a good resource for bitonic sorting
 * network. The core implementations of the vectorized qsort functions
 * avx512_qsort<T>(T*, int64_t) are modified versions of avx2 quicksort
 * presented in the paper [2] and source code associated with that paper [3].
 *
 * [1] Fast and Robust Vectorized In-Place Sorting of Primitive Types
 *     https://drops.dagstuhl.de/opus/volltexte/2021/13775/
 *
 * [2] A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel
 * Skylake https://arxiv.org/pdf/1704.08579.pdf
 *
 * [3] https://github.com/simd-sorting/fast-and-robust: SPDX-License-Identifier: MIT
 *
 * [4] http://mitp-content-server.mit.edu:18180/books/content/sectbyfn?collid=books_pres_0&fn=Chapter%2027.pdf&id=8030
 *
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include "simd/simd.h"

#define X86_SIMD_SORT_INFINITY std::numeric_limits<double>::infinity()
#define X86_SIMD_SORT_INFINITYF std::numeric_limits<float>::infinity()
#define X86_SIMD_SORT_MAX_UINT16 std::numeric_limits<uint16_t>::max()
#define X86_SIMD_SORT_MAX_INT16 std::numeric_limits<int16_t>::max()
#define X86_SIMD_SORT_MIN_INT16 std::numeric_limits<int16_t>::min()
#define X86_SIMD_SORT_MAX_UINT32 std::numeric_limits<uint32_t>::max()
#define X86_SIMD_SORT_MAX_INT32 std::numeric_limits<int32_t>::max()
#define X86_SIMD_SORT_MIN_INT32 std::numeric_limits<int32_t>::min()
#define X86_SIMD_SORT_MAX_UINT64 std::numeric_limits<uint64_t>::max()
#define X86_SIMD_SORT_MAX_INT64 std::numeric_limits<int64_t>::max()
#define X86_SIMD_SORT_MIN_INT64 std::numeric_limits<int64_t>::min()
#define ZMM_MAX_DOUBLE _mm512_set1_pd(X86_SIMD_SORT_INFINITY)
#define ZMM_MAX_UINT64 _mm512_set1_epi64(X86_SIMD_SORT_MAX_UINT64)
#define ZMM_MAX_INT64 _mm512_set1_epi64(X86_SIMD_SORT_MAX_INT64)
#define ZMM_MAX_FLOAT _mm512_set1_ps(X86_SIMD_SORT_INFINITYF)
#define ZMM_MAX_UINT _mm512_set1_epi32(X86_SIMD_SORT_MAX_UINT32)
#define ZMM_MAX_INT _mm512_set1_epi32(X86_SIMD_SORT_MAX_INT32)
#define ZMM_MAX_UINT16 _mm512_set1_epi16(X86_SIMD_SORT_MAX_UINT16)
#define ZMM_MAX_INT16 _mm512_set1_epi16(X86_SIMD_SORT_MAX_INT16)
#define SHUFFLE_MASK(a, b, c, d) (a << 6) | (b << 4) | (c << 2) | d

template <typename type>
struct vector;

template <typename T>
void avx512_qsort(T *arr, int64_t arrsize);

/*
 * COEX == Compare and Exchange two registers by swapping min and max values
 */
template <typename vtype, typename mm_t>
static void COEX(mm_t &a, mm_t &b)
{
    mm_t temp = a;
    a = vtype::min(a, b);
    b = vtype::max(temp, b);
}

template <typename vtype,
          typename zmm_t = typename vtype::zmm_t,
          typename opmask_t = typename vtype::opmask_t>
static inline zmm_t cmp_merge(zmm_t in1, zmm_t in2, opmask_t mask)
{
    zmm_t min = vtype::min(in2, in1);
    zmm_t max = vtype::max(in2, in1);
    return vtype::mask_mov(min, mask, max); // 0 -> min, 1 -> max
}

/*
 * Parition one ZMM register based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype, typename type_t, typename zmm_t>
static inline int32_t partition_vec(type_t *arr,
                                    int64_t left,
                                    int64_t right,
                                    const zmm_t curr_vec,
                                    const zmm_t pivot_vec,
                                    zmm_t *smallest_vec,
                                    zmm_t *biggest_vec)
{
    /* which elements are larger than the pivot */
    typename vtype::opmask_t gt_mask = vtype::ge(curr_vec, pivot_vec);
    int32_t amount_gt_pivot = _mm_popcnt_u32((int32_t)gt_mask);
    vtype::mask_compressstoreu(
            arr + left, vtype::knot_opmask(gt_mask), curr_vec);
    vtype::mask_compressstoreu(
            arr + right - amount_gt_pivot, gt_mask, curr_vec);
    *smallest_vec = vtype::min(curr_vec, *smallest_vec);
    *biggest_vec = vtype::max(curr_vec, *biggest_vec);
    return amount_gt_pivot;
}

/*
 * Parition an array based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype, typename type_t>
static inline int64_t partition_avx512(type_t *arr,
                                       int64_t left,
                                       int64_t right,
                                       type_t pivot,
                                       type_t *smallest,
                                       type_t *biggest)
{
    /* make array length divisible by vtype::numlanes , shortening the array */
    for (int32_t i = (right - left) % vtype::numlanes; i > 0; --i) {
        *smallest = std::min(*smallest, arr[left]);
        *biggest = std::max(*biggest, arr[left]);
        if (arr[left] > pivot) { std::swap(arr[left], arr[--right]); }
        else {
            ++left;
        }
    }

    if (left == right)
        return left; /* less than vtype::numlanes elements in the array */

    using zmm_t = typename vtype::zmm_t;
    zmm_t pivot_vec = vtype::set1(pivot);
    zmm_t min_vec = vtype::set1(*smallest);
    zmm_t max_vec = vtype::set1(*biggest);

    if (right - left == vtype::numlanes) {
        zmm_t vec = vtype::loadu(arr + left);
        int32_t amount_gt_pivot = partition_vec<vtype>(arr,
                                                       left,
                                                       left + vtype::numlanes,
                                                       vec,
                                                       pivot_vec,
                                                       &min_vec,
                                                       &max_vec);
        *smallest = vtype::reducemin(min_vec);
        *biggest = vtype::reducemax(max_vec);
        return left + (vtype::numlanes - amount_gt_pivot);
    }

    // first and last vtype::numlanes values are partitioned at the end
    zmm_t vec_left = vtype::loadu(arr + left);
    zmm_t vec_right = vtype::loadu(arr + (right - vtype::numlanes));
    // store points of the vectors
    int64_t r_store = right - vtype::numlanes;
    int64_t l_store = left;
    // indices for loading the elements
    left += vtype::numlanes;
    right -= vtype::numlanes;
    while (right - left != 0) {
        zmm_t curr_vec;
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if ((r_store + vtype::numlanes) - right < left - l_store) {
            right -= vtype::numlanes;
            curr_vec = vtype::loadu(arr + right);
        }
        else {
            curr_vec = vtype::loadu(arr + left);
            left += vtype::numlanes;
        }
        // partition the current vector and save it on both sides of the array
        int32_t amount_gt_pivot
                = partition_vec<vtype>(arr,
                                       l_store,
                                       r_store + vtype::numlanes,
                                       curr_vec,
                                       pivot_vec,
                                       &min_vec,
                                       &max_vec);
        ;
        r_store -= amount_gt_pivot;
        l_store += (vtype::numlanes - amount_gt_pivot);
    }

    /* partition and save vec_left and vec_right */
    int32_t amount_gt_pivot = partition_vec<vtype>(arr,
                                                   l_store,
                                                   r_store + vtype::numlanes,
                                                   vec_left,
                                                   pivot_vec,
                                                   &min_vec,
                                                   &max_vec);
    l_store += (vtype::numlanes - amount_gt_pivot);
    amount_gt_pivot = partition_vec<vtype>(arr,
                                           l_store,
                                           l_store + vtype::numlanes,
                                           vec_right,
                                           pivot_vec,
                                           &min_vec,
                                           &max_vec);
    l_store += (vtype::numlanes - amount_gt_pivot);
    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}
#endif // __AVX512_QSORT_COMMON__
