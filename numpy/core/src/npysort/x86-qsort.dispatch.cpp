/*@targets
 * $maxopt $keep_baseline avx512_skx
 */
// policy $keep_baseline is used to avoid skip building avx512_skx
// when its part of baseline features (--cpu-baseline), since
// 'baseline' option isn't specified within targets.

#include "x86-qsort.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#ifdef NPY_HAVE_AVX512_SKX
#include "numpy/npy_math.h"

#include "npy_sort.h"
#include "numpy_tag.h"

#include "simd/simd.h"
#include <immintrin.h>

template <typename Tag, typename type>
NPY_NO_EXPORT int
heapsort_(type *start, npy_intp n);

/*
 * Quicksort using AVX-512 for int, uint32 and float. The ideas and code are
 * based on these two research papers:
 * (1) Fast and Robust Vectorized In-Place Sorting of Primitive Types
 *     https://drops.dagstuhl.de/opus/volltexte/2021/13775/
 * (2) A Novel Hybrid Quicksort Algorithm Vectorized using AVX-512 on Intel
 * Skylake https://arxiv.org/pdf/1704.08579.pdf
 *
 * High level idea: Vectorize the quicksort partitioning using AVX-512
 * compressstore instructions. The algorithm to pick the pivot is to use median
 * of 72 elements picked at random. If the array size is < 128, then use
 * Bitonic sorting network. Good resource for bitonic sorting network:
 * http://mitp-content-server.mit.edu:18180/books/content/sectbyfn?collid=books_pres_0&fn=Chapter%2027.pdf&id=8030
 *
 * Refer to https://github.com/numpy/numpy/pull/20133#issuecomment-958110340
 * for potential problems when converting this code to universal intrinsics
 * framework.
 */

/*
 * Constants used in sorting 16 elements in a ZMM registers. Based on Bitonic
 * sorting network (see
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg)
 */
#define NETWORK1 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1
#define NETWORK2 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
#define NETWORK3 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7
#define NETWORK4 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2
#define NETWORK5 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define NETWORK6 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4
#define NETWORK7 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
#define ZMM_MAX_FLOAT _mm512_set1_ps(NPY_INFINITYF)
#define ZMM_MAX_UINT _mm512_set1_epi32(NPY_MAX_UINT32)
#define ZMM_MAX_INT _mm512_set1_epi32(NPY_MAX_INT32)
#define SHUFFLE_MASK(a, b, c, d) (a << 6) | (b << 4) | (c << 2) | d
#define SHUFFLE_ps(ZMM, MASK) _mm512_shuffle_ps(zmm, zmm, MASK)
#define SHUFFLE_epi32(ZMM, MASK) _mm512_shuffle_epi32(zmm, MASK)

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*
 * Vectorized random number generator xoroshiro128+. Broken into 2 parts:
 * (1) vnext generates 2 64-bit random integers
 * (2) rnd_epu32 converts this to 4 32-bit random integers and bounds it to
 *     the length of the array
 */
#define VROTL(x, k) /* rotate each uint64_t value in vector */ \
    _mm256_or_si256(_mm256_slli_epi64((x), (k)),               \
                    _mm256_srli_epi64((x), 64 - (k)))

static inline __m256i
vnext(__m256i *s0, __m256i *s1)
{
    *s1 = _mm256_xor_si256(*s0, *s1); /* modify vectors s1 and s0 */
    *s0 = _mm256_xor_si256(_mm256_xor_si256(VROTL(*s0, 24), *s1),
                           _mm256_slli_epi64(*s1, 16));
    *s1 = VROTL(*s1, 37);
    return _mm256_add_epi64(*s0, *s1); /* return random vector */
}

/* transform random numbers to the range between 0 and bound - 1 */
static inline __m256i
rnd_epu32(__m256i rnd_vec, __m256i bound)
{
    __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(rnd_vec, bound), 32);
    __m256i odd = _mm256_mul_epu32(_mm256_srli_epi64(rnd_vec, 32), bound);
    return _mm256_blend_epi32(odd, even, 0b01010101);
}

template <typename type>
struct vector;

template <>
struct vector<npy_int> {
    using tag = npy::int_tag;
    using type_t = npy_int;
    using zmm_t = __m512i;
    using ymm_t = __m256i;

    static type_t type_max() { return NPY_MAX_INT32; }
    static type_t type_min() { return NPY_MIN_INT32; }
    static zmm_t zmm_max() { return _mm512_set1_epi32(type_max()); }

    static __mmask16 ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_NLT);
    }
    template <int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static zmm_t loadu(void const *mem) { return _mm512_loadu_si512(mem); }
    static zmm_t max(zmm_t x, zmm_t y) { return _mm512_max_epi32(x, y); }
    static void mask_compressstoreu(void *mem, __mmask16 mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, __mmask16 mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, __mmask16 mask, zmm_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, __mmask16 mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y) { return _mm512_min_epi32(x, y); }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(zmm_t v) { return npyv_reduce_max_s32(v); }
    static type_t reducemin(zmm_t v) { return npyv_reduce_min_s32(v); }
    static zmm_t set1(type_t v) { return _mm512_set1_epi32(v); }
    template<__mmask16 mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y) { return _mm256_max_epi32(x, y); }
    static ymm_t min(ymm_t x, ymm_t y) { return _mm256_min_epi32(x, y); }
};
template <>
struct vector<npy_uint> {
    using tag = npy::uint_tag;
    using type_t = npy_uint;
    using zmm_t = __m512i;
    using ymm_t = __m256i;

    static type_t type_max() { return NPY_MAX_UINT32; }
    static type_t type_min() { return 0; }
    static zmm_t zmm_max() { return _mm512_set1_epi32(type_max()); }

    template<int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_epi32(index, base, scale);
    }
    static __mmask16 ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_epu32_mask(x, y, _MM_CMPINT_NLT);
    }
    static zmm_t loadu(void const *mem) { return _mm512_loadu_si512(mem); }
    static zmm_t max(zmm_t x, zmm_t y) { return _mm512_max_epu32(x, y); }
    static void mask_compressstoreu(void *mem, __mmask16 mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_epi32(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, __mmask16 mask, void const *mem)
    {
        return _mm512_mask_loadu_epi32(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, __mmask16 mask, zmm_t y)
    {
        return _mm512_mask_mov_epi32(x, mask, y);
    }
    static void mask_storeu(void *mem, __mmask16 mask, zmm_t x)
    {
        return _mm512_mask_storeu_epi32(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y) { return _mm512_min_epu32(x, y); }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_epi32(idx, zmm);
    }
    static type_t reducemax(zmm_t v) { return npyv_reduce_max_u32(v); }
    static type_t reducemin(zmm_t v) { return npyv_reduce_min_u32(v); }
    static zmm_t set1(type_t v) { return _mm512_set1_epi32(v); }
    template<__mmask16 mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_epi32(zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x)
    {
        return _mm512_storeu_si512(mem, x);
    }

    static ymm_t max(ymm_t x, ymm_t y) { return _mm256_max_epu32(x, y); }
    static ymm_t min(ymm_t x, ymm_t y) { return _mm256_min_epu32(x, y); }
};
template <>
struct vector<npy_float> {
    using tag = npy::float_tag;
    using type_t = npy_float;
    using zmm_t = __m512;
    using ymm_t = __m256;

    static type_t type_max() { return NPY_INFINITYF; }
    static type_t type_min() { return -NPY_INFINITYF; }
    static zmm_t zmm_max() { return _mm512_set1_ps(type_max()); }

    static __mmask16 ge(zmm_t x, zmm_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
    }
    template<int scale>
    static ymm_t i64gather(__m512i index, void const *base)
    {
        return _mm512_i64gather_ps(index, base, scale);
    }
    static zmm_t loadu(void const *mem) { return _mm512_loadu_ps(mem); }
    static zmm_t max(zmm_t x, zmm_t y) { return _mm512_max_ps(x, y); }
    static void mask_compressstoreu(void *mem, __mmask16 mask, zmm_t x)
    {
        return _mm512_mask_compressstoreu_ps(mem, mask, x);
    }
    static zmm_t mask_loadu(zmm_t x, __mmask16 mask, void const *mem)
    {
        return _mm512_mask_loadu_ps(x, mask, mem);
    }
    static zmm_t mask_mov(zmm_t x, __mmask16 mask, zmm_t y)
    {
        return _mm512_mask_mov_ps(x, mask, y);
    }
    static void mask_storeu(void *mem, __mmask16 mask, zmm_t x)
    {
        return _mm512_mask_storeu_ps(mem, mask, x);
    }
    static zmm_t min(zmm_t x, zmm_t y) { return _mm512_min_ps(x, y); }
    static zmm_t permutexvar(__m512i idx, zmm_t zmm)
    {
        return _mm512_permutexvar_ps(idx, zmm);
    }
    static type_t reducemax(zmm_t v) { return npyv_reduce_max_f32(v); }
    static type_t reducemin(zmm_t v) { return npyv_reduce_min_f32(v); }
    static zmm_t set1(type_t v) { return _mm512_set1_ps(v); }
    template<__mmask16 mask>
    static zmm_t shuffle(zmm_t zmm)
    {
        return _mm512_shuffle_ps(zmm, zmm, (_MM_PERM_ENUM)mask);
    }
    static void storeu(void *mem, zmm_t x) { return _mm512_storeu_ps(mem, x); }

    static ymm_t max(ymm_t x, ymm_t y) { return _mm256_max_ps(x, y); }
    static ymm_t min(ymm_t x, ymm_t y) { return _mm256_min_ps(x, y); }
};

/*
 * COEX == Compare and Exchange two registers by swapping min and max values
 */
template <typename vtype, typename mm_t>
void
COEX(mm_t &a, mm_t &b)
{
    mm_t temp = a;
    a = vtype::min(a, b);
    b = vtype::max(temp, b);
}

template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline zmm_t
cmp_merge(zmm_t in1, zmm_t in2, __mmask16 mask)
{
    zmm_t min = vtype::min(in2, in1);
    zmm_t max = vtype::max(in2, in1);
    return vtype::mask_mov(min, mask, max);  // 0 -> min, 1 -> max
}

/*
 * Assumes zmm is random and performs a full sorting network defined in
 * https://en.wikipedia.org/wiki/Bitonic_sorter#/media/File:BitonicSort.svg
 */
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline zmm_t
sort_zmm(zmm_t zmm)
{
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
                           0xAAAA);
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(0, 1, 2, 3)>(zmm),
                           0xCCCC);
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
                           0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(_mm512_set_epi32(NETWORK3), zmm), 0xF0F0);
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
                           0xCCCC);
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
                           0xAAAA);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(_mm512_set_epi32(NETWORK5), zmm), 0xFF00);
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(_mm512_set_epi32(NETWORK6), zmm), 0xF0F0);
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
                           0xCCCC);
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
                           0xAAAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline zmm_t
bitonic_merge_zmm(zmm_t zmm)
{
    // 1) half_cleaner[16]: compare 1-9, 2-10, 3-11 etc ..
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(_mm512_set_epi32(NETWORK7), zmm), 0xFF00);
    // 2) half_cleaner[8]: compare 1-5, 2-6, 3-7 etc ..
    zmm = cmp_merge<vtype>(
            zmm, vtype::permutexvar(_mm512_set_epi32(NETWORK6), zmm), 0xF0F0);
    // 3) half_cleaner[4]
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(1, 0, 3, 2)>(zmm),
                           0xCCCC);
    // 3) half_cleaner[1]
    zmm = cmp_merge<vtype>(zmm, vtype::template shuffle<SHUFFLE_MASK(2, 3, 0, 1)>(zmm),
                           0xAAAA);
    return zmm;
}

// Assumes zmm1 and zmm2 are sorted and performs a recursive half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void
bitonic_merge_two_zmm(zmm_t *zmm1, zmm_t *zmm2)
{
    // 1) First step of a merging network: coex of zmm1 and zmm2 reversed
    *zmm2 = vtype::permutexvar(_mm512_set_epi32(NETWORK5), *zmm2);
    zmm_t zmm3 = vtype::min(*zmm1, *zmm2);
    zmm_t zmm4 = vtype::max(*zmm1, *zmm2);
    // 2) Recursive half cleaner for each
    *zmm1 = bitonic_merge_zmm<vtype>(zmm3);
    *zmm2 = bitonic_merge_zmm<vtype>(zmm4);
}

// Assumes [zmm0, zmm1] and [zmm2, zmm3] are sorted and performs a recursive
// half cleaner
template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void
bitonic_merge_four_zmm(zmm_t *zmm)
{
    zmm_t zmm2r = vtype::permutexvar(_mm512_set_epi32(NETWORK5), zmm[2]);
    zmm_t zmm3r = vtype::permutexvar(_mm512_set_epi32(NETWORK5), zmm[3]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm3r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm2r);
    zmm_t zmm_t3 = vtype::permutexvar(_mm512_set_epi32(NETWORK5),
                                      vtype::max(zmm[1], zmm2r));
    zmm_t zmm_t4 = vtype::permutexvar(_mm512_set_epi32(NETWORK5),
                                      vtype::max(zmm[0], zmm3r));
    zmm_t zmm0 = vtype::min(zmm_t1, zmm_t2);
    zmm_t zmm1 = vtype::max(zmm_t1, zmm_t2);
    zmm_t zmm2 = vtype::min(zmm_t3, zmm_t4);
    zmm_t zmm3 = vtype::max(zmm_t3, zmm_t4);
    zmm[0] = bitonic_merge_zmm<vtype>(zmm0);
    zmm[1] = bitonic_merge_zmm<vtype>(zmm1);
    zmm[2] = bitonic_merge_zmm<vtype>(zmm2);
    zmm[3] = bitonic_merge_zmm<vtype>(zmm3);
}

template <typename vtype, typename zmm_t = typename vtype::zmm_t>
static inline void
bitonic_merge_eight_zmm(zmm_t *zmm)
{
    zmm_t zmm4r = vtype::permutexvar(_mm512_set_epi32(NETWORK5), zmm[4]);
    zmm_t zmm5r = vtype::permutexvar(_mm512_set_epi32(NETWORK5), zmm[5]);
    zmm_t zmm6r = vtype::permutexvar(_mm512_set_epi32(NETWORK5), zmm[6]);
    zmm_t zmm7r = vtype::permutexvar(_mm512_set_epi32(NETWORK5), zmm[7]);
    zmm_t zmm_t1 = vtype::min(zmm[0], zmm7r);
    zmm_t zmm_t2 = vtype::min(zmm[1], zmm6r);
    zmm_t zmm_t3 = vtype::min(zmm[2], zmm5r);
    zmm_t zmm_t4 = vtype::min(zmm[3], zmm4r);
    zmm_t zmm_t5 = vtype::permutexvar(_mm512_set_epi32(NETWORK5),
                                      vtype::max(zmm[3], zmm4r));
    zmm_t zmm_t6 = vtype::permutexvar(_mm512_set_epi32(NETWORK5),
                                      vtype::max(zmm[2], zmm5r));
    zmm_t zmm_t7 = vtype::permutexvar(_mm512_set_epi32(NETWORK5),
                                      vtype::max(zmm[1], zmm6r));
    zmm_t zmm_t8 = vtype::permutexvar(_mm512_set_epi32(NETWORK5),
                                      vtype::max(zmm[0], zmm7r));
    COEX<vtype>(zmm_t1, zmm_t3);
    COEX<vtype>(zmm_t2, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t7);
    COEX<vtype>(zmm_t6, zmm_t8);
    COEX<vtype>(zmm_t1, zmm_t2);
    COEX<vtype>(zmm_t3, zmm_t4);
    COEX<vtype>(zmm_t5, zmm_t6);
    COEX<vtype>(zmm_t7, zmm_t8);
    zmm[0] = bitonic_merge_zmm<vtype>(zmm_t1);
    zmm[1] = bitonic_merge_zmm<vtype>(zmm_t2);
    zmm[2] = bitonic_merge_zmm<vtype>(zmm_t3);
    zmm[3] = bitonic_merge_zmm<vtype>(zmm_t4);
    zmm[4] = bitonic_merge_zmm<vtype>(zmm_t5);
    zmm[5] = bitonic_merge_zmm<vtype>(zmm_t6);
    zmm[6] = bitonic_merge_zmm<vtype>(zmm_t7);
    zmm[7] = bitonic_merge_zmm<vtype>(zmm_t8);
}

template <typename vtype, typename type_t>
static inline void
sort_16(type_t *arr, npy_int N)
{
    __mmask16 load_mask = (0x0001 << N) - 0x0001;
    typename vtype::zmm_t zmm =
            vtype::mask_loadu(vtype::zmm_max(), load_mask, arr);
    vtype::mask_storeu(arr, load_mask, sort_zmm<vtype>(zmm));
}

template <typename vtype, typename type_t>
static inline void
sort_32(type_t *arr, npy_int N)
{
    if (N <= 16) {
        sort_16<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    zmm_t zmm1 = vtype::loadu(arr);
    __mmask16 load_mask = (0x0001 << (N - 16)) - 0x0001;
    zmm_t zmm2 = vtype::mask_loadu(vtype::zmm_max(), load_mask, arr + 16);
    zmm1 = sort_zmm<vtype>(zmm1);
    zmm2 = sort_zmm<vtype>(zmm2);
    bitonic_merge_two_zmm<vtype>(&zmm1, &zmm2);
    vtype::storeu(arr, zmm1);
    vtype::mask_storeu(arr + 16, load_mask, zmm2);
}

template <typename vtype, typename type_t>
static inline void
sort_64(type_t *arr, npy_int N)
{
    if (N <= 32) {
        sort_32<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    zmm_t zmm[4];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 16);
    __mmask16 load_mask1 = 0xFFFF, load_mask2 = 0xFFFF;
    if (N < 48) {
        load_mask1 = (0x0001 << (N - 32)) - 0x0001;
        load_mask2 = 0x0000;
    }
    else if (N < 64) {
        load_mask2 = (0x0001 << (N - 48)) - 0x0001;
    }
    zmm[2] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 32);
    zmm[3] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 48);
    zmm[0] = sort_zmm<vtype>(zmm[0]);
    zmm[1] = sort_zmm<vtype>(zmm[1]);
    zmm[2] = sort_zmm<vtype>(zmm[2]);
    zmm[3] = sort_zmm<vtype>(zmm[3]);
    bitonic_merge_two_zmm<vtype>(&zmm[0], &zmm[1]);
    bitonic_merge_two_zmm<vtype>(&zmm[2], &zmm[3]);
    bitonic_merge_four_zmm<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 16, zmm[1]);
    vtype::mask_storeu(arr + 32, load_mask1, zmm[2]);
    vtype::mask_storeu(arr + 48, load_mask2, zmm[3]);
}

template <typename vtype, typename type_t>
static inline void
sort_128(type_t *arr, npy_int N)
{
    if (N <= 64) {
        sort_64<vtype>(arr, N);
        return;
    }
    using zmm_t = typename vtype::zmm_t;
    zmm_t zmm[8];
    zmm[0] = vtype::loadu(arr);
    zmm[1] = vtype::loadu(arr + 16);
    zmm[2] = vtype::loadu(arr + 32);
    zmm[3] = vtype::loadu(arr + 48);
    zmm[0] = sort_zmm<vtype>(zmm[0]);
    zmm[1] = sort_zmm<vtype>(zmm[1]);
    zmm[2] = sort_zmm<vtype>(zmm[2]);
    zmm[3] = sort_zmm<vtype>(zmm[3]);
    __mmask16 load_mask1 = 0xFFFF, load_mask2 = 0xFFFF;
    __mmask16 load_mask3 = 0xFFFF, load_mask4 = 0xFFFF;
    if (N < 80) {
        load_mask1 = (0x0001 << (N - 64)) - 0x0001;
        load_mask2 = 0x0000;
        load_mask3 = 0x0000;
        load_mask4 = 0x0000;
    }
    else if (N < 96) {
        load_mask2 = (0x0001 << (N - 80)) - 0x0001;
        load_mask3 = 0x0000;
        load_mask4 = 0x0000;
    }
    else if (N < 112) {
        load_mask3 = (0x0001 << (N - 96)) - 0x0001;
        load_mask4 = 0x0000;
    }
    else {
        load_mask4 = (0x0001 << (N - 112)) - 0x0001;
    }
    zmm[4] = vtype::mask_loadu(vtype::zmm_max(), load_mask1, arr + 64);
    zmm[5] = vtype::mask_loadu(vtype::zmm_max(), load_mask2, arr + 80);
    zmm[6] = vtype::mask_loadu(vtype::zmm_max(), load_mask3, arr + 96);
    zmm[7] = vtype::mask_loadu(vtype::zmm_max(), load_mask4, arr + 112);
    zmm[4] = sort_zmm<vtype>(zmm[4]);
    zmm[5] = sort_zmm<vtype>(zmm[5]);
    zmm[6] = sort_zmm<vtype>(zmm[6]);
    zmm[7] = sort_zmm<vtype>(zmm[7]);
    bitonic_merge_two_zmm<vtype>(&zmm[0], &zmm[1]);
    bitonic_merge_two_zmm<vtype>(&zmm[2], &zmm[3]);
    bitonic_merge_two_zmm<vtype>(&zmm[4], &zmm[5]);
    bitonic_merge_two_zmm<vtype>(&zmm[6], &zmm[7]);
    bitonic_merge_four_zmm<vtype>(zmm);
    bitonic_merge_four_zmm<vtype>(zmm + 4);
    bitonic_merge_eight_zmm<vtype>(zmm);
    vtype::storeu(arr, zmm[0]);
    vtype::storeu(arr + 16, zmm[1]);
    vtype::storeu(arr + 32, zmm[2]);
    vtype::storeu(arr + 48, zmm[3]);
    vtype::mask_storeu(arr + 64, load_mask1, zmm[4]);
    vtype::mask_storeu(arr + 80, load_mask2, zmm[5]);
    vtype::mask_storeu(arr + 96, load_mask3, zmm[6]);
    vtype::mask_storeu(arr + 112, load_mask4, zmm[7]);
}

template <typename type_t>
static inline void
swap(type_t *arr, npy_intp ii, npy_intp jj)
{
    type_t temp = arr[ii];
    arr[ii] = arr[jj];
    arr[jj] = temp;
}

// Median of 3 strategy
// template<typename type_t>
// static inline
// npy_intp get_pivot_index(type_t *arr, const npy_intp left, const npy_intp
// right) {
//    return (rand() % (right + 1 - left)) + left;
//    //npy_intp middle = ((right-left)/2) + left;
//    //type_t a = arr[left], b = arr[middle], c = arr[right];
//    //if ((b >= a && b <= c) || (b <= a && b >= c))
//    //    return middle;
//    //if ((a >= b && a <= c) || (a <= b && a >= c))
//    //    return left;
//    //else
//    //    return right;
//}

/*
 * Picking the pivot: Median of 72 array elements chosen at random.
 */

template <typename vtype, typename type_t>
static inline type_t
get_pivot(type_t *arr, const npy_intp left, const npy_intp right)
{
    /* seeds for vectorized random number generator */
    __m256i s0 = _mm256_setr_epi64x(8265987198341093849, 3762817312854612374,
                                    1324281658759788278, 6214952190349879213);
    __m256i s1 = _mm256_setr_epi64x(2874178529384792648, 1257248936691237653,
                                    7874578921548791257, 1998265912745817298);
    s0 = _mm256_add_epi64(s0, _mm256_set1_epi64x(left));
    s1 = _mm256_sub_epi64(s1, _mm256_set1_epi64x(right));

    npy_intp arrsize = right - left + 1;
    __m256i bound =
            _mm256_set1_epi32(arrsize > INT32_MAX ? INT32_MAX : arrsize);
    __m512i left_vec = _mm512_set1_epi64(left);
    __m512i right_vec = _mm512_set1_epi64(right);
    using ymm_t = typename vtype::ymm_t;
    ymm_t v[9];
    /* fill 9 vectors with random numbers */
    for (npy_int i = 0; i < 9; ++i) {
        __m256i rand_64 = vnext(&s0, &s1); /* vector with 4 random uint64_t */
        __m512i rand_32 = _mm512_cvtepi32_epi64(rnd_epu32(
                rand_64, bound)); /* random numbers between 0 and bound - 1 */
        __m512i indices;
        if (i < 5)
            indices =
                    _mm512_add_epi64(left_vec, rand_32); /* indices for arr */
        else
            indices =
                    _mm512_sub_epi64(right_vec, rand_32); /* indices for arr */

        v[i] = vtype::template i64gather<sizeof(type_t)>(indices, arr);
    }

    /* median network for 9 elements */
    COEX<vtype>(v[0], v[1]);
    COEX<vtype>(v[2], v[3]);
    COEX<vtype>(v[4], v[5]);
    COEX<vtype>(v[6], v[7]);
    COEX<vtype>(v[0], v[2]);
    COEX<vtype>(v[1], v[3]);
    COEX<vtype>(v[4], v[6]);
    COEX<vtype>(v[5], v[7]);
    COEX<vtype>(v[0], v[4]);
    COEX<vtype>(v[1], v[2]);
    COEX<vtype>(v[5], v[6]);
    COEX<vtype>(v[3], v[7]);
    COEX<vtype>(v[1], v[5]);
    COEX<vtype>(v[2], v[6]);
    COEX<vtype>(v[3], v[5]);
    COEX<vtype>(v[2], v[4]);
    COEX<vtype>(v[3], v[4]);
    COEX<vtype>(v[3], v[8]);
    COEX<vtype>(v[4], v[8]);

    // technically v[4] needs to be sorted before we pick the correct median,
    // picking the 4th element works just as well for performance
    type_t *temp = (type_t *)&v[4];

    return temp[4];
}

/*
 * Partition one ZMM register based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype, typename type_t, typename zmm_t>
static inline npy_int
partition_vec(type_t *arr, npy_intp left, npy_intp right, const zmm_t curr_vec,
              const zmm_t pivot_vec, zmm_t *smallest_vec, zmm_t *biggest_vec)
{
    /* which elements are larger than the pivot */
    __mmask16 gt_mask = vtype::ge(curr_vec, pivot_vec);
    npy_int amount_gt_pivot = _mm_popcnt_u32((npy_int)gt_mask);
    vtype::mask_compressstoreu(arr + left, _mm512_knot(gt_mask), curr_vec);
    vtype::mask_compressstoreu(arr + right - amount_gt_pivot, gt_mask,
                               curr_vec);
    *smallest_vec = vtype::min(curr_vec, *smallest_vec);
    *biggest_vec = vtype::max(curr_vec, *biggest_vec);
    return amount_gt_pivot;
}

/*
 * Partition an array based on the pivot and returns the index of the
 * last element that is less than equal to the pivot.
 */
template <typename vtype, typename type_t>
static inline npy_intp
partition_avx512(type_t *arr, npy_intp left, npy_intp right, type_t pivot,
                 type_t *smallest, type_t *biggest)
{
    /* make array length divisible by 16 , shortening the array */
    for (npy_int i = (right - left) % 16; i > 0; --i) {
        *smallest = MIN(*smallest, arr[left]);
        *biggest = MAX(*biggest, arr[left]);
        if (arr[left] > pivot) {
            swap(arr, left, --right);
        }
        else {
            ++left;
        }
    }

    if (left == right)
        return left; /* less than 16 elements in the array */

    using zmm_t = typename vtype::zmm_t;
    zmm_t pivot_vec = vtype::set1(pivot);
    zmm_t min_vec = vtype::set1(*smallest);
    zmm_t max_vec = vtype::set1(*biggest);

    if (right - left == 16) {
        zmm_t vec = vtype::loadu(arr + left);
        npy_int amount_gt_pivot = partition_vec<vtype>(
                arr, left, left + 16, vec, pivot_vec, &min_vec, &max_vec);
        *smallest = vtype::reducemin(min_vec);
        *biggest = vtype::reducemax(max_vec);
        return left + (16 - amount_gt_pivot);
    }

    // first and last 16 values are partitioned at the end
    zmm_t vec_left = vtype::loadu(arr + left);
    zmm_t vec_right = vtype::loadu(arr + (right - 16));
    // store points of the vectors
    npy_intp r_store = right - 16;
    npy_intp l_store = left;
    // indices for loading the elements
    left += 16;
    right -= 16;
    while (right - left != 0) {
        zmm_t curr_vec;
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if ((r_store + 16) - right < left - l_store) {
            right -= 16;
            curr_vec = vtype::loadu(arr + right);
        }
        else {
            curr_vec = vtype::loadu(arr + left);
            left += 16;
        }
        // partition the current vector and save it on both sides of the array
        npy_int amount_gt_pivot =
                partition_vec<vtype>(arr, l_store, r_store + 16, curr_vec,
                                     pivot_vec, &min_vec, &max_vec);
        ;
        r_store -= amount_gt_pivot;
        l_store += (16 - amount_gt_pivot);
    }

    /* partition and save vec_left and vec_right */
    npy_int amount_gt_pivot =
            partition_vec<vtype>(arr, l_store, r_store + 16, vec_left,
                                 pivot_vec, &min_vec, &max_vec);
    l_store += (16 - amount_gt_pivot);
    amount_gt_pivot =
            partition_vec<vtype>(arr, l_store, l_store + 16, vec_right,
                                 pivot_vec, &min_vec, &max_vec);
    l_store += (16 - amount_gt_pivot);
    *smallest = vtype::reducemin(min_vec);
    *biggest = vtype::reducemax(max_vec);
    return l_store;
}

template <typename vtype, typename type_t>
static inline void
qsort_(type_t *arr, npy_intp left, npy_intp right, npy_int max_iters)
{
    /*
     * Resort to heapsort if quicksort isn't making any progress
     */
    if (max_iters <= 0) {
        heapsort_<typename vtype::tag>(arr + left, right + 1 - left);
        return;
    }
    /*
     * Base case: use bitonic networks to sort arrays <= 128
     */
    if (right + 1 - left <= 128) {
        sort_128<vtype>(arr + left, (npy_int)(right + 1 - left));
        return;
    }

    type_t pivot = get_pivot<vtype>(arr, left, right);
    type_t smallest = vtype::type_max();
    type_t biggest = vtype::type_min();
    npy_intp pivot_index = partition_avx512<vtype>(arr, left, right + 1, pivot,
                                                   &smallest, &biggest);
    if (pivot != smallest)
        qsort_<vtype>(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qsort_<vtype>(arr, pivot_index, right, max_iters - 1);
}

static inline npy_intp
replace_nan_with_inf(npy_float *arr, npy_intp arrsize)
{
    npy_intp nan_count = 0;
    __mmask16 loadmask = 0xFFFF;
    while (arrsize > 0) {
        if (arrsize < 16) {
            loadmask = (0x0001 << arrsize) - 0x0001;
        }
        __m512 in_zmm = _mm512_maskz_loadu_ps(loadmask, arr);
        __mmask16 nanmask = _mm512_cmp_ps_mask(in_zmm, in_zmm, _CMP_NEQ_UQ);
        nan_count += _mm_popcnt_u32((npy_int)nanmask);
        _mm512_mask_storeu_ps(arr, nanmask, ZMM_MAX_FLOAT);
        arr += 16;
        arrsize -= 16;
    }
    return nan_count;
}

static inline void
replace_inf_with_nan(npy_float *arr, npy_intp arrsize, npy_intp nan_count)
{
    for (npy_intp ii = arrsize - 1; nan_count > 0; --ii) {
        arr[ii] = NPY_NANF;
        nan_count -= 1;
    }
}

/***************************************
 * C > C++ dispatch
 ***************************************/

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_int)(void *arr, npy_intp arrsize)
{
    if (arrsize > 1) {
        qsort_<vector<npy_int>, npy_int>((npy_int *)arr, 0, arrsize - 1,
                                         2 * (npy_int)log2(arrsize));
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_uint)(void *arr, npy_intp arrsize)
{
    if (arrsize > 1) {
        qsort_<vector<npy_uint>, npy_uint>((npy_uint *)arr, 0, arrsize - 1,
                                           2 * (npy_int)log2(arrsize));
    }
}

NPY_NO_EXPORT void
NPY_CPU_DISPATCH_CURFX(x86_quicksort_float)(void *arr, npy_intp arrsize)
{
    if (arrsize > 1) {
        npy_intp nan_count = replace_nan_with_inf((npy_float *)arr, arrsize);
        qsort_<vector<npy_float>, npy_float>((npy_float *)arr, 0, arrsize - 1,
                                             2 * (npy_int)log2(arrsize));
        replace_inf_with_nan((npy_float *)arr, arrsize, nan_count);
    }
}

#endif  // NPY_HAVE_AVX512_SKX
