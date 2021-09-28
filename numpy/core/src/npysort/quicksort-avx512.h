#if defined HAVE_ATTRIBUTE_TARGET_AVX512_SKX_WITH_INTRINSICS
#include <immintrin.h>
#include "numpy/npy_math.h"

#define NETWORK1 14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1
#define NETWORK2 12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
#define NETWORK3 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
#define NETWORK4 13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2
#define NETWORK5 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 
#define NETWORK6 11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4
#define NETWORK7 7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8
#define ZMM_INF _mm512_set1_ps(NPY_INFINITYF)

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define COEX_ZMM(a, b) {                                                    \
    __m512 temp = a;                                                        \
    a = _mm512_min_ps(a,b);                                                 \
    b = _mm512_max_ps(temp, b);}                                            \

#define COEX_YMM(a, b){                                                     \
    __m256 temp = a;                                                        \
    a = _mm256_min_ps(a, b);                                                \
    b = _mm256_max_ps(temp, b);}

#define SHUFFLE_MASK(a,b,c,d) (a << 6) | (b << 4) | (c << 2) | d
    
static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
__m512 cmp_merge(__m512 in1, __m512 in2, __mmask16 mask)
{
    __m512 min = _mm512_min_ps(in2, in1);
    __m512 max = _mm512_max_ps(in2, in1);
    return _mm512_mask_mov_ps(min, mask, max); // 0 -> min, 1 -> max
}

// Assumes zmm is random and performs a full sorting network
static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
__m512 sort_zmm(__m512 zmm)
{
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(2,3,0,1)), 0xAAAA);
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(0,1,2,3)), 0xCCCC);
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(2,3,0,1)), 0xAAAA);
    zmm = cmp_merge(zmm, _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK3),zmm), 0xF0F0);
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(1,0,3,2)), 0xCCCC);
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(2,3,0,1)), 0xAAAA);
    zmm = cmp_merge(zmm, _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5),zmm), 0xFF00);
    zmm = cmp_merge(zmm, _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK6),zmm), 0xF0F0);
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(1,0,3,2)), 0xCCCC);
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(2,3,0,1)), 0xAAAA);
    return zmm;
}

// Assumes zmm is bitonic and performs a recursive half cleaner
static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
__m512 bitonic_merge_zmm(__m512 zmm)
{
    // 1) half_cleaner[16]: compare 1-9, 2-10, 3-11 etc ..
    zmm = cmp_merge(zmm, _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK7), zmm), 0xFF00);
    // 2) half_cleaner[8]: compare 1-5, 2-6, 3-7 etc ..
    zmm = cmp_merge(zmm, _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK6),zmm), 0xF0F0);
    // 3) half_cleaner[4]
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(1,0,3,2)), 0xCCCC);
    // 3) half_cleaner[1]
    zmm = cmp_merge(zmm, _mm512_shuffle_ps(zmm, zmm, SHUFFLE_MASK(2,3,0,1)), 0xAAAA);
    return zmm;
}

// Assumes zmm1 and zmm2 are sorted and performs a recursive half cleaner
static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void bitonic_merge_two_zmm(__m512* zmm1, __m512* zmm2)
{
    // 1) First step of a merging network: coex of zmm1 and zmm2 reversed
    *zmm2 = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), *zmm2);
    __m512 zmm3 = _mm512_min_ps(*zmm1, *zmm2);
    __m512 zmm4 = _mm512_max_ps(*zmm1, *zmm2);
    // 2) Recursive half cleaner for each
    *zmm1 = bitonic_merge_zmm(zmm3);
    *zmm2 = bitonic_merge_zmm(zmm4);
}

// Assumes [zmm0, zmm1] and [zmm2, zmm3] are sorted and performs a recursive half cleaner
static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void bitonic_merge_four_zmm(__m512* zmm)
{
    __m512 zmm2r = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), zmm[2]);
    __m512 zmm3r = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), zmm[3]);
    __m512 zmm_t1 = _mm512_min_ps(zmm[0], zmm3r);
    __m512 zmm_t2 = _mm512_min_ps(zmm[1], zmm2r);
    __m512 zmm_t3 = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), _mm512_max_ps(zmm[1], zmm2r));
    __m512 zmm_t4 = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), _mm512_max_ps(zmm[0], zmm3r));
    __m512 zmm0 = _mm512_min_ps(zmm_t1, zmm_t2);
    __m512 zmm1 = _mm512_max_ps(zmm_t1, zmm_t2);
    __m512 zmm2 = _mm512_min_ps(zmm_t3, zmm_t4);
    __m512 zmm3 = _mm512_max_ps(zmm_t3, zmm_t4);
    zmm[0] = bitonic_merge_zmm(zmm0);
    zmm[1] = bitonic_merge_zmm(zmm1);
    zmm[2] = bitonic_merge_zmm(zmm2);
    zmm[3] = bitonic_merge_zmm(zmm3);
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void bitonic_merge_eight_zmm(__m512* zmm)
{
    __m512 zmm4r = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), zmm[4]);
    __m512 zmm5r = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), zmm[5]);
    __m512 zmm6r = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), zmm[6]);
    __m512 zmm7r = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), zmm[7]);
    __m512 zmm_t1 = _mm512_min_ps(zmm[0], zmm7r);
    __m512 zmm_t2 = _mm512_min_ps(zmm[1], zmm6r);
    __m512 zmm_t3 = _mm512_min_ps(zmm[2], zmm5r);
    __m512 zmm_t4 = _mm512_min_ps(zmm[3], zmm4r);
    __m512 zmm_t5 = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), _mm512_max_ps(zmm[3], zmm4r));
    __m512 zmm_t6 = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), _mm512_max_ps(zmm[2], zmm5r));
    __m512 zmm_t7 = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), _mm512_max_ps(zmm[1], zmm6r));
    __m512 zmm_t8 = _mm512_permutexvar_ps(_mm512_set_epi32(NETWORK5), _mm512_max_ps(zmm[0], zmm7r));
    COEX_ZMM(zmm_t1, zmm_t3);
    COEX_ZMM(zmm_t2, zmm_t4);
    COEX_ZMM(zmm_t5, zmm_t7);
    COEX_ZMM(zmm_t6, zmm_t8);
    COEX_ZMM(zmm_t1, zmm_t2);
    COEX_ZMM(zmm_t3, zmm_t4);
    COEX_ZMM(zmm_t5, zmm_t6);
    COEX_ZMM(zmm_t7, zmm_t8);
    zmm[0] = bitonic_merge_zmm(zmm_t1);
    zmm[1] = bitonic_merge_zmm(zmm_t2);
    zmm[2] = bitonic_merge_zmm(zmm_t3);
    zmm[3] = bitonic_merge_zmm(zmm_t4);
    zmm[4] = bitonic_merge_zmm(zmm_t5);
    zmm[5] = bitonic_merge_zmm(zmm_t6);
    zmm[6] = bitonic_merge_zmm(zmm_t7);
    zmm[7] = bitonic_merge_zmm(zmm_t8);
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void sort_16(npy_float* arr, npy_int N)
{
    __mmask16 load_mask = (0x0001 << N) - 0x0001;
    __m512 zmm = _mm512_mask_loadu_ps(ZMM_INF, load_mask, arr);
    _mm512_mask_storeu_ps(arr, load_mask, sort_zmm(zmm));
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void sort_32(npy_float* arr, npy_int N)
{
    if (N <= 16) {
        sort_16(arr, N);
        return;
    }
    __m512 zmm1 = _mm512_loadu_ps(arr);
    __mmask16 load_mask = (0x0001 << (N-16)) - 0x0001;
    __m512 zmm2 = _mm512_mask_loadu_ps(ZMM_INF, load_mask, arr + 16);
    zmm1 = sort_zmm(zmm1);
    zmm2 = sort_zmm(zmm2);
    bitonic_merge_two_zmm(&zmm1, &zmm2);
    _mm512_storeu_ps(arr, zmm1);
    _mm512_mask_storeu_ps(arr + 16, load_mask, zmm2);
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void sort_64(npy_float* arr, npy_int N)
{
    if (N <= 32) {
        sort_32(arr, N);
        return;
    }
    __m512 zmm[4];
    zmm[0] = _mm512_loadu_ps(arr);
    zmm[1] = _mm512_loadu_ps(arr + 16);
    __mmask16 load_mask1 = 0xFFFF, load_mask2 = 0xFFFF;
    if (N < 48) {
        load_mask1 = (0x0001 << (N-32)) - 0x0001;
        load_mask2 = 0x0000;
    }
    else if (N < 64) {
        load_mask2 = (0x0001 << (N-48)) - 0x0001;
    }
    zmm[2] = _mm512_mask_loadu_ps(ZMM_INF, load_mask1, arr + 32);
    zmm[3] = _mm512_mask_loadu_ps(ZMM_INF, load_mask2, arr + 48);
    zmm[0] = sort_zmm(zmm[0]);
    zmm[1] = sort_zmm(zmm[1]);
    zmm[2] = sort_zmm(zmm[2]);
    zmm[3] = sort_zmm(zmm[3]);
    bitonic_merge_two_zmm(&zmm[0], &zmm[1]);
    bitonic_merge_two_zmm(&zmm[2], &zmm[3]);
    bitonic_merge_four_zmm(zmm);
    _mm512_storeu_ps(arr, zmm[0]);
    _mm512_storeu_ps(arr + 16, zmm[1]);
    _mm512_mask_storeu_ps(arr + 32, load_mask1, zmm[2]);
    _mm512_mask_storeu_ps(arr + 48, load_mask2, zmm[3]);
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void sort_128(npy_float* arr, npy_int N)
{
    if (N <= 64) {
        sort_64(arr, N);
        return;
    }
    __m512 zmm[8];
    zmm[0] = _mm512_loadu_ps(arr);
    zmm[1] = _mm512_loadu_ps(arr + 16);
    zmm[2] = _mm512_loadu_ps(arr + 32);
    zmm[3] = _mm512_loadu_ps(arr + 48);
    zmm[0] = sort_zmm(zmm[0]);
    zmm[1] = sort_zmm(zmm[1]);
    zmm[2] = sort_zmm(zmm[2]);
    zmm[3] = sort_zmm(zmm[3]);
    __mmask16 load_mask1 = 0xFFFF, load_mask2 = 0xFFFF;
    __mmask16 load_mask3 = 0xFFFF, load_mask4 = 0xFFFF;
    if (N < 80) {
        load_mask1 = (0x0001 << (N-64)) - 0x0001;
        load_mask2 = 0x0000;
        load_mask3 = 0x0000;
        load_mask4 = 0x0000;
    }
    else if (N < 96) {
        load_mask2 = (0x0001 << (N-80)) - 0x0001;
        load_mask3 = 0x0000;
        load_mask4 = 0x0000;
    }
    else if (N < 112) {
        load_mask3 = (0x0001 << (N-96)) - 0x0001;
        load_mask4 = 0x0000;
    }
    else {
        load_mask4 = (0x0001 << (N-112)) - 0x0001;
    }
    zmm[4] = _mm512_mask_loadu_ps(ZMM_INF, load_mask1, arr + 64);
    zmm[5] = _mm512_mask_loadu_ps(ZMM_INF, load_mask2, arr + 80);
    zmm[6] = _mm512_mask_loadu_ps(ZMM_INF, load_mask3, arr + 96);
    zmm[7] = _mm512_mask_loadu_ps(ZMM_INF, load_mask4, arr + 112);
    zmm[4] = sort_zmm(zmm[4]);
    zmm[5] = sort_zmm(zmm[5]);
    zmm[6] = sort_zmm(zmm[6]);
    zmm[7] = sort_zmm(zmm[7]);
    bitonic_merge_two_zmm(&zmm[0], &zmm[1]);
    bitonic_merge_two_zmm(&zmm[2], &zmm[3]);
    bitonic_merge_two_zmm(&zmm[4], &zmm[5]);
    bitonic_merge_two_zmm(&zmm[6], &zmm[7]);
    bitonic_merge_four_zmm(zmm);
    bitonic_merge_four_zmm(zmm + 4);
    bitonic_merge_eight_zmm(zmm);
    _mm512_storeu_ps(arr, zmm[0]);
    _mm512_storeu_ps(arr + 16, zmm[1]);
    _mm512_storeu_ps(arr + 32, zmm[2]);
    _mm512_storeu_ps(arr + 48, zmm[3]);
    _mm512_mask_storeu_ps(arr + 64, load_mask1, zmm[4]);
    _mm512_mask_storeu_ps(arr + 80, load_mask2, zmm[5]);
    _mm512_mask_storeu_ps(arr + 96, load_mask3, zmm[6]);
    _mm512_mask_storeu_ps(arr + 112, load_mask4, zmm[7]);
}


static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void swap(npy_float *arr, npy_intp ii, npy_intp jj) {
    npy_float temp = arr[ii];
    arr[ii] = arr[jj];
    arr[jj] = temp;
}

// Median of 3 stratergy
//static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
//npy_intp get_pivot_index(npy_float *arr, const npy_intp left, const npy_intp right) {
//    return (rand() % (right + 1 - left)) + left;
//    //npy_intp middle = ((right-left)/2) + left;
//    //npy_float a = arr[left], b = arr[middle], c = arr[right];
//    //if ((b >= a && b <= c) || (b <= a && b >= c)) 
//    //    return middle;
//    //if ((a >= b && a <= c) || (a <= b && a >= c)) 
//    //    return left;
//    //else
//    //    return right;
//}

/* vectorized random number generator xoroshiro128+ */
#define VROTL(x, k) /* rotate each uint64_t value in vector */               \
  _mm256_or_si256(_mm256_slli_epi64((x),(k)),_mm256_srli_epi64((x),64-(k)))

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
__m256i vnext(__m256i* s0, __m256i* s1) {
    *s1 = _mm256_xor_si256(*s0, *s1); /* modify vectors s1 and s0 */
    *s0 = _mm256_xor_si256(_mm256_xor_si256(VROTL(*s0, 24), *s1),
                           _mm256_slli_epi64(*s1, 16));
    *s1 = VROTL(*s1, 37);
    return _mm256_add_epi64(*s0, *s1); /* return random vector */
}

/* transform random numbers to the range between 0 and bound - 1 */
static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
__m256i rnd_epu32(__m256i rnd_vec, __m256i bound) {
    __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(rnd_vec, bound), 32);
    __m256i odd = _mm256_mul_epu32(_mm256_srli_epi64(rnd_vec, 32), bound);
    return _mm256_blend_epi32(odd, even, 0b01010101);
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
npy_float get_pivot(npy_float *arr, const npy_intp left, const npy_intp right) {
    /* seeds for vectorized random number generator */
    __m256i s0 = _mm256_setr_epi64x(8265987198341093849, 3762817312854612374,
                                    1324281658759788278, 6214952190349879213);
    __m256i s1 = _mm256_setr_epi64x(2874178529384792648, 1257248936691237653,
                                    7874578921548791257, 1998265912745817298);
    s0 = _mm256_add_epi64(s0, _mm256_set1_epi64x(left));
    s1 = _mm256_sub_epi64(s1, _mm256_set1_epi64x(right));

    npy_intp arrsize = right - left + 1;
    __m256i bound = _mm256_set1_epi32(arrsize > INT32_MAX ? INT32_MAX : arrsize);
    __m512i left_vec = _mm512_set1_epi64(left);
    __m512i right_vec = _mm512_set1_epi64(right);
    __m256 v[9];
    /* fill 9 vectors with random numbers */
    for (npy_int i = 0; i < 9; ++i) {
        __m256i rand_64 = vnext(&s0, &s1); /* vector with 4 random uint64_t */
        __m512i rand_32 = _mm512_cvtepi32_epi64(rnd_epu32(rand_64, bound)); /* random numbers between 0 and bound - 1 */
        __m512i indices;
        if (i < 5)
            indices = _mm512_add_epi64(left_vec, rand_32); /* indices for arr */
        else 
            indices = _mm512_sub_epi64(right_vec, rand_32); /* indices for arr */

        v[i] = _mm512_i64gather_ps(indices, arr, sizeof(npy_float));
    }

    /* median network for 9 elements */
    COEX_YMM(v[0], v[1]); COEX_YMM(v[2], v[3]); /* step 1 */
    COEX_YMM(v[4], v[5]); COEX_YMM(v[6], v[7]);
    COEX_YMM(v[0], v[2]); COEX_YMM(v[1], v[3]); /* step 2 */
    COEX_YMM(v[4], v[6]); COEX_YMM(v[5], v[7]);
    COEX_YMM(v[0], v[4]); COEX_YMM(v[1], v[2]); /* step 3 */
    COEX_YMM(v[5], v[6]); COEX_YMM(v[3], v[7]);
    COEX_YMM(v[1], v[5]); COEX_YMM(v[2], v[6]); /* step 4 */
    COEX_YMM(v[3], v[5]); COEX_YMM(v[2], v[4]); /* step 5 */
    COEX_YMM(v[3], v[4]);                      /* step 6 */
    COEX_YMM(v[3], v[8]);                      /* step 7 */
    COEX_YMM(v[4], v[8]);                      /* step 8 */

    npy_float* temp = (npy_float*) &v[4];

    return temp[4]; 
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
npy_int partition_vec(npy_float* arr, npy_intp left, npy_intp right,
                      const __m512 curr_vec, const __m512 pivot_vec,
                      __m512* smallest_vec, __m512* biggest_vec)
{
    /* which elements are larger than the pivot */
    __mmask16 gt_mask = _mm512_cmp_ps_mask(curr_vec, pivot_vec, _CMP_GT_OQ);
    npy_int amount_gt_pivot = _mm_popcnt_u32((npy_int)gt_mask);
    _mm512_mask_compressstoreu_ps(arr + left, _knot_mask16(gt_mask), curr_vec);
    _mm512_mask_compressstoreu_ps(arr + right - amount_gt_pivot, gt_mask, curr_vec);
    *smallest_vec = _mm512_min_ps(curr_vec, *smallest_vec);
    *biggest_vec = _mm512_max_ps(curr_vec, *biggest_vec);
    return amount_gt_pivot;
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
npy_intp partition_avx512(npy_float* arr, npy_intp left, npy_intp right,
                          npy_float pivot, npy_float* smallest, npy_float* biggest)
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

    if(left == right) 
      return left; /* less than 16 elements in the array */

    __m512 pivot_vec = _mm512_set1_ps(pivot);
    __m512 min_vec = _mm512_set1_ps(*smallest);
    __m512 max_vec = _mm512_set1_ps(*biggest);

    if(right - left == 16) {
        __m512 vec = _mm512_loadu_ps(arr + left);
        npy_int amount_gt_pivot = partition_vec(arr, left, left + 16, vec, pivot_vec, &min_vec, &max_vec);
        *smallest = _mm512_reduce_min_ps(min_vec);
        *biggest = _mm512_reduce_max_ps(max_vec);
        return left + (16 - amount_gt_pivot);
    }

    // first and last 16 values are partitioned at the end
    __m512 vec_left = _mm512_loadu_ps(arr + left);
    __m512 vec_right = _mm512_loadu_ps(arr + (right - 16));
    // store points of the vectors
    npy_intp r_store = right - 16;
    npy_intp l_store = left;
    // indices for loading the elements
    left += 16;
    right -= 16;
    while(right - left != 0) {
        __m512 curr_vec;
        /*
         * if fewer elements are stored on the right side of the array,
         * then next elements are loaded from the right side,
         * otherwise from the left side
         */
        if((r_store + 16) - right < left - l_store) {
            right -= 16;
            curr_vec = _mm512_loadu_ps(arr + right);
        }
        else {
            curr_vec = _mm512_loadu_ps(arr + left);
            left += 16;
        }
        // partition the current vector and save it on both sides of the array
        npy_int amount_gt_pivot = partition_vec(arr, l_store, r_store + 16, curr_vec, pivot_vec, &min_vec, &max_vec);;
        r_store -= amount_gt_pivot; l_store += (16 - amount_gt_pivot);
    }

    /* partition and save vec_left and vec_right */
    npy_int amount_gt_pivot = partition_vec(arr, l_store, r_store + 16, vec_left, pivot_vec, &min_vec, &max_vec);
    l_store += (16 - amount_gt_pivot);
    amount_gt_pivot = partition_vec(arr, l_store, l_store + 16, vec_right, pivot_vec, &min_vec, &max_vec);
    l_store += (16 - amount_gt_pivot);
    *smallest = _mm512_reduce_min_ps(min_vec);
    *biggest = _mm512_reduce_max_ps(max_vec);
    return l_store;
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void qs_sort(npy_float* arr, npy_intp left, npy_intp right, npy_int max_iters)
{
    if (max_iters <= 0) {
        heapsort_float((void*)(arr + left), right + 1 - left, NULL);
        return;
    }
    if (right + 1 - left <= 128) {
        sort_128(arr + left, right + 1 -left);
        return;
    }
    
    npy_float pivot = get_pivot(arr, left, right);
    npy_float smallest = NPY_INFINITYF;
    npy_float biggest = -NPY_INFINITYF;
    npy_intp pivot_index = partition_avx512(arr, left, right+1, pivot, &smallest, &biggest);
    if (pivot != smallest)
        qs_sort(arr, left, pivot_index - 1, max_iters - 1);
    if (pivot != biggest)
        qs_sort(arr, pivot_index, right, max_iters - 1);
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
npy_intp replace_nan_with_inf(npy_float* arr, npy_intp arrsize)
{
    npy_intp nan_count = 0;
    __mmask16 loadmask = 0xFFFF;
    while (arrsize > 0) {
        if (arrsize < 16) {
            loadmask = (0x0001 << arrsize) - 0x0001;
        }
        __m512 in_zmm = _mm512_maskz_loadu_ps(loadmask, arr);
        __mmask16 nanmask = _mm512_cmpunord_ps_mask(in_zmm, in_zmm);
        nan_count += _mm_popcnt_u32((npy_int) nanmask);
        _mm512_mask_storeu_ps(arr, nanmask, ZMM_INF);
        arr += 16;
        arrsize -= 16;
    }
    return nan_count;
}

static NPY_INLINE NPY_GCC_OPT_3 NPY_GCC_TARGET_AVX512_SKX
void replace_inf_with_nan(npy_float* arr, npy_intp arrsize, npy_intp nan_count)
{
    for (npy_intp ii = arrsize-1; nan_count > 0; --ii) {
        arr[ii] = NPY_NANF;
        nan_count -= 1;
    }
}

void avx512_qs_sort_float(npy_float* arr, npy_intp arrsize)
{
    if (arrsize > 1) {
        npy_intp nan_count = replace_nan_with_inf(arr, arrsize);
        qs_sort(arr, 0, arrsize-1, 2*log2(arrsize));
        replace_inf_with_nan(arr, arrsize, nan_count);
    }
}
#endif // HAVE_ATTRIBUTE_TARGET_AVX512_SKX_WITH_INTRINSICS
