/* -*- c -*- */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"

#include "npy_binsearch.h"
#include "npy_sort.h"
#include "numpy_tag.hpp"

#include <array>
#include <functional>  // for std::less and std::less_equal
#include <type_traits>

// Enumerators for the variant of binsearch
enum arg_t
{
    noarg,
    arg
};
enum side_t
{
    left,
    right
};

// Mapping from enumerators to comparators
template <class Tag, side_t side>
struct side_to_cmp;

template <class Tag>
struct side_to_cmp<Tag, left> {
    static constexpr auto value = Tag::less;
};

template <class Tag>
struct side_to_cmp<Tag, right> {
    static constexpr auto value = Tag::less_equal;
};

template <side_t side>
struct side_to_generic_cmp;

template <>
struct side_to_generic_cmp<left> {
    using type = std::less<int>;
};

template <>
struct side_to_generic_cmp<right> {
    using type = std::less_equal<int>;
};

/*
 *****************************************************************************
 **                            NUMERIC SEARCHES                             **
 *****************************************************************************
 */
/*
 * Plain (non NaN-aware) comparators, used for floating point types when
 * neither the haystack nor the current block of keys contains NaNs. In that
 * case they give exactly the same answers as Tag::less / Tag::less_equal.
 */
template <class T, side_t side>
struct plain_cmp;

template <class T>
struct plain_cmp<T, left> {
    static inline int value(T a, T b) { return a < b; }
};

template <class T>
struct plain_cmp<T, right> {
    static inline int value(T a, T b) { return a <= b; }
};

/*
 * Batched binary search over one block of keys (see the comment in
 * `binsearch` below). The keys and the running bases are kept in small
 * local buffers so every pass of the batched search stays in L1 cache.
 */
template <class T, class Cmp>
static inline void
binsearch_block(const char *arr, npy_intp arr_len, npy_intp arr_str,
                const T *keys, npy_intp *bases, npy_intp nkeys, Cmp cmp)
{
    npy_intp interval_length = arr_len;
    npy_intp half = interval_length >> 1;
    interval_length -= half;  // length -> ceil(length / 2)

    const T first_mid = *(const T *)(arr + half * arr_str);
    for (npy_intp i = 0; i < nkeys; ++i) {
        bases[i] = cmp(first_mid, keys[i]) * half;
    }

    if (arr_str == (npy_intp)sizeof(T)) {
        const T *a = (const T *)arr;
        while (interval_length > 1) {
            npy_intp h = interval_length >> 1;
            interval_length -= h;
            for (npy_intp i = 0; i < nkeys; ++i) {
                const npy_intp b = bases[i];
                bases[i] = cmp(a[b + h], keys[i]) ? b + h : b;
            }
        }
        for (npy_intp i = 0; i < nkeys; ++i) {
            bases[i] += cmp(a[bases[i]], keys[i]);
        }
    }
    else {
        while (interval_length > 1) {
            npy_intp h = interval_length >> 1;
            interval_length -= h;
            for (npy_intp i = 0; i < nkeys; ++i) {
                const npy_intp b = bases[i];
                const T mid_val = *(const T *)(arr + (b + h) * arr_str);
                bases[i] = cmp(mid_val, keys[i]) ? b + h : b;
            }
        }
        for (npy_intp i = 0; i < nkeys; ++i) {
            bases[i] += cmp(*(const T *)(arr + bases[i] * arr_str), keys[i]);
        }
    }
}

template <class Tag, side_t side>
static void
binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
          npy_intp key_len, npy_intp arr_str, npy_intp key_str,
          npy_intp ret_str, PyArrayObject *, PyArrayObject *,
          PyArray_BinSearchCompareFunc *)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;

    // If the array length is 0 we return all 0s
    if (arr_len <= 0) {
        for (npy_intp i = 0; i < key_len; ++i) {
            *(npy_intp *)(ret + i * ret_str) = 0;
        }
        return;
    }

    /*
    In this binary search implementation, the candidate insertion indices for
    the j-th key are in the range [base_j, base_j+length] and on each
    iteration we pick a pivot at the mid-point of the range to compare against
    the j-th key. Depending on the comparison result, we adjust the base_j and
    halve the length of the interval.

    To batch multiple queries, we process all bases of a block of keys with a
    fixed length. The length is halved on each iteration of an outer loop and
    all bases of the block are updated in an inner loop. Blocks are small
    enough that keys and bases stay in L1 cache for all passes.

    There are two benefits of this approach:

    1. Cache locallity of pivots. In early iterations each key is compared
    against the same set of pivots. For example, in the first iteration all
    keys are compared against the median. In the second iteration, all keys
    end up being compared against 1st and 3rd quartiles.

    2. Independent calculations for out-of-order execution. In the single-key
    version, step i+1 depends on computation of step i. Meaning that step i+1
    must wait for step i to complete before proceeding. When batching multiple
    keys, we compute each step for all keys before continuing on the next
    step. All the computations at a given step are independent across
    different keys. Meaning that the CPU can execute multiple keys
    out-of-order in parallel.

    Invariant (for every j):
    - cmp(arr[i], key_val_j) == true  for all i < base_j
    - cmp(arr[i], key_val_j) == false for all i >= base_j + length

    where cmp(a, b) operator depends on side input:
    - For side = "left", cmp operator is <
    - For side = "right", cmp operator is <=

    The insertion index candidates are in range [base, base+length] and
    on each iteration we shrink the range into either
        [base, ceil(length / 2)]
    or
        [base + floor(length / 2), ceil(length / 2)]

    The outer loop terminates when length = 1. At that point, for each j
    the insertion order is either base_j or base_j + 1. An additional
    comparison is required to determine which of the two values.
    If cmp(arr[base_j], key_val_j) == true, insertion index is base_j + 1.
    Otherwise the insertion order is base_j.

    For floating point types, when neither the haystack nor a block of keys
    contains NaN, the NaN-aware comparison reduces to the plain one, which
    is cheaper and branch free.
    */
    constexpr npy_intp BLOCK = 256;
    T keys[BLOCK];
    npy_intp bases[BLOCK];

    constexpr bool is_float = std::is_floating_point<T>::value;
    /*
     * Checking the haystack for NaNs is O(arr_len), so only do it when the
     * searches themselves cost at least as much (many keys relative to the
     * haystack size). Otherwise keep the NaN-aware comparison, which gives
     * the same answers; this keeps few-key searches O(log arr_len).
     */
    bool arr_has_nan = true;
    if constexpr (is_float) {
        if (arr_len <= 8 * key_len) {
            arr_has_nan = false;
        }
        for (npy_intp i = 0; !arr_has_nan && i < arr_len; ++i) {
            const T v = *(const T *)(arr + i * arr_str);
            if (v != v) {
                arr_has_nan = true;
                break;
            }
        }
    }

    for (npy_intp start = 0; start < key_len; start += BLOCK) {
        const npy_intp nkeys =
                (key_len - start < BLOCK) ? key_len - start : BLOCK;
        const char *kp = key + start * key_str;
        bool block_has_nan = false;
        if constexpr (is_float) {
            int nan_count = 0;
            for (npy_intp i = 0; i < nkeys; ++i) {
                const T v = *(const T *)(kp + i * key_str);
                keys[i] = v;
                nan_count |= (v != v);
            }
            block_has_nan = nan_count != 0;
        }
        else {
            for (npy_intp i = 0; i < nkeys; ++i) {
                keys[i] = *(const T *)(kp + i * key_str);
            }
        }

        bool done = false;
        if constexpr (is_float) {
            if (!arr_has_nan && !block_has_nan) {
                binsearch_block<T>(arr, arr_len, arr_str, keys, bases, nkeys,
                                   plain_cmp<T, side>::value);
                done = true;
            }
        }
        if (!done) {
            binsearch_block<T>(arr, arr_len, arr_str, keys, bases, nkeys,
                               cmp);
        }

        char *rp = ret + start * ret_str;
        for (npy_intp i = 0; i < nkeys; ++i) {
            *(npy_intp *)(rp + i * ret_str) = bases[i];
        }
    }
}

template <class Tag, side_t side>
static int
argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
             npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
             npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
             PyArrayObject *, PyArrayObject *, PyArray_BinSearchCompareFunc *)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;

    // If the array length is 0 we return all 0s
    if (arr_len <= 0) {
        for (npy_intp i = 0; i < key_len; ++i) {
            *(npy_intp *)(ret + i * ret_str) = 0;
        }
        return 0;
    }

    npy_intp interval_length = arr_len;
    npy_intp half = interval_length >> 1;
    interval_length -= half; // length -> ceil(length / 2)

    npy_intp base = 0;
    npy_intp mid_idx = *(npy_intp *)(sort + (base + half) * sort_str);
    if (mid_idx < 0 || mid_idx >= arr_len) {
        return -1;
    }
    const T mid_val = *(const T *)(arr + mid_idx * arr_str);

    for (npy_intp i = 0; i < key_len; ++i) {
        const T key_val = *(const T *)(key + i * key_str);
        *(npy_intp *)(ret + i * ret_str) = cmp(mid_val, key_val) * half;
    }

    while (interval_length > 1) {
        npy_intp half = interval_length >> 1;
        interval_length -= half; // length -> ceil(length / 2)

        for (npy_intp i = 0; i < key_len; ++i) {
            npy_intp &base = *(npy_intp *)(ret + i * ret_str);
            npy_intp mid_idx = *(npy_intp *)(sort + (base + half) * sort_str);
            if (mid_idx < 0 || mid_idx >= arr_len) {
                return -1;
            }
            const T mid_val = *(const T *)(arr + mid_idx * arr_str);
            const T key_val = *(const T *)(key + i * key_str);
            base += cmp(mid_val, key_val) * half;
        }
    }

    for (npy_intp i = 0; i < key_len; ++i) {
        npy_intp &base = *(npy_intp *)(ret + i * ret_str);
        npy_intp mid_idx = *(npy_intp *)(sort + base * sort_str);
        if (mid_idx < 0 || mid_idx >= arr_len) {
            return -1;
        }
        const T key_val = *(const T *)(key + i * key_str);
        base += cmp(*(const T *)(arr + mid_idx * arr_str), key_val);
    }
    return 0;
}

/*
 *****************************************************************************
 **                             GENERIC SEARCH                              **
 *****************************************************************************
 */

template <side_t side>
static void
npy_binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
              npy_intp key_len, npy_intp arr_str, npy_intp key_str,
              npy_intp ret_str, PyArrayObject *key_arr, PyArrayObject *arr_arr,
              PyArray_BinSearchCompareFunc *compare)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        /* last_key and key are both elements of the key array */
        if (Cmp{}(compare(last_key, key, key_arr, key_arr), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const char *arr_ptr = arr + mid_idx * arr_str;

            /* arr_ptr belongs to the haystack, key to the key array */
            if (Cmp{}(compare(arr_ptr, key, arr_arr, key_arr), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
}

template <side_t side>
static int
npy_argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
                 npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
                 npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
                 PyArrayObject *key_arr, PyArrayObject *arr_arr,
                 PyArray_BinSearchCompareFunc *compare)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        /* last_key and key are both elements of the key array */
        if (Cmp{}(compare(last_key, key, key_arr, key_arr), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const npy_intp sort_idx = *(npy_intp *)(sort + mid_idx * sort_str);
            const char *arr_ptr;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            arr_ptr = arr + sort_idx * arr_str;

            /* arr_ptr belongs to the haystack, key to the key array */
            if (Cmp{}(compare(arr_ptr, key, arr_arr, key_arr), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
    return 0;
}

/*
 *****************************************************************************
 **                             GENERATOR                                   **
 *****************************************************************************
 */

template <arg_t arg>
struct binsearch_base;

template <>
struct binsearch_base<arg> {
    using function_type = PyArray_ArgBinSearchFunc *;
    struct value_type {
        int typenum;
        function_type binsearch[NPY_NSEARCHSIDES];
    };
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&argbinsearch<Tags, left>,
                            (function_type)argbinsearch<Tags, right>}}...};
    }
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_argbinsearch<left>,
            (function_type)&npy_argbinsearch<right>};
};
constexpr std::array<binsearch_base<arg>::function_type, 2>
        binsearch_base<arg>::npy_map;

template <>
struct binsearch_base<noarg> {
    using function_type = PyArray_BinSearchFunc *;
    struct value_type {
        int typenum;
        function_type binsearch[NPY_NSEARCHSIDES];
    };
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&binsearch<Tags, left>,
                            (function_type)binsearch<Tags, right>}}...};
    }
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_binsearch<left>,
            (function_type)&npy_binsearch<right>};
};
constexpr std::array<binsearch_base<noarg>::function_type, 2>
        binsearch_base<noarg>::npy_map;

// Handle generation of all binsearch variants
template <arg_t arg>
struct binsearch_t : binsearch_base<arg> {
    using binsearch_base<arg>::make_binsearch_map;
    using value_type = typename binsearch_base<arg>::value_type;

    using taglist = npy::taglist<
            /* If adding new types, make sure to keep them ordered by type num
             */
            npy::bool_tag, npy::byte_tag, npy::ubyte_tag, npy::short_tag,
            npy::ushort_tag, npy::int_tag, npy::uint_tag, npy::long_tag,
            npy::ulong_tag, npy::longlong_tag, npy::ulonglong_tag,
            npy::float_tag, npy::double_tag, npy::longdouble_tag, 
            npy::cfloat_tag, npy::cdouble_tag, npy::clongdouble_tag, 
            npy::datetime_tag, npy::timedelta_tag, npy::half_tag>;

    static constexpr std::array<value_type, taglist::size> map =
            make_binsearch_map(taglist());
};

template <arg_t arg>
constexpr std::array<typename binsearch_t<arg>::value_type,
                     binsearch_t<arg>::taglist::size>
        binsearch_t<arg>::map;

template <arg_t arg>
static inline typename binsearch_t<arg>::function_type
_get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    using binsearch = binsearch_t<arg>;
    npy_intp nfuncs = binsearch::map.size();
    npy_intp min_idx = 0;
    npy_intp max_idx = nfuncs;
    int type = dtype->type_num;

    if ((int)side >= (int)NPY_NSEARCHSIDES) {
        return NULL;
    }

    /*
     * It seems only fair that a binary search function be searched for
     * using a binary search...
     */
    while (min_idx < max_idx) {
        npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);

        if (binsearch::map[mid_idx].typenum < type) {
            min_idx = mid_idx + 1;
        }
        else {
            max_idx = mid_idx;
        }
    }

    if (min_idx < nfuncs && binsearch::map[min_idx].typenum == type) {
        return binsearch::map[min_idx].binsearch[side];
    }

    if (PyDataType_GetArrFuncs(dtype)->compare) {
        return binsearch::npy_map[side];
    }

    return NULL;
}

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */
extern "C" {
NPY_NO_EXPORT PyArray_BinSearchFunc *
get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return _get_binsearch_func<noarg>(dtype, side);
}

NPY_NO_EXPORT PyArray_ArgBinSearchFunc *
get_argbinsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return _get_binsearch_func<arg>(dtype, side);
}
}
