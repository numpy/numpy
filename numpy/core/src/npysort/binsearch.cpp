/* -*- c -*- */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"

#include "npy_binsearch.h"
#include "npy_sort.h"
#include "numpy_tag.h"

#include <array>
#include <functional>  // for std::less and std::less_equal

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
template <class Tag, side_t side>
static void
binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
          npy_intp key_len, npy_intp arr_str, npy_intp key_str,
          npy_intp ret_str, PyArrayObject *)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    T last_key_val;

    if (key_len == 0) {
        return;
    }
    last_key_val = *(const T *)key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        const T key_val = *(const T *)key;
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (cmp(last_key_val, key_val)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key_val = key_val;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const T mid_val = *(const T *)(arr + mid_idx * arr_str);
            if (cmp(mid_val, key_val)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
}

template <class Tag, side_t side>
static int
argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
             npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
             npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
             PyArrayObject *)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    T last_key_val;

    if (key_len == 0) {
        return 0;
    }
    last_key_val = *(const T *)key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        const T key_val = *(const T *)key;
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (cmp(last_key_val, key_val)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key_val = key_val;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const npy_intp sort_idx = *(npy_intp *)(sort + mid_idx * sort_str);
            T mid_val;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            mid_val = *(const T *)(arr + sort_idx * arr_str);

            if (cmp(mid_val, key_val)) {
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
 **                             GENERIC SEARCH                              **
 *****************************************************************************
 */

template <side_t side>
static void
npy_binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
              npy_intp key_len, npy_intp arr_str, npy_intp key_str,
              npy_intp ret_str, PyArrayObject *cmp)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    PyArray_CompareFunc *compare = PyArray_DESCR(cmp)->f->compare;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (Cmp{}(compare(last_key, key, cmp), 0)) {
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

            if (Cmp{}(compare(arr_ptr, key, cmp), 0)) {
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
                 PyArrayObject *cmp)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    PyArray_CompareFunc *compare = PyArray_DESCR(cmp)->f->compare;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (Cmp{}(compare(last_key, key, cmp), 0)) {
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

            if (Cmp{}(compare(arr_ptr, key, cmp), 0)) {
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

    if (dtype->f->compare) {
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
