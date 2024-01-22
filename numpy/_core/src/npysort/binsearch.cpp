#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_binsearch.h"

#include "heapsort.hpp" // np::sort::LessThan()

namespace np::sort {

template <bool right_side, typename T>
inline bool BinaryCompare(const T &a, const T &b)
{
    if constexpr (right_side) {
        return !LessThan(b, a);
    }
    else {
        return LessThan(a, b);
    }
}

template <bool right_side>
inline bool BinaryGenericCompare(int a, int b)
{
    if constexpr (right_side) {
        return a <= b;
    }
    else {
        return a < b;
    }
}

template <bool right_side>
void BinaryGenericSearch(const char *arr, const char *key, char *ret, SSize arr_len,
                         SSize key_len, SSize arr_str, SSize key_str,
                         SSize ret_str, PyArrayObject *cmp)
{
    PyArray_CompareFunc *compare = PyArray_DESCR(cmp)->f->compare;
    SSize min_idx = 0;
    SSize max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (BinaryGenericCompare<right_side>(compare(last_key, key, cmp), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const SSize mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const char *arr_ptr = arr + mid_idx * arr_str;

            if (BinaryGenericCompare<right_side>(compare(arr_ptr, key, cmp), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(SSize *)ret = min_idx;
    }
}

template <bool right_side>
int BinaryArgGenericSearch(const char *arr, const char *key, const char *sort, char *ret,
                           SSize arr_len, SSize key_len, SSize arr_str,
                           SSize key_str, SSize sort_str, SSize ret_str,
                           PyArrayObject *cmp)
{
    PyArray_CompareFunc *compare = PyArray_DESCR(cmp)->f->compare;
    SSize min_idx = 0;
    SSize max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        if (BinaryGenericCompare<right_side>(compare(last_key, key, cmp), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const SSize mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const SSize sort_idx = *(SSize *)(sort + mid_idx * sort_str);
            const char *arr_ptr;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            arr_ptr = arr + sort_idx * arr_str;

            if (BinaryGenericCompare<right_side>(compare(arr_ptr, key, cmp), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(SSize *)ret = min_idx;
    }
    return 0;
}

template <bool right_side, typename T>
void BinarySearch(const char *arr, const char *key, char *ret, SSize arr_len,
                  SSize key_len, SSize arr_str, SSize key_str,
                  SSize ret_str, PyArrayObject *)
{
    SSize min_idx = 0;
    SSize max_idx = arr_len;
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
        if (BinaryCompare<right_side>(last_key_val, key_val)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key_val = key_val;

        while (min_idx < max_idx) {
            const SSize mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const T mid_val = *(const T *)(arr + mid_idx * arr_str);
            if (BinaryCompare<right_side>(mid_val, key_val)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(SSize *)ret = min_idx;
    }
}

template <>
void BinarySearch<false, void>(const char *arr, const char *key, char *ret, SSize arr_len,
                               SSize key_len, SSize arr_str, SSize key_str,
                               SSize ret_str, PyArrayObject *cmp)
{
    BinaryGenericSearch<false>(arr, key, ret, arr_len, key_len, arr_str, key_str, ret_str, cmp);
}
template <>
void BinarySearch<true, void>(const char *arr, const char *key, char *ret, SSize arr_len,
                               SSize key_len, SSize arr_str, SSize key_str,
                               SSize ret_str, PyArrayObject *cmp)
{
    BinaryGenericSearch<true>(arr, key, ret, arr_len, key_len, arr_str, key_str, ret_str, cmp);
}


template <bool right_side, typename T>
int BinaryArgSearch(const char *arr, const char *key, const char *sort, char *ret,
                    SSize arr_len, SSize key_len, SSize arr_str,
                    SSize key_str, SSize sort_str, SSize ret_str,
                    PyArrayObject *)
{
    SSize min_idx = 0;
    SSize max_idx = arr_len;
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
        if (BinaryCompare<right_side>(last_key_val, key_val)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key_val = key_val;

        while (min_idx < max_idx) {
            const SSize mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const SSize sort_idx = *(SSize *)(sort + mid_idx * sort_str);
            T mid_val;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            mid_val = *(const T *)(arr + sort_idx * arr_str);

            if (BinaryCompare<right_side>(mid_val, key_val)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(SSize *)ret = min_idx;
    }
    return 0;
}

template <>
int BinaryArgSearch<false, void>(const char *arr, const char *key, const char *sort, char *ret,
                                SSize arr_len, SSize key_len, SSize arr_str,
                                SSize key_str, SSize sort_str, SSize ret_str,
                                PyArrayObject *cmp)
{
    return BinaryArgGenericSearch<false>(arr, key, sort, ret, arr_len, key_len, arr_str,
                                         key_str, sort_str, ret_str, cmp);
}
template <>
int BinaryArgSearch<true, void>(const char *arr, const char *key, const char *sort, char *ret,
                                SSize arr_len, SSize key_len, SSize arr_str,
                                SSize key_str, SSize sort_str, SSize ret_str,
                                PyArrayObject *cmp)
{
    return BinaryArgGenericSearch<true>(arr, key, sort, ret, arr_len, key_len, arr_str,
                                        key_str, sort_str, ret_str, cmp);
}

template <typename ...T>
struct BinarySearchTab {
    PyArray_BinSearchFunc *ptrs[2][sizeof...(T)] = {
        { BinarySearch<false, T>... },
        { BinarySearch<true, T>... }
    };
    PyArray_ArgBinSearchFunc *arg_ptrs[2][sizeof...(T)] = {
        { BinaryArgSearch<false, T>... },
        { BinaryArgSearch<true, T>... }
    };
};

static BinarySearchTab<
    np::Bool, np::Byte, np::UByte, np::Short,
    np::UShort, np::Int, np::UInt, np::Long,
    np::ULong, np::LongLong, np::ULongLong,
    np::Float, np::Double, np::LongDouble,
    np::CFloat, np::CDouble, np::CLongDouble,
    void/*Object*/, void/*String*/, void/*Unicode*/, void/*void*/,
    np::DateTime, np::TimeDelta, np::Half
> binary_search_tab;

template <bool arg>
inline std::conditional_t<arg, PyArray_ArgBinSearchFunc*, PyArray_BinSearchFunc*>
GetBinarySearchFunc(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    if ((int)side >= (int)NPY_NSEARCHSIDES) {
        return nullptr;
    }
    int type_id = dtype->type_num;
    if (type_id < (int)kBool || type_id > (int)kHalf) {
        if (!dtype->f->compare) {
            return nullptr;
        }
        type_id = (int)kObject;
    }
    if constexpr (arg) {
        return binary_search_tab.arg_ptrs[side][type_id];
    }
    else {
        return binary_search_tab.ptrs[side][type_id];
    }
}

} // namespace np::sort

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */
NPY_NO_EXPORT PyArray_BinSearchFunc *
get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return np::sort::GetBinarySearchFunc<false>(dtype, side);
}

NPY_NO_EXPORT PyArray_ArgBinSearchFunc *
get_argbinsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return np::sort::GetBinarySearchFunc<true>(dtype, side);
}
