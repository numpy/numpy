#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"

#include "../common/numpy_tag.h"
#include <cstdlib>
#include <type_traits>

/*
 *****************************************************************************
 **                            INTEGER SORTS                                **
 *****************************************************************************
 */

// Reference: https://github.com/eloj/radix-sorting#-key-derivation
template <bool reverse, class T, class UT>
UT
KEY_OF(UT x)
{
    UT key;

    if constexpr (std::is_signed_v<T>) {
        // For signed ints, we flip the sign bit so the negatives are below the
        // positives.
        key = x ^ (UT(1) << (sizeof(UT) * 8 - 1));
    }
    else {
        key = x;
    }

    if constexpr (reverse) {
        return ~key;
    }
    else {
        return key;
    }
}

template <class T>
static inline npy_ubyte
nth_byte(T key, npy_intp l)
{
    return (key >> (l << 3)) & 0xFF;
}

template <class T, class UT, bool reverse>
static UT *
radixsort0(UT *start, UT *aux, npy_intp num)
{
    npy_intp cnt[sizeof(UT)][1 << 8] = {{0}};
    UT key0 = KEY_OF<reverse, T>(start[0]);

    for (npy_intp i = 0; i < num; i++) {
        UT k = KEY_OF<reverse, T>(start[i]);

        for (size_t l = 0; l < sizeof(UT); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    size_t ncols = 0;
    npy_ubyte cols[sizeof(UT)];
    for (size_t l = 0; l < sizeof(UT); l++) {
        if (cnt[l][nth_byte(key0, l)] != num) {
            cols[ncols++] = l;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        for (npy_intp i = 0; i < 256; i++) {
            npy_intp b = cnt[cols[l]][i];
            cnt[cols[l]][i] = a;
            a += b;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        UT *temp;
        for (npy_intp i = 0; i < num; i++) {
            UT k = KEY_OF<reverse, T>(start[i]);
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = start[i];
        }

        temp = aux;
        aux = start;
        start = temp;
    }

    return start;
}

template <class T, class UT, bool reverse>
static int
radixsort_(UT *start, npy_intp num)
{
    if (num < 2) {
        return 0;
    }

    npy_bool all_sorted = 1;
    UT k1 = KEY_OF<reverse, T>(start[0]);
    for (npy_intp i = 1; i < num; i++) {
        UT k2 = KEY_OF<reverse, T>(start[i]);
        if (k1 > k2) {
            all_sorted = 0;
            break;
        }
        k1 = k2;
    }

    if (all_sorted) {
        return 0;
    }

    UT *aux = (UT *)malloc(num * sizeof(UT));
    if (aux == nullptr) {
        return -NPY_ENOMEM;
    }

    UT *sorted = radixsort0<T, UT, reverse>(start, aux, num);
    if (sorted != start) {
        memcpy(start, sorted, num * sizeof(UT));
    }

    free(aux);
    return 0;
}

template <class T, bool reverse = false>
static int
radixsort(void *start, npy_intp num)
{
    using UT = typename std::make_unsigned<T>::type;
    return radixsort_<T, UT, reverse>((UT *)start, num);
}

// ``PyArray_SortFunc``-shaped trampoline.
template <typename Tag, typename type, bool reverse = false>
static int
radixsort_impl(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return radixsort<type, reverse>(start, num);
}

template <class T, class UT, bool reverse>
static npy_intp *
aradixsort0(UT *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    npy_intp cnt[sizeof(UT)][1 << 8] = {{0}};
    UT key0 = KEY_OF<reverse, T>(start[0]);

    for (npy_intp i = 0; i < num; i++) {
        UT k = KEY_OF<reverse, T>(start[i]);

        for (size_t l = 0; l < sizeof(UT); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    size_t ncols = 0;
    npy_ubyte cols[sizeof(UT)];
    for (size_t l = 0; l < sizeof(UT); l++) {
        if (cnt[l][nth_byte(key0, l)] != num) {
            cols[ncols++] = l;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        npy_intp a = 0;
        for (npy_intp i = 0; i < 256; i++) {
            npy_intp b = cnt[cols[l]][i];
            cnt[cols[l]][i] = a;
            a += b;
        }
    }

    for (size_t l = 0; l < ncols; l++) {
        npy_intp *temp;
        for (npy_intp i = 0; i < num; i++) {
            UT k = KEY_OF<reverse, T>(start[tosort[i]]);
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = tosort[i];
        }

        temp = aux;
        aux = tosort;
        tosort = temp;
    }

    return tosort;
}

template <class T, class UT, bool reverse>
static int
aradixsort_(UT *start, npy_intp *tosort, npy_intp num)
{
    npy_intp *sorted;
    npy_intp *aux;
    UT k1, k2;
    npy_bool all_sorted = 1;

    if (num < 2) {
        return 0;
    }

    k1 = KEY_OF<reverse, T>(start[tosort[0]]);
    for (npy_intp i = 1; i < num; i++) {
        k2 = KEY_OF<reverse, T>(start[tosort[i]]);
        if (k1 > k2) {
            all_sorted = 0;
            break;
        }
        k1 = k2;
    }

    if (all_sorted) {
        return 0;
    }

    aux = (npy_intp *)malloc(num * sizeof(npy_intp));
    if (aux == NULL) {
        return -NPY_ENOMEM;
    }

    sorted = aradixsort0<T, UT, reverse>(start, aux, tosort, num);
    if (sorted != tosort) {
        memcpy(tosort, sorted, num * sizeof(npy_intp));
    }

    free(aux);
    return 0;
}

template <class T, bool reverse = false>
static int
aradixsort(void *start, npy_intp *tosort, npy_intp num)
{
    using UT = typename std::make_unsigned<T>::type;
    return aradixsort_<T, UT, reverse>((UT *)start, tosort, num);
}

// ``PyArray_ArgSortFunc``-shaped trampoline.
template <typename Tag, typename type, bool reverse = false>
static int
aradixsort_impl(void *start, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return aradixsort<type, reverse>(start, tosort, num);
}
