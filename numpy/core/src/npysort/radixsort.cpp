#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"

#include "../common/numpy_tag.h"
#include <stdlib.h>
#include <type_traits>

/*
 *****************************************************************************
 **                            INTEGER SORTS                                **
 *****************************************************************************
 */

// Reference: https://github.com/eloj/radix-sorting#-key-derivation
template <class T>
T
KEY_OF(T x)
{
    // Floating-point is currently disabled.
    // Floating-point tests succeed for double and float on macOS but not on
    // Windows/Linux. Basic sorting tests succeed but others relying on sort
    // fail. Possibly related to floating-point normalisation or multiple NaN
    // reprs? Not sure.
    if (std::is_floating_point<T>::value) {
        // For floats, we invert the key if the sign bit is set, else we invert
        // the sign bit.
        return ((x) ^ (-((x) >> (sizeof(T) * 8 - 1)) |
                       ((T)1 << (sizeof(T) * 8 - 1))));
    }
    else if (std::is_signed<T>::value) {
        // For signed ints, we flip the sign bit so the negatives are below the
        // positives.
        return ((x) ^ ((T)1 << (sizeof(T) * 8 - 1)));
    }
    else {
        return x;
    }
}

template <class T>
static inline npy_ubyte
nth_byte(T key, npy_intp l)
{
    return (key >> (l << 3)) & 0xFF;
}

template <class T>
static T *
radixsort0(T *start, T *aux, npy_intp num)
{
    npy_intp cnt[sizeof(T)][1 << 8] = {{0}};
    T key0 = KEY_OF(start[0]);

    for (npy_intp i = 0; i < num; i++) {
        T k = KEY_OF(start[i]);

        for (size_t l = 0; l < sizeof(T); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    size_t ncols = 0;
    npy_ubyte cols[sizeof(T)];
    for (size_t l = 0; l < sizeof(T); l++) {
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
        T *temp;
        for (npy_intp i = 0; i < num; i++) {
            T k = KEY_OF(start[i]);
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = start[i];
        }

        temp = aux;
        aux = start;
        start = temp;
    }

    return start;
}

template <class T>
static int
radixsort_(T *start, npy_intp num)
{
    if (num < 2) {
        return 0;
    }

    npy_bool all_sorted = 1;
    T k1 = KEY_OF(start[0]), k2;
    for (npy_intp i = 1; i < num; i++) {
        k2 = KEY_OF(start[i]);
        if (k1 > k2) {
            all_sorted = 0;
            break;
        }
        k1 = k2;
    }

    if (all_sorted) {
        return 0;
    }

    T *aux = (T *)malloc(num * sizeof(T));
    if (aux == nullptr) {
        return -NPY_ENOMEM;
    }

    T *sorted = radixsort0(start, aux, num);
    if (sorted != start) {
        memcpy(start, sorted, num * sizeof(T));
    }

    free(aux);
    return 0;
}

template <class T>
static int
radixsort(void *start, npy_intp num)
{
    return radixsort_((T *)start, num);
}

template <class T>
static npy_intp *
aradixsort0(T *start, npy_intp *aux, npy_intp *tosort, npy_intp num)
{
    npy_intp cnt[sizeof(T)][1 << 8] = {{0}};
    T key0 = KEY_OF(start[0]);

    for (npy_intp i = 0; i < num; i++) {
        T k = KEY_OF(start[i]);

        for (size_t l = 0; l < sizeof(T); l++) {
            cnt[l][nth_byte(k, l)]++;
        }
    }

    size_t ncols = 0;
    npy_ubyte cols[sizeof(T)];
    for (size_t l = 0; l < sizeof(T); l++) {
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
            T k = KEY_OF(start[tosort[i]]);
            npy_intp dst = cnt[cols[l]][nth_byte(k, cols[l])]++;
            aux[dst] = tosort[i];
        }

        temp = aux;
        aux = tosort;
        tosort = temp;
    }

    return tosort;
}

template <class T>
static int
aradixsort_(T *start, npy_intp *tosort, npy_intp num)
{
    npy_intp *sorted;
    npy_intp *aux;
    T k1, k2;
    npy_bool all_sorted = 1;

    if (num < 2) {
        return 0;
    }

    k1 = KEY_OF(start[tosort[0]]);
    for (npy_intp i = 1; i < num; i++) {
        k2 = KEY_OF(start[tosort[i]]);
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

    sorted = aradixsort0(start, aux, tosort, num);
    if (sorted != tosort) {
        memcpy(tosort, sorted, num * sizeof(npy_intp));
    }

    free(aux);
    return 0;
}

template <class T>
static int
aradixsort(void *start, npy_intp *tosort, npy_intp num)
{
    return aradixsort_((T *)start, tosort, num);
}

extern "C" {
NPY_NO_EXPORT int
radixsort_bool(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_bool>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_byte(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_byte>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ubyte(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ubyte>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_short(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_short>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ushort(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ushort>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_int(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_int>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_uint(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_uint>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_long(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_long>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ulong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ulong>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_longlong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_longlong>(vec, cnt);
}
NPY_NO_EXPORT int
radixsort_ulonglong(void *vec, npy_intp cnt, void *NPY_UNUSED(null))
{
    return radixsort<npy_ulonglong>(vec, cnt);
}
NPY_NO_EXPORT int
aradixsort_bool(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_bool>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_byte(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_byte>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ubyte(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    return aradixsort<npy_ubyte>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_short(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    return aradixsort<npy_short>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ushort(void *vec, npy_intp *ind, npy_intp cnt,
                  void *NPY_UNUSED(null))
{
    return aradixsort<npy_ushort>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_int(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_int>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_uint(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_uint>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_long(void *vec, npy_intp *ind, npy_intp cnt, void *NPY_UNUSED(null))
{
    return aradixsort<npy_long>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ulong(void *vec, npy_intp *ind, npy_intp cnt,
                 void *NPY_UNUSED(null))
{
    return aradixsort<npy_ulong>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_longlong(void *vec, npy_intp *ind, npy_intp cnt,
                    void *NPY_UNUSED(null))
{
    return aradixsort<npy_longlong>(vec, ind, cnt);
}
NPY_NO_EXPORT int
aradixsort_ulonglong(void *vec, npy_intp *ind, npy_intp cnt,
                     void *NPY_UNUSED(null))
{
    return aradixsort<npy_ulonglong>(vec, ind, cnt);
}
}
