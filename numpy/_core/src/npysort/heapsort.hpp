#ifndef NUMPY_SRC_COMMON_NPYSORT_HEAPSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HEAPSORT_HPP

#include "common.hpp"

namespace np::sort {

template <typename T>
inline bool LessThan(const T &a, const T &b)
{
    if constexpr(std::is_same_v<T, Half>) {
        bool a_nn = !a.IsNaN();
        return b.IsNaN() ? a_nn : a_nn && a.Less(b);
    }
    else if constexpr (kIsFloat<T>) {
        return a < b || (b != b && a == a);
    }
    else if constexpr (kIsComplex<T>) {
        auto a_real = a.real();
        auto a_imag = a.imag();
        auto b_real = b.real();
        auto b_imag = b.imag();
        if (a_real < b_real) {
            return a_imag == a_imag || b_imag != b_imag;
        }
        else if (a_real > b_real) {
            return b_imag != b_imag && a_imag == a_imag;
        }
        else if (a_real == b_real || (a_real != a_real && b_real != b_real)) {
            return  a_imag < b_imag || (b_imag != b_imag && a_imag == a_imag);
        }
        else {
            return b_real != b_real;
        }
    }
    else if constexpr (kIsTime<T>) {
        return (b.IsNaT() || static_cast<int64_t>(a) < static_cast<int64_t>(b)) && a.IsFinite();
    }
    else {
        return a < b;
    }
}

template <typename T>
inline bool LessThan(const T *a, const T *b, SSize len)
{
    bool ret = false;
    for (SSize i = 0; i < len; ++i) {
        if (a[i] != b[i]) {
            ret = a[i] < b[i];
            break;
        }
    }
    return ret;
}

template <typename T>
inline void Swap(T *s1, T *s2, SSize len)
{
    while (len--) {
        const T t = *s1;
        *s1++ = *s2;
        *s2++ = t;
    }
}


// NUMERIC SORTS
template <typename T>
inline void Heap(T *start, SSize n)
{
    SSize i, j, l;
    // The array needs to be offset by one for heapsort indexing
    T tmp, *a = start - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && LessThan(a[j], a[j + 1])) {
                j += 1;
            }
            if (LessThan(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && LessThan(a[j], a[j + 1])) {
                j++;
            }
            if (LessThan(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }
}
// Argsort
template <typename T>
inline void Heap(T *vv, SSize *tosort, SSize n)
{
    T *v = vv;
    SSize *a, i, j, l, tmp;
    // The arrays need to be offset by one for heapsort indexing
    a = tosort - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && LessThan(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (LessThan(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && LessThan(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (LessThan(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }
}
// string sort
template <typename T>
inline int Heap(T *start, SSize n, PyArrayObject *arr)
{
    SSize arr_len = PyArray_ITEMSIZE(arr);
    if (arr_len == 0) {
        return 0;  /* no need for sorting if strings are empty */
    }
    size_t len = arr_len / sizeof(T);

    T *tmp = (T *)malloc(arr_len);
    T *a = (T *)start - len;
    SSize i, j, l;

    if (tmp == NULL) {
        return -NPY_ENOMEM;
    }

    for (l = n >> 1; l > 0; --l) {
        memcpy(tmp, a + l * len, arr_len);
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && LessThan(a + j * len, a + (j + 1) * len, len))
                j += 1;
            if (LessThan(tmp, a + j * len, len)) {
                memcpy(a + i * len, a + j * len, arr_len);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        memcpy(a + i * len, tmp, arr_len);
    }

    for (; n > 1;) {
        memcpy(tmp, a + n * len, arr_len);
        memcpy(a + n * len, a + len, arr_len);
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && LessThan(a + j * len, a + (j + 1) * len, len))
                j++;
            if (LessThan(tmp, a + j * len, len)) {
                memcpy(a + i * len, a + j * len, arr_len);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        memcpy(a + i * len, tmp, arr_len);
    }

    free(tmp);
    return 0;
}

template <typename T>
inline int Heap(T *vv, SSize *tosort, SSize n, PyArrayObject *arr)
{
    T *v = vv;
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(T);
    SSize *a, i, j, l, tmp;

    /* The array needs to be offset by one for heapsort indexing */
    a = tosort - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && LessThan(v + a[j] * len, v + a[j + 1] * len, len))
                j += 1;
            if (LessThan(v + tmp * len, v + a[j] * len, len)) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && LessThan(v + a[j] * len, v + a[j + 1] * len, len))
                j++;
            if (LessThan(v + tmp * len, v + a[j] * len, len)) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    return 0;
}

} // namespace np::sort
#endif
