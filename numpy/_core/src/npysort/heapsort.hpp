#ifndef NUMPY_SRC_COMMON_NPYSORT_HEAPSORT_HPP
#define NUMPY_SRC_COMMON_NPYSORT_HEAPSORT_HPP

#include "common.hpp"

namespace np::sort {

// Strict-weak-less comparator: returns true iff ``a`` sorts before ``b``.
// ``reverse=true`` flips the relational test (so it really means "greater"),
// while keeping NaN-at-end semantics for floating-point types.
template <bool reverse = false, typename T>
constexpr bool Cmp(const T &a, const T &b)
{
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (reverse) {
            return a > b || (b != b && a == a);
        }
        else {
            return a < b || (b != b && a == a);
        }
    }
    else if constexpr(std::is_same_v<T, Half>) {
        bool a_nn = !a.IsNaN();
        if constexpr (reverse) {
            return b.IsNaN() ? a_nn : a_nn && b.Less(a);
        }
        else {
            return b.IsNaN() ? a_nn : a_nn && a.Less(b);
        }
    }
    else {
        if constexpr (reverse) {
            return a > b;
        }
        else {
            return a < b;
        }
    }
}

// NUMERIC SORTS
template <bool reverse = false, typename T>
inline void Heap(T *start, SSize n)
{
    SSize i, j, l;
    // The array needs to be offset by one for heapsort indexing
    T tmp, *a = start - 1;

    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && Cmp<reverse>(a[j], a[j + 1])) {
                j += 1;
            }
            if (Cmp<reverse>(tmp, a[j])) {
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
            if (j < n && Cmp<reverse>(a[j], a[j + 1])) {
                j++;
            }
            if (Cmp<reverse>(tmp, a[j])) {
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
} // namespace np::sort
#endif
