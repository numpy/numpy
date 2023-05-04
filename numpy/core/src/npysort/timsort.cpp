/* -*- c -*- */

/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario can
 * be slower than the merge and heap sorts.  The merge sort requires
 * extra memory and so for large arrays may not be useful.
 *
 * The merge sort is *stable*, meaning that equal components
 * are unmoved from their entry versions, so it can be used to
 * implement lexicographic sorting on multiple keys.
 *
 * The heap sort is included for completeness.
 */

/* For details of Timsort, refer to
 * https://github.com/python/cpython/blob/3.7/Objects/listsort.txt
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include <cstdlib>
#include <utility>

/* enough for 32 * 1.618 ** 128 elements */
#define TIMSORT_STACK_SIZE 128

static npy_intp
compute_min_run(npy_intp num)
{
    npy_intp r = 0;

    while (64 < num) {
        r |= num & 1;
        num >>= 1;
    }

    return num + r;
}

typedef struct {
    npy_intp s; /* start pointer */
    npy_intp l; /* length */
} run;

/* buffer for argsort. Declared here to avoid multiple declarations. */
typedef struct {
    npy_intp *pw;
    npy_intp size;
} buffer_intp;

/* buffer method */
static inline int
resize_buffer_intp(buffer_intp *buffer, npy_intp new_size)
{
    if (new_size <= buffer->size) {
        return 0;
    }

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        buffer->pw = (npy_intp *)malloc(new_size * sizeof(npy_intp));
    }
    else {
        buffer->pw =
                (npy_intp *)realloc(buffer->pw, new_size * sizeof(npy_intp));
    }

    buffer->size = new_size;

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        return 0;
    }
}

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag>
struct buffer_ {
    typename Tag::type *pw;
    npy_intp size;
};

template <typename Tag>
static inline int
resize_buffer_(buffer_<Tag> *buffer, npy_intp new_size)
{
    using type = typename Tag::type;
    if (new_size <= buffer->size) {
        return 0;
    }

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        buffer->pw = (type *)malloc(new_size * sizeof(type));
    }
    else {
        buffer->pw = (type *)realloc(buffer->pw, new_size * sizeof(type));
    }

    buffer->size = new_size;

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        return 0;
    }
}

template <typename Tag, typename type>
static npy_intp
count_run_(type *arr, npy_intp l, npy_intp num, npy_intp minrun)
{
    npy_intp sz;
    type vc, *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = arr + l;

    /* (not strictly) ascending sequence */
    if (!Tag::less(*(pl + 1), *pl)) {
        for (pi = pl + 1; pi < arr + num - 1 && !Tag::less(*(pi + 1), *pi);
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1; pi < arr + num - 1 && Tag::less(*(pi + 1), *pi);
             ++pi) {
        }

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    ++pi;
    sz = pi - pl;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* insertion sort */
        for (; pi < pr; ++pi) {
            vc = *pi;
            pj = pi;

            while (pl < pj && Tag::less(vc, *(pj - 1))) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vc;
        }
    }

    return sz;
}

/* when the left part of the array (p1) is smaller, copy p1 to buffer
 * and merge from left to right
 */
template <typename Tag, typename type>
static void
merge_left_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3)
{
    type *end = p2 + l2;
    memcpy(p3, p1, sizeof(type) * l1);
    /* first element must be in p2 otherwise skipped in the caller */
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        if (Tag::less(*p2, *p3)) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(type) * (p2 - p1));
    }
}

/* when the right part of the array (p2) is smaller, copy p2 to buffer
 * and merge from right to left
 */
template <typename Tag, typename type>
static void
merge_right_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3)
{
    npy_intp ofs;
    type *start = p1 - 1;
    memcpy(p3, p2, sizeof(type) * l2);
    p1 += l1 - 1;
    p2 += l2 - 1;
    p3 += l2 - 1;
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;

    while (p1 < p2 && start < p1) {
        if (Tag::less(*p3, *p1)) {
            *p2-- = *p1--;
        }
        else {
            *p2-- = *p3--;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + 1, p3 - ofs + 1, sizeof(type) * ofs);
    }
}

/* Note: the naming convention of gallop functions are different from that of
 * CPython. For example, here gallop_right means gallop from left toward right,
 * whereas in CPython gallop_right means gallop
 * and find the right most element among equal elements
 */
template <typename Tag, typename type>
static npy_intp
gallop_right_(const type *arr, const npy_intp size, const type key)
{
    npy_intp last_ofs, ofs, m;

    if (Tag::less(key, arr[0])) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (Tag::less(key, arr[ofs])) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[last_ofs] <= key < arr[ofs] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (Tag::less(key, arr[m])) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[ofs-1] <= key < arr[ofs] */
    return ofs;
}

template <typename Tag, typename type>
static npy_intp
gallop_left_(const type *arr, const npy_intp size, const type key)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (Tag::less(arr[size - 1], key)) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (Tag::less(arr[size - ofs - 1], key)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[size-ofs-1] < key <= arr[size-last_ofs-1] */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (Tag::less(arr[m], key)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[r-1] < key <= arr[r] */
    return r;
}

template <typename Tag, typename type>
static int
merge_at_(type *arr, const run *stack, const npy_intp at, buffer_<Tag> *buffer)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    type *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* arr[s2] belongs to arr[s1+k].
     * if try to comment this out for debugging purpose, remember
     * in the merging process the first element is skipped
     */
    k = gallop_right_<Tag>(arr + s1, l1, arr[s2]);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = arr + s1 + k;
    l1 -= k;
    p2 = arr + s2;
    /* arr[s2-1] belongs to arr[s2+l2] */
    l2 = gallop_left_<Tag>(arr + s2, l2, arr[s2 - 1]);

    if (l2 < l1) {
        ret = resize_buffer_<Tag>(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_right_<Tag>(p1, l1, p2, l2, buffer->pw);
    }
    else {
        ret = resize_buffer_<Tag>(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_left_<Tag>(p1, l1, p2, l2, buffer->pw);
    }

    return 0;
}

template <typename Tag, typename type>
static int
try_collapse_(type *arr, run *stack, npy_intp *stack_ptr, buffer_<Tag> *buffer)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = merge_at_<Tag>(arr, stack, top - 3, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

template <typename Tag, typename type>
static int
force_collapse_(type *arr, run *stack, npy_intp *stack_ptr,
                buffer_<Tag> *buffer)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = merge_at_<Tag>(arr, stack, top - 3, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

template <typename Tag>
static int
timsort_(void *start, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    buffer_<Tag> buffer;
    run stack[TIMSORT_STACK_SIZE];
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run(num);

    for (l = 0; l < num;) {
        n = count_run_<Tag>((type *)start, l, num, minrun);
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = try_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = force_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;
cleanup:

    free(buffer.pw);

    return ret;
}

/* argsort */

template <typename Tag, typename type>
static npy_intp
acount_run_(type *arr, npy_intp *tosort, npy_intp l, npy_intp num,
            npy_intp minrun)
{
    npy_intp sz;
    type vc;
    npy_intp vi;
    npy_intp *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = tosort + l;

    /* (not strictly) ascending sequence */
    if (!Tag::less(arr[*(pl + 1)], arr[*pl])) {
        for (pi = pl + 1;
             pi < tosort + num - 1 && !Tag::less(arr[*(pi + 1)], arr[*pi]);
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1;
             pi < tosort + num - 1 && Tag::less(arr[*(pi + 1)], arr[*pi]);
             ++pi) {
        }

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    ++pi;
    sz = pi - pl;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* insertion sort */
        for (; pi < pr; ++pi) {
            vi = *pi;
            vc = arr[*pi];
            pj = pi;

            while (pl < pj && Tag::less(vc, arr[*(pj - 1)])) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vi;
        }
    }

    return sz;
}

template <typename Tag, typename type>
static npy_intp
agallop_right_(const type *arr, const npy_intp *tosort, const npy_intp size,
               const type key)
{
    npy_intp last_ofs, ofs, m;

    if (Tag::less(key, arr[tosort[0]])) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (Tag::less(key, arr[tosort[ofs]])) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[last_ofs]] <= key < arr[tosort[ofs]] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (Tag::less(key, arr[tosort[m]])) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[tosort[ofs-1]] <= key < arr[tosort[ofs]] */
    return ofs;
}

template <typename Tag, typename type>
static npy_intp
agallop_left_(const type *arr, const npy_intp *tosort, const npy_intp size,
              const type key)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (Tag::less(arr[tosort[size - 1]], key)) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (Tag::less(arr[tosort[size - ofs - 1]], key)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[size-ofs-1]] < key <= arr[tosort[size-last_ofs-1]]
     */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (Tag::less(arr[tosort[m]], key)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[tosort[r-1]] < key <= arr[tosort[r]] */
    return r;
}

template <typename Tag, typename type>
static void
amerge_left_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
             npy_intp *p3)
{
    npy_intp *end = p2 + l2;
    memcpy(p3, p1, sizeof(npy_intp) * l1);
    /* first element must be in p2 otherwise skipped in the caller */
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        if (Tag::less(arr[*p2], arr[*p3])) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(npy_intp) * (p2 - p1));
    }
}

template <typename Tag, typename type>
static void
amerge_right_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
              npy_intp *p3)
{
    npy_intp ofs;
    npy_intp *start = p1 - 1;
    memcpy(p3, p2, sizeof(npy_intp) * l2);
    p1 += l1 - 1;
    p2 += l2 - 1;
    p3 += l2 - 1;
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;

    while (p1 < p2 && start < p1) {
        if (Tag::less(arr[*p3], arr[*p1])) {
            *p2-- = *p1--;
        }
        else {
            *p2-- = *p3--;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + 1, p3 - ofs + 1, sizeof(npy_intp) * ofs);
    }
}

template <typename Tag, typename type>
static int
amerge_at_(type *arr, npy_intp *tosort, const run *stack, const npy_intp at,
           buffer_intp *buffer)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    npy_intp *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* tosort[s2] belongs to tosort[s1+k] */
    k = agallop_right_<Tag>(arr, tosort + s1, l1, arr[tosort[s2]]);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = tosort + s1 + k;
    l1 -= k;
    p2 = tosort + s2;
    /* tosort[s2-1] belongs to tosort[s2+l2] */
    l2 = agallop_left_<Tag>(arr, tosort + s2, l2, arr[tosort[s2 - 1]]);

    if (l2 < l1) {
        ret = resize_buffer_intp(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_right_<Tag>(arr, p1, l1, p2, l2, buffer->pw);
    }
    else {
        ret = resize_buffer_intp(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_left_<Tag>(arr, p1, l1, p2, l2, buffer->pw);
    }

    return 0;
}

template <typename Tag, typename type>
static int
atry_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
               buffer_intp *buffer)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

template <typename Tag, typename type>
static int
aforce_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                 buffer_intp *buffer)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

template <typename Tag>
static int
atimsort_(void *v, npy_intp *tosort, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    buffer_intp buffer;
    run stack[TIMSORT_STACK_SIZE];
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run(num);

    for (l = 0; l < num;) {
        n = acount_run_<Tag>((type *)v, tosort, l, num, minrun);
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = atry_collapse_<Tag>((type *)v, tosort, stack, &stack_ptr,
                                  &buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = aforce_collapse_<Tag>((type *)v, tosort, stack, &stack_ptr, &buffer);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;
cleanup:

    if (buffer.pw != NULL) {
        free(buffer.pw);
    }

    return ret;
}

/* For string sorts and generic sort, element comparisons are very expensive,
 * and the time cost of insertion sort (involves N**2 comparison) clearly
 * hurts. Implementing binary insertion sort and probably gallop mode during
 * merging process can hopefully boost the performance. Here as a temporary
 * workaround we use shorter run length to reduce the cost of insertion sort.
 */

static npy_intp
compute_min_run_short(npy_intp num)
{
    npy_intp r = 0;

    while (16 < num) {
        r |= num & 1;
        num >>= 1;
    }

    return num + r;
}

/*
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag>
struct string_buffer_ {
    typename Tag::type *pw;
    npy_intp size;
    size_t len;
};

template <typename Tag>
static inline int
resize_buffer_(string_buffer_<Tag> *buffer, npy_intp new_size)
{
    using type = typename Tag::type;
    if (new_size <= buffer->size) {
        return 0;
    }

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        buffer->pw = (type *)malloc(sizeof(type) * new_size * buffer->len);
    }
    else {
        buffer->pw = (type *)realloc(buffer->pw,
                                     sizeof(type) * new_size * buffer->len);
    }

    buffer->size = new_size;

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        return 0;
    }
}

template <typename Tag, typename type>
static npy_intp
count_run_(type *arr, npy_intp l, npy_intp num, npy_intp minrun, type *vp,
           size_t len)
{
    npy_intp sz;
    type *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = arr + l * len;

    /* (not strictly) ascending sequence */
    if (!Tag::less(pl + len, pl, len)) {
        for (pi = pl + len;
             pi < arr + (num - 1) * len && !Tag::less(pi + len, pi, len);
             pi += len) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + len;
             pi < arr + (num - 1) * len && Tag::less(pi + len, pi, len);
             pi += len) {
        }

        for (pj = pl, pr = pi; pj < pr; pj += len, pr -= len) {
            Tag::swap(pj, pr, len);
        }
    }

    pi += len;
    sz = (pi - pl) / len;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz * len;

        /* insertion sort */
        for (; pi < pr; pi += len) {
            Tag::copy(vp, pi, len);
            pj = pi;

            while (pl < pj && Tag::less(vp, pj - len, len)) {
                Tag::copy(pj, pj - len, len);
                pj -= len;
            }

            Tag::copy(pj, vp, len);
        }
    }

    return sz;
}

template <typename Tag>
static npy_intp
gallop_right_(const typename Tag::type *arr, const npy_intp size,
              const typename Tag::type *key, size_t len)
{
    npy_intp last_ofs, ofs, m;

    if (Tag::less(key, arr, len)) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (Tag::less(key, arr + ofs * len, len)) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[last_ofs*len] <= key < arr[ofs*len] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (Tag::less(key, arr + m * len, len)) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[(ofs-1)*len] <= key < arr[ofs*len] */
    return ofs;
}

template <typename Tag>
static npy_intp
gallop_left_(const typename Tag::type *arr, const npy_intp size,
             const typename Tag::type *key, size_t len)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (Tag::less(arr + (size - 1) * len, key, len)) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (Tag::less(arr + (size - ofs - 1) * len, key, len)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[(size-ofs-1)*len] < key <= arr[(size-last_ofs-1)*len] */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (Tag::less(arr + m * len, key, len)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[(r-1)*len] < key <= arr[r*len] */
    return r;
}

template <typename Tag>
static void
merge_left_(typename Tag::type *p1, npy_intp l1, typename Tag::type *p2,
            npy_intp l2, typename Tag::type *p3, size_t len)
{
    using type = typename Tag::type;
    type *end = p2 + l2 * len;
    memcpy(p3, p1, sizeof(type) * l1 * len);
    /* first element must be in p2 otherwise skipped in the caller */
    Tag::copy(p1, p2, len);
    p1 += len;
    p2 += len;

    while (p1 < p2 && p2 < end) {
        if (Tag::less(p2, p3, len)) {
            Tag::copy(p1, p2, len);
            p1 += len;
            p2 += len;
        }
        else {
            Tag::copy(p1, p3, len);
            p1 += len;
            p3 += len;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(type) * (p2 - p1));
    }
}

template <typename Tag, typename type>
static void
merge_right_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3,
             size_t len)
{
    npy_intp ofs;
    type *start = p1 - len;
    memcpy(p3, p2, sizeof(type) * l2 * len);
    p1 += (l1 - 1) * len;
    p2 += (l2 - 1) * len;
    p3 += (l2 - 1) * len;
    /* first element must be in p1 otherwise skipped in the caller */
    Tag::copy(p2, p1, len);
    p2 -= len;
    p1 -= len;

    while (p1 < p2 && start < p1) {
        if (Tag::less(p3, p1, len)) {
            Tag::copy(p2, p1, len);
            p2 -= len;
            p1 -= len;
        }
        else {
            Tag::copy(p2, p3, len);
            p2 -= len;
            p3 -= len;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + len, p3 - ofs + len, sizeof(type) * ofs);
    }
}

template <typename Tag, typename type>
static int
merge_at_(type *arr, const run *stack, const npy_intp at,
          string_buffer_<Tag> *buffer, size_t len)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    type *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* arr[s2] belongs to arr[s1+k] */
    Tag::copy(buffer->pw, arr + s2 * len, len);
    k = gallop_right_<Tag>(arr + s1 * len, l1, buffer->pw, len);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = arr + (s1 + k) * len;
    l1 -= k;
    p2 = arr + s2 * len;
    /* arr[s2-1] belongs to arr[s2+l2] */
    Tag::copy(buffer->pw, arr + (s2 - 1) * len, len);
    l2 = gallop_left_<Tag>(arr + s2 * len, l2, buffer->pw, len);

    if (l2 < l1) {
        ret = resize_buffer_<Tag>(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_right_<Tag>(p1, l1, p2, l2, buffer->pw, len);
    }
    else {
        ret = resize_buffer_<Tag>(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_left_<Tag>(p1, l1, p2, l2, buffer->pw, len);
    }

    return 0;
}

template <typename Tag, typename type>
static int
try_collapse_(type *arr, run *stack, npy_intp *stack_ptr,
              string_buffer_<Tag> *buffer, size_t len)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = merge_at_<Tag>(arr, stack, top - 3, buffer, len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

template <typename Tag, typename type>
static int
force_collapse_(type *arr, run *stack, npy_intp *stack_ptr,
                string_buffer_<Tag> *buffer, size_t len)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = merge_at_<Tag>(arr, stack, top - 3, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = merge_at_<Tag>(arr, stack, top - 2, buffer, len);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

template <typename Tag>
NPY_NO_EXPORT int
string_timsort_(void *start, npy_intp num, void *varr)
{
    using type = typename Tag::type;
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(varr);
    size_t elsize = PyArray_ITEMSIZE(arr);
    size_t len = elsize / sizeof(type);
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    run stack[TIMSORT_STACK_SIZE];
    string_buffer_<Tag> buffer;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    buffer.pw = NULL;
    buffer.size = 0;
    buffer.len = len;
    stack_ptr = 0;
    minrun = compute_min_run_short(num);
    /* used for insertion sort and gallop key */
    ret = resize_buffer_<Tag>(&buffer, 1);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    for (l = 0; l < num;) {
        n = count_run_<Tag>((type *)start, l, num, minrun, buffer.pw, len);
        /* both s and l are scaled by len */
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = try_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer,
                                 len);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = force_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer, len);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (buffer.pw != NULL) {
        free(buffer.pw);
    }
    return ret;
}

/* argsort */

template <typename Tag, typename type>
static npy_intp
acount_run_(type *arr, npy_intp *tosort, npy_intp l, npy_intp num,
            npy_intp minrun, size_t len)
{
    npy_intp sz;
    npy_intp vi;
    npy_intp *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = tosort + l;

    /* (not strictly) ascending sequence */
    if (!Tag::less(arr + (*(pl + 1)) * len, arr + (*pl) * len, len)) {
        for (pi = pl + 1;
             pi < tosort + num - 1 &&
             !Tag::less(arr + (*(pi + 1)) * len, arr + (*pi) * len, len);
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1;
             pi < tosort + num - 1 &&
             Tag::less(arr + (*(pi + 1)) * len, arr + (*pi) * len, len);
             ++pi) {
        }

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    ++pi;
    sz = pi - pl;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* insertion sort */
        for (; pi < pr; ++pi) {
            vi = *pi;
            pj = pi;

            while (pl < pj &&
                   Tag::less(arr + vi * len, arr + (*(pj - 1)) * len, len)) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vi;
        }
    }

    return sz;
}

template <typename Tag, typename type>
static npy_intp
agallop_left_(const type *arr, const npy_intp *tosort, const npy_intp size,
              const type *key, size_t len)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (Tag::less(arr + tosort[size - 1] * len, key, len)) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (Tag::less(arr + tosort[size - ofs - 1] * len, key, len)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[size-ofs-1]*len] < key <=
     * arr[tosort[size-last_ofs-1]*len] */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (Tag::less(arr + tosort[m] * len, key, len)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[tosort[r-1]*len] < key <= arr[tosort[r]*len] */
    return r;
}

template <typename Tag, typename type>
static npy_intp
agallop_right_(const type *arr, const npy_intp *tosort, const npy_intp size,
               const type *key, size_t len)
{
    npy_intp last_ofs, ofs, m;

    if (Tag::less(key, arr + tosort[0] * len, len)) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (Tag::less(key, arr + tosort[ofs] * len, len)) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[last_ofs]*len] <= key < arr[tosort[ofs]*len] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (Tag::less(key, arr + tosort[m] * len, len)) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[tosort[ofs-1]*len] <= key < arr[tosort[ofs]*len] */
    return ofs;
}

template <typename Tag, typename type>
static void
amerge_left_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
             npy_intp *p3, size_t len)
{
    npy_intp *end = p2 + l2;
    memcpy(p3, p1, sizeof(npy_intp) * l1);
    /* first element must be in p2 otherwise skipped in the caller */
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        if (Tag::less(arr + (*p2) * len, arr + (*p3) * len, len)) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(npy_intp) * (p2 - p1));
    }
}

template <typename Tag, typename type>
static void
amerge_right_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
              npy_intp *p3, size_t len)
{
    npy_intp ofs;
    npy_intp *start = p1 - 1;
    memcpy(p3, p2, sizeof(npy_intp) * l2);
    p1 += l1 - 1;
    p2 += l2 - 1;
    p3 += l2 - 1;
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;

    while (p1 < p2 && start < p1) {
        if (Tag::less(arr + (*p3) * len, arr + (*p1) * len, len)) {
            *p2-- = *p1--;
        }
        else {
            *p2-- = *p3--;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + 1, p3 - ofs + 1, sizeof(npy_intp) * ofs);
    }
}

template <typename Tag, typename type>
static int
amerge_at_(type *arr, npy_intp *tosort, const run *stack, const npy_intp at,
           buffer_intp *buffer, size_t len)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    npy_intp *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* tosort[s2] belongs to tosort[s1+k] */
    k = agallop_right_<Tag>(arr, tosort + s1, l1, arr + tosort[s2] * len, len);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = tosort + s1 + k;
    l1 -= k;
    p2 = tosort + s2;
    /* tosort[s2-1] belongs to tosort[s2+l2] */
    l2 = agallop_left_<Tag>(arr, tosort + s2, l2, arr + tosort[s2 - 1] * len,
                            len);

    if (l2 < l1) {
        ret = resize_buffer_intp(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_right_<Tag>(arr, p1, l1, p2, l2, buffer->pw, len);
    }
    else {
        ret = resize_buffer_intp(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_left_<Tag>(arr, p1, l1, p2, l2, buffer->pw, len);
    }

    return 0;
}

template <typename Tag, typename type>
static int
atry_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
               buffer_intp *buffer, size_t len)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer,
                                      len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer,
                                      len);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

template <typename Tag, typename type>
static int
aforce_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                 buffer_intp *buffer, size_t len)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer, len);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer, len);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

template <typename Tag>
NPY_NO_EXPORT int
string_atimsort_(void *start, npy_intp *tosort, npy_intp num, void *varr)
{
    using type = typename Tag::type;
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(varr);
    size_t elsize = PyArray_ITEMSIZE(arr);
    size_t len = elsize / sizeof(type);
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    run stack[TIMSORT_STACK_SIZE];
    buffer_intp buffer;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run_short(num);

    for (l = 0; l < num;) {
        n = acount_run_<Tag>((type *)start, tosort, l, num, minrun, len);
        /* both s and l are scaled by len */
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = atry_collapse_<Tag>((type *)start, tosort, stack, &stack_ptr,
                                  &buffer, len);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = aforce_collapse_<Tag>((type *)start, tosort, stack, &stack_ptr,
                                &buffer, len);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (buffer.pw != NULL) {
        free(buffer.pw);
    }
    return ret;
}

/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

typedef struct {
    char *pw;
    npy_intp size;
    size_t len;
} buffer_char;

static inline int
resize_buffer_char(buffer_char *buffer, npy_intp new_size)
{
    if (new_size <= buffer->size) {
        return 0;
    }

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        buffer->pw = (char *)malloc(sizeof(char) * new_size * buffer->len);
    }
    else {
        buffer->pw = (char *)realloc(buffer->pw,
                                     sizeof(char) * new_size * buffer->len);
    }

    buffer->size = new_size;

    if (NPY_UNLIKELY(buffer->pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        return 0;
    }
}

static npy_intp
npy_count_run(char *arr, npy_intp l, npy_intp num, npy_intp minrun, char *vp,
              size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp sz;
    char *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = arr + l * len;

    /* (not strictly) ascending sequence */
    if (cmp(pl, pl + len, py_arr) <= 0) {
        for (pi = pl + len;
             pi < arr + (num - 1) * len && cmp(pi, pi + len, py_arr) <= 0;
             pi += len) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + len;
             pi < arr + (num - 1) * len && cmp(pi + len, pi, py_arr) < 0;
             pi += len) {
        }

        for (pj = pl, pr = pi; pj < pr; pj += len, pr -= len) {
            GENERIC_SWAP(pj, pr, len);
        }
    }

    pi += len;
    sz = (pi - pl) / len;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz * len;

        /* insertion sort */
        for (; pi < pr; pi += len) {
            GENERIC_COPY(vp, pi, len);
            pj = pi;

            while (pl < pj && cmp(vp, pj - len, py_arr) < 0) {
                GENERIC_COPY(pj, pj - len, len);
                pj -= len;
            }

            GENERIC_COPY(pj, vp, len);
        }
    }

    return sz;
}

static npy_intp
npy_gallop_right(const char *arr, const npy_intp size, const char *key,
                 size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, m;

    if (cmp(key, arr, py_arr) < 0) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (cmp(key, arr + ofs * len, py_arr) < 0) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[last_ofs*len] <= key < arr[ofs*len] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (cmp(key, arr + m * len, py_arr) < 0) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[(ofs-1)*len] <= key < arr[ofs*len] */
    return ofs;
}

static npy_intp
npy_gallop_left(const char *arr, const npy_intp size, const char *key,
                size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (cmp(arr + (size - 1) * len, key, py_arr) < 0) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (cmp(arr + (size - ofs - 1) * len, key, py_arr) < 0) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[(size-ofs-1)*len] < key <= arr[(size-last_ofs-1)*len] */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (cmp(arr + m * len, key, py_arr) < 0) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[(r-1)*len] < key <= arr[r*len] */
    return r;
}

static void
npy_merge_left(char *p1, npy_intp l1, char *p2, npy_intp l2, char *p3,
               size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    char *end = p2 + l2 * len;
    memcpy(p3, p1, sizeof(char) * l1 * len);
    /* first element must be in p2 otherwise skipped in the caller */
    GENERIC_COPY(p1, p2, len);
    p1 += len;
    p2 += len;

    while (p1 < p2 && p2 < end) {
        if (cmp(p2, p3, py_arr) < 0) {
            GENERIC_COPY(p1, p2, len);
            p1 += len;
            p2 += len;
        }
        else {
            GENERIC_COPY(p1, p3, len);
            p1 += len;
            p3 += len;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(char) * (p2 - p1));
    }
}

static void
npy_merge_right(char *p1, npy_intp l1, char *p2, npy_intp l2, char *p3,
                size_t len, PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp ofs;
    char *start = p1 - len;
    memcpy(p3, p2, sizeof(char) * l2 * len);
    p1 += (l1 - 1) * len;
    p2 += (l2 - 1) * len;
    p3 += (l2 - 1) * len;
    /* first element must be in p1 otherwise skipped in the caller */
    GENERIC_COPY(p2, p1, len);
    p2 -= len;
    p1 -= len;

    while (p1 < p2 && start < p1) {
        if (cmp(p3, p1, py_arr) < 0) {
            GENERIC_COPY(p2, p1, len);
            p2 -= len;
            p1 -= len;
        }
        else {
            GENERIC_COPY(p2, p3, len);
            p2 -= len;
            p3 -= len;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + len, p3 - ofs + len, sizeof(char) * ofs);
    }
}

static int
npy_merge_at(char *arr, const run *stack, const npy_intp at,
             buffer_char *buffer, size_t len, PyArray_CompareFunc *cmp,
             PyArrayObject *py_arr)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    char *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* arr[s2] belongs to arr[s1+k] */
    GENERIC_COPY(buffer->pw, arr + s2 * len, len);
    k = npy_gallop_right(arr + s1 * len, l1, buffer->pw, len, cmp, py_arr);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = arr + (s1 + k) * len;
    l1 -= k;
    p2 = arr + s2 * len;
    /* arr[s2-1] belongs to arr[s2+l2] */
    GENERIC_COPY(buffer->pw, arr + (s2 - 1) * len, len);
    l2 = npy_gallop_left(arr + s2 * len, l2, buffer->pw, len, cmp, py_arr);

    if (l2 < l1) {
        ret = resize_buffer_char(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        npy_merge_right(p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);
    }
    else {
        ret = resize_buffer_char(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        npy_merge_left(p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);
    }

    return 0;
}

static int
npy_try_collapse(char *arr, run *stack, npy_intp *stack_ptr,
                 buffer_char *buffer, size_t len, PyArray_CompareFunc *cmp,
                 PyArrayObject *py_arr)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = npy_merge_at(arr, stack, top - 3, buffer, len, cmp,
                                   py_arr);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp,
                                   py_arr);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp, py_arr);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

static int
npy_force_collapse(char *arr, run *stack, npy_intp *stack_ptr,
                   buffer_char *buffer, size_t len, PyArray_CompareFunc *cmp,
                   PyArrayObject *py_arr)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = npy_merge_at(arr, stack, top - 3, buffer, len, cmp, py_arr);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp, py_arr);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = npy_merge_at(arr, stack, top - 2, buffer, len, cmp, py_arr);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

NPY_NO_EXPORT int
npy_timsort(void *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(varr);
    size_t len = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    run stack[TIMSORT_STACK_SIZE];
    buffer_char buffer;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    buffer.pw = NULL;
    buffer.size = 0;
    buffer.len = len;
    stack_ptr = 0;
    minrun = compute_min_run_short(num);

    /* used for insertion sort and gallop key */
    ret = resize_buffer_char(&buffer, len);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    for (l = 0; l < num;) {
        n = npy_count_run((char *)start, l, num, minrun, buffer.pw, len, cmp,
                          arr);

        /* both s and l are scaled by len */
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = npy_try_collapse((char *)start, stack, &stack_ptr, &buffer, len,
                               cmp, arr);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = npy_force_collapse((char *)start, stack, &stack_ptr, &buffer, len,
                             cmp, arr);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (buffer.pw != NULL) {
        free(buffer.pw);
    }
    return ret;
}

/* argsort */

static npy_intp
npy_acount_run(char *arr, npy_intp *tosort, npy_intp l, npy_intp num,
               npy_intp minrun, size_t len, PyArray_CompareFunc *cmp,
               PyArrayObject *py_arr)
{
    npy_intp sz;
    npy_intp vi;
    npy_intp *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = tosort + l;

    /* (not strictly) ascending sequence */
    if (cmp(arr + (*pl) * len, arr + (*(pl + 1)) * len, py_arr) <= 0) {
        for (pi = pl + 1;
             pi < tosort + num - 1 &&
             cmp(arr + (*pi) * len, arr + (*(pi + 1)) * len, py_arr) <= 0;
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1;
             pi < tosort + num - 1 &&
             cmp(arr + (*(pi + 1)) * len, arr + (*pi) * len, py_arr) < 0;
             ++pi) {
        }

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    ++pi;
    sz = pi - pl;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* insertion sort */
        for (; pi < pr; ++pi) {
            vi = *pi;
            pj = pi;

            while (pl < pj &&
                   cmp(arr + vi * len, arr + (*(pj - 1)) * len, py_arr) < 0) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vi;
        }
    }

    return sz;
}

static npy_intp
npy_agallop_left(const char *arr, const npy_intp *tosort, const npy_intp size,
                 const char *key, size_t len, PyArray_CompareFunc *cmp,
                 PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (cmp(arr + tosort[size - 1] * len, key, py_arr) < 0) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (cmp(arr + tosort[size - ofs - 1] * len, key, py_arr) < 0) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[size-ofs-1]*len] < key <=
     * arr[tosort[size-last_ofs-1]*len] */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (cmp(arr + tosort[m] * len, key, py_arr) < 0) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[tosort[r-1]*len] < key <= arr[tosort[r]*len] */
    return r;
}

static npy_intp
npy_agallop_right(const char *arr, const npy_intp *tosort, const npy_intp size,
                  const char *key, size_t len, PyArray_CompareFunc *cmp,
                  PyArrayObject *py_arr)
{
    npy_intp last_ofs, ofs, m;

    if (cmp(key, arr + tosort[0] * len, py_arr) < 0) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (cmp(key, arr + tosort[ofs] * len, py_arr) < 0) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[last_ofs]*len] <= key < arr[tosort[ofs]*len] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (cmp(key, arr + tosort[m] * len, py_arr) < 0) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[tosort[ofs-1]*len] <= key < arr[tosort[ofs]*len] */
    return ofs;
}

static void
npy_amerge_left(char *arr, npy_intp *p1, npy_intp l1, npy_intp *p2,
                npy_intp l2, npy_intp *p3, size_t len,
                PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp *end = p2 + l2;
    memcpy(p3, p1, sizeof(npy_intp) * l1);
    /* first element must be in p2 otherwise skipped in the caller */
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        if (cmp(arr + (*p2) * len, arr + (*p3) * len, py_arr) < 0) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(npy_intp) * (p2 - p1));
    }
}

static void
npy_amerge_right(char *arr, npy_intp *p1, npy_intp l1, npy_intp *p2,
                 npy_intp l2, npy_intp *p3, size_t len,
                 PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    npy_intp ofs;
    npy_intp *start = p1 - 1;
    memcpy(p3, p2, sizeof(npy_intp) * l2);
    p1 += l1 - 1;
    p2 += l2 - 1;
    p3 += l2 - 1;
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;

    while (p1 < p2 && start < p1) {
        if (cmp(arr + (*p3) * len, arr + (*p1) * len, py_arr) < 0) {
            *p2-- = *p1--;
        }
        else {
            *p2-- = *p3--;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + 1, p3 - ofs + 1, sizeof(npy_intp) * ofs);
    }
}

static int
npy_amerge_at(char *arr, npy_intp *tosort, const run *stack, const npy_intp at,
              buffer_intp *buffer, size_t len, PyArray_CompareFunc *cmp,
              PyArrayObject *py_arr)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    npy_intp *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* tosort[s2] belongs to tosort[s1+k] */
    k = npy_agallop_right(arr, tosort + s1, l1, arr + tosort[s2] * len, len,
                          cmp, py_arr);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = tosort + s1 + k;
    l1 -= k;
    p2 = tosort + s2;
    /* tosort[s2-1] belongs to tosort[s2+l2] */
    l2 = npy_agallop_left(arr, tosort + s2, l2, arr + tosort[s2 - 1] * len,
                          len, cmp, py_arr);

    if (l2 < l1) {
        ret = resize_buffer_intp(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        npy_amerge_right(arr, p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);
    }
    else {
        ret = resize_buffer_intp(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        npy_amerge_left(arr, p1, l1, p2, l2, buffer->pw, len, cmp, py_arr);
    }

    return 0;
}

static int
npy_atry_collapse(char *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                  buffer_intp *buffer, size_t len, PyArray_CompareFunc *cmp,
                  PyArrayObject *py_arr)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = npy_amerge_at(arr, tosort, stack, top - 3, buffer, len,
                                    cmp, py_arr);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len,
                                    cmp, py_arr);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len, cmp,
                                py_arr);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

static int
npy_aforce_collapse(char *arr, npy_intp *tosort, run *stack,
                    npy_intp *stack_ptr, buffer_intp *buffer, size_t len,
                    PyArray_CompareFunc *cmp, PyArrayObject *py_arr)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = npy_amerge_at(arr, tosort, stack, top - 3, buffer, len, cmp,
                                py_arr);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len, cmp,
                                py_arr);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = npy_amerge_at(arr, tosort, stack, top - 2, buffer, len, cmp,
                            py_arr);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

NPY_NO_EXPORT int
npy_atimsort(void *start, npy_intp *tosort, npy_intp num, void *varr)
{
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(varr);
    size_t len = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyArray_DESCR(arr)->f->compare;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    run stack[TIMSORT_STACK_SIZE];
    buffer_intp buffer;

    /* Items that have zero size don't make sense to sort */
    if (len == 0) {
        return 0;
    }

    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run_short(num);

    for (l = 0; l < num;) {
        n = npy_acount_run((char *)start, tosort, l, num, minrun, len, cmp,
                           arr);
        /* both s and l are scaled by len */
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = npy_atry_collapse((char *)start, tosort, stack, &stack_ptr,
                                &buffer, len, cmp, arr);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = npy_aforce_collapse((char *)start, tosort, stack, &stack_ptr,
                              &buffer, len, cmp, arr);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (buffer.pw != NULL) {
        free(buffer.pw);
    }
    return ret;
}

/***************************************
 * C > C++ dispatch
 ***************************************/

NPY_NO_EXPORT int
timsort_bool(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::bool_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_byte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::byte_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_ubyte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::ubyte_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_short(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::short_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_ushort(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::ushort_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_int(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::int_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_uint(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::uint_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_long(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::long_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_ulong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::ulong_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_longlong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::longlong_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_ulonglong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::ulonglong_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_half(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::half_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_float(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::float_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_double(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::double_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_longdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::longdouble_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_cfloat(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::cfloat_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_cdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::cdouble_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_clongdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::clongdouble_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_datetime(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::datetime_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_timedelta(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return timsort_<npy::timedelta_tag>(start, num);
}
NPY_NO_EXPORT int
timsort_string(void *start, npy_intp num, void *varr)
{
    return string_timsort_<npy::string_tag>(start, num, varr);
}
NPY_NO_EXPORT int
timsort_unicode(void *start, npy_intp num, void *varr)
{
    return string_timsort_<npy::unicode_tag>(start, num, varr);
}

NPY_NO_EXPORT int
atimsort_bool(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::bool_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_byte(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::byte_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_ubyte(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::ubyte_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_short(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::short_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_ushort(void *v, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return atimsort_<npy::ushort_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_int(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::int_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_uint(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::uint_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_long(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::long_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_ulong(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::ulong_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_longlong(void *v, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return atimsort_<npy::longlong_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_ulonglong(void *v, npy_intp *tosort, npy_intp num,
                   void *NPY_UNUSED(varr))
{
    return atimsort_<npy::ulonglong_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_half(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::half_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_float(void *v, npy_intp *tosort, npy_intp num, void *NPY_UNUSED(varr))
{
    return atimsort_<npy::float_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_double(void *v, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return atimsort_<npy::double_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_longdouble(void *v, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    return atimsort_<npy::longdouble_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_cfloat(void *v, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    return atimsort_<npy::cfloat_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_cdouble(void *v, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    return atimsort_<npy::cdouble_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_clongdouble(void *v, npy_intp *tosort, npy_intp num,
                     void *NPY_UNUSED(varr))
{
    return atimsort_<npy::clongdouble_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_datetime(void *v, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    return atimsort_<npy::datetime_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_timedelta(void *v, npy_intp *tosort, npy_intp num,
                   void *NPY_UNUSED(varr))
{
    return atimsort_<npy::timedelta_tag>(v, tosort, num);
}
NPY_NO_EXPORT int
atimsort_string(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    return string_atimsort_<npy::string_tag>(v, tosort, num, varr);
}
NPY_NO_EXPORT int
atimsort_unicode(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    return string_atimsort_<npy::unicode_tag>(v, tosort, num, varr);
}
