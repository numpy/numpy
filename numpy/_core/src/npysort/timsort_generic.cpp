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

#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include <cstdlib>
#include <utility>

/* enough for 32 * 1.618 ** 128 elements.
   If powersort was used in all cases, 90 would suffice, as  32 * 2 ** 90  >=  32 * 1.618 ** 128  */
#define RUN_STACK_SIZE 128

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

typedef struct {
    npy_intp s; /* start pointer */
    npy_intp l; /* length */
    int power;  /* node "level" for powersort merge strategy */
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

    npy_intp *new_pw = (npy_intp *)realloc(buffer->pw, new_size * sizeof(npy_intp));

    buffer->size = new_size;

    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        buffer->pw = new_pw;
        return 0;
    }
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

    char *new_pw = (char *)realloc(buffer->pw, sizeof(char) * new_size * buffer->len);
    buffer->size = new_size;

    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        buffer->pw = new_pw;
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
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    run stack[RUN_STACK_SIZE];
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
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    run stack[RUN_STACK_SIZE];
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
