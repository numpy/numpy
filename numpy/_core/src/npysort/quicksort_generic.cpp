/* -*- c -*- */

/*
 * Comparator-function-driven version of the introsort implemented in
 * ``quicksort.hpp``.  Used by dtypes that register a
 * ``PyArray_CompareFunc`` rather than the type-specialised path.
 * See ``quicksort.hpp`` for the algorithm description.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"

#include <cstdlib>

/*
 * pushing largest partition has upper bound of log2(n) space
 * we store two pointers each time
 */
#define PYA_QS_STACK (NPY_BITSOF_INTP * 2)
#define SMALL_QUICKSORT 15

/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

NPY_NO_EXPORT int
npy_quicksort(void *start, npy_intp num, void *varr)
{
    npy_intp elsize;
    PyArray_CompareFunc *cmp;
    get_sort_data_from_array(varr, &elsize, &cmp);

    return npy_quicksort_impl(start, num, varr, elsize, cmp);
}

NPY_NO_EXPORT int
npy_quicksort_impl(void *start, npy_intp num, void *varr,
                   npy_intp elsize, PyArray_CompareFunc *cmp)
{
    void *arr = varr;
    char *vp;
    char *pl = (char *)start;
    char *pr = pl + (num - 1) * elsize;
    char *stack[PYA_QS_STACK];
    char **sptr = stack;
    char *pm, *pi, *pj, *pk;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    vp = (char *)malloc(elsize);
    if (vp == NULL) {
        return -NPY_ENOMEM;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_heapsort_impl(pl, (pr - pl) / elsize + 1, varr, elsize, cmp);
            goto stack_pop;
        }
        while (pr - pl > SMALL_QUICKSORT * elsize) {
            /* quicksort partition */
            pm = pl + (((pr - pl) / elsize) >> 1) * elsize;
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            if (cmp(pr, pm, arr) < 0) {
                GENERIC_SWAP(pr, pm, elsize);
            }
            if (cmp(pm, pl, arr) < 0) {
                GENERIC_SWAP(pm, pl, elsize);
            }
            GENERIC_COPY(vp, pm, elsize);
            pi = pl;
            pj = pr - elsize;
            GENERIC_SWAP(pm, pj, elsize);
            /*
             * Generic comparisons may be buggy, so don't rely on the sentinels
             * to keep the pointers from going out of bounds.
             */
            for (;;) {
                do {
                    pi += elsize;
                } while (cmp(pi, vp, arr) < 0 && pi < pj);
                do {
                    pj -= elsize;
                } while (cmp(vp, pj, arr) < 0 && pi < pj);
                if (pi >= pj) {
                    break;
                }
                GENERIC_SWAP(pi, pj, elsize);
            }
            pk = pr - elsize;
            GENERIC_SWAP(pi, pk, elsize);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + elsize;
                *sptr++ = pr;
                pr = pi - elsize;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - elsize;
                pl = pi + elsize;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + elsize; pi <= pr; pi += elsize) {
            GENERIC_COPY(vp, pi, elsize);
            pj = pi;
            pk = pi - elsize;
            while (pj > pl && cmp(vp, pk, arr) < 0) {
                GENERIC_COPY(pj, pk, elsize);
                pj -= elsize;
                pk -= elsize;
            }
            GENERIC_COPY(pj, vp, elsize);
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    free(vp);
    return 0;
}

NPY_NO_EXPORT int
npy_aquicksort(void *vv, npy_intp *tosort, npy_intp num, void *varr)
{
    npy_intp elsize;
    PyArray_CompareFunc *cmp;
    get_sort_data_from_array(varr, &elsize, &cmp);

    return npy_aquicksort_impl(vv, tosort, num, varr, elsize, cmp);
}

NPY_NO_EXPORT int
npy_aquicksort_impl(void *vv, npy_intp *tosort, npy_intp num, void *varr,
                    npy_intp elsize, PyArray_CompareFunc *cmp)
{
    void *arr = varr;
    char *v = (char *)vv;
    char *vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int *psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            npy_aheapsort_impl(vv, pl, pr - pl + 1, varr, elsize, cmp);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            if (cmp(v + (*pr) * elsize, v + (*pm) * elsize, arr) < 0) {
                INTP_SWAP(*pr, *pm);
            }
            if (cmp(v + (*pm) * elsize, v + (*pl) * elsize, arr) < 0) {
                INTP_SWAP(*pm, *pl);
            }
            vp = v + (*pm) * elsize;
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (cmp(v + (*pi) * elsize, vp, arr) < 0 && pi < pj);
                do {
                    --pj;
                } while (cmp(vp, v + (*pj) * elsize, arr) < 0 && pi < pj);
                if (pi >= pj) {
                    break;
                }
                INTP_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi, *pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v + vi * elsize;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && cmp(vp, v + (*pk) * elsize, arr) < 0) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}
