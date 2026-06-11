#ifndef __NPY_SORT_H__
#define __NPY_SORT_H__

/* Python include is for future object sorts */
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>
#include <dtypemeta.h>

#define NPY_ENOMEM 1
#define NPY_ECOMP 2

static inline int npy_get_msb(npy_uintp unum)
{
    int depth_limit = 0;
    while (unum >>= 1)  {
        depth_limit++;
    }
    return depth_limit;
}

#ifdef __cplusplus
extern "C" {
#endif


/*
 *****************************************************************************
 **                       NEW SORT METHOD REGISTRATIONS                     **
 *****************************************************************************
 */

NPY_NO_EXPORT int register_all_sorts(void);


/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */


NPY_NO_EXPORT int npy_quicksort(void *vec, npy_intp cnt, void *arr);
NPY_NO_EXPORT int npy_timsort(void *vec, npy_intp cnt, void *arr);
NPY_NO_EXPORT int npy_aquicksort(void *vec, npy_intp *ind, npy_intp cnt, void *arr);
NPY_NO_EXPORT int npy_atimsort(void *vec, npy_intp *ind, npy_intp cnt, void *arr);

/*
 *****************************************************************************
 **                         NEW-STYLE GENERIC SORT                          **
 *****************************************************************************
 */

NPY_NO_EXPORT int npy_default_sort_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata);
NPY_NO_EXPORT int npy_default_argsort_loop(PyArrayMethod_Context *context,
        char *const *data, const npy_intp *dimensions, const npy_intp *strides,
        NpyAuxData *transferdata);

/*
 *****************************************************************************
 **                      GENERIC SORT IMPLEMENTATIONS                       **
 *****************************************************************************
 */

typedef int (PyArray_SortImpl)(void *start, npy_intp num, void *varr,
                               npy_intp elsize, PyArray_CompareFunc *cmp);
typedef int (PyArray_ArgSortImpl)(void *vv, npy_intp *tosort, npy_intp n,
                                  void *varr, npy_intp elsize,
                                  PyArray_CompareFunc *cmp);

NPY_NO_EXPORT int npy_quicksort_impl(void *start, npy_intp num, void *varr,
                                     npy_intp elsize, PyArray_CompareFunc *cmp);
NPY_NO_EXPORT int npy_heapsort_impl(void *start, npy_intp num, void *varr,
                                    npy_intp elsize, PyArray_CompareFunc *cmp);
NPY_NO_EXPORT int npy_timsort_impl(void *start, npy_intp num, void *varr,
                                   npy_intp elsize, PyArray_CompareFunc *cmp);
NPY_NO_EXPORT int npy_aquicksort_impl(void *vv, npy_intp *tosort, npy_intp num, void *varr,
                                      npy_intp elsize, PyArray_CompareFunc *cmp);
NPY_NO_EXPORT int npy_aheapsort_impl(void *vv, npy_intp *tosort, npy_intp num, void *varr,
                                     npy_intp elsize, PyArray_CompareFunc *cmp);
NPY_NO_EXPORT int npy_atimsort_impl(void *v, npy_intp *tosort, npy_intp num, void *varr,
                                    npy_intp elsize, PyArray_CompareFunc *cmp);


#ifdef __cplusplus
}
#endif

#endif
