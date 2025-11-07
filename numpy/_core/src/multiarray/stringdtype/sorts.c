#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

#include "dtypemeta.h"
#include "gil_utils.h"
#include "sorts.h"
#include "static_string.h"

int
_cmp(char *a, char *b, PyArray_StringDTypeObject *descr, int descending)
{
    npy_string_allocator *allocator = descr->allocator;
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    int has_nan_na = descr->has_nan_na;
    npy_static_string *default_string = &descr->default_string;
    const npy_packed_static_string *ps_a = (npy_packed_static_string *)a;
    npy_static_string s_a = {0, NULL};
    int a_is_null = NpyString_load(allocator, ps_a, &s_a);
    const npy_packed_static_string *ps_b = (npy_packed_static_string *)b;
    npy_static_string s_b = {0, NULL};
    int b_is_null = NpyString_load(allocator, ps_b, &s_b);
    if (NPY_UNLIKELY(a_is_null == -1 || b_is_null == -1)) {
        char *msg = "Failed to load string in string comparison";
        npy_gil_error(PyExc_MemoryError, msg);
        return 0;
    }
    else if (NPY_UNLIKELY(a_is_null || b_is_null)) {
        if (has_null && !has_string_na) {
            if (has_nan_na) {
                // we do not consider descending here, as NaNs are always
                // sorted to the end
                if (a_is_null) {
                    return 1;
                }
                else if (b_is_null) {
                    return -1;
                }
            }
            else {
                npy_gil_error(PyExc_ValueError,
                              "Cannot compare null that is not a nan-like value");
                return 0;
            }
        }
        else {
            if (a_is_null) {
                s_a = *default_string;
            }
            if (b_is_null) {
                s_b = *default_string;
            }
        }
    }
    int cmp = NpyString_cmp(&s_a, &s_b);
    if (descending) {
        cmp = -cmp;
    }
    return cmp;
}

static void
string_mergesort0(char *pl, char *pr, char *pw, char *vp, npy_intp elsize,
                  PyArray_StringDTypeObject *sdescr, int descending)
{
    char *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_STRING_MERGESORT * elsize) {
        /* merge sort */
        pm = pl + (((pr - pl) / elsize) >> 1) * elsize;
        string_mergesort0(pl, pm, pw, vp, elsize, sdescr, descending);
        string_mergesort0(pm, pr, pw, vp, elsize, sdescr, descending);
        memcpy(pw, pl, pm - pl);
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (_cmp(pm, pj, sdescr, descending) < 0) {
                memcpy(pk, pm, elsize);
                pm += elsize;
                pk += elsize;
            }
            else {
                memcpy(pk, pj, elsize);
                pj += elsize;
                pk += elsize;
            }
        }
        memcpy(pk, pj, pi - pj);
    }
    else {
        /* insertion sort */
        for (pi = pl + elsize; pi < pr; pi += elsize) {
            memcpy(vp, pi, elsize);
            pj = pi;
            pk = pi - elsize;
            while (pj > pl && _cmp(vp, pk, sdescr, descending) < 0) {
                memcpy(pj, pk, elsize);
                pj -= elsize;
                pk -= elsize;
            }
            memcpy(pj, vp, elsize);
        }
    }
}

int
stringdtype_stablesort(PyArrayMethod_Context *context, char *const *data,
                       const npy_intp *dimensions, const npy_intp *NPY_UNUSED(strides),
                       NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr *descr = context->descriptors[0];
    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)descr;
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);

    PyArrayMethod_SortParameters *parameters = (PyArrayMethod_SortParameters *)context->parameters;
    int descending = (parameters->flags & NPY_SORT_DESCENDING);

    npy_intp num = dimensions[0];
    npy_intp elsize = descr->elsize;
    char *pl = (char *)data[0];
    char *pr = pl + num * elsize;
    char *pw;
    char *vp;
    int err = -1;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    pw = (char *)malloc((num >> 1) * elsize);
    vp = (char *)malloc(elsize);

    if (pw != NULL && vp != NULL) {
        string_mergesort0(pl, pr, pw, vp, elsize, sdescr, descending);
        err = 0;
    }

    free(vp);
    free(pw);
    NpyString_release_allocator(allocator);

    return err;
}

int
stringdtype_sort_get_loop(PyArrayMethod_Context *context, int aligned,
                          int move_references, const npy_intp *strides,
                          PyArrayMethod_StridedLoop **out_loop,
                          NpyAuxData **out_transferdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &stringdtype_stablesort;  // default to mergesort
    return 0;
}

static void
string_amergesort0(npy_intp *pl, npy_intp *pr, char *v, npy_intp *pw, npy_intp elsize,
                   PyArray_StringDTypeObject *sdescr, int descending)
{
    char *vp;
    npy_intp vi, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_STRING_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);
        string_amergesort0(pl, pm, v, pw, elsize, sdescr, descending);
        string_amergesort0(pm, pr, v, pw, elsize, sdescr, descending);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (_cmp(v + (*pm) * elsize, v + (*pj) * elsize, sdescr, descending) < 0) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v + vi * elsize;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && _cmp(vp, v + (*pk) * elsize, sdescr, descending) < 0) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

NPY_NO_EXPORT int
stringdtype_stableargsort(PyArrayMethod_Context *context, char *const *data,
                          const npy_intp *dimensions,
                          const npy_intp *NPY_UNUSED(strides),
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr *descr = context->descriptors[0];
    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)descr;
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);

    PyArrayMethod_SortParameters *parameters = (PyArrayMethod_SortParameters *)context->parameters;
    int descending = (parameters->flags & NPY_SORT_DESCENDING);

    npy_intp num = dimensions[0];
    npy_intp elsize = descr->elsize;
    npy_intp *pl, *pr, *pw;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;
    }

    pl = (npy_intp *)data[1];
    pr = pl + num;
    pw = (npy_intp *)malloc((num >> 1) * sizeof(npy_intp));
    if (pw == NULL) {
        return -2;
    }
    string_amergesort0(pl, pr, (char *)data[0], pw, elsize, sdescr, descending);
    free(pw);
    NpyString_release_allocator(allocator);

    return 0;
}

int
stringdtype_argsort_get_loop(PyArrayMethod_Context *context, int aligned,
                             int move_references, const npy_intp *strides,
                             PyArrayMethod_StridedLoop **out_loop,
                             NpyAuxData **out_transferdata,
                             NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags |= NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &stringdtype_stableargsort;  // default to mergesort
    return 0;
}

int
init_stringdtype_sorts(void)
{
    PyArray_DTypeMeta *dtypes[2] = {&PyArray_StringDType, &PyArray_StringDType};
    PyType_Slot sort_slots[2] = {{NPY_METH_get_loop, &stringdtype_sort_get_loop},
                                 {0, NULL}};
    PyArrayMethod_Spec sort_spec = {
            .name = "stringdtype_sort",
            .nin = 1,
            .nout = 1,
            .dtypes = dtypes,
            .slots = sort_slots,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
    };

    PyType_Slot argsort_slots[2] = {{NPY_METH_get_loop, &stringdtype_argsort_get_loop},
                                    {0, NULL}};
    PyArrayMethod_Spec argsort_spec = {
            .name = "stringdtype_argsort",
            .nin = 1,
            .nout = 1,
            .dtypes = dtypes,
            .slots = argsort_slots,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
    };

    PyBoundArrayMethodObject *sort_method = PyArrayMethod_FromSpec_int(&sort_spec, 1);
    if (sort_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(&PyArray_StringDType)->sort_meth =
            (PyArrayMethodObject *)sort_method->method;
    Py_INCREF(sort_method->method);
    Py_DECREF(sort_method);

    PyBoundArrayMethodObject *argsort_method = PyArrayMethod_FromSpec_int(&argsort_spec, 1);
    if (argsort_method == NULL) {
        return -1;
    }
    NPY_DT_SLOTS(&PyArray_StringDType)->argsort_meth =
            (PyArrayMethodObject *)argsort_method->method;
    Py_INCREF(argsort_method->method);
    Py_DECREF(argsort_method);

    return 0;
}