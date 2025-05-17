#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/npy_math.h"
#include "npy_argparse.h"
#include "npy_config.h"
#include "templ_common.h" /* for npy_mul_sizes_with_overflow */
#include "lowlevel_strided_loops.h" /* for npy_bswap8 */
#include "alloc.h"
#include "ctors.h"
#include "common.h"
#include "dtypemeta.h"
#include "simd/simd.h"

#include <string.h>

typedef enum {
    PACK_ORDER_LITTLE = 0,
    PACK_ORDER_BIG
} PACK_ORDER;

/*
 * Returns -1 if the array is monotonic decreasing,
 * +1 if the array is monotonic increasing,
 * and 0 if the array is not monotonic.
 */
static int
check_array_monotonic(const double *a, npy_intp lena)
{
    npy_intp i;
    double next;
    double last;

    if (lena == 0) {
        /* all bin edges hold the same value */
        return 1;
    }
    last = a[0];

    /* Skip repeated values at the beginning of the array */
    for (i = 1; (i < lena) && (a[i] == last); i++);

    if (i == lena) {
        /* all bin edges hold the same value */
        return 1;
    }

    next = a[i];
    if (last < next) {
        /* Possibly monotonic increasing */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last > next) {
                return 0;
            }
        }
        return 1;
    }
    else {
        /* last > next, possibly monotonic decreasing */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last < next) {
                return 0;
            }
        }
        return -1;
    }
}

/* Find the minimum and maximum of an integer array */
static void
minmax(const npy_intp *data, npy_intp data_len, npy_intp *mn, npy_intp *mx)
{
    npy_intp min = *data;
    npy_intp max = *data;

    while (--data_len) {
        const npy_intp val = *(++data);
        if (val < min) {
            min = val;
        }
        else if (val > max) {
            max = val;
        }
    }

    *mn = min;
    *mx = max;
}

/*
 * arr_bincount is registered as bincount.
 *
 * bincount accepts one, two or three arguments. The first is an array of
 * non-negative integers. The second, if present, is an array of weights,
 * which must be promotable to double. Call these arguments list and
 * weight. Both must be one-dimensional with len(weight) == len(list). If
 * weight is not present then bincount(list)[i] is the number of occurrences
 * of i in list.  If weight is present then bincount(self,list, weight)[i]
 * is the sum of all weight[j] where list [j] == i.  Self is not used.
 * The third argument, if present, is a minimum length desired for the
 * output array.
 */
NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *NPY_UNUSED(self), PyObject *const *args,
                            Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *list = NULL, *weight = Py_None, *mlength = NULL;
    PyArrayObject *lst = NULL, *ans = NULL, *wts = NULL;
    npy_intp *numbers, *ians, len, mx, mn, ans_size;
    npy_intp minlength = 0;
    npy_intp i;
    double *weights , *dans;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("bincount", args, len_args, kwnames,
                "list", NULL, &list,
                "|weights", NULL, &weight,
                "|minlength", NULL, &mlength,
                NULL, NULL, NULL) < 0) {
        return NULL;
    }

    /*
     * Accepting arbitrary lists that are cast to NPY_INTP, possibly
     * losing precision because of unsafe casts, is deprecated.  We
     * continue to use PyArray_ContiguousFromAny(list, NPY_INTP, 1, 1)
     * to convert the input during the deprecation period, but we also
     * check to see if a deprecation warning should be generated.
     * Some refactoring will be needed when the deprecation expires.
     */

    /* Check to see if we should generate a deprecation warning. */
    if (!PyArray_Check(list)) {
        /* list is not a numpy array, so convert it. */
        PyArrayObject *tmp1 = (PyArrayObject *)PyArray_FromAny(
                                                    list, NULL, 1, 1,
                                                    NPY_ARRAY_DEFAULT, NULL);
        if (tmp1 == NULL) {
            goto fail;
        }
        if (PyArray_SIZE(tmp1) > 0) {
            /* The input is not empty, so convert it to NPY_INTP. */
            int flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;
            if (PyArray_ISINTEGER(tmp1)) {
                flags = flags | NPY_ARRAY_FORCECAST;
            }
            PyArray_Descr* local_dtype = PyArray_DescrFromType(NPY_INTP);
            lst = (PyArrayObject *)PyArray_FromAny((PyObject *)tmp1, local_dtype, 1, 1, flags, NULL);
            Py_DECREF(tmp1);
            if (lst == NULL) {
                /* Failed converting to NPY_INTP. */
                if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                    PyErr_Clear();
                    /* Deprecated 2024-08-02, NumPy 2.1 */
                    if (DEPRECATE("Non-integer input passed to bincount. In a "
                                  "future version of NumPy, this will be an "
                                  "error. (Deprecated NumPy 2.1)") < 0) {
                        goto fail;
                    }
                }
                else {
                    /* Failure was not a TypeError. */
                    goto fail;
                }
            }
        }
        else {
            /* Got an empty list. */
            Py_DECREF(tmp1);
        }
    }

    if (lst == NULL) {
        int flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;
        if (PyArray_Check((PyObject *)list) &&
            PyArray_ISINTEGER((PyArrayObject *)list)) {
            flags = flags | NPY_ARRAY_FORCECAST;
        }
        PyArray_Descr* local_dtype = PyArray_DescrFromType(NPY_INTP);
        lst = (PyArrayObject *)PyArray_FromAny(list, local_dtype, 1, 1, flags, NULL);
        if (lst == NULL) {
            goto fail;
        }
    }
    len = PyArray_SIZE(lst);

    /*
     * This if/else if can be removed by changing the argspec above,
     */
    if (mlength == Py_None) {
        PyErr_SetString(PyExc_TypeError,
             "use 0 instead of None for minlength");
        goto fail;
    }
    else if (mlength != NULL) {
        minlength = PyArray_PyIntAsIntp(mlength);
        if (error_converting(minlength)) {
            goto fail;
        }
    }

    if (minlength < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "'minlength' must not be negative");
        goto fail;
    }

    /* handle empty list */
    if (len == 0) {
        ans = (PyArrayObject *)PyArray_ZEROS(1, &minlength, NPY_INTP, 0);
        if (ans == NULL){
            goto fail;
        }
        Py_DECREF(lst);
        return (PyObject *)ans;
    }

    numbers = (npy_intp *)PyArray_DATA(lst);
    minmax(numbers, len, &mn, &mx);
    if (mn < 0) {
        PyErr_SetString(PyExc_ValueError,
                "'list' argument must have no negative elements");
        goto fail;
    }
    ans_size = mx + 1;
    if (mlength != Py_None) {
        if (ans_size < minlength) {
            ans_size = minlength;
        }
    }
    if (weight == Py_None) {
        ans = (PyArrayObject *)PyArray_ZEROS(1, &ans_size, NPY_INTP, 0);
        if (ans == NULL) {
            goto fail;
        }
        ians = (npy_intp *)PyArray_DATA(ans);
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < len; i++)
            ians[numbers[i]] += 1;
        NPY_END_ALLOW_THREADS;
        Py_DECREF(lst);
    }
    else {
        wts = (PyArrayObject *)PyArray_ContiguousFromAny(
                                                weight, NPY_DOUBLE, 1, 1);
        if (wts == NULL) {
            goto fail;
        }
        weights = (double *)PyArray_DATA(wts);
        if (PyArray_SIZE(wts) != len) {
            PyErr_SetString(PyExc_ValueError,
                    "The weights and list don't have the same length.");
            goto fail;
        }
        ans = (PyArrayObject *)PyArray_ZEROS(1, &ans_size, NPY_DOUBLE, 0);
        if (ans == NULL) {
            goto fail;
        }
        dans = (double *)PyArray_DATA(ans);
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < len; i++) {
            dans[numbers[i]] += weights[i];
        }
        NPY_END_ALLOW_THREADS;
        Py_DECREF(lst);
        Py_DECREF(wts);
    }
    return (PyObject *)ans;

fail:
    Py_XDECREF(lst);
    Py_XDECREF(wts);
    Py_XDECREF(ans);
    return NULL;
}

/* Internal function to expose check_array_monotonic to python */
NPY_NO_EXPORT PyObject *
arr__monotonicity(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"x", NULL};
    PyObject *obj_x = NULL;
    PyArrayObject *arr_x = NULL;
    long monotonic;
    npy_intp len_x;
    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:_monotonicity", kwlist,
                                     &obj_x)) {
        return NULL;
    }

    /*
     * TODO:
     *  `x` could be strided, needs change to check_array_monotonic
     *  `x` is forced to double for this check
     */
    arr_x = (PyArrayObject *)PyArray_FROMANY(
        obj_x, NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY_RO);
    if (arr_x == NULL) {
        return NULL;
    }

    len_x = PyArray_SIZE(arr_x);
    NPY_BEGIN_THREADS_THRESHOLDED(len_x)
    monotonic = check_array_monotonic(
        (const double *)PyArray_DATA(arr_x), len_x);
    NPY_END_THREADS
    Py_DECREF(arr_x);

    return PyLong_FromLong(monotonic);
}

/*
 * Returns input array with values inserted sequentially into places
 * indicated by the mask
 */
NPY_NO_EXPORT PyObject *
arr_place(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{
    char *src, *dest;
    npy_bool *mask_data;
    PyArray_Descr *dtype;
    PyArray_CopySwapFunc *copyswap;
    PyObject *array0, *mask0, *values0;
    PyArrayObject *array, *mask, *values;
    npy_intp i, j, chunk, nm, ni, nv;

    static char *kwlist[] = {"input", "mask", "vals", NULL};
    NPY_BEGIN_THREADS_DEF;
    values = mask = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O!OO:place", kwlist,
                &PyArray_Type, &array0, &mask0, &values0)) {
        return NULL;
    }

    array = (PyArrayObject *)PyArray_FromArray((PyArrayObject *)array0, NULL,
                                    NPY_ARRAY_CARRAY | NPY_ARRAY_WRITEBACKIFCOPY);
    if (array == NULL) {
        goto fail;
    }

    ni = PyArray_SIZE(array);
    dest = PyArray_DATA(array);
    chunk = PyArray_ITEMSIZE(array);
    mask = (PyArrayObject *)PyArray_FROM_OTF(mask0, NPY_BOOL,
                                NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST);
    if (mask == NULL) {
        goto fail;
    }

    nm = PyArray_SIZE(mask);
    if (nm != ni) {
        PyErr_SetString(PyExc_ValueError,
                        "place: mask and data must be "
                        "the same size");
        goto fail;
    }

    mask_data = PyArray_DATA(mask);
    dtype = PyArray_DESCR(array);
    Py_INCREF(dtype);

    values = (PyArrayObject *)PyArray_FromAny(values0, dtype,
                                    0, 0, NPY_ARRAY_CARRAY, NULL);
    if (values == NULL) {
        goto fail;
    }

    nv = PyArray_SIZE(values); /* zero if null array */
    if (nv <= 0) {
        npy_bool allFalse = 1;
        i = 0;

        while (allFalse && i < ni) {
            if (mask_data[i]) {
                allFalse = 0;
            } else {
                i++;
            }
        }
        if (!allFalse) {
            PyErr_SetString(PyExc_ValueError,
                            "Cannot insert from an empty array!");
            goto fail;
        } else {
            Py_XDECREF(values);
            Py_XDECREF(mask);
            PyArray_ResolveWritebackIfCopy(array);
            Py_XDECREF(array);
            Py_RETURN_NONE;
        }
    }

    src = PyArray_DATA(values);
    j = 0;

    copyswap = PyDataType_GetArrFuncs(PyArray_DESCR(array))->copyswap;
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(array));
    for (i = 0; i < ni; i++) {
        if (mask_data[i]) {
            if (j >= nv) {
                j = 0;
            }

            copyswap(dest + i*chunk, src + j*chunk, 0, array);
            j++;
        }
    }
    NPY_END_THREADS;

    Py_XDECREF(values);
    Py_XDECREF(mask);
    PyArray_ResolveWritebackIfCopy(array);
    Py_DECREF(array);
    Py_RETURN_NONE;

 fail:
    Py_XDECREF(mask);
    PyArray_ResolveWritebackIfCopy(array);
    Py_XDECREF(array);
    Py_XDECREF(values);
    return NULL;
}

#define LIKELY_IN_CACHE_SIZE 8

#ifdef __INTEL_COMPILER
#pragma intel optimization_level 0
#endif
static inline npy_intp
_linear_search(const npy_double key, const npy_double *arr, const npy_intp len, const npy_intp i0)
{
    npy_intp i;

    for (i = i0; i < len && key >= arr[i]; i++);
    return i - 1;
}

/** @brief find index of a sorted array such that arr[i] <= key < arr[i + 1].
 *
 * If an starting index guess is in-range, the array values around this
 * index are first checked.  This allows for repeated calls for well-ordered
 * keys (a very common case) to use the previous index as a very good guess.
 *
 * If the guess value is not useful, bisection of the array is used to
 * find the index.  If there is no such index, the return values are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @param guess initial guess of index
 * @return index
 */
static npy_intp
binary_search_with_guess(const npy_double key, const npy_double *arr,
                         npy_intp len, npy_intp guess)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    /* Handle keys outside of the arr range first */
    if (key > arr[len - 1]) {
        return len;
    }
    else if (key < arr[0]) {
        return -1;
    }

    /*
     * If len <= 4 use linear search.
     * From above we know key >= arr[0] when we start.
     */
    if (len <= 4) {
        return _linear_search(key, arr, len, 1);
    }

    if (guess > len - 3) {
        guess = len - 3;
    }
    if (guess < 1)  {
        guess = 1;
    }

    /* check most likely values: guess - 1, guess, guess + 1 */
    if (key < arr[guess]) {
        if (key < arr[guess - 1]) {
            imax = guess - 1;
            /* last attempt to restrict search to items in cache */
            if (guess > LIKELY_IN_CACHE_SIZE &&
                        key >= arr[guess - LIKELY_IN_CACHE_SIZE]) {
                imin = guess - LIKELY_IN_CACHE_SIZE;
            }
        }
        else {
            /* key >= arr[guess - 1] */
            return guess - 1;
        }
    }
    else {
        /* key >= arr[guess] */
        if (key < arr[guess + 1]) {
            return guess;
        }
        else {
            /* key >= arr[guess + 1] */
            if (key < arr[guess + 2]) {
                return guess + 1;
            }
            else {
                /* key >= arr[guess + 2] */
                imin = guess + 2;
                /* last attempt to restrict search to items in cache */
                if (guess < len - LIKELY_IN_CACHE_SIZE - 1 &&
                            key < arr[guess + LIKELY_IN_CACHE_SIZE]) {
                    imax = guess + LIKELY_IN_CACHE_SIZE;
                }
            }
        }
    }

    /* finally, find index by bisection */
    while (imin < imax) {
        const npy_intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

#undef LIKELY_IN_CACHE_SIZE

NPY_NO_EXPORT PyObject *
arr_interp(PyObject *NPY_UNUSED(self), PyObject *const *args, Py_ssize_t len_args,
                             PyObject *kwnames)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;
    npy_double lval, rval;
    const npy_double *dy, *dx, *dz;
    npy_double *dres, *slopes = NULL;

    NPY_BEGIN_THREADS_DEF;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("interp", args, len_args, kwnames,
                "x", NULL, &x,
                "xp", NULL, &xp,
                "fp", NULL, &fp,
                "|left", NULL, &left,
                "|right", NULL, &right,
                NULL, NULL, NULL) < 0) {
        return NULL;
    }

    afp = (PyArrayObject *)PyArray_ContiguousFromAny(fp, NPY_DOUBLE, 1, 1);
    if (afp == NULL) {
        return NULL;
    }
    axp = (PyArrayObject *)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) {
        goto fail;
    }
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 0, 0);
    if (ax == NULL) {
        goto fail;
    }
    lenxp = PyArray_SIZE(axp);
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (PyArray_SIZE(afp) != lenxp) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    af = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ax),
                                            PyArray_DIMS(ax), NPY_DOUBLE);
    if (af == NULL) {
        goto fail;
    }
    lenx = PyArray_SIZE(ax);

    dy = (const npy_double *)PyArray_DATA(afp);
    dx = (const npy_double *)PyArray_DATA(axp);
    dz = (const npy_double *)PyArray_DATA(ax);
    dres = (npy_double *)PyArray_DATA(af);
    /* Get left and right fill values. */
    if ((left == NULL) || (left == Py_None)) {
        lval = dy[0];
    }
    else {
        lval = PyFloat_AsDouble(left);
        if (error_converting(lval)) {
            goto fail;
        }
    }
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp - 1];
    }
    else {
        rval = PyFloat_AsDouble(right);
        if (error_converting(rval)) {
            goto fail;
        }
    }

    /* binary_search_with_guess needs at least a 3 item long array */
    if (lenxp == 1) {
        const npy_double xp_val = dx[0];
        const npy_double fp_val = dy[0];

        NPY_BEGIN_THREADS_THRESHOLDED(lenx);
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];
            dres[i] = (x_val < xp_val) ? lval :
                                         ((x_val > xp_val) ? rval : fp_val);
        }
        NPY_END_THREADS;
    }
    else {
        npy_intp j = 0;

        /* only pre-calculate slopes if there are relatively few of them. */
        if (lenxp <= lenx) {
            slopes = PyArray_malloc((lenxp - 1) * sizeof(npy_double));
            if (slopes == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
        }

        NPY_BEGIN_THREADS;

        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                slopes[i] = (dy[i+1] - dy[i]) / (dx[i+1] - dx[i]);
            }
        }

        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];

            if (npy_isnan(x_val)) {
                dres[i] = x_val;
                continue;
            }

            j = binary_search_with_guess(x_val, dx, lenxp, j);
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (dx[j] == x_val) {
                /* Avoid potential non-finite interpolation */
                dres[i] = dy[j];
            }
            else {
                const npy_double slope =
                        (slopes != NULL) ? slopes[j] :
                        (dy[j+1] - dy[j]) / (dx[j+1] - dx[j]);

                /* If we get nan in one direction, try the other */
                dres[i] = slope*(x_val - dx[j]) + dy[j];
                if (NPY_UNLIKELY(npy_isnan(dres[i]))) {
                    dres[i] = slope*(x_val - dx[j+1]) + dy[j+1];
                    if (NPY_UNLIKELY(npy_isnan(dres[i])) && dy[j] == dy[j+1]) {
                        dres[i] = dy[j];
                    }
                }
            }
        }

        NPY_END_THREADS;
    }

    PyArray_free(slopes);
    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);
    return PyArray_Return(af);

fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}

/* As for arr_interp but for complex fp values */
NPY_NO_EXPORT PyObject *
arr_interp_complex(PyObject *NPY_UNUSED(self), PyObject *const *args, Py_ssize_t len_args,
                             PyObject *kwnames)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;

    const npy_double *dx, *dz;
    const npy_cdouble *dy;
    npy_cdouble lval, rval;
    npy_cdouble *dres, *slopes = NULL;

    NPY_BEGIN_THREADS_DEF;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("interp_complex", args, len_args, kwnames,
                "x", NULL, &x,
                "xp", NULL, &xp,
                "fp", NULL, &fp,
                "|left", NULL, &left,
                "|right", NULL, &right,
                NULL, NULL, NULL) < 0) {
        return NULL;
    }

    afp = (PyArrayObject *)PyArray_ContiguousFromAny(fp, NPY_CDOUBLE, 1, 1);

    if (afp == NULL) {
        return NULL;
    }

    axp = (PyArrayObject *)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) {
        goto fail;
    }
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 0, 0);
    if (ax == NULL) {
        goto fail;
    }
    lenxp = PyArray_SIZE(axp);
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (PyArray_SIZE(afp) != lenxp) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    lenx = PyArray_SIZE(ax);
    dx = (const npy_double *)PyArray_DATA(axp);
    dz = (const npy_double *)PyArray_DATA(ax);

    af = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ax),
                                            PyArray_DIMS(ax), NPY_CDOUBLE);
    if (af == NULL) {
        goto fail;
    }

    dy = (const npy_cdouble *)PyArray_DATA(afp);
    dres = (npy_cdouble *)PyArray_DATA(af);
    /* Get left and right fill values. */
    if ((left == NULL) || (left == Py_None)) {
        lval = dy[0];
    }
    else {
        npy_csetreal(&lval, PyComplex_RealAsDouble(left));
        if (error_converting(npy_creal(lval))) {
            goto fail;
        }
        npy_csetimag(&lval, PyComplex_ImagAsDouble(left));
        if (error_converting(npy_cimag(lval))) {
            goto fail;
        }
    }

    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp - 1];
    }
    else {
        npy_csetreal(&rval, PyComplex_RealAsDouble(right));
        if (error_converting(npy_creal(rval))) {
            goto fail;
        }
        npy_csetimag(&rval, PyComplex_ImagAsDouble(right));
        if (error_converting(npy_cimag(rval))) {
            goto fail;
        }
    }

    /* binary_search_with_guess needs at least a 3 item long array */
    if (lenxp == 1) {
        const npy_double xp_val = dx[0];
        const npy_cdouble fp_val = dy[0];

        NPY_BEGIN_THREADS_THRESHOLDED(lenx);
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];
            dres[i] = (x_val < xp_val) ? lval :
              ((x_val > xp_val) ? rval : fp_val);
        }
        NPY_END_THREADS;
    }
    else {
        npy_intp j = 0;

        /* only pre-calculate slopes if there are relatively few of them. */
        if (lenxp <= lenx) {
            slopes = PyArray_malloc((lenxp - 1) * sizeof(npy_cdouble));
            if (slopes == NULL) {
                PyErr_NoMemory();
                goto fail;
            }
        }

        NPY_BEGIN_THREADS;

        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                const double inv_dx = 1.0 / (dx[i+1] - dx[i]);
                npy_csetreal(&slopes[i], (npy_creal(dy[i+1]) - npy_creal(dy[i])) * inv_dx);
                npy_csetimag(&slopes[i], (npy_cimag(dy[i+1]) - npy_cimag(dy[i])) * inv_dx);
            }
        }

        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];

            if (npy_isnan(x_val)) {
                npy_csetreal(&dres[i], x_val);
                npy_csetimag(&dres[i], 0.0);
                continue;
            }

            j = binary_search_with_guess(x_val, dx, lenxp, j);
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (dx[j] == x_val) {
                /* Avoid potential non-finite interpolation */
                dres[i] = dy[j];
            }
            else {
                npy_cdouble slope;
                if (slopes != NULL) {
                    slope = slopes[j];
                }
                else {
                    const npy_double inv_dx = 1.0 / (dx[j+1] - dx[j]);
                    npy_csetreal(&slope, (npy_creal(dy[j+1]) - npy_creal(dy[j])) * inv_dx);
                    npy_csetimag(&slope, (npy_cimag(dy[j+1]) - npy_cimag(dy[j])) * inv_dx);
                }

                /* If we get nan in one direction, try the other */
                npy_csetreal(&dres[i], npy_creal(slope)*(x_val - dx[j]) + npy_creal(dy[j]));
                if (NPY_UNLIKELY(npy_isnan(npy_creal(dres[i])))) {
                    npy_csetreal(&dres[i], npy_creal(slope)*(x_val - dx[j+1]) + npy_creal(dy[j+1]));
                    if (NPY_UNLIKELY(npy_isnan(npy_creal(dres[i]))) &&
                            npy_creal(dy[j]) == npy_creal(dy[j+1])) {
                        npy_csetreal(&dres[i], npy_creal(dy[j]));
                    }
                }
                npy_csetimag(&dres[i], npy_cimag(slope)*(x_val - dx[j]) + npy_cimag(dy[j]));
                if (NPY_UNLIKELY(npy_isnan(npy_cimag(dres[i])))) {
                    npy_csetimag(&dres[i], npy_cimag(slope)*(x_val - dx[j+1]) + npy_cimag(dy[j+1]));
                    if (NPY_UNLIKELY(npy_isnan(npy_cimag(dres[i]))) &&
                            npy_cimag(dy[j]) == npy_cimag(dy[j+1])) {
                        npy_csetimag(&dres[i], npy_cimag(dy[j]));
                    }
                }
            }
        }

        NPY_END_THREADS;
    }
    PyArray_free(slopes);

    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);
    return PyArray_Return(af);

fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}

static const char EMPTY_SEQUENCE_ERR_MSG[] = "indices must be integral: the provided " \
    "empty sequence was inferred as float. Wrap it with " \
    "'np.array(indices, dtype=np.intp)'";

static const char NON_INTEGRAL_ERROR_MSG[] = "only int indices permitted";

/* Convert obj to an ndarray with integer dtype or fail */
static PyArrayObject *
astype_anyint(PyObject *obj) {
    PyArrayObject *ret;

    if (!PyArray_Check(obj)) {
        /* prefer int dtype */
        PyArray_Descr *dtype_guess = NULL;
        if (PyArray_DTypeFromObject(obj, NPY_MAXDIMS, &dtype_guess) < 0) {
            return NULL;
        }
        if (dtype_guess == NULL) {
            if (PySequence_Check(obj) && PySequence_Size(obj) == 0) {
                PyErr_SetString(PyExc_TypeError, EMPTY_SEQUENCE_ERR_MSG);
            }
            return NULL;
        }
        ret = (PyArrayObject*)PyArray_FromAny(obj, dtype_guess, 0, 0, 0, NULL);
        if (ret == NULL) {
            return NULL;
        }
    }
    else {
        ret = (PyArrayObject *)obj;
        Py_INCREF(ret);
    }

    if (!(PyArray_ISINTEGER(ret) || PyArray_ISBOOL(ret))) {
        /* ensure dtype is int-based */
        PyErr_SetString(PyExc_TypeError, NON_INTEGRAL_ERROR_MSG);
        Py_DECREF(ret);
        return NULL;
    }

    return ret;
}

/*
 * Converts a Python sequence into 'count' PyArrayObjects
 *
 * seq         - Input Python object, usually a tuple but any sequence works.
 *               Must have integral content.
 * paramname   - The name of the parameter that produced 'seq'.
 * count       - How many arrays there should be (errors if it doesn't match).
 * op          - Where the arrays are placed.
 */
static int int_sequence_to_arrays(PyObject *seq,
                              char *paramname,
                              int count,
                              PyArrayObject **op
                              )
{
    int i;

    if (!PySequence_Check(seq) || PySequence_Size(seq) != count) {
        PyErr_Format(PyExc_ValueError,
                "parameter %s must be a sequence of length %d",
                paramname, count);
        return -1;
    }

    for (i = 0; i < count; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);
        if (item == NULL) {
            goto fail;
        }
        op[i] = astype_anyint(item);
        Py_DECREF(item);
        if (op[i] == NULL) {
            goto fail;
        }
    }

    return 0;

fail:
    while (--i >= 0) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
    }
    return -1;
}

/* Inner loop for ravel_multi_index */
static int
ravel_multi_index_loop(int ravel_ndim, npy_intp *ravel_dims,
                        npy_intp *ravel_strides,
                        npy_intp count,
                        NPY_CLIPMODE *modes,
                        char **coords, npy_intp *coords_strides)
{
    int i;
    char invalid;
    npy_intp j, m;

    /*
     * Check for 0-dimensional axes unless there is nothing to do.
     * An empty array/shape cannot be indexed at all.
     */
    if (count != 0) {
        for (i = 0; i < ravel_ndim; ++i) {
            if (ravel_dims[i] == 0) {
                PyErr_SetString(PyExc_ValueError,
                        "cannot unravel if shape has zero entries (is empty).");
                return NPY_FAIL;
            }
        }
    }

    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    while (count--) {
        npy_intp raveled = 0;
        for (i = 0; i < ravel_ndim; ++i) {
            m = ravel_dims[i];
            j = *(npy_intp *)coords[i];
            switch (modes[i]) {
                case NPY_RAISE:
                    if (j < 0 || j >= m) {
                        invalid = 1;
                        goto end_while;
                    }
                    break;
                case NPY_WRAP:
                    if (j < 0) {
                        j += m;
                        if (j < 0) {
                            j = j % m;
                            if (j != 0) {
                                j += m;
                            }
                        }
                    }
                    else if (j >= m) {
                        j -= m;
                        if (j >= m) {
                            j = j % m;
                        }
                    }
                    break;
                case NPY_CLIP:
                    if (j < 0) {
                        j = 0;
                    }
                    else if (j >= m) {
                        j = m - 1;
                    }
                    break;

            }
            raveled += j * ravel_strides[i];

            coords[i] += coords_strides[i];
        }
        *(npy_intp *)coords[ravel_ndim] = raveled;
        coords[ravel_ndim] += coords_strides[ravel_ndim];
    }
end_while:
    NPY_END_ALLOW_THREADS;
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
              "invalid entry in coordinates array");
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* ravel_multi_index implementation - see add_newdocs.py */
NPY_NO_EXPORT PyObject *
arr_ravel_multi_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    int i;
    PyObject *mode0=NULL, *coords0=NULL;
    PyArrayObject *ret = NULL;
    PyArray_Dims dimensions={0,0};
    npy_intp s, ravel_strides[NPY_MAXDIMS];
    NPY_ORDER order = NPY_CORDER;
    NPY_CLIPMODE modes[NPY_MAXDIMS];

    PyArrayObject *op[NPY_MAXARGS];
    PyArray_Descr *dtype[NPY_MAXARGS];
    npy_uint32 op_flags[NPY_MAXARGS];

    NpyIter *iter = NULL;

    static char *kwlist[] = {"multi_index", "dims", "mode", "order", NULL};

    memset(op, 0, sizeof(op));
    dtype[0] = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                        "OO&|OO&:ravel_multi_index", kwlist,
                     &coords0,
                     PyArray_IntpConverter, &dimensions,
                     &mode0,
                     PyArray_OrderConverter, &order)) {
        goto fail;
    }

    if (dimensions.len+1 > NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                    "too many dimensions passed to ravel_multi_index");
        goto fail;
    }

    if (!PyArray_ConvertClipmodeSequence(mode0, modes, dimensions.len)) {
       goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            s = 1;
            for (i = dimensions.len-1; i >= 0; --i) {
                ravel_strides[i] = s;
                if (npy_mul_sizes_with_overflow(&s, s, dimensions.ptr[i])) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid dims: array size defined by dims is larger "
                        "than the maximum possible size.");
                    goto fail;
                }
            }
            break;
        case NPY_FORTRANORDER:
            s = 1;
            for (i = 0; i < dimensions.len; ++i) {
                ravel_strides[i] = s;
                if (npy_mul_sizes_with_overflow(&s, s, dimensions.ptr[i])) {
                    PyErr_SetString(PyExc_ValueError,
                        "invalid dims: array size defined by dims is larger "
                        "than the maximum possible size.");
                    goto fail;
                }
            }
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    /* Get the multi_index into op */
    if (int_sequence_to_arrays(coords0, "multi_index", dimensions.len, op) < 0) {
        goto fail;
    }

    for (i = 0; i < dimensions.len; ++i) {
        op_flags[i] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
    }
    op_flags[dimensions.len] = NPY_ITER_WRITEONLY|
                               NPY_ITER_ALIGNED|
                               NPY_ITER_ALLOCATE;
    dtype[0] = PyArray_DescrFromType(NPY_INTP);
    for (i = 1; i <= dimensions.len; ++i) {
        dtype[i] = dtype[0];
    }

    iter = NpyIter_MultiNew(dimensions.len+1, op, NPY_ITER_BUFFERED|
                                                  NPY_ITER_EXTERNAL_LOOP|
                                                  NPY_ITER_ZEROSIZE_OK,
                                                  NPY_KEEPORDER,
                                                  NPY_SAME_KIND_CASTING,
                                                  op_flags, dtype);
    if (iter == NULL) {
        goto fail;
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            if (ravel_multi_index_loop(dimensions.len, dimensions.ptr,
                        ravel_strides, *countptr, modes,
                        dataptr, strides) != NPY_SUCCEED) {
                goto fail;
            }
        } while(iternext(iter));
    }

    ret = NpyIter_GetOperandArray(iter)[dimensions.len];
    Py_INCREF(ret);

    Py_DECREF(dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    npy_free_cache_dim_obj(dimensions);
    NpyIter_Deallocate(iter);
    return PyArray_Return(ret);

fail:
    Py_XDECREF(dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    npy_free_cache_dim_obj(dimensions);
    NpyIter_Deallocate(iter);
    return NULL;
}


/*
 * Inner loop for unravel_index
 * order must be NPY_CORDER or NPY_FORTRANORDER
 */
static int
unravel_index_loop(int unravel_ndim, npy_intp const *unravel_dims,
                   npy_intp unravel_size, npy_intp count,
                   char *indices, npy_intp indices_stride,
                   npy_intp *coords, NPY_ORDER order)
{
    int i, idx;
    int idx_start = (order == NPY_CORDER) ? unravel_ndim - 1: 0;
    int idx_step = (order == NPY_CORDER) ? -1 : 1;
    char invalid = 0;
    npy_intp val = 0;

    NPY_BEGIN_ALLOW_THREADS;
    /* NPY_KEEPORDER or NPY_ANYORDER have no meaning in this setting */
    assert(order == NPY_CORDER || order == NPY_FORTRANORDER);
    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            invalid = 1;
            break;
        }
        idx = idx_start;
        for (i = 0; i < unravel_ndim; ++i) {
            /*
             * Using a local seems to enable single-divide optimization
             * but only if the / precedes the %
             */
            npy_intp tmp = val / unravel_dims[idx];
            coords[idx] = val % unravel_dims[idx];
            val = tmp;
            idx += idx_step;
        }
        coords += unravel_ndim;
        indices += indices_stride;
    }
    NPY_END_ALLOW_THREADS;
    if (invalid) {
        PyErr_Format(PyExc_ValueError,
            "index %" NPY_INTP_FMT " is out of bounds for array with size "
            "%" NPY_INTP_FMT,
            val, unravel_size
        );
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* unravel_index implementation - see add_newdocs.py */
NPY_NO_EXPORT PyObject *
arr_unravel_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *indices0 = NULL;
    PyObject *ret_tuple = NULL;
    PyArrayObject *ret_arr = NULL;
    PyArrayObject *indices = NULL;
    PyArray_Descr *dtype = NULL;
    PyArray_Dims dimensions = {0, 0};
    NPY_ORDER order = NPY_CORDER;
    npy_intp unravel_size;

    NpyIter *iter = NULL;
    int i, ret_ndim;
    npy_intp ret_dims[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    static char *kwlist[] = {"indices", "shape", "order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|O&:unravel_index",
                    kwlist,
                    &indices0,
                    PyArray_IntpConverter, &dimensions,
                    PyArray_OrderConverter, &order)) {
        goto fail;
    }

    unravel_size = PyArray_OverflowMultiplyList(dimensions.ptr, dimensions.len);
    if (unravel_size == -1) {
        PyErr_SetString(PyExc_ValueError,
                        "dimensions are too large; arrays and shapes with "
                        "a total size greater than 'intp' are not supported.");
        goto fail;
    }

    indices = astype_anyint(indices0);
    if (indices == NULL) {
        goto fail;
    }

    dtype = PyArray_DescrFromType(NPY_INTP);
    if (dtype == NULL) {
        goto fail;
    }

    iter = NpyIter_New(indices, NPY_ITER_READONLY|
                                NPY_ITER_ALIGNED|
                                NPY_ITER_BUFFERED|
                                NPY_ITER_ZEROSIZE_OK|
                                NPY_ITER_DONT_NEGATE_STRIDES|
                                NPY_ITER_MULTI_INDEX,
                                NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                                dtype);
    if (iter == NULL) {
        goto fail;
    }

    /*
     * Create the return array with a layout compatible with the indices
     * and with a dimension added to the end for the multi-index
     */
    ret_ndim = PyArray_NDIM(indices) + 1;
    if (NpyIter_GetShape(iter, ret_dims) != NPY_SUCCEED) {
        goto fail;
    }
    ret_dims[ret_ndim-1] = dimensions.len;
    if (NpyIter_CreateCompatibleStrides(iter,
                dimensions.len*sizeof(npy_intp), ret_strides) != NPY_SUCCEED) {
        goto fail;
    }
    ret_strides[ret_ndim-1] = sizeof(npy_intp);

    /* Remove the multi-index and inner loop */
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        goto fail;
    }
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        goto fail;
    }

    ret_arr = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                            ret_ndim, ret_dims, ret_strides, NULL, 0, NULL);
    dtype = NULL;
    if (ret_arr == NULL) {
        goto fail;
    }

    if (order != NPY_CORDER && order != NPY_FORTRANORDER) {
        PyErr_SetString(PyExc_ValueError,
                        "only 'C' or 'F' order is permitted");
        goto fail;
    }
    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr, count;
        npy_intp *coordsptr = (npy_intp *)PyArray_DATA(ret_arr);

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            count = *countptr;
            if (unravel_index_loop(dimensions.len, dimensions.ptr,
                                   unravel_size, count, *dataptr, *strides,
                                   coordsptr, order) != NPY_SUCCEED) {
                goto fail;
            }
            coordsptr += count * dimensions.len;
        } while (iternext(iter));
    }


    if (dimensions.len == 0 && PyArray_NDIM(indices) != 0) {
        /*
         * There's no index meaning "take the only element 10 times"
         * on a zero-d array, so we have no choice but to error. (See gh-580)
         *
         * Do this check after iterating, so we give a better error message
         * for invalid indices.
         */
        PyErr_SetString(PyExc_ValueError,
                "multiple indices are not supported for 0d arrays");
        goto fail;
    }

    /* Now make a tuple of views, one per index */
    ret_tuple = PyTuple_New(dimensions.len);
    if (ret_tuple == NULL) {
        goto fail;
    }
    for (i = 0; i < dimensions.len; ++i) {
        PyArrayObject *view;

        view = (PyArrayObject *)PyArray_NewFromDescrAndBase(
                &PyArray_Type, PyArray_DescrFromType(NPY_INTP),
                ret_ndim - 1, ret_dims, ret_strides,
                PyArray_BYTES(ret_arr) + i*sizeof(npy_intp),
                NPY_ARRAY_WRITEABLE, NULL, (PyObject *)ret_arr);
        if (view == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(ret_tuple, i, PyArray_Return(view));
    }

    Py_DECREF(ret_arr);
    Py_XDECREF(indices);
    npy_free_cache_dim_obj(dimensions);
    NpyIter_Deallocate(iter);

    return ret_tuple;

fail:
    Py_XDECREF(ret_tuple);
    Py_XDECREF(ret_arr);
    Py_XDECREF(dtype);
    Py_XDECREF(indices);
    npy_free_cache_dim_obj(dimensions);
    NpyIter_Deallocate(iter);
    return NULL;
}


/* Can only be called if doc is currently NULL */
NPY_NO_EXPORT PyObject *
arr_add_docstring(PyObject *NPY_UNUSED(dummy), PyObject *const *args, Py_ssize_t len_args)
{
    PyObject *obj;
    PyObject *str;
    const char *docstr;
    static const char msg[] = "already has a different docstring";

    /* Don't add docstrings */
#if PY_VERSION_HEX > 0x030b0000
    if (npy_static_cdata.optimize > 1) {
#else
    if (Py_OptimizeFlag > 1) {
#endif
        Py_RETURN_NONE;
    }

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("add_docstring", args, len_args, NULL,
            "", NULL, &obj,
            "", NULL, &str,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }
    if (!PyUnicode_Check(str)) {
        PyErr_SetString(PyExc_TypeError,
                "argument docstring of add_docstring should be a str");
        return NULL;
    }

    docstr = PyUnicode_AsUTF8(str);
    if (docstr == NULL) {
        return NULL;
    }

#define _ADDDOC(doc, name)                                              \
        if (!(doc)) {                                                   \
            doc = docstr;                                               \
            Py_INCREF(str);  /* hold on to string (leaks reference) */  \
        }                                                               \
        else if (strcmp(doc, docstr) != 0) {                            \
            PyErr_Format(PyExc_RuntimeError, "%s method %s", name, msg); \
            return NULL;                                                \
        }

    if (Py_TYPE(obj) == &PyCFunction_Type) {
        PyCFunctionObject *new = (PyCFunctionObject *)obj;
        _ADDDOC(new->m_ml->ml_doc, new->m_ml->ml_name);
    }
    else if (PyObject_TypeCheck(obj, &PyType_Type)) {
        /*
         * We add it to both `tp_doc` and `__doc__` here.  Note that in theory
         * `tp_doc` extracts the signature line, but we currently do not use
         * it.  It may make sense to only add it as `__doc__` and
         * `__text_signature__` to the dict in the future.
         * The dictionary path is only necessary for heaptypes (currently not
         * used) and metaclasses.
         * If `__doc__` as stored in `tp_dict` is None, we assume this was
         * filled in by `PyType_Ready()` and should also be replaced.
         */
        PyTypeObject *new = (PyTypeObject *)obj;
        _ADDDOC(new->tp_doc, new->tp_name);
        if (new->tp_dict != NULL && PyDict_CheckExact(new->tp_dict) &&
                PyDict_GetItemString(new->tp_dict, "__doc__") == Py_None) {
            /* Warning: Modifying `tp_dict` is not generally safe! */
            if (PyDict_SetItemString(new->tp_dict, "__doc__", str) < 0) {
                return NULL;
            }
        }
    }
    else if (Py_TYPE(obj) == &PyMemberDescr_Type) {
        PyMemberDescrObject *new = (PyMemberDescrObject *)obj;
        _ADDDOC(new->d_member->doc, new->d_member->name);
    }
    else if (Py_TYPE(obj) == &PyGetSetDescr_Type) {
        PyGetSetDescrObject *new = (PyGetSetDescrObject *)obj;
        _ADDDOC(new->d_getset->doc, new->d_getset->name);
    }
    else if (Py_TYPE(obj) == &PyMethodDescr_Type) {
        PyMethodDescrObject *new = (PyMethodDescrObject *)obj;
        _ADDDOC(new->d_method->ml_doc, new->d_method->ml_name);
    }
    else {
        PyObject *doc_attr;

        doc_attr = PyObject_GetAttrString(obj, "__doc__");
        if (doc_attr != NULL && doc_attr != Py_None &&
                (PyUnicode_Compare(doc_attr, str) != 0)) {
            Py_DECREF(doc_attr);
            if (PyErr_Occurred()) {
                /* error during PyUnicode_Compare */
                return NULL;
            }
            PyErr_Format(PyExc_RuntimeError, "object %s", msg);
            return NULL;
        }
        Py_XDECREF(doc_attr);

        if (PyObject_SetAttrString(obj, "__doc__", str) < 0) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot set a docstring for that object");
            return NULL;
        }
        Py_RETURN_NONE;
    }

#undef _ADDDOC

    Py_RETURN_NONE;
}

/*
 * This function packs boolean values in the input array into the bits of a
 * byte array. Truth values are determined as usual: 0 is false, everything
 * else is true.
 */
static NPY_GCC_OPT_3 inline void
pack_inner(const char *inptr,
           npy_intp element_size,   /* in bytes */
           npy_intp n_in,
           npy_intp in_stride,
           char *outptr,
           npy_intp n_out,
           npy_intp out_stride,
           PACK_ORDER order)
{
    /*
     * Loop through the elements of inptr.
     * Determine whether or not it is nonzero.
     *  Yes: set corresponding bit (and adjust build value)
     *  No:  move on
     * Every 8th value, set the value of build and increment the outptr
     */
    npy_intp index = 0;
    int remain = n_in % 8;              /* uneven bits */

#if NPY_SIMD
    if (in_stride == 1 && element_size == 1 && n_out > 2) {
        npyv_u8 v_zero = npyv_zero_u8();
        /* don't handle non-full 8-byte remainder */
        npy_intp vn_out = n_out - (remain ? 1 : 0);
        const int vstep = npyv_nlanes_u64;
        const int vstepx4 = vstep * 4;
        const int isAligned = npy_is_aligned(outptr, sizeof(npy_uint64));
        vn_out -= (vn_out & (vstep - 1));
        for (; index <= vn_out - vstepx4; index += vstepx4, inptr += npyv_nlanes_u8 * 4) {
            npyv_u8 v0 = npyv_load_u8((const npy_uint8*)inptr);
            npyv_u8 v1 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 1);
            npyv_u8 v2 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 2);
            npyv_u8 v3 = npyv_load_u8((const npy_uint8*)inptr + npyv_nlanes_u8 * 3);
            if (order == PACK_ORDER_BIG) {
                v0 = npyv_rev64_u8(v0);
                v1 = npyv_rev64_u8(v1);
                v2 = npyv_rev64_u8(v2);
                v3 = npyv_rev64_u8(v3);
            }
            npy_uint64 bb[4];
            bb[0] = npyv_tobits_b8(npyv_cmpneq_u8(v0, v_zero));
            bb[1] = npyv_tobits_b8(npyv_cmpneq_u8(v1, v_zero));
            bb[2] = npyv_tobits_b8(npyv_cmpneq_u8(v2, v_zero));
            bb[3] = npyv_tobits_b8(npyv_cmpneq_u8(v3, v_zero));
            if(out_stride == 1 && 
                (!NPY_ALIGNMENT_REQUIRED || isAligned)) {
                npy_uint64 *ptr64 = (npy_uint64*)outptr;
            #if NPY_SIMD_WIDTH == 16
                npy_uint64 bcomp = bb[0] | (bb[1] << 16) | (bb[2] << 32) | (bb[3] << 48);
                ptr64[0] = bcomp;
            #elif NPY_SIMD_WIDTH == 32
                ptr64[0] = bb[0] | (bb[1] << 32);
                ptr64[1] = bb[2] | (bb[3] << 32);
            #else
                ptr64[0] = bb[0]; ptr64[1] = bb[1];
                ptr64[2] = bb[2]; ptr64[3] = bb[3];
            #endif
                outptr += vstepx4;
            } else {
                for(int i = 0; i < 4; i++) {
                    for (int j = 0; j < vstep; j++) {
                        memcpy(outptr, (char*)&bb[i] + j, 1);
                        outptr += out_stride;
                    }
                }
            }
        }
        for (; index < vn_out; index += vstep, inptr += npyv_nlanes_u8) {
            npyv_u8 va = npyv_load_u8((const npy_uint8*)inptr);
            if (order == PACK_ORDER_BIG) {
                va = npyv_rev64_u8(va);
            }
            npy_uint64 bb = npyv_tobits_b8(npyv_cmpneq_u8(va, v_zero));
            for (int i = 0; i < vstep; ++i) {
                memcpy(outptr, (char*)&bb + i, 1);
                outptr += out_stride;
            }
        }
    }
#endif

    if (remain == 0) {                  /* assumes n_in > 0 */
        remain = 8;
    }
    /* Don't reset index. Just handle remainder of above block */
    for (; index < n_out; index++) {
        unsigned char build = 0;
        int maxi = (index == n_out - 1) ? remain : 8;
        if (order == PACK_ORDER_BIG) {
            for (int i = 0; i < maxi; i++) {
                build <<= 1;
                for (npy_intp j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0);
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build <<= 8 - remain;
            }
        }
        else
        {
            for (int i = 0; i < maxi; i++) {
                build >>= 1;
                for (npy_intp j = 0; j < element_size; j++) {
                    build |= (inptr[j] != 0) ? 128 : 0;
                }
                inptr += in_stride;
            }
            if (index == n_out - 1) {
                build >>= 8 - remain;
            }
        }
        *outptr = (char)build;
        outptr += out_stride;
    }
}

static PyObject *
pack_bits(PyObject *input, int axis, char order)
{
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    NPY_BEGIN_THREADS_DEF;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (!PyArray_ISBOOL(inp) && !PyArray_ISINTEGER(inp)) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of integer or boolean data type");
        Py_DECREF(inp);
        goto fail;
    }

    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(new) == 0) {
        char *optr, *iptr;

        out = (PyArrayObject *)PyArray_NewFromDescr(
                Py_TYPE(new), PyArray_DescrFromType(NPY_UBYTE),
                0, NULL, NULL, NULL,
                0, NULL);
        if (out == NULL) {
            goto fail;
        }
        optr = PyArray_DATA(out);
        iptr = PyArray_DATA(new);
        *optr = 0;
        for (i = 0; i < PyArray_ITEMSIZE(new); i++) {
            if (*iptr != 0) {
                *optr = 1;
                break;
            }
            iptr++;
        }
        goto finish;
    }


    /* Setup output shape */
    for (i = 0; i < PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    /*
     * Divide axis dimension by 8
     * 8 -> 1, 9 -> 2, 16 -> 2, 17 -> 3 etc..
     */
    outdims[axis] = ((outdims[axis] - 1) >> 3) + 1;

    /* Create output array */
    out = (PyArrayObject *)PyArray_NewFromDescr(
            Py_TYPE(new), PyArray_DescrFromType(NPY_UBYTE),
            PyArray_NDIM(new), outdims, NULL, NULL,
            PyArray_ISFORTRAN(new), NULL);
    if (out == NULL) {
        goto fail;
    }
    /* Setup iterators to iterate over all but given axis */
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)new, &axis);
    ot = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }
    const PACK_ORDER ordere = order == 'b' ? PACK_ORDER_BIG : PACK_ORDER_LITTLE;
    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_DIM(out, axis));
    while (PyArray_ITER_NOTDONE(it)) {
        pack_inner(PyArray_ITER_DATA(it), PyArray_ITEMSIZE(new),
                   PyArray_DIM(new, axis), PyArray_STRIDE(new, axis),
                   PyArray_ITER_DATA(ot), PyArray_DIM(out, axis),
                   PyArray_STRIDE(out, axis), ordere);
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    NPY_END_THREADS;

    Py_DECREF(it);
    Py_DECREF(ot);

finish:
    Py_DECREF(new);
    return (PyObject *)out;

fail:
    Py_XDECREF(new);
    Py_XDECREF(out);
    return NULL;
}

static PyObject *
unpack_bits(PyObject *input, int axis, PyObject *count_obj, char order)
{
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    npy_intp count, in_n, in_tail, out_pad, in_stride, out_stride;
    NPY_BEGIN_THREADS_DEF;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (PyArray_TYPE(inp) != NPY_UBYTE) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of unsigned byte data type");
        Py_DECREF(inp);
        goto fail;
    }

    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(new) == 0) {
        /* Handle 0-d array by converting it to a 1-d array */
        PyArrayObject *temp;
        PyArray_Dims newdim = {NULL, 1};
        npy_intp shape = 1;

        newdim.ptr = &shape;
        temp = (PyArrayObject *)PyArray_Newshape(new, &newdim, NPY_CORDER);
        Py_DECREF(new);
        if (temp == NULL) {
            return NULL;
        }
        new = temp;
    }

    /* Setup output shape */
    for (i = 0; i < PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    /* Multiply axis dimension by 8 */
    outdims[axis] *= 8;
    if (count_obj != Py_None) {
        count = PyArray_PyIntAsIntp(count_obj);
        if (error_converting(count)) {
            goto fail;
        }
        if (count < 0) {
            outdims[axis] += count;
            if (outdims[axis] < 0) {
                PyErr_Format(PyExc_ValueError,
                             "-count larger than number of elements");
                goto fail;
            }
        }
        else {
            outdims[axis] = count;
        }
    }

    /* Create output array */
    out = (PyArrayObject *)PyArray_NewFromDescr(
            Py_TYPE(new), PyArray_DescrFromType(NPY_UBYTE),
            PyArray_NDIM(new), outdims, NULL, NULL,
            PyArray_ISFORTRAN(new), NULL);
    if (out == NULL) {
        goto fail;
    }

    /* Setup iterators to iterate over all but given axis */
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)new, &axis);
    ot = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }

    count = PyArray_DIM(new, axis) * 8;
    if (outdims[axis] > count) {
        in_n = count / 8;
        in_tail = 0;
        out_pad = outdims[axis] - count;
    }
    else {
        in_n = outdims[axis] / 8;
        in_tail = outdims[axis] % 8;
        out_pad = 0;
    }

    in_stride = PyArray_STRIDE(new, axis);
    out_stride = PyArray_STRIDE(out, axis);

    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_Size((PyObject *)out) / 8);

    while (PyArray_ITER_NOTDONE(it)) {
        npy_intp index;
        unsigned const char *inptr = PyArray_ITER_DATA(it);
        char *outptr = PyArray_ITER_DATA(ot);

        if (out_stride == 1) {
            /* for unity stride we can just copy out of the lookup table */
            if (order == 'b') {
                for (index = 0; index < in_n; index++) {
                    npy_uint64 v = npy_static_cdata.unpack_lookup_big[*inptr].uint64;
                    memcpy(outptr, &v, 8);
                    outptr += 8;
                    inptr += in_stride;
                }
            }
            else {
                for (index = 0; index < in_n; index++) {
                    npy_uint64 v = npy_static_cdata.unpack_lookup_big[*inptr].uint64;
                    if (order != 'b') {
                        v = npy_bswap8(v);
                    }
                    memcpy(outptr, &v, 8);
                    outptr += 8;
                    inptr += in_stride;
                }
            }
            /* Clean up the tail portion */
            if (in_tail) {
                npy_uint64 v = npy_static_cdata.unpack_lookup_big[*inptr].uint64;
                if (order != 'b') {
                    v = npy_bswap8(v);
                }
                memcpy(outptr, &v, in_tail);
            }
            /* Add padding */
            else if (out_pad) {
                memset(outptr, 0, out_pad);
            }
        }
        else {
            if (order == 'b') {
                for (index = 0; index < in_n; index++) {
                    for (i = 0; i < 8; i++) {
                        *outptr = ((*inptr & (128 >> i)) != 0);
                        outptr += out_stride;
                    }
                    inptr += in_stride;
                }
                /* Clean up the tail portion */
                for (i = 0; i < in_tail; i++) {
                    *outptr = ((*inptr & (128 >> i)) != 0);
                    outptr += out_stride;
                }
            }
            else {
                for (index = 0; index < in_n; index++) {
                    for (i = 0; i < 8; i++) {
                        *outptr = ((*inptr & (1 << i)) != 0);
                        outptr += out_stride;
                    }
                    inptr += in_stride;
                }
                /* Clean up the tail portion */
                for (i = 0; i < in_tail; i++) {
                    *outptr = ((*inptr & (1 << i)) != 0);
                    outptr += out_stride;
                }
            }
            /* Add padding */
            for (index = 0; index < out_pad; index++) {
                *outptr = 0;
                outptr += out_stride;
            }
        }

        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    NPY_END_THREADS;

    Py_DECREF(it);
    Py_DECREF(ot);

    Py_DECREF(new);
    return (PyObject *)out;

fail:
    Py_XDECREF(new);
    Py_XDECREF(out);
    return NULL;
}


NPY_NO_EXPORT PyObject *
io_pack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_RAVEL_AXIS;
    static char *kwlist[] = {"in", "axis", "bitorder", NULL};
    char c = 'b';
    const char * order_str = NULL;

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&s:pack" , kwlist,
                &obj, PyArray_AxisConverter, &axis, &order_str)) {
        return NULL;
    }
    if (order_str != NULL) {
        if (strncmp(order_str, "little", 6) == 0)
            c = 'l';
        else if (strncmp(order_str, "big", 3) == 0)
            c = 'b';
        else {
            PyErr_SetString(PyExc_ValueError,
                    "'order' must be either 'little' or 'big'");
            return NULL;
        }
    }
    return pack_bits(obj, axis, c);
}


NPY_NO_EXPORT PyObject *
io_unpack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_RAVEL_AXIS;
    PyObject *count = Py_None;
    static char *kwlist[] = {"in", "axis", "count", "bitorder", NULL};
    const char * c = NULL;

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&Os:unpack" , kwlist,
                &obj, PyArray_AxisConverter, &axis, &count, &c)) {
        return NULL;
    }
    if (c == NULL) {
        c = "b";
    }
    if (c[0] != 'l' && c[0] != 'b') {
        PyErr_SetString(PyExc_ValueError,
                    "'order' must begin with 'l' or 'b'");
        return NULL;
    }
    return unpack_bits(obj, axis, count, c[0]);
}
