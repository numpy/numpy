#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <Python.h>
#include <structmember.h>
#include <string.h>

#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "numpy/npy_math.h"
#include "npy_config.h"
#include "templ_common.h" /* for npy_mul_with_overflow_intp */
#include "lowlevel_strided_loops.h" /* for npy_bswap8 */
#include "alloc.h"
#include "common.h"


/*
 * Returns -1 if the array is monotonic decreasing,
 * +1 if the array is monotonic increasing,
 * and 0 if the array is not monotonic.
 */
static int
check_array_monotonic(const double *a, npy_int lena)
{
    npy_intp i;
    double next;
    double last = a[0];

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
 * non-negative integers The second, if present, is an array of weights,
 * which must be promotable to double. Call these arguments list and
 * weight. Both must be one-dimensional with len(weight) == len(list). If
 * weight is not present then bincount(list)[i] is the number of occurrences
 * of i in list.  If weight is present then bincount(self,list, weight)[i]
 * is the sum of all weight[j] where list [j] == i.  Self is not used.
 * The third argument, if present, is a minimum length desired for the
 * output array.
 */
NPY_NO_EXPORT PyObject *
arr_bincount(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *list = NULL, *weight = Py_None, *mlength = NULL;
    PyArrayObject *lst = NULL, *ans = NULL, *wts = NULL;
    npy_intp *numbers, *ians, len, mx, mn, ans_size;
    npy_intp minlength = 0;
    npy_intp i;
    double *weights , *dans;
    static char *kwlist[] = {"list", "weights", "minlength", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO:bincount",
                kwlist, &list, &weight, &mlength)) {
            goto fail;
    }

    lst = (PyArrayObject *)PyArray_ContiguousFromAny(list, NPY_INTP, 1, 1);
    if (lst == NULL) {
        goto fail;
    }
    len = PyArray_SIZE(lst);

    /*
     * This if/else if can be removed by changing the argspec to O|On above,
     * once we retire the deprecation
     */
    if (mlength == Py_None) {
        /* NumPy 1.14, 2017-06-01 */
        if (DEPRECATE("0 should be passed as minlength instead of None; "
                      "this will error in future.") < 0) {
            goto fail;
        }
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

/*
 * digitize(x, bins, right=False) returns an array of integers the same length
 * as x. The values i returned are such that bins[i - 1] <= x < bins[i] if
 * bins is monotonically increasing, or bins[i - 1] > x >= bins[i] if bins
 * is monotonically decreasing.  Beyond the bounds of bins, returns either
 * i = 0 or i = len(bins) as appropriate. If right == True the comparison
 * is bins [i - 1] < x <= bins[i] or bins [i - 1] >= x > bins[i]
 */
NPY_NO_EXPORT PyObject *
arr_digitize(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj_x = NULL;
    PyObject *obj_bins = NULL;
    PyArrayObject *arr_x = NULL;
    PyArrayObject *arr_bins = NULL;
    PyObject *ret = NULL;
    npy_intp len_bins;
    int monotonic, right = 0;
    NPY_BEGIN_THREADS_DEF

    static char *kwlist[] = {"x", "bins", "right", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i:digitize", kwlist,
                                     &obj_x, &obj_bins, &right)) {
        goto fail;
    }

    /* PyArray_SearchSorted will make `x` contiguous even if we don't */
    arr_x = (PyArrayObject *)PyArray_FROMANY(obj_x, NPY_DOUBLE, 0, 0,
                                             NPY_ARRAY_CARRAY_RO);
    if (arr_x == NULL) {
        goto fail;
    }

    /* TODO: `bins` could be strided, needs change to check_array_monotonic */
    arr_bins = (PyArrayObject *)PyArray_FROMANY(obj_bins, NPY_DOUBLE, 1, 1,
                                               NPY_ARRAY_CARRAY_RO);
    if (arr_bins == NULL) {
        goto fail;
    }

    len_bins = PyArray_SIZE(arr_bins);
    if (len_bins == 0) {
        PyErr_SetString(PyExc_ValueError, "bins must have non-zero length");
        goto fail;
    }

    NPY_BEGIN_THREADS_THRESHOLDED(len_bins)
    monotonic = check_array_monotonic((const double *)PyArray_DATA(arr_bins),
                                      len_bins);
    NPY_END_THREADS

    if (monotonic == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "bins must be monotonically increasing or decreasing");
        goto fail;
    }

    /* PyArray_SearchSorted needs an increasing array */
    if (monotonic == - 1) {
        PyArrayObject *arr_tmp = NULL;
        npy_intp shape = PyArray_DIM(arr_bins, 0);
        npy_intp stride = -PyArray_STRIDE(arr_bins, 0);
        void *data = (void *)(PyArray_BYTES(arr_bins) - stride * (shape - 1));

        arr_tmp = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &shape,
                                               NPY_DOUBLE, &stride, data, 0,
                                               PyArray_FLAGS(arr_bins), NULL);
        if (!arr_tmp) {
            goto fail;
        }

        if (PyArray_SetBaseObject(arr_tmp, (PyObject *)arr_bins) < 0) {

            Py_DECREF(arr_tmp);
            goto fail;
        }
        arr_bins = arr_tmp;
    }

    ret = PyArray_SearchSorted(arr_bins, (PyObject *)arr_x,
                               right ? NPY_SEARCHLEFT : NPY_SEARCHRIGHT, NULL);
    if (!ret) {
        goto fail;
    }

    /* If bins is decreasing, ret has bins from end, not start */
    if (monotonic == -1) {
        npy_intp *ret_data =
                        (npy_intp *)PyArray_DATA((PyArrayObject *)ret);
        npy_intp len_ret = PyArray_SIZE((PyArrayObject *)ret);

        NPY_BEGIN_THREADS_THRESHOLDED(len_ret)
        while (len_ret--) {
            *ret_data = len_bins - *ret_data;
            ret_data++;
        }
        NPY_END_THREADS
    }

    fail:
        Py_XDECREF(arr_x);
        Py_XDECREF(arr_bins);
        return ret;
}

/*
 * Returns input array with values inserted sequentially into places
 * indicated by the mask
 */
NPY_NO_EXPORT PyObject *
arr_insert(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
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
    chunk = PyArray_DESCR(array)->elsize;
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
            Py_XDECREF(array);
            Py_RETURN_NONE;
        }
    }

    src = PyArray_DATA(values);
    j = 0;

    copyswap = PyArray_DESCR(array)->f->copyswap;
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
    Py_XDECREF(array);
    Py_XDECREF(values);
    return NULL;
}

#define LIKELY_IN_CACHE_SIZE 8

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
        npy_intp i;

        for (i = 1; i < len && key >= arr[i]; ++i);
        return i - 1;
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
arr_interp(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;
    npy_double lval, rval;
    const npy_double *dy, *dx, *dz;
    npy_double *dres, *slopes = NULL;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO:interp", kwlist,
                                     &x, &xp, &fp, &left, &right)) {
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
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 0);
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
            else {
                const npy_double slope = (slopes != NULL) ? slopes[j] :
                                         (dy[j+1] - dy[j]) / (dx[j+1] - dx[j]);
                dres[i] = slope*(x_val - dx[j]) + dy[j];
            }
        }

        NPY_END_THREADS;
    }

    PyArray_free(slopes);
    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);
    return (PyObject *)af;

fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}

/* As for arr_interp but for complex fp values */
NPY_NO_EXPORT PyObject *
arr_interp_complex(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;

    const npy_double *dx, *dz;
    const npy_cdouble *dy; 
    npy_cdouble lval, rval; 
    npy_cdouble *dres, *slopes = NULL;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO:interp_complex",
                                     kwlist, &x, &xp, &fp, &left, &right)) {
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
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 0);
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
        lval.real = PyComplex_RealAsDouble(left);
        if (error_converting(lval.real)) {
            goto fail;
        }
        lval.imag = PyComplex_ImagAsDouble(left);
        if (error_converting(lval.imag)) {
            goto fail;
        }
    }
        
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp - 1];
    }
    else {
        rval.real = PyComplex_RealAsDouble(right);
        if (error_converting(rval.real)) {
            goto fail;
        }
        rval.imag = PyComplex_ImagAsDouble(right);
        if (error_converting(rval.imag)) {
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
                goto fail;
            }
        }
            
        NPY_BEGIN_THREADS;
        
        if (slopes != NULL) {
            for (i = 0; i < lenxp - 1; ++i) {
                const double inv_dx = 1.0 / (dx[i+1] - dx[i]);
                slopes[i].real = (dy[i+1].real - dy[i].real) * inv_dx;
                slopes[i].imag = (dy[i+1].imag - dy[i].imag) * inv_dx;
            }
        }
        
        for (i = 0; i < lenx; ++i) {
            const npy_double x_val = dz[i];
            
            if (npy_isnan(x_val)) {
                dres[i].real = x_val;
                dres[i].imag = 0.0;
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
            else {
                if (slopes!=NULL) {
                    dres[i].real = slopes[j].real*(x_val - dx[j]) + dy[j].real;
                    dres[i].imag = slopes[j].imag*(x_val - dx[j]) + dy[j].imag;
                }
                else {
                    const npy_double inv_dx = 1.0 / (dx[j+1] - dx[j]);
                    dres[i].real = (dy[j+1].real - dy[j].real)*(x_val - dx[j])*
			inv_dx + dy[j].real;
                    dres[i].imag = (dy[j+1].imag - dy[j].imag)*(x_val - dx[j])*
			inv_dx + dy[j].imag;
                }
            }
        }
        
        NPY_END_THREADS;
    } 
    PyArray_free(slopes);
    
    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);
    return (PyObject *)af;

fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}

/*
 * Converts a Python sequence into 'count' PyArrayObjects
 *
 * seq       - Input Python object, usually a tuple but any sequence works.
 * op        - Where the arrays are placed.
 * count     - How many arrays there should be (errors if it doesn't match).
 * paramname - The name of the parameter that produced 'seq'.
 */
static int sequence_to_arrays(PyObject *seq,
                                PyArrayObject **op, int count,
                                char *paramname)
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
            while (--i >= 0) {
                Py_DECREF(op[i]);
                op[i] = NULL;
            }
            return -1;
        }

        op[i] = (PyArrayObject *)PyArray_FROM_O(item);
        if (op[i] == NULL) {
            while (--i >= 0) {
                Py_DECREF(op[i]);
                op[i] = NULL;
            }
            Py_DECREF(item);
            return -1;
        }

        Py_DECREF(item);
    }

    return 0;
}

/* Inner loop for unravel_index */
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

    char *kwlist[] = {"multi_index", "dims", "mode", "order", NULL};

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
                if (npy_mul_with_overflow_intp(&s, s, dimensions.ptr[i])) {
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
                if (npy_mul_with_overflow_intp(&s, s, dimensions.ptr[i])) {
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
    if (sequence_to_arrays(coords0, op, dimensions.len, "multi_index") < 0) {
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

/* C-order inner loop for unravel_index */
static int
unravel_index_loop_corder(int unravel_ndim, npy_intp *unravel_dims,
                        npy_intp unravel_size, npy_intp count,
                        char *indices, npy_intp indices_stride,
                        npy_intp *coords)
{
    int i;
    char invalid;
    npy_intp val;

    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            invalid = 1;
            break;
        }
        for (i = unravel_ndim-1; i >= 0; --i) {
            coords[i] = val % unravel_dims[i];
            val /= unravel_dims[i];
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

/* Fortran-order inner loop for unravel_index */
static int
unravel_index_loop_forder(int unravel_ndim, npy_intp *unravel_dims,
                        npy_intp unravel_size, npy_intp count,
                        char *indices, npy_intp indices_stride,
                        npy_intp *coords)
{
    int i;
    char invalid;
    npy_intp val;

    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            invalid = 1;
            break;
        }
        for (i = 0; i < unravel_ndim; ++i) {
            *coords++ = val % unravel_dims[i];
            val /= unravel_dims[i];
        }
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
    PyObject *indices0 = NULL, *ret_tuple = NULL;
    PyArrayObject *ret_arr = NULL;
    PyArrayObject *indices = NULL;
    PyArray_Descr *dtype = NULL;
    PyArray_Dims dimensions={0,0};
    NPY_ORDER order = NPY_CORDER;
    npy_intp unravel_size;

    NpyIter *iter = NULL;
    int i, ret_ndim;
    npy_intp ret_dims[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    char *kwlist[] = {"indices", "dims", "order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|O&:unravel_index",
                    kwlist,
                    &indices0,
                    PyArray_IntpConverter, &dimensions,
                    PyArray_OrderConverter, &order)) {
        goto fail;
    }

    unravel_size = PyArray_MultiplyList(dimensions.ptr, dimensions.len);

    if (!PyArray_Check(indices0)) {
        indices = (PyArrayObject*)PyArray_FROM_O(indices0);
        if (indices == NULL) {
            goto fail;
        }
    }
    else {
        indices = (PyArrayObject *)indices0;
        Py_INCREF(indices);
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

    if (order == NPY_CORDER) {
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
                if (unravel_index_loop_corder(dimensions.len, dimensions.ptr,
                            unravel_size, count, *dataptr, *strides,
                            coordsptr) != NPY_SUCCEED) {
                    goto fail;
                }
                coordsptr += count*dimensions.len;
            } while(iternext(iter));
        }
    }
    else if (order == NPY_FORTRANORDER) {
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
                if (unravel_index_loop_forder(dimensions.len, dimensions.ptr,
                            unravel_size, count, *dataptr, *strides,
                            coordsptr) != NPY_SUCCEED) {
                    goto fail;
                }
                coordsptr += count*dimensions.len;
            } while(iternext(iter));
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "only 'C' or 'F' order is permitted");
        goto fail;
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

        view = (PyArrayObject *)PyArray_New(&PyArray_Type, ret_ndim-1,
                                ret_dims, NPY_INTP,
                                ret_strides,
                                PyArray_BYTES(ret_arr) + i*sizeof(npy_intp),
                                0, NPY_ARRAY_WRITEABLE, NULL);
        if (view == NULL) {
            goto fail;
        }
        Py_INCREF(ret_arr);
        if (PyArray_SetBaseObject(view, (PyObject *)ret_arr) < 0) {
            Py_DECREF(view);
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
arr_add_docstring(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *obj;
    PyObject *str;
    char *docstr;
    static char *msg = "already has a docstring";
    PyObject *tp_dict = PyArrayDescr_Type.tp_dict;
    PyObject *myobj;
    static PyTypeObject *PyMemberDescr_TypePtr = NULL;
    static PyTypeObject *PyGetSetDescr_TypePtr = NULL;
    static PyTypeObject *PyMethodDescr_TypePtr = NULL;

    /* Don't add docstrings */
    if (Py_OptimizeFlag > 1) {
        Py_RETURN_NONE;
    }

    if (PyGetSetDescr_TypePtr == NULL) {
        /* Get "subdescr" */
        myobj = PyDict_GetItemString(tp_dict, "fields");
        if (myobj != NULL) {
            PyGetSetDescr_TypePtr = Py_TYPE(myobj);
        }
    }
    if (PyMemberDescr_TypePtr == NULL) {
        myobj = PyDict_GetItemString(tp_dict, "alignment");
        if (myobj != NULL) {
            PyMemberDescr_TypePtr = Py_TYPE(myobj);
        }
    }
    if (PyMethodDescr_TypePtr == NULL) {
        myobj = PyDict_GetItemString(tp_dict, "newbyteorder");
        if (myobj != NULL) {
            PyMethodDescr_TypePtr = Py_TYPE(myobj);
        }
    }

#if defined(NPY_PY3K)
    if (!PyArg_ParseTuple(args, "OO!:add_docstring", &obj, &PyUnicode_Type, &str)) {
        return NULL;
    }

    docstr = PyBytes_AS_STRING(PyUnicode_AsUTF8String(str));
#else
    if (!PyArg_ParseTuple(args, "OO!:add_docstring", &obj, &PyString_Type, &str)) {
        return NULL;
    }

    docstr = PyString_AS_STRING(str);
#endif

#define _TESTDOC1(typebase) (Py_TYPE(obj) == &Py##typebase##_Type)
#define _TESTDOC2(typebase) (Py_TYPE(obj) == Py##typebase##_TypePtr)
#define _ADDDOC(typebase, doc, name) do {                               \
        Py##typebase##Object *new = (Py##typebase##Object *)obj;        \
        if (!(doc)) {                                                   \
            doc = docstr;                                               \
        }                                                               \
        else {                                                          \
            PyErr_Format(PyExc_RuntimeError, "%s method %s", name, msg); \
            return NULL;                                                \
        }                                                               \
    } while (0)

    if (_TESTDOC1(CFunction)) {
        _ADDDOC(CFunction, new->m_ml->ml_doc, new->m_ml->ml_name);
    }
    else if (_TESTDOC1(Type)) {
        _ADDDOC(Type, new->tp_doc, new->tp_name);
    }
    else if (_TESTDOC2(MemberDescr)) {
        _ADDDOC(MemberDescr, new->d_member->doc, new->d_member->name);
    }
    else if (_TESTDOC2(GetSetDescr)) {
        _ADDDOC(GetSetDescr, new->d_getset->doc, new->d_getset->name);
    }
    else if (_TESTDOC2(MethodDescr)) {
        _ADDDOC(MethodDescr, new->d_method->ml_doc, new->d_method->ml_name);
    }
    else {
        PyObject *doc_attr;

        doc_attr = PyObject_GetAttrString(obj, "__doc__");
        if (doc_attr != NULL && doc_attr != Py_None) {
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

#undef _TESTDOC1
#undef _TESTDOC2
#undef _ADDDOC

    Py_INCREF(str);
    Py_RETURN_NONE;
}

#if defined NPY_HAVE_SSE2_INTRINSICS
#include <emmintrin.h>
#endif

/*
 * This function packs boolean values in the input array into the bits of a
 * byte array. Truth values are determined as usual: 0 is false, everything
 * else is true.
 */
static NPY_INLINE void
pack_inner(const char *inptr,
           npy_intp element_size,   /* in bytes */
           npy_intp n_in,
           npy_intp in_stride,
           char *outptr,
           npy_intp n_out,
           npy_intp out_stride)
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

#if defined NPY_HAVE_SSE2_INTRINSICS && defined HAVE__M_FROM_INT64
    if (in_stride == 1 && element_size == 1 && n_out > 2) {
        __m128i zero = _mm_setzero_si128();
        /* don't handle non-full 8-byte remainder */
        npy_intp vn_out = n_out - (remain ? 1 : 0);
        vn_out -= (vn_out & 1);
        for (index = 0; index < vn_out; index += 2) {
            unsigned int r;
            /* swap as packbits is "big endian", note x86 can load unaligned */
            npy_uint64 a = npy_bswap8(*(npy_uint64*)inptr);
            npy_uint64 b = npy_bswap8(*(npy_uint64*)(inptr + 8));
            __m128i v = _mm_set_epi64(_m_from_int64(b), _m_from_int64(a));
            /* false -> 0x00 and true -> 0xFF (there is no cmpneq) */
            v = _mm_cmpeq_epi8(v, zero);
            v = _mm_cmpeq_epi8(v, zero);
            /* extract msb of 16 bytes and pack it into 16 bit */
            r = _mm_movemask_epi8(v);
            /* store result */
            memcpy(outptr, &r, 1);
            outptr += out_stride;
            memcpy(outptr, (char*)&r + 1, 1);
            outptr += out_stride;
            inptr += 16;
        }
    }
#endif

    if (remain == 0) {                  /* assumes n_in > 0 */
        remain = 8;
    }
    /* don't reset index to handle remainder of above block */
    for (; index < n_out; index++) {
        char build = 0;
        int i, maxi;
        npy_intp j;

        maxi = (index == n_out - 1) ? remain : 8;
        for (i = 0; i < maxi; i++) {
            build <<= 1;
            for (j = 0; j < element_size; j++) {
                build |= (inptr[j] != 0);
            }
            inptr += in_stride;
        }
        if (index == n_out - 1) {
            build <<= 8 - remain;
        }
        *outptr = build;
        outptr += out_stride;
    }
}

static PyObject *
pack_bits(PyObject *input, int axis)
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
        goto fail;
    }

    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(new) == 0) {
        char *optr, *iptr;

        out = (PyArrayObject *)PyArray_New(Py_TYPE(new), 0, NULL, NPY_UBYTE,
                NULL, NULL, 0, 0, NULL);
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
    out = (PyArrayObject *)PyArray_New(Py_TYPE(new),
                        PyArray_NDIM(new), outdims, NPY_UBYTE,
                        NULL, NULL, 0, PyArray_ISFORTRAN(new), NULL);
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

    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_DIM(out, axis));
    while (PyArray_ITER_NOTDONE(it)) {
        pack_inner(PyArray_ITER_DATA(it), PyArray_ITEMSIZE(new),
                   PyArray_DIM(new, axis), PyArray_STRIDE(new, axis),
                   PyArray_ITER_DATA(ot), PyArray_DIM(out, axis),
                   PyArray_STRIDE(out, axis));
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
unpack_bits(PyObject *input, int axis)
{
    static int unpack_init = 0;
    static char unpack_lookup[256][8];
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    npy_intp n_in, in_stride, out_stride;
    NPY_BEGIN_THREADS_DEF;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (PyArray_TYPE(inp) != NPY_UBYTE) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of unsigned byte data type");
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
        if (temp == NULL) {
            goto fail;
        }
        Py_DECREF(new);
        new = temp;
    }

    /* Setup output shape */
    for (i=0; i<PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    /* Multiply axis dimension by 8 */
    outdims[axis] <<= 3;

    /* Create output array */
    out = (PyArrayObject *)PyArray_New(Py_TYPE(new),
                        PyArray_NDIM(new), outdims, NPY_UBYTE,
                        NULL, NULL, 0, PyArray_ISFORTRAN(new), NULL);
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

    /* setup lookup table under GIL, big endian 0..256 as bytes */
    if (unpack_init == 0) {
        npy_uint64 j;
        npy_uint64 * unpack_lookup_64 = (npy_uint64 *)unpack_lookup;
        for (j=0; j < 256; j++) {
            npy_uint64 v = 0;
            v |= (npy_uint64)((j &   1) ==   1);
            v |= (npy_uint64)((j &   2) ==   2) << 8;
            v |= (npy_uint64)((j &   4) ==   4) << 16;
            v |= (npy_uint64)((j &   8) ==   8) << 24;
            v |= (npy_uint64)((j &  16) ==  16) << 32;
            v |= (npy_uint64)((j &  32) ==  32) << 40;
            v |= (npy_uint64)((j &  64) ==  64) << 48;
            v |= (npy_uint64)((j & 128) == 128) << 56;
#if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN
            v = npy_bswap8(v);
#endif
            unpack_lookup_64[j] = v;
        }
        unpack_init = 1;
    }

    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_DIM(new, axis));

    n_in = PyArray_DIM(new, axis);
    in_stride = PyArray_STRIDE(new, axis);
    out_stride = PyArray_STRIDE(out, axis);

    while (PyArray_ITER_NOTDONE(it)) {
        npy_intp index;
        unsigned const char *inptr = PyArray_ITER_DATA(it);
        char *outptr = PyArray_ITER_DATA(ot);

        if (out_stride == 1) {
            /* for unity stride we can just copy out of the lookup table */
            for (index = 0; index < n_in; index++) {
                memcpy(outptr, unpack_lookup[*inptr], 8);
                outptr += 8;
                inptr += in_stride;
            }
        }
        else {
            for (index = 0; index < n_in; index++) {
                unsigned char mask = 128;

                for (i = 0; i < 8; i++) {
                    *outptr = ((mask & (*inptr)) != 0);
                    outptr += out_stride;
                    mask >>= 1;
                }
                inptr += in_stride;
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
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&:pack" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return pack_bits(obj, axis);
}

NPY_NO_EXPORT PyObject *
io_unpack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&:unpack" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return unpack_bits(obj, axis);
}
