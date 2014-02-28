#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "Python.h"
#include "structmember.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "npy_config.h"
#include "numpy/ufuncobject.h"
#include "string.h"

static npy_intp
incr_slot_(double x, double *bins, npy_intp lbins)
{
    npy_intp i;

    for ( i = 0; i < lbins; i ++ ) {
        if ( x < bins [i] ) {
            return i;
        }
    }
    return lbins;
}

static npy_intp
decr_slot_(double x, double * bins, npy_intp lbins)
{
    npy_intp i;

    for ( i = lbins - 1; i >= 0; i -- ) {
        if (x < bins [i]) {
            return i + 1;
        }
    }
    return 0;
}

static npy_intp
incr_slot_right_(double x, double *bins, npy_intp lbins)
{
    npy_intp i;

    for ( i = 0; i < lbins; i ++ ) {
        if ( x <= bins [i] ) {
            return i;
        }
    }
    return lbins;
}

static npy_intp
decr_slot_right_(double x, double * bins, npy_intp lbins)
{
    npy_intp i;

    for ( i = lbins - 1; i >= 0; i -- ) {
        if (x <= bins [i]) {
            return i + 1;
        }
    }
    return 0;
}

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


/*
 * Return the maximum value in an array of non-negative npy_intp integers.
 * Return -1 if there are any negative values.
 */

static NPY_INLINE npy_intp
max_loop(const char *arr, npy_intp stride, npy_intp len)
{
    npy_intp max = 0;

    while (len--) {
        const npy_intp val = *(const npy_intp *)arr;
        arr += stride;

        if (val < 0) {
            max = -1;
            break;
        }
        if (val > max) {
            max = val;
        }
    }

    return max;
}

static npy_intp
get_non_negative_max(PyArrayObject* arr)
{
    npy_intp max_arr;
    PyArray_Descr *type_intp = PyArray_DescrFromType(NPY_INTP);
    npy_bool no_iter = PyArray_ISALIGNED(arr) &&
                       PyArray_ISNOTSWAPPED(arr) &&
                       PyArray_EquivTypes(PyArray_DESCR(arr), type_intp) &&
                       (PyArray_NDIM(arr) <= 1 || PyArray_ISCONTIGUOUS(arr));

    if (!PyArray_SIZE(arr)) {
        /* special handling for empty arrays */
        max_arr = -1;
    }
    else if (no_iter) {
        npy_intp stride = PyArray_NDIM(arr) ?
                          PyArray_STRIDE(arr, PyArray_NDIM(arr) - 1) : 0;
        NPY_BEGIN_ALLOW_THREADS;
        max_arr = max_loop((const char *)PyArray_DATA(arr),
                           stride,
                           PyArray_SIZE(arr));
        NPY_END_ALLOW_THREADS;
    }
    else {
        NpyIter *iter = NULL;
        NpyIter_IterNextFunc *iternext = NULL;
        char **dataptr;
        npy_intp *strideptr, *innersizeptr;

        iter = NpyIter_New(arr,
                           NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP |
                           NPY_ITER_BUFFERED | NPY_ITER_GROWINNER |
                           NPY_ITER_NBO | NPY_ITER_ALIGNED,
                           NPY_KEEPORDER, NPY_UNSAFE_CASTING, type_intp);
        if (!(iter && (iternext = NpyIter_GetIterNext(iter, NULL)))) {
            if (iter) {
                NpyIter_Deallocate(iter);
            }
            Py_DECREF(type_intp);
            return -1;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strideptr = NpyIter_GetInnerStrideArray(iter);
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        do {
            NPY_BEGIN_ALLOW_THREADS;
            max_arr = max_loop((const char *)dataptr[0], *strideptr,
                               *innersizeptr);
            NPY_END_ALLOW_THREADS;
        } while (max_arr >= 0 && iternext(iter));

        NpyIter_Deallocate(iter);
    }

    Py_DECREF(type_intp);
    return max_arr;
}

/*
 * Write to 'out[i]' the number of times that 'i' comes up in 'arr', when
 * both 'arr' and 'out' are of type npy_intp. 'out' array must be contiguous
 * and long enough to hold the largest value in `arr`, typically calculated
 * with a prior call to 'max_loop'.
 */

static NPY_INLINE void
count_loop(const char *arr, npy_intp *out, npy_intp stride, npy_intp len)
{
    while (len--) {
        const npy_intp val = *(const npy_intp *)arr;

        arr += stride;
        out[val]++;
    }
}

/*
 * Add to 'out[i]' the sum of all the items in 'weights' at positions
 * where 'arr' holds the value 'i', with 'arr' of type npy_intp and 'weights'
 * and 'out' of type npy_double. 'out' must be contiguous and long enough to
 * hold the largest value in `arr`, typically calculated with a prior call
 * to 'max_loop'
 */

static NPY_INLINE void
weights_loop(const char *arr, const char *weights, npy_double *out,
             npy_intp stride_arr, npy_intp stride_wts, npy_intp len)
{
    while (len--) {
        const npy_intp val_arr = *(const npy_intp *)arr;
        const npy_double val_wts = *(const npy_double *)weights;

        arr += stride_arr;
        weights += stride_wts;
        out[val_arr] += val_wts;
    }
}

/*
 * On succes, returns 0 and fills 'axes' with the bounds-checked axes found
 * in 'axes_in', and 'axes_len' with the number of axes found. On failure
 * sets an exception and returns -1;
 */

static int
parse_axes(PyObject *axes_in, int *axes, int *axes_len, int ndim)
{
    npy_intp naxes;
    npy_intp j;

    if (!axes_in || axes_in == Py_None) {
        /* Convert blank argument or 'None' into all the axes */
        naxes = ndim;
        for (j = 0; j < naxes; j++) {
            axes[j] = j;
        }
    }
    else if (PyTuple_Check(axes_in)) {
        naxes = PyTuple_Size(axes_in);
        if (naxes < 0 || naxes > NPY_MAXDIMS) {
            PyErr_SetString(PyExc_ValueError, "too many values for 'axis'");
            return -1;
        }
        for (j = 0; j < naxes; j++) {
            PyObject *axis_obj = PyTuple_GET_ITEM(axes_in, j);
            long axis;

            if (!PyInt_Check(axis_obj)) {
                /* Don't allow floats or the like to slip through as ints */
                PyErr_SetString(PyExc_TypeError,
                        "'axis' must be an integer or tuple of integers");
                return -1;
            }
            axis = PyInt_AsLong(axis_obj);
            if (axis == -1 && PyErr_Occurred()) {
                return -1;
            }
            if (axis < 0) {
                axis += ndim;
            }
            if (axis < 0 || axis >= ndim) {
                PyErr_SetString(PyExc_ValueError,
                                "'axis' entry is out of bounds");
                return -1;
            }
            axes[j] = (int)axis;
        }
    }
    else if (PyInt_Check(axes_in)){
        /* `axis` can also be a single integer */
        long axis = PyInt_AsLong(axes_in);
        if (axis == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (axis < 0) {
            axis += ndim;
        }
        /* Special case letting axis={0 or -1} slip through for scalars */
        if (ndim == 0 && (axis == 0 || axis == -1)) {
            axis = 0;
        }
        else if (axis < 0 || axis >= ndim) {
            PyErr_SetString(PyExc_ValueError,
                            "'axis' entry is out of bounds");
            return -1;
        }
        axes[0] = (int)axis;
        naxes = 1;
    } else {
        /* Don't allow floats or the like to slip through as ints */
        PyErr_SetString(PyExc_TypeError,
                        "'axis' must be an integer or tuple of integers");
        return -1;
    }

    *axes_len = naxes;
    return 0;
}

/*
 * arr_ndbincount is registered as bincount.
 *
 * bincount accepts one, two or three arguments. The first is an array of
 * non-negative integers. The second, if present, is an array of weights,
 * which must be promotable to double. Call these arguments 'arr' and
 * 'weights'. Their shapes must broadcast. The fourth argument, call
 * it `axis', holds a list of axes, and defaults to all axes. The
 * return will have the shape of the broadcasted arrays, with the axes in
 * `axis` removed, and an extra dimension added at the end
 * If 'weights' is not present then 'bincount(arr)[..., i]` is the number of
 * occurrences of 'i' in 'arr' over the removed 'axis' at each of the
 * non-removed positions.
 * If 'weights' is present then 'bincount(arr, weights)[..., i]' is the sum
 * of all 'weights[j]' where 'arr[j] == i' over the removed 'axis' at each
 * of the non-removed positions.
 * The third argument, call it 'minlength', if present, is a minimum length
 * for the last dimension of the output array.
 * The entries in the fourth argument, 'axis', refer to the original shape of
 * the 'arr' argument before broadcasting.
 */

static PyObject*
arr_ndbincount(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    /* Raw input arguments  */
    PyObject *arr = NULL;
    PyObject *weights = NULL;
    PyObject *axis = NULL;
    PyObject *minlength = Py_None;
    static char *kwlist[] = {"arr", "weights", "minlength", "axis", NULL};
    /* Array version of the inputs and output */
    PyArrayObject *arr_arr = NULL;
    PyArrayObject *arr_wts = NULL;
    PyArrayObject *arr_ret = NULL;
    /* Metadata of the input arguments */
    int axes[NPY_MAXDIMS];
    int naxes;
    npy_bool axis_flags[NPY_MAXDIMS];
    int ndim_arr, ndim_wts, ndim_ret;
    npy_intp shape_ret[NPY_MAXDIMS];
    npy_intp shape_array_arr[NPY_MAXDIMS];
    npy_intp *shape_arr = shape_array_arr;
    npy_intp max_arr = 0;
    npy_intp min_len = 0;
    npy_intp j, k;
    /* We have to take ownership of this type references */
    PyArray_Descr *type_intp = PyArray_DescrFromType(NPY_INTP);
    PyArray_Descr *type_double = PyArray_DescrFromType(NPY_DOUBLE);
    /* Iterator related declarations */
    NpyIter *iter_outer = NULL, *iter_inner = NULL;
    int ndim_outer, ndim_inner;
    int op_axes_outer_arrays[3][NPY_MAXDIMS];
    int op_axes_inner_arrays[2][NPY_MAXDIMS];
    npy_bool is_empty_arr = 0, no_iter_count = 0, no_iter_weights = 0;
    NPY_BEGIN_THREADS_DEF;


    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOO", kwlist,
                                     &arr, &weights, &minlength, &axis)) {
        goto fail;
    }

    /*
     * Have to handle conversion of minlength to a npy_intp manually to
     * allow None passed in without throwing an error. Ideally we would
     * simply set its type to 'n' in the parser with a default of 0.
     */
    if (minlength == Py_None) {
        min_len = 0;
    }
    else if (PyInt_Check(minlength)) {
        min_len = (npy_intp)PyInt_AsSsize_t(minlength);
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "minlength must be an integer.");
        goto fail;
    }

    /* Convert arr to an array. Cast to NPY_INTP for 1-D arrays only */
    if (!PyArray_Check(arr)) {
        arr_arr = (PyArrayObject *)PyArray_FROMANY(arr, NPY_INTP, 0, 0,
                                                   NPY_ARRAY_CARRAY_RO);
    }
    else if (PyArray_ISINTEGER((PyArrayObject *)arr)) {
        if (PyArray_NDIM((PyArrayObject *)arr) == 1) {
            arr_arr = (PyArrayObject *)PyArray_FROMANY(arr, NPY_INTP, 0, 0,
                                                       NPY_ARRAY_ALIGNED |
                                                       NPY_ARRAY_FORCECAST);
        }
         else {
            arr_arr = (PyArrayObject *)arr;
            Py_INCREF(arr_arr);
         }
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "input must be an array of integers");
    }
    if (!arr_arr) {

        goto fail;
    }
    ndim_arr = PyArray_NDIM(arr_arr);
    is_empty_arr = PyArray_SIZE(arr_arr) ? 0 : 1;

    /* Check axis in bounds */
    if (parse_axes(axis, axes, &naxes, ndim_arr) < 0) {
        goto fail;
    }
    memset(axis_flags, 0, ndim_arr);
    for (j = 0; j < naxes; j++) {
        int axis = axes[j];
        if (axis_flags[axis]) {
            PyErr_SetString(PyExc_ValueError, "duplicate value in 'axis'");
            goto fail;;
        }
        axis_flags[axis] = 1;
    }

    /* Convert weights to an array. Cast to NPY_DOUBLE for 1-D arrays only */
    if (weights && weights != Py_None) {

        if (!PyArray_Check(weights)) {
            arr_wts = (PyArrayObject *)PyArray_FROMANY(weights,
                                                       NPY_DOUBLE, 0, 0,
                                                       NPY_ARRAY_CARRAY_RO);
        }
        else if (PyArray_CanCastTypeTo(PyArray_DESCR((PyArrayObject *)weights),
                                       type_double, NPY_SAFE_CASTING)) {
            if (PyArray_NDIM((PyArrayObject *)weights) <= 1) {
                arr_wts = (PyArrayObject *)PyArray_FROMANY(weights,
                                                    NPY_DOUBLE, 0, 0,
                                                    NPY_ARRAY_ALIGNED |
                                                    NPY_ARRAY_FORCECAST);
            }
             else {
                arr_wts = (PyArrayObject *)weights;
                Py_INCREF(arr_wts);
             }
        } else {
            PyErr_SetString(PyExc_TypeError,
                            "cannot cast 'weights' to float");
        }
        if (!arr_wts) {
            goto fail;
        }
        ndim_wts = PyArray_NDIM(arr_wts);
    }

    /* Broadacst the arrays to create the return array shape */
    shape_arr = PyArray_DIMS(arr_arr);

    if (arr_wts) {
        npy_intp *shape_wts = PyArray_DIMS(arr_wts);

        j = ndim_wts > ndim_arr ? ndim_arr - ndim_wts : 0;
        k = ndim_wts > ndim_arr ? 0 : ndim_arr - ndim_wts;

        for (ndim_inner = 0, ndim_outer = 0; j < ndim_arr; j++, k++) {

            if (j < 0 || !axis_flags[j]) {
                op_axes_outer_arrays[0][ndim_outer] = j < 0 ? -1 : j;
                op_axes_outer_arrays[1][ndim_outer] = k < 0 ? -1 : k;
                op_axes_outer_arrays[2][ndim_outer] = ndim_outer;
                /*
                 * We are not validating whether the arrays broadcast,
                 * the iterator will handle that. But if they broadcast,
                 * then this is the broadcast shape.
                 */
                if (j < 0 || (k >= 0 && shape_arr[j] == 1)) {
                    shape_ret[ndim_outer++] = shape_wts[k];
                }
                else  {
                    shape_ret[ndim_outer++] = shape_arr[j];
                }
            }
            else {
                op_axes_inner_arrays[0][ndim_inner] = j;
                op_axes_inner_arrays[1][ndim_inner++] = k < 0 ? -1 : k;
            }
        }
    }
    else {
        for(j = 0, ndim_inner = 0, ndim_outer = 0; j < ndim_arr; j++) {
            if (!axis_flags[j]) {
                op_axes_outer_arrays[0][ndim_outer] = j;
                op_axes_outer_arrays[2][ndim_outer] = ndim_outer;
                shape_ret[ndim_outer++] = shape_arr[j];
            }
            else {
                op_axes_inner_arrays[0][ndim_inner++] = j;
            }
        }
    }
    ndim_ret = ndim_outer;

    /* Find maximum entry in arr_arr and check for negative entries. */
    max_arr = get_non_negative_max(arr_arr);
    if (max_arr == -1 && !is_empty_arr) {
        /*
         * If arr is empty, we want max_arr = -1 not to raise an error,
         * as it will set the return length to the right value.
         */
        PyErr_SetString(PyExc_ValueError, "wrong value in input array");
        goto fail;
    }

    /* Finish creating the return array shape */
    shape_ret[ndim_ret++] = (min_len <= max_arr) ? max_arr + 1 : min_len;

    /* Create the return array */
    arr_ret = (PyArrayObject *)PyArray_ZEROS(ndim_ret, shape_ret,
                                             arr_wts ? NPY_DOUBLE : NPY_INTP,
                                             0);

    /* Do we need an iterator? */
    no_iter_count = PyArray_ISALIGNED(arr_arr) &&
                    PyArray_ISNOTSWAPPED(arr_arr) &&
                    PyArray_EquivTypes(PyArray_DESCR(arr_arr), type_intp) &&
                    ndim_outer == 0 &&
                    (ndim_arr <= 1 || PyArray_ISCONTIGUOUS(arr_arr));
    no_iter_weights = no_iter_count && arr_wts &&
                      PyArray_ISALIGNED(arr_wts) &&
                      PyArray_ISNOTSWAPPED(arr_wts) &&
                      PyArray_EquivTypes(PyArray_DESCR(arr_wts),
                                         type_double) &&
                      PyArray_SIZE(arr_arr) == PyArray_SIZE(arr_wts) &&
                      (ndim_wts <= 1 || PyArray_ISCONTIGUOUS(arr_wts));

    /* Finally, lets count */
    if (is_empty_arr) {
        goto finish;
    }
    else if (no_iter_count && !arr_wts) {
        npy_intp stride = ndim_arr ?
                          PyArray_STRIDE(arr_arr, ndim_arr - 1) : 0;
        NPY_BEGIN_THREADS;
        count_loop((const char *)PyArray_DATA(arr_arr),
                    (npy_intp *)PyArray_DATA(arr_ret),
                    stride,
                    PyArray_SIZE(arr_arr));
        NPY_END_THREADS;
    }
    else if (no_iter_weights) {
        npy_intp stride_arr = ndim_arr ?
                              PyArray_STRIDE(arr_arr, ndim_arr - 1) : 0;
        npy_intp stride_wts = ndim_wts ?
                              PyArray_STRIDE(arr_wts, ndim_wts - 1) : 0;
        NPY_BEGIN_THREADS;
        weights_loop((const char *)PyArray_DATA(arr_arr),
                     (const char *)PyArray_DATA(arr_wts),
                     (npy_double *)PyArray_DATA(arr_ret),
                     stride_arr, stride_wts,
                     PyArray_SIZE(arr_arr));
        NPY_END_THREADS;
    }
    else {
        PyArrayObject *op[3];
        int *op_axes_outer[3], *op_axes_inner[2];
        npy_uint32 op_flags_outer[3], op_flags_inner[2];
        npy_uint32 flags_inner;
        PyArray_Descr *op_dtypes[2];
        NpyIter_IterNextFunc *iternext_outer, *iternext_inner;
        char **dataptrs_outer, **dataptrs_inner;
        npy_intp *strideptr_inner;
        npy_intp *innersizeptr_inner;
        int nops = arr_wts ? 3 : 2;
        int nop_ret = nops - 1;

        /* We need nested iterators, set them up... */
        op[0] = arr_arr;
        op[1] = arr_wts;
        /* nop_ret will overwrite arr_wts if arr_wts == NULL */
        op[nop_ret] = arr_ret;

        /* ...the outer... */
        op_axes_outer[0] = op_axes_outer_arrays[0];
        op_axes_outer[1] = op_axes_outer_arrays[1];
        op_axes_outer[nop_ret] = op_axes_outer_arrays[2];
        op_flags_outer[0] = NPY_ITER_READONLY;
        op_flags_outer[1] = NPY_ITER_READONLY;
        op_flags_outer[nop_ret] = NPY_ITER_READWRITE;
        iter_outer = NpyIter_AdvancedNew(nops, op, 0, NPY_KEEPORDER,
                                         NPY_NO_CASTING, op_flags_outer,
                                         NULL, ndim_outer, op_axes_outer,
                                         NULL, -1);
        if (!iter_outer ||
            !(iternext_outer = NpyIter_GetIterNext(iter_outer, NULL))) {
                goto fail;
        }
        dataptrs_outer = NpyIter_GetDataPtrArray(iter_outer);

        /* ...and the inner */
        op_axes_inner[0] = op_axes_inner_arrays[0];
        op_axes_inner[1] = op_axes_inner_arrays[1];
        flags_inner = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED |
                      NPY_ITER_GROWINNER;
        op_flags_inner[0] = NPY_ITER_READONLY | NPY_ITER_NBO |
                            NPY_ITER_ALIGNED;
        op_flags_inner[1] = NPY_ITER_READONLY | NPY_ITER_NBO |
                            NPY_ITER_ALIGNED;
        op_dtypes[0] = type_intp;
        op_dtypes[1] = type_double;
        iter_inner = NpyIter_AdvancedNew(nops - 1, op, flags_inner,
                                         NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                         op_flags_inner, op_dtypes,
                                         ndim_inner, op_axes_inner,
                                         NULL, -1);
        if (!iter_inner ||
            !(iternext_inner = NpyIter_GetIterNext(iter_inner, NULL))) {
                goto fail;
        }
        dataptrs_inner = NpyIter_GetDataPtrArray(iter_inner);
        strideptr_inner = NpyIter_GetInnerStrideArray(iter_inner);
        innersizeptr_inner = NpyIter_GetInnerLoopSizePtr(iter_inner);

        if (arr_wts) {
            do {
                npy_double *data_ret = (npy_double *)dataptrs_outer[2];
                NpyIter_ResetBasePointers(iter_inner, dataptrs_outer, NULL);
                do {
                    NPY_BEGIN_THREADS;
                    weights_loop((const char *)dataptrs_inner[0],
                                 (const char *)dataptrs_inner[1],
                                 data_ret, strideptr_inner[0],
                                 strideptr_inner[1], *innersizeptr_inner);
                    NPY_END_THREADS;
                } while (iternext_inner(iter_inner));
            } while (iternext_outer(iter_outer));
        }
        else {
            do {
                npy_intp *data_ret = (npy_intp *)dataptrs_outer[1];
                NpyIter_ResetBasePointers(iter_inner, dataptrs_outer, NULL);
                do {
                    NPY_BEGIN_THREADS;
                    count_loop((const char *)dataptrs_inner[0], data_ret,
                               strideptr_inner[0], *innersizeptr_inner);
                    NPY_END_THREADS;
                } while (iternext_inner(iter_inner));
            } while (iternext_outer(iter_outer));
        }

        NpyIter_Deallocate(iter_inner);
        NpyIter_Deallocate(iter_outer);
    }

    finish:
        Py_DECREF(type_intp);
        Py_DECREF(type_double);
        Py_XDECREF(arr_arr);
        Py_XDECREF(arr_wts);
        return (PyObject *)arr_ret;

    fail:
        if (iter_inner) {
            NpyIter_Deallocate(iter_inner);
        }
        if (iter_outer) {
            NpyIter_Deallocate(iter_outer);
        }
        Py_DECREF(type_intp);
        Py_DECREF(type_double);
        Py_XDECREF(arr_arr);
        Py_XDECREF(arr_wts);
        Py_XDECREF(arr_ret);
        return NULL;
}

/*
 * digitize (x, bins, right=False) returns an array of python integers the same
 * length of x. The values i returned are such that bins [i - 1] <= x <
 * bins [i] if bins is monotonically increasing, or bins [i - 1] > x >=
 * bins [i] if bins is monotonically decreasing.  Beyond the bounds of
 * bins, returns either i = 0 or i = len (bins) as appropriate.
 * if right == True the comparison is bins [i - 1] < x <= bins[i]
 * or bins [i - 1] >= x > bins[i]
 */
static PyObject *
arr_digitize(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    /* self is not used */
    PyObject *ox, *obins;
    PyArrayObject *ax = NULL, *abins = NULL, *aret = NULL;
    double *dx, *dbins;
    npy_intp lbins, lx;             /* lengths */
    npy_intp right = 0; /* whether right or left is inclusive */
    npy_intp *iret;
    int m, i;
    static char *kwlist[] = {"x", "bins", "right", NULL};
    PyArray_Descr *type;
    char bins_non_monotonic = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist, &ox, &obins,
                &right)) {
        goto fail;
    }
    type = PyArray_DescrFromType(NPY_DOUBLE);
    ax = (PyArrayObject *)PyArray_FromAny(ox, type,
                                        1, 1, NPY_ARRAY_CARRAY, NULL);
    if (ax == NULL) {
        goto fail;
    }
    Py_INCREF(type);
    abins = (PyArrayObject *)PyArray_FromAny(obins, type,
                                        1, 1, NPY_ARRAY_CARRAY, NULL);
    if (abins == NULL) {
        goto fail;
    }

    lx = PyArray_SIZE(ax);
    dx = (double *)PyArray_DATA(ax);
    lbins = PyArray_SIZE(abins);
    dbins = (double *)PyArray_DATA(abins);
    aret = (PyArrayObject *)PyArray_SimpleNew(1, &lx, NPY_INTP);
    if (aret == NULL) {
        goto fail;
    }
    iret = (npy_intp *)PyArray_DATA(aret);

    if (lx <= 0 || lbins < 0) {
        PyErr_SetString(PyExc_ValueError,
                "Both x and bins must have non-zero length");
            goto fail;
    }
    NPY_BEGIN_ALLOW_THREADS;
    if (lbins == 1)  {
        if (right == 0) {
            for (i = 0; i < lx; i++) {
                if (dx [i] >= dbins[0]) {
                    iret[i] = 1;
                }
                else {
                    iret[i] = 0;
                }
            }
        }
        else {
            for (i = 0; i < lx; i++) {
                if (dx [i] > dbins[0]) {
                    iret[i] = 1;
                }
                else {
                    iret[i] = 0;
                }
            }

        }
    }
    else {
        m = check_array_monotonic(dbins, lbins);
        if (right == 0) {
            if ( m == -1 ) {
                for ( i = 0; i < lx; i ++ ) {
                    iret [i] = decr_slot_ ((double)dx[i], dbins, lbins);
                }
            }
            else if ( m == 1 ) {
                for ( i = 0; i < lx; i ++ ) {
                    iret [i] = incr_slot_ ((double)dx[i], dbins, lbins);
                }
            }
            else {
            /* defer PyErr_SetString until after NPY_END_ALLOW_THREADS */
                bins_non_monotonic = 1;
            }
        }
        else {
            if ( m == -1 ) {
                for ( i = 0; i < lx; i ++ ) {
                    iret [i] = decr_slot_right_ ((double)dx[i], dbins,
                                                                lbins);
                }
            }
            else if ( m == 1 ) {
                for ( i = 0; i < lx; i ++ ) {
                    iret [i] = incr_slot_right_ ((double)dx[i], dbins,
                                                               lbins);
                }
            }
            else {
            /* defer PyErr_SetString until after NPY_END_ALLOW_THREADS */
                bins_non_monotonic = 1;
            }

        }
    }
    NPY_END_ALLOW_THREADS;
    if (bins_non_monotonic) {
        PyErr_SetString(PyExc_ValueError,
                "The bins must be monotonically increasing or decreasing");
        goto fail;
    }
    Py_DECREF(ax);
    Py_DECREF(abins);
    return (PyObject *)aret;

fail:
    Py_XDECREF(ax);
    Py_XDECREF(abins);
    Py_XDECREF(aret);
    return NULL;
}



static char arr_insert__doc__[] = "Insert vals sequentially into equivalent 1-d positions indicated by mask.";

/*
 * Insert values from an input array into an output array, at positions
 * indicated by a mask. If the arrays are of dtype object (indicated by
 * the objarray flag), take care of reference counting.
 *
 * This function implements the copying logic of arr_insert() defined
 * below.
 */
static void
arr_insert_loop(char *mptr, char *vptr, char *input_data, char *zero,
                char *avals_data, int melsize, int delsize, int objarray,
                int totmask, int numvals, int nd, npy_intp *instrides,
                npy_intp *inshape)
{
    int mindx, rem_indx, indx, i, copied;

    /*
     * Walk through mask array, when non-zero is encountered
     * copy next value in the vals array to the input array.
     * If we get through the value array, repeat it as necessary.
     */
    copied = 0;
    for (mindx = 0; mindx < totmask; mindx++) {
        if (memcmp(mptr,zero,melsize) != 0) {
            /* compute indx into input array */
            rem_indx = mindx;
            indx = 0;
            for (i = nd - 1; i > 0; --i) {
                indx += (rem_indx % inshape[i]) * instrides[i];
                rem_indx /= inshape[i];
            }
            indx += rem_indx * instrides[0];
            /* fprintf(stderr, "mindx = %d, indx=%d\n", mindx, indx); */
            /* Copy value element over to input array */
            memcpy(input_data+indx,vptr,delsize);
            if (objarray) {
                Py_INCREF(*((PyObject **)vptr));
            }
            vptr += delsize;
            copied += 1;
            /* If we move past value data.  Reset */
            if (copied >= numvals) {
                vptr = avals_data;
            }
        }
        mptr += melsize;
    }
}

/*
 * Returns input array with values inserted sequentially into places
 * indicated by the mask
 */
static PyObject *
arr_insert(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{
    PyObject *mask = NULL, *vals = NULL;
    PyArrayObject *ainput = NULL, *amask = NULL, *avals = NULL, *tmp = NULL;
    int numvals, totmask, sameshape;
    char *input_data, *mptr, *vptr, *zero = NULL;
    int melsize, delsize, nd, objarray, k;
    npy_intp *instrides, *inshape;

    static char *kwlist[] = {"input", "mask", "vals", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O&OO", kwlist,
                PyArray_Converter, &ainput,
                &mask, &vals)) {
        goto fail;
    }

    amask = (PyArrayObject *)PyArray_FROM_OF(mask, NPY_ARRAY_CARRAY);
    if (amask == NULL) {
        goto fail;
    }
    /* Cast an object array */
    if (PyArray_DESCR(amask)->type_num == NPY_OBJECT) {
        tmp = (PyArrayObject *)PyArray_Cast(amask, NPY_INTP);
        if (tmp == NULL) {
            goto fail;
        }
        Py_DECREF(amask);
        amask = tmp;
    }

    sameshape = 1;
    if (PyArray_NDIM(amask) == PyArray_NDIM(ainput)) {
        for (k = 0; k < PyArray_NDIM(amask); k++) {
            if (PyArray_DIMS(amask)[k] != PyArray_DIMS(ainput)[k]) {
                sameshape = 0;
            }
        }
    }
    else {
        /* Test to see if amask is 1d */
        if (PyArray_NDIM(amask) != 1) {
            sameshape = 0;
        }
        else if ((PyArray_SIZE(ainput)) != PyArray_SIZE(amask)) {
            sameshape = 0;
        }
    }
    if (!sameshape) {
        PyErr_SetString(PyExc_TypeError,
                        "mask array must be 1-d or same shape as input array");
        goto fail;
    }

    avals = (PyArrayObject *)PyArray_FromObject(vals,
                                        PyArray_DESCR(ainput)->type_num, 0, 1);
    if (avals == NULL) {
        goto fail;
    }
    numvals = PyArray_SIZE(avals);
    nd = PyArray_NDIM(ainput);
    input_data = PyArray_DATA(ainput);
    mptr = PyArray_DATA(amask);
    melsize = PyArray_DESCR(amask)->elsize;
    vptr = PyArray_DATA(avals);
    delsize = PyArray_DESCR(avals)->elsize;
    zero = PyArray_Zero(amask);
    if (zero == NULL) {
        goto fail;
    }
    objarray = (PyArray_DESCR(ainput)->type_num == NPY_OBJECT);

    /* Handle zero-dimensional case separately */
    if (nd == 0) {
        if (memcmp(mptr,zero,melsize) != 0) {
            /* Copy value element over to input array */
            memcpy(input_data,vptr,delsize);
            if (objarray) {
                Py_INCREF(*((PyObject **)vptr));
            }
        }
        Py_DECREF(amask);
        Py_DECREF(avals);
        PyDataMem_FREE(zero);
        Py_DECREF(ainput);
        Py_INCREF(Py_None);
        return Py_None;
    }

    totmask = (int) PyArray_SIZE(amask);
    instrides = PyArray_STRIDES(ainput);
    inshape = PyArray_DIMS(ainput);
    if (objarray) {
        /* object array, need to refcount, can't release the GIL */
        arr_insert_loop(mptr, vptr, input_data, zero, PyArray_DATA(avals),
                        melsize, delsize, objarray, totmask, numvals, nd,
                        instrides, inshape);
    }
    else {
        /* No increfs take place in arr_insert_loop, so release the GIL */
        NPY_BEGIN_ALLOW_THREADS;
        arr_insert_loop(mptr, vptr, input_data, zero, PyArray_DATA(avals),
                        melsize, delsize, objarray, totmask, numvals, nd,
                        instrides, inshape);
        NPY_END_ALLOW_THREADS;
    }

    Py_DECREF(amask);
    Py_DECREF(avals);
    PyDataMem_FREE(zero);
    Py_DECREF(ainput);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    PyDataMem_FREE(zero);
    Py_XDECREF(ainput);
    Py_XDECREF(amask);
    Py_XDECREF(avals);
    return NULL;
}

/** @brief Use bisection on a sorted array to find first entry > key.
 *
 * Use bisection to find an index i s.t. arr[i] <= key < arr[i + 1]. If there is
 * no such i the error returns are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @return index
 */
static npy_intp
binary_search(double key, double arr [], npy_intp len)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    if (key > arr[len - 1]) {
        return len;
    }
    while (imin < imax) {
        npy_intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

static PyObject *
arr_interp(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;
    double lval, rval;
    double *dy, *dx, *dz, *dres, *slopes;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO", kwlist,
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
    lenxp = PyArray_DIMS(axp)[0];
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (PyArray_DIMS(afp)[0] != lenxp) {
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

    dy = (double *)PyArray_DATA(afp);
    dx = (double *)PyArray_DATA(axp);
    dz = (double *)PyArray_DATA(ax);
    dres = (double *)PyArray_DATA(af);

    /* Get left and right fill values. */
    if ((left == NULL) || (left == Py_None)) {
        lval = dy[0];
    }
    else {
        lval = PyFloat_AsDouble(left);
        if ((lval == -1) && PyErr_Occurred()) {
            goto fail;
        }
    }
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp-1];
    }
    else {
        rval = PyFloat_AsDouble(right);
        if ((rval == -1) && PyErr_Occurred()) {
            goto fail;
        }
    }

    /* only pre-calculate slopes if there are relatively few of them. */
    if (lenxp <= lenx) {
        slopes = (double *) PyArray_malloc((lenxp - 1)*sizeof(double));
        if (! slopes) {
            goto fail;
        }
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < lenxp - 1; i++) {
            slopes[i] = (dy[i + 1] - dy[i])/(dx[i + 1] - dx[i]);
        }
        for (i = 0; i < lenx; i++) {
            npy_intp j = binary_search(dz[i], dx, lenxp);

            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else {
                dres[i] = slopes[j]*(dz[i] - dx[j]) + dy[j];
            }
        }
        NPY_END_ALLOW_THREADS;
        PyArray_free(slopes);
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < lenx; i++) {
            npy_intp j = binary_search(dz[i], dx, lenxp);

            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else {
                double slope = (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j]);
                dres[i] = slope*(dz[i] - dx[j]) + dy[j];
            }
        }
        NPY_END_ALLOW_THREADS;
    }

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

        op[i] = (PyArrayObject *)PyArray_FromAny(item, NULL, 0, 0, 0, NULL);
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
static PyObject *
arr_ravel_multi_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    int i, s;
    PyObject *mode0=NULL, *coords0=NULL;
    PyArrayObject *ret = NULL;
    PyArray_Dims dimensions={0,0};
    npy_intp ravel_strides[NPY_MAXDIMS];
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
                s *= dimensions.ptr[i];
            }
            break;
        case NPY_FORTRANORDER:
            s = 1;
            for (i = 0; i < dimensions.len; ++i) {
                ravel_strides[i] = s;
                s *= dimensions.ptr[i];
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
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);
    return PyArray_Return(ret);

fail:
    Py_XDECREF(dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    PyDimMem_FREE(dimensions.ptr);
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
        PyErr_SetString(PyExc_ValueError,
              "invalid entry in index array");
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
        PyErr_SetString(PyExc_ValueError,
              "invalid entry in index array");
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* unravel_index implementation - see add_newdocs.py */
static PyObject *
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

    if (dimensions.len == 0) {
        PyErr_SetString(PyExc_ValueError,
                "dims must have at least one value");
        goto fail;
    }

    unravel_size = PyArray_MultiplyList(dimensions.ptr, dimensions.len);

    if (!PyArray_Check(indices0)) {
        indices = (PyArrayObject*)PyArray_FromAny(indices0,
                                                    NULL, 0, 0, 0, NULL);
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
                                0, 0, NULL);
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
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);

    return ret_tuple;

fail:
    Py_XDECREF(ret_tuple);
    Py_XDECREF(ret_arr);
    Py_XDECREF(dtype);
    Py_XDECREF(indices);
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);
    return NULL;
}


static PyTypeObject *PyMemberDescr_TypePtr = NULL;
static PyTypeObject *PyGetSetDescr_TypePtr = NULL;
static PyTypeObject *PyMethodDescr_TypePtr = NULL;

/* Can only be called if doc is currently NULL */
static PyObject *
arr_add_docstring(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *obj;
    PyObject *str;
    char *docstr;
    static char *msg = "already has a docstring";

    /* Don't add docstrings */
    if (Py_OptimizeFlag > 1) {
        Py_INCREF(Py_None);
        return Py_None;
    }
#if defined(NPY_PY3K)
    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyUnicode_Type, &str)) {
        return NULL;
    }

    docstr = PyBytes_AS_STRING(PyUnicode_AsUTF8String(str));
#else
    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyString_Type, &str)) {
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
        Py_INCREF(Py_None);
        return Py_None;
    }

#undef _TESTDOC1
#undef _TESTDOC2
#undef _ADDDOC

    Py_INCREF(str);
    Py_INCREF(Py_None);
    return Py_None;
}


/* docstring in numpy.add_newdocs.py */
static PyObject *
add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyUFuncObject *ufunc;
    PyObject *str;
    char *docstr, *newdocstr;

#if defined(NPY_PY3K)
    if (!PyArg_ParseTuple(args, "O!O!", &PyUFunc_Type, &ufunc,
                                        &PyUnicode_Type, &str)) {
        return NULL;
    }
    docstr = PyBytes_AS_STRING(PyUnicode_AsUTF8String(str));
#else
    if (!PyArg_ParseTuple(args, "O!O!", &PyUFunc_Type, &ufunc,
                                         &PyString_Type, &str)) {
         return NULL;
    }
    docstr = PyString_AS_STRING(str);
#endif

    if (NULL != ufunc->doc) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot change docstring of ufunc with non-NULL docstring");
        return NULL;
    }

    /*
     * This introduces a memory leak, as the memory allocated for the doc
     * will not be freed even if the ufunc itself is deleted. In practice
     * this should not be a problem since the user would have to
     * repeatedly create, document, and throw away ufuncs.
     */
    newdocstr = malloc(strlen(docstr) + 1);
    strcpy(newdocstr, docstr);
    ufunc->doc = newdocstr;

    Py_INCREF(Py_None);
    return Py_None;
}

/*  PACKBITS
 *
 *  This function packs binary (0 or 1) 1-bit per pixel arrays
 *  into contiguous bytes.
 *
 */

static void
_packbits( void *In,
           int element_size,  /* in bytes */
           npy_intp in_N,
           npy_intp in_stride,
           void *Out,
           npy_intp out_N,
           npy_intp out_stride
)
{
    char build;
    int i, index;
    npy_intp out_Nm1;
    int maxi, remain, nonzero, j;
    char *outptr,*inptr;

    outptr = Out;    /* pointer to output buffer */
    inptr  = In;     /* pointer to input buffer */

    /*
     * Loop through the elements of In
     * Determine whether or not it is nonzero.
     *  Yes: set correspdoning bit (and adjust build value)
     *  No:  move on
     * Every 8th value, set the value of build and increment the outptr
     */

    remain = in_N % 8;                      /* uneven bits */
    if (remain == 0) {
        remain = 8;
    }
    out_Nm1 = out_N - 1;
    for (index = 0; index < out_N; index++) {
        build = 0;
        maxi = (index != out_Nm1 ? 8 : remain);
        for (i = 0; i < maxi; i++) {
            build <<= 1;
            nonzero = 0;
            for (j = 0; j < element_size; j++) {
                nonzero += (*(inptr++) != 0);
            }
            inptr += (in_stride - element_size);
            build += (nonzero != 0);
        }
        if (index == out_Nm1) build <<= (8-remain);
        /* printf("Here: %d %d %d %d\n",build,slice,index,maxi); */
        *outptr = build;
        outptr += out_stride;
    }
    return;
}


static void
_unpackbits(void *In,
        int NPY_UNUSED(el_size),  /* unused */
        npy_intp in_N,
        npy_intp in_stride,
        void *Out,
        npy_intp NPY_UNUSED(out_N),
        npy_intp out_stride
        )
{
    unsigned char mask;
    int i, index;
    char *inptr, *outptr;

    outptr = Out;
    inptr  = In;
    for (index = 0; index < in_N; index++) {
        mask = 128;
        for (i = 0; i < 8; i++) {
            *outptr = ((mask & (unsigned char)(*inptr)) != 0);
            outptr += out_stride;
            mask >>= 1;
        }
        inptr += in_stride;
    }
    return;
}

/* Fixme -- pack and unpack should be separate routines */
static PyObject *
pack_or_unpack_bits(PyObject *input, int axis, int unpack)
{
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    void (*thefunc)(void *, int, npy_intp, npy_intp, void *, npy_intp, npy_intp);
    PyArrayIterObject *it, *ot;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (unpack) {
        if (PyArray_TYPE(inp) != NPY_UBYTE) {
            PyErr_SetString(PyExc_TypeError,
                    "Expected an input array of unsigned byte data type");
            goto fail;
        }
    }
    else if (!PyArray_ISINTEGER(inp)) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of integer data type");
        goto fail;
    }

    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }
    /* Handle zero-dim array separately */
    if (PyArray_SIZE(new) == 0) {
        return PyArray_Copy(new);
    }

    if (PyArray_NDIM(new) == 0) {
        if (unpack) {
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
        else {
            char *optr, *iptr;
            out = (PyArrayObject *)PyArray_New(Py_TYPE(new), 0, NULL, NPY_UBYTE,
                    NULL, NULL, 0, 0, NULL);
            if (out == NULL) {
                goto fail;
            }
            optr = PyArray_DATA(out);
            iptr = PyArray_DATA(new);
            *optr = 0;
            for (i = 0; i<PyArray_ITEMSIZE(new); i++) {
                if (*iptr != 0) {
                    *optr = 1;
                    break;
                }
                iptr++;
            }
            goto finish;
        }
    }


    /* Setup output shape */
    for (i=0; i<PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    if (unpack) {
        /* Multiply axis dimension by 8 */
        outdims[axis] <<= 3;
        thefunc = _unpackbits;
    }
    else {
        /*
         * Divide axis dimension by 8
         * 8 -> 1, 9 -> 2, 16 -> 2, 17 -> 3 etc..
         */
        outdims[axis] = ((outdims[axis] - 1) >> 3) + 1;
        thefunc = _packbits;
    }

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

    while(PyArray_ITER_NOTDONE(it)) {
        thefunc(PyArray_ITER_DATA(it), PyArray_ITEMSIZE(new),
                PyArray_DIM(new, axis), PyArray_STRIDE(new, axis),
                PyArray_ITER_DATA(ot), PyArray_DIM(out, axis),
                PyArray_STRIDE(out, axis));
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
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
io_pack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return pack_or_unpack_bits(obj, axis, 0);
}

static PyObject *
io_unpack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return pack_or_unpack_bits(obj, axis, 1);
}

/* The docstrings for many of these methods are in add_newdocs.py. */
static struct PyMethodDef methods[] = {
    {"_insert", (PyCFunction)arr_insert,
        METH_VARARGS | METH_KEYWORDS, arr_insert__doc__},
    {"bincount", (PyCFunction)arr_ndbincount,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"digitize", (PyCFunction)arr_digitize,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"interp", (PyCFunction)arr_interp,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ravel_multi_index", (PyCFunction)arr_ravel_multi_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unravel_index", (PyCFunction)arr_unravel_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"add_docstring", (PyCFunction)arr_add_docstring,
        METH_VARARGS, NULL},
    {"add_newdoc_ufunc", (PyCFunction)add_newdoc_ufunc,
        METH_VARARGS, NULL},
    {"packbits", (PyCFunction)io_pack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unpackbits", (PyCFunction)io_unpack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}    /* sentinel */
};

static void
define_types(void)
{
    PyObject *tp_dict;
    PyObject *myobj;

    tp_dict = PyArrayDescr_Type.tp_dict;
    /* Get "subdescr" */
    myobj = PyDict_GetItemString(tp_dict, "fields");
    if (myobj == NULL) {
        return;
    }
    PyGetSetDescr_TypePtr = Py_TYPE(myobj);
    myobj = PyDict_GetItemString(tp_dict, "alignment");
    if (myobj == NULL) {
        return;
    }
    PyMemberDescr_TypePtr = Py_TYPE(myobj);
    myobj = PyDict_GetItemString(tp_dict, "newbyteorder");
    if (myobj == NULL) {
        return;
    }
    PyMethodDescr_TypePtr = Py_TYPE(myobj);
    return;
}

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_compiled_base",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit__compiled_base(void)
#else
#define RETVAL
PyMODINIT_FUNC
init_compiled_base(void)
#endif
{
    PyObject *m, *d;

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_compiled_base", methods);
#endif
    if (!m) {
        return RETVAL;
    }

    /* Import the array objects */
    import_array();
    import_umath();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "numpy.lib.error".
     * This is for backward compatibility with existing code.
     */
    PyDict_SetItemString(d, "error", PyExc_Exception);


    /* define PyGetSetDescr_Type and PyMemberDescr_Type */
    define_types();

    return RETVAL;
}
