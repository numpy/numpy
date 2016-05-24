#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "numpy/arrayobject.h"
#include "arrayobject.h"

#include "npy_config.h"

#include "npy_pycompat.h"
#include "npy_import.h"

#include "common.h"
#include "iterators.h"
#include "mapping.h"
#include "lowlevel_strided_loops.h"
#include "item_selection.h"


#define HAS_INTEGER 1
#define HAS_NEWAXIS 2
#define HAS_SLICE 4
#define HAS_ELLIPSIS 8
/* HAS_FANCY can be mixed with HAS_0D_BOOL, be careful when to use & or == */
#define HAS_FANCY 16
#define HAS_BOOL 32
/* NOTE: Only set if it is neither fancy nor purely integer index! */
#define HAS_SCALAR_ARRAY 64
/*
 * Indicate that this is a fancy index that comes from a 0d boolean.
 * This means that the index does not operate along a real axis. The
 * corresponding index type is just HAS_FANCY.
 */
#define HAS_0D_BOOL (HAS_FANCY | 128)


static int
_nonzero_indices(PyObject *myBool, PyArrayObject **arrays);

/******************************************************************************
 ***                    IMPLEMENT MAPPING PROTOCOL                          ***
 *****************************************************************************/

NPY_NO_EXPORT Py_ssize_t
array_length(PyArrayObject *self)
{
    if (PyArray_NDIM(self) != 0) {
        return PyArray_DIMS(self)[0];
    } else {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object");
        return -1;
    }
}


/* -------------------------------------------------------------- */

/*NUMPY_API
 *
 */
NPY_NO_EXPORT void
PyArray_MapIterSwapAxes(PyArrayMapIterObject *mit, PyArrayObject **ret, int getmap)
{
    PyObject *new;
    int n1, n2, n3, val, bnd;
    int i;
    PyArray_Dims permute;
    npy_intp d[NPY_MAXDIMS];
    PyArrayObject *arr;

    permute.ptr = d;
    permute.len = mit->nd;

    /*
     * arr might not have the right number of dimensions
     * and need to be reshaped first by pre-pending ones
     */
    arr = *ret;
    if (PyArray_NDIM(arr) != mit->nd) {
        for (i = 1; i <= PyArray_NDIM(arr); i++) {
            permute.ptr[mit->nd-i] = PyArray_DIMS(arr)[PyArray_NDIM(arr)-i];
        }
        for (i = 0; i < mit->nd-PyArray_NDIM(arr); i++) {
            permute.ptr[i] = 1;
        }
        new = PyArray_Newshape(arr, &permute, NPY_ANYORDER);
        Py_DECREF(arr);
        *ret = (PyArrayObject *)new;
        if (new == NULL) {
            return;
        }
    }

    /*
     * Setting and getting need to have different permutations.
     * On the get we are permuting the returned object, but on
     * setting we are permuting the object-to-be-set.
     * The set permutation is the inverse of the get permutation.
     */

    /*
     * For getting the array the tuple for transpose is
     * (n1,...,n1+n2-1,0,...,n1-1,n1+n2,...,n3-1)
     * n1 is the number of dimensions of the broadcast index array
     * n2 is the number of dimensions skipped at the start
     * n3 is the number of dimensions of the result
     */

    /*
     * For setting the array the tuple for transpose is
     * (n2,...,n1+n2-1,0,...,n2-1,n1+n2,...n3-1)
     */
    n1 = mit->nd_fancy;
    n2 = mit->consec; /* axes to insert at */
    n3 = mit->nd;

    /* use n1 as the boundary if getting but n2 if setting */
    bnd = getmap ? n1 : n2;
    val = bnd;
    i = 0;
    while (val < n1 + n2) {
        permute.ptr[i++] = val++;
    }
    val = 0;
    while (val < bnd) {
        permute.ptr[i++] = val++;
    }
    val = n1 + n2;
    while (val < n3) {
        permute.ptr[i++] = val++;
    }
    new = PyArray_Transpose(*ret, &permute);
    Py_DECREF(*ret);
    *ret = (PyArrayObject *)new;
}


/**
 * Prepare an npy_index_object from the python slicing object.
 *
 * This function handles all index preparations with the exception
 * of field access. It fills the array of index_info structs correctly.
 * It already handles the boolean array special case for fancy indexing,
 * i.e. if the index type is boolean, it is exactly one matching boolean
 * array. If the index type is fancy, the boolean array is already
 * converted to integer arrays. There is (as before) no checking of the
 * boolean dimension.
 *
 * Checks everything but the bounds.
 *
 * @param the array being indexed
 * @param the index object
 * @param index info struct being filled (size of NPY_MAXDIMS * 2 + 1)
 * @param number of indices found
 * @param dimension of the indexing result
 * @param dimension of the fancy/advanced indices part
 * @param whether to allow the boolean special case
 *
 * @returns the index_type or -1 on failure and fills the number of indices.
 */
NPY_NO_EXPORT int
prepare_index(PyArrayObject *self, PyObject *index,
              npy_index_info *indices,
              int *num, int *ndim, int *out_fancy_ndim, int allow_boolean)
{
    int new_ndim, fancy_ndim, used_ndim, index_ndim;
    int curr_idx, get_idx;

    int i;
    npy_intp n;

    npy_bool make_tuple = 0;
    PyObject *obj = NULL;
    PyArrayObject *arr;

    int index_type = 0;
    int ellipsis_pos = -1;
    static PyObject *VisibleDeprecation = NULL;

    npy_cache_import(
        "numpy", "VisibleDeprecationWarning", &VisibleDeprecation);

    /*
     * The index might be a multi-dimensional index, but not yet a tuple
     * this makes it a tuple in that case.
     *
     * TODO: Refactor into its own function.
     */
    if (!PyTuple_CheckExact(index)
            /* Next three are just to avoid slow checks */
#if !defined(NPY_PY3K)
            && (!PyInt_CheckExact(index))
#else
            && (!PyLong_CheckExact(index))
#endif
            && (index != Py_None)
            && (!PySlice_Check(index))
            && (!PyArray_Check(index))
            && (PySequence_Check(index))) {
        /*
         * Sequences < NPY_MAXDIMS with any slice objects
         * or newaxis, Ellipsis or other arrays or sequences
         * embedded, are considered equivalent to an indexing
         * tuple. (`a[[[1,2], [3,4]]] == a[[1,2], [3,4]]`)
         */

        if (PyTuple_Check(index)) {
            /* If it is already a tuple, make it an exact tuple anyway */
            n = 0;
            make_tuple = 1;
        }
        else {
            n = PySequence_Size(index);
        }
        if (n < 0 || n >= NPY_MAXDIMS) {
            n = 0;
        }
        for (i = 0; i < n; i++) {
            PyObject *tmp_obj = PySequence_GetItem(index, i);
            /* if getitem fails (unusual) treat this as a single index */
            if (tmp_obj == NULL) {
                PyErr_Clear();
                make_tuple = 0;
                break;
            }
            if (PyArray_Check(tmp_obj) || PySequence_Check(tmp_obj)
                    || PySlice_Check(tmp_obj) || tmp_obj == Py_Ellipsis
                    || tmp_obj == Py_None) {
                make_tuple = 1;
                Py_DECREF(tmp_obj);
                break;
            }
            Py_DECREF(tmp_obj);
        }

        if (make_tuple) {
            /* We want to interpret it as a tuple, so make it one */
            index = PySequence_Tuple(index);
            if (index == NULL) {
                return -1;
            }
        }
    }

    /* If the index is not a tuple, handle it the same as (index,) */
    if (!PyTuple_CheckExact(index)) {
        obj = index;
        index_ndim = 1;
    }
    else {
        n = PyTuple_GET_SIZE(index);
        if (n > NPY_MAXDIMS * 2) {
            PyErr_SetString(PyExc_IndexError,
                            "too many indices for array");
            goto fail;
        }
        index_ndim = (int)n;
        obj = NULL;
    }

    /*
     * Parse all indices into the `indices` array of index_info structs
     */
    used_ndim = 0;
    new_ndim = 0;
    fancy_ndim = 0;
    get_idx = 0;
    curr_idx = 0;

    while (get_idx < index_ndim) {
        if (curr_idx > NPY_MAXDIMS * 2) {
            PyErr_SetString(PyExc_IndexError,
                            "too many indices for array");
            goto failed_building_indices;
        }

        /* Check for single index. obj is already set then. */
        if ((curr_idx != 0) || (obj == NULL)) {
            obj = PyTuple_GET_ITEM(index, get_idx++);
        }
        else {
            /* only one loop */
            get_idx += 1;
        }

        /**** Try the cascade of possible indices ****/

        /* Index is an ellipsis (`...`) */
        if (obj == Py_Ellipsis) {
            /*
             * If there is more then one Ellipsis, it is replaced. Deprecated,
             * since it is hard to imagine anyone using two Ellipsis and
             * actually planning on all but the first being automatically
             * replaced with a slice.
             */
            if (index_type & HAS_ELLIPSIS) {
                /* 2013-04-14, 1.8 */
                if (PyErr_WarnEx(
                        VisibleDeprecation,
                        "an index can only have a single Ellipsis (`...`); "
                        "replace all but one with slices (`:`).",
                        1) < 0) {
                    goto failed_building_indices;
                }
                index_type |= HAS_SLICE;

                indices[curr_idx].type = HAS_SLICE;
                indices[curr_idx].object = PySlice_New(NULL, NULL, NULL);

                if (indices[curr_idx].object == NULL) {
                    goto failed_building_indices;
                }

                used_ndim += 1;
                new_ndim += 1;
                curr_idx += 1;
                continue;
            }
            index_type |= HAS_ELLIPSIS;

            indices[curr_idx].type = HAS_ELLIPSIS;
            indices[curr_idx].object = NULL;
            /* number of slices it is worth, won't update if it is 0: */
            indices[curr_idx].value = 0;

            ellipsis_pos = curr_idx;
            /* the used and new ndim will be found later */
            used_ndim += 0;
            new_ndim += 0;
            curr_idx += 1;
            continue;
        }

        /* Index is np.newaxis/None */
        else if (obj == Py_None) {
            index_type |= HAS_NEWAXIS;

            indices[curr_idx].type = HAS_NEWAXIS;
            indices[curr_idx].object = NULL;

            used_ndim += 0;
            new_ndim += 1;
            curr_idx += 1;
            continue;
        }

        /* Index is a slice object. */
        else if (PySlice_Check(obj)) {
            index_type |= HAS_SLICE;

            Py_INCREF(obj);
            indices[curr_idx].object = obj;
            indices[curr_idx].type = HAS_SLICE;
            used_ndim += 1;
            new_ndim += 1;
            curr_idx += 1;
            continue;
        }

        /*
         * Special case to allow 0-d boolean indexing with scalars.
         * Should be removed after boolean as integer deprecation.
         * Since this is always an error if it was not a boolean, we can
         * allow the 0-d special case before the rest.
         */
        else if (PyArray_NDIM(self) != 0) {
            /*
             * Single integer index, there are two cases here.
             * It could be an array, a 0-d array is handled
             * a bit weird however, so need to special case it.
             */
#if !defined(NPY_PY3K)
            if (PyInt_CheckExact(obj) || !PyArray_Check(obj)) {
#else
            if (PyLong_CheckExact(obj) || !PyArray_Check(obj)) {
#endif
                npy_intp ind = PyArray_PyIntAsIntp(obj);

                if ((ind == -1) && PyErr_Occurred()) {
                    PyErr_Clear();
                }
                else {
                    index_type |= HAS_INTEGER;
                    indices[curr_idx].object = NULL;
                    indices[curr_idx].value = ind;
                    indices[curr_idx].type = HAS_INTEGER;
                    used_ndim += 1;
                    new_ndim += 0;
                    curr_idx += 1;
                    continue;
                }
            }
        }

        /*
         * At this point, we must have an index array (or array-like).
         * It might still be a (purely) bool special case, a 0-d integer
         * array (an array scalar) or something invalid.
         */

        if (!PyArray_Check(obj)) {
            PyArrayObject *tmp_arr;
            tmp_arr = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
            if (tmp_arr == NULL) {
                /* TODO: Should maybe replace the error here? */
                goto failed_building_indices;
            }

            /*
             * For example an empty list can be cast to an integer array,
             * however it will default to a float one.
             */
            if (PyArray_SIZE(tmp_arr) == 0) {
                PyArray_Descr *indtype = PyArray_DescrFromType(NPY_INTP);

                arr = (PyArrayObject *)PyArray_FromArray(tmp_arr, indtype,
                                                         NPY_ARRAY_FORCECAST);
                Py_DECREF(tmp_arr);
                if (arr == NULL) {
                    goto failed_building_indices;
                }
            }
            /*
             * Special case to allow 0-d boolean indexing with
             * scalars. Should be removed after boolean-array
             * like as integer-array like deprecation.
             * (does not cover ufunc.at, because it does not use the
             * boolean special case, but that should not matter...)
             * Since all but strictly boolean indices are invalid,
             * there is no need for any further conversion tries.
             */
            else if (PyArray_NDIM(self) == 0) {
                arr = tmp_arr;
            }
            else {
                /*
                 * These Checks can be removed after deprecation, since
                 * they should then be either correct already or error out
                 * later just like a normal array.
                 */
                if (PyArray_ISBOOL(tmp_arr)) {
                    /* 2013-04-14, 1.8 */
                    if (DEPRECATE_FUTUREWARNING(
                            "in the future, boolean array-likes will be "
                            "handled as a boolean array index") < 0) {
                        Py_DECREF(tmp_arr);
                        goto failed_building_indices;
                    }
                    if (PyArray_NDIM(tmp_arr) == 0) {
                        /*
                         * Need to raise an error here, since the
                         * DeprecationWarning before was not triggered.
                         * TODO: A `False` triggers a Deprecation *not* a
                         *       a FutureWarning.
                         */
                        PyErr_SetString(PyExc_IndexError,
                                "in the future, 0-d boolean arrays will be "
                                "interpreted as a valid boolean index");
                        Py_DECREF(tmp_arr);
                        goto failed_building_indices;
                    }
                    else {
                        arr = tmp_arr;
                    }
                }
                /*
                 * Note: Down the road, the integers will be cast to intp.
                 *       The user has to make sure they can be safely cast.
                 *       If not, we might index wrong instead of an giving
                 *       an error.
                 */
                else if (!PyArray_ISINTEGER(tmp_arr)) {
                    if (PyArray_NDIM(tmp_arr) == 0) {
                        /* match integer deprecation warning */
                        /* 2013-09-25, 1.8 */
                        if (PyErr_WarnEx(
                                VisibleDeprecation,
                                "using a non-integer number instead of an "
                                "integer will result in an error in the "
                                "future",
                                1) < 0) {

                            /* The error message raised in the future */
                            PyErr_SetString(PyExc_IndexError,
                                "only integers, slices (`:`), ellipsis (`...`), "
                                "numpy.newaxis (`None`) and integer or boolean "
                                "arrays are valid indices");
                            Py_DECREF((PyObject *)tmp_arr);
                            goto failed_building_indices;
                        }
                    }
                    else {
                        /* 2013-09-25, 1.8 */
                        if (PyErr_WarnEx(
                                VisibleDeprecation,
                                "non integer (and non boolean) array-likes "
                                "will not be accepted as indices in the "
                                "future",
                                1) < 0) {

                            /* Error message to be raised in the future */
                            PyErr_SetString(PyExc_IndexError,
                                "non integer (and non boolean) array-likes will "
                                "not be accepted as indices in the future");
                            Py_DECREF((PyObject *)tmp_arr);
                            goto failed_building_indices;
                        }
                    }
                }

                arr = (PyArrayObject *)PyArray_FromArray(tmp_arr,
                                            PyArray_DescrFromType(NPY_INTP),
                                            NPY_ARRAY_FORCECAST);

                if (arr == NULL) {
                    /* Since this will be removed, handle this later */
                    PyErr_Clear();
                    arr = tmp_arr;
                }
                else {
                    Py_DECREF((PyObject *)tmp_arr);
                }
            }
        }
        else {
            Py_INCREF(obj);
            arr = (PyArrayObject *)obj;
        }

        /* Check if the array is valid and fill the information */
        if (PyArray_ISBOOL(arr)) {
            /*
             * There are two types of boolean indices (which are equivalent,
             * for the most part though). A single boolean index of matching
             * dimensionality and size is a boolean index.
             * If this is not the case, it is instead expanded into (multiple)
             * integer array indices.
             */
            PyArrayObject *nonzero_result[NPY_MAXDIMS];

            if ((index_ndim == 1) && allow_boolean) {
                /*
                 * If ndim and size match, this can be optimized as a single
                 * boolean index. The size check is necessary only to support
                 * old non-matching sizes by using fancy indexing instead.
                 * The reason for that is that fancy indexing uses nonzero,
                 * and only the result of nonzero is checked for legality.
                 */
                if ((PyArray_NDIM(arr) == PyArray_NDIM(self))
                        && PyArray_SIZE(arr) == PyArray_SIZE(self)) {

                    index_type = HAS_BOOL;
                    indices[curr_idx].type = HAS_BOOL;
                    indices[curr_idx].object = (PyObject *)arr;

                    /* keep track anyway, just to be complete */
                    used_ndim = PyArray_NDIM(self);
                    fancy_ndim = PyArray_NDIM(self);
                    curr_idx += 1;
                    break;
                }
            }

            if (PyArray_NDIM(arr) == 0) {
                /*
                 * TODO, WARNING: This code block cannot be used due to
                 *                FutureWarnings at this time. So instead
                 *                just raise an IndexError.
                 */
                PyErr_SetString(PyExc_IndexError,
                        "in the future, 0-d boolean arrays will be "
                        "interpreted as a valid boolean index");
                Py_DECREF((PyObject *)arr);
                goto failed_building_indices;
                /*
                 * This can actually be well defined. A new axis is added,
                 * but at the same time no axis is "used". So if we have True,
                 * we add a new axis (a bit like with np.newaxis). If it is
                 * False, we add a new axis, but this axis has 0 entries.
                 */

                index_type |= HAS_FANCY;
                indices[curr_idx].type = HAS_0D_BOOL;
                indices[curr_idx].value = 1;

                /* TODO: This can't fail, right? Is there a faster way? */
                if (PyObject_IsTrue((PyObject *)arr)) {
                    n = 1;
                }
                else {
                    n = 0;
                }
                indices[curr_idx].object = PyArray_Zeros(1, &n,
                                            PyArray_DescrFromType(NPY_INTP), 0);
                Py_DECREF(arr);

                if (indices[curr_idx].object == NULL) {
                    goto failed_building_indices;
                }

                used_ndim += 0;
                if (fancy_ndim < 1) {
                    fancy_ndim = 1;
                }
                curr_idx += 1;
                continue;
            }

            /* Convert the boolean array into multiple integer ones */
            n = _nonzero_indices((PyObject *)arr, nonzero_result);
            Py_DECREF(arr);

            if (n < 0) {
                goto failed_building_indices;
            }

            /* Check that we will not run out of indices to store new ones */
            if (curr_idx + n >= NPY_MAXDIMS * 2) {
                PyErr_SetString(PyExc_IndexError,
                                "too many indices for array");
                for (i=0; i < n; i++) {
                    Py_DECREF(nonzero_result[i]);
                }
                goto failed_building_indices;
            }

            /* Add the arrays from the nonzero result to the index */
            index_type |= HAS_FANCY;
            for (i=0; i < n; i++) {
                indices[curr_idx].type = HAS_FANCY;
                indices[curr_idx].value = PyArray_DIM(arr, i);
                indices[curr_idx].object = (PyObject *)nonzero_result[i];

                used_ndim += 1;
                curr_idx += 1;
            }

            /* All added indices have 1 dimension */
            if (fancy_ndim < 1) {
                fancy_ndim = 1;
            }
            continue;
        }

        /* Normal case of an integer array */
        else if (PyArray_ISINTEGER(arr)) {
            if (PyArray_NDIM(arr) == 0) {
                /*
                 * A 0-d integer array is an array scalar and can
                 * be dealt with the HAS_SCALAR_ARRAY flag.
                 * We could handle 0-d arrays early on, but this makes
                 * sure that array-likes or odder arrays are always
                 * handled right.
                 */
                npy_intp ind = PyArray_PyIntAsIntp((PyObject *)arr);

                Py_DECREF(arr);
                if ((ind == -1) && PyErr_Occurred()) {
                    goto failed_building_indices;
                }
                else {
                    index_type |= (HAS_INTEGER | HAS_SCALAR_ARRAY);
                    indices[curr_idx].object = NULL;
                    indices[curr_idx].value = ind;
                    indices[curr_idx].type = HAS_INTEGER;
                    used_ndim += 1;
                    new_ndim += 0;
                    curr_idx += 1;
                    continue;
                }
            }

            index_type |= HAS_FANCY;
            indices[curr_idx].type = HAS_FANCY;
            indices[curr_idx].value = -1;
            indices[curr_idx].object = (PyObject *)arr;

            used_ndim += 1;
            if (fancy_ndim < PyArray_NDIM(arr)) {
                fancy_ndim = PyArray_NDIM(arr);
            }
            curr_idx += 1;
            continue;
        }

        /*
         * The array does not have a valid type.
         */
        if ((PyObject *)arr == obj) {
            /* The input was an array already */
            PyErr_SetString(PyExc_IndexError,
                "arrays used as indices must be of integer (or boolean) type");
        }
        else {
            /* The input was not an array, so give a general error message */
            PyErr_SetString(PyExc_IndexError,
                    "only integers, slices (`:`), ellipsis (`...`), "
                    "numpy.newaxis (`None`) and integer or boolean "
                    "arrays are valid indices");
        }
        Py_DECREF(arr);
        goto failed_building_indices;
    }

    /*
     * Compare dimension of the index to the real ndim. this is
     * to find the ellipsis value or append an ellipsis if necessary.
     */
    if (used_ndim < PyArray_NDIM(self)) {
       if (index_type & HAS_ELLIPSIS) {
           indices[ellipsis_pos].value = PyArray_NDIM(self) - used_ndim;
           used_ndim = PyArray_NDIM(self);
           new_ndim += indices[ellipsis_pos].value;
       }
       else {
           /*
            * There is no ellipsis yet, but it is not a full index
            * so we append an ellipsis to the end.
            */
           index_type |= HAS_ELLIPSIS;
           indices[curr_idx].object = NULL;
           indices[curr_idx].type = HAS_ELLIPSIS;
           indices[curr_idx].value = PyArray_NDIM(self) - used_ndim;
           ellipsis_pos = curr_idx;

           used_ndim = PyArray_NDIM(self);
           new_ndim += indices[curr_idx].value;
           curr_idx += 1;
       }
    }
    else if (used_ndim > PyArray_NDIM(self)) {
        PyErr_SetString(PyExc_IndexError,
                        "too many indices for array");
        goto failed_building_indices;
    }
    else if (index_ndim == 0) {
        /*
         * 0-d index into 0-d array, i.e. array[()]
         * We consider this an integer index. Which means it will return
         * the scalar.
         * This makes sense, because then array[...] gives
         * an array and array[()] gives the scalar.
         */
        used_ndim = 0;
        index_type = HAS_INTEGER;
    }

    /* HAS_SCALAR_ARRAY requires cleaning up the index_type */
    if (index_type & HAS_SCALAR_ARRAY) {
        /* clear as info is unnecessary and makes life harder later */
        if (index_type & HAS_FANCY) {
            index_type -= HAS_SCALAR_ARRAY;
        }
        /* A full integer index sees array scalars as part of itself */
        else if (index_type == (HAS_INTEGER | HAS_SCALAR_ARRAY)) {
            index_type -= HAS_SCALAR_ARRAY;
        }
    }

    /*
     * At this point indices are all set correctly, no bounds checking
     * has been made and the new array may still have more dimensions
     * than is possible and boolean indexing arrays may have an incorrect shape.
     *
     * Check this now so we do not have to worry about it later.
     * It can happen for fancy indexing or with newaxis.
     * This means broadcasting errors in the case of too many dimensions
     * take less priority.
     */
    if (index_type & (HAS_NEWAXIS | HAS_FANCY)) {
        if (new_ndim + fancy_ndim > NPY_MAXDIMS) {
            PyErr_Format(PyExc_IndexError,
                         "number of dimensions must be within [0, %d], "
                         "indexing result would have %d",
                         NPY_MAXDIMS, (new_ndim + fancy_ndim));
            goto failed_building_indices;
        }

        /*
         * If we had a fancy index, we may have had a boolean array index.
         * So check if this had the correct shape now that we can find out
         * which axes it acts on.
         */
        used_ndim = 0;
        for (i = 0; i < curr_idx; i++) {
            if ((indices[i].type == HAS_FANCY) && indices[i].value > 0) {
                if (indices[i].value != PyArray_DIM(self, used_ndim)) {
                    static PyObject *warning = NULL;

                    char err_msg[174];
                    PyOS_snprintf(err_msg, sizeof(err_msg),
                        "boolean index did not match indexed array along "
                        "dimension %d; dimension is %" NPY_INTP_FMT
                        " but corresponding boolean dimension is %" NPY_INTP_FMT,
                        used_ndim, PyArray_DIM(self, used_ndim),
                        indices[i].value);

                    npy_cache_import(
                        "numpy", "VisibleDeprecationWarning", &warning);
                    if (warning == NULL) {
                        goto failed_building_indices;
                    }

                    if (PyErr_WarnEx(warning, err_msg, 1) < 0) {
                        goto failed_building_indices;
                    }
                    break;
                }
            }

            if (indices[i].type == HAS_ELLIPSIS) {
                used_ndim += indices[i].value;
            }
            else if ((indices[i].type == HAS_NEWAXIS) ||
                     (indices[i].type == HAS_0D_BOOL)) {
                used_ndim += 0;
            }
            else {
                used_ndim += 1;
            }
        }
    }

    *num = curr_idx;
    *ndim = new_ndim + fancy_ndim;
    *out_fancy_ndim = fancy_ndim;

    if (make_tuple) {
        Py_DECREF(index);
    }

    return index_type;

  failed_building_indices:
    for (i=0; i < curr_idx; i++) {
        Py_XDECREF(indices[i].object);
    }
  fail:
    if (make_tuple) {
        Py_DECREF(index);
    }
    return -1;
}


/**
 * Get pointer for an integer index.
 *
 * For a purely integer index, set ptr to the memory address.
 * Returns 0 on success, -1 on failure.
 * The caller must ensure that the index is a full integer
 * one.
 *
 * @param Array being indexed
 * @param result pointer
 * @param parsed index information
 * @param number of indices
 *
 * @return 0 on success -1 on failure
 */
static int
get_item_pointer(PyArrayObject *self, char **ptr,
                    npy_index_info *indices, int index_num) {
    int i;
    *ptr = PyArray_BYTES(self);
    for (i=0; i < index_num; i++) {
        if ((check_and_adjust_index(&(indices[i].value),
                               PyArray_DIMS(self)[i], i, NULL)) < 0) {
            return -1;
        }
        *ptr += PyArray_STRIDE(self, i) * indices[i].value;
    }
    return 0;
}


/**
 * Get view into an array using all non-array indices.
 *
 * For any index, get a view of the subspace into the original
 * array. If there are no fancy indices, this is the result of
 * the indexing operation.
 * Ensure_array allows to fetch a safe subspace view for advanced
 * indexing.
 *
 * @param Array being indexed
 * @param resulting array (new reference)
 * @param parsed index information
 * @param number of indices
 * @param Whether result should inherit the type from self
 *
 * @return 0 on success -1 on failure
 */
static int
get_view_from_index(PyArrayObject *self, PyArrayObject **view,
                    npy_index_info *indices, int index_num, int ensure_array) {
    npy_intp new_strides[NPY_MAXDIMS];
    npy_intp new_shape[NPY_MAXDIMS];
    int i, j;
    int new_dim = 0;
    int orig_dim = 0;
    char *data_ptr = PyArray_BYTES(self);

    /* for slice parsing */
    npy_intp start, stop, step, n_steps;

    for (i=0; i < index_num; i++) {
        switch (indices[i].type) {
            case HAS_INTEGER:
                if ((check_and_adjust_index(&indices[i].value,
                                PyArray_DIMS(self)[orig_dim], orig_dim,
                                NULL)) < 0) {
                    return -1;
                }
                data_ptr += PyArray_STRIDE(self, orig_dim) * indices[i].value;

                new_dim += 0;
                orig_dim += 1;
                break;
            case HAS_ELLIPSIS:
                for (j=0; j < indices[i].value; j++) {
                    new_strides[new_dim] = PyArray_STRIDE(self, orig_dim);
                    new_shape[new_dim] = PyArray_DIMS(self)[orig_dim];
                    new_dim += 1;
                    orig_dim += 1;
                }
                break;
            case HAS_SLICE:
                if (slice_GetIndices((PySliceObject *)indices[i].object,
                                     PyArray_DIMS(self)[orig_dim],
                                     &start, &stop, &step, &n_steps) < 0) {
                    if (!PyErr_Occurred()) {
                        PyErr_SetString(PyExc_IndexError, "invalid slice");
                    }
                    return -1;
                }
                if (n_steps <= 0) {
                    /* TODO: Always points to start then, could change that */
                    n_steps = 0;
                    step = 1;
                    start = 0;
                }

                data_ptr += PyArray_STRIDE(self, orig_dim) * start;
                new_strides[new_dim] = PyArray_STRIDE(self, orig_dim) * step;
                new_shape[new_dim] = n_steps;
                new_dim += 1;
                orig_dim += 1;
                break;
            case HAS_NEWAXIS:
                new_strides[new_dim] = 0;
                new_shape[new_dim] = 1;
                new_dim += 1;
                break;
            /* Fancy and 0-d boolean indices are ignored here */
            case HAS_0D_BOOL:
                break;
            default:
                new_dim += 0;
                orig_dim += 1;
                break;
        }
    }

    /* Create the new view and set the base array */
    Py_INCREF(PyArray_DESCR(self));
    *view = (PyArrayObject *)PyArray_NewFromDescr(
                                ensure_array ? &PyArray_Type : Py_TYPE(self),
                                PyArray_DESCR(self),
                                new_dim, new_shape,
                                new_strides, data_ptr,
                                PyArray_FLAGS(self),
                                ensure_array ? NULL : (PyObject *)self);
    if (*view == NULL) {
        return -1;
    }

    Py_INCREF(self);
    if (PyArray_SetBaseObject(*view, (PyObject *)self) < 0) {
        Py_DECREF(*view);
        return -1;
    }

    return 0;
}


/*
 * Implements boolean indexing. This produces a one-dimensional
 * array which picks out all of the elements of 'self' for which
 * the corresponding element of 'op' is True.
 *
 * This operation is somewhat unfortunate, because to produce
 * a one-dimensional output array, it has to choose a particular
 * iteration order, in the case of NumPy that is always C order even
 * though this function allows different choices.
 */
NPY_NO_EXPORT PyArrayObject *
array_boolean_subscript(PyArrayObject *self,
                        PyArrayObject *bmask, NPY_ORDER order)
{
    npy_intp size, itemsize;
    char *ret_data;
    PyArray_Descr *dtype;
    PyArrayObject *ret;
    int needs_api = 0;

    size = count_boolean_trues(PyArray_NDIM(bmask), PyArray_DATA(bmask),
                                PyArray_DIMS(bmask), PyArray_STRIDES(bmask));

    /* Allocate the output of the boolean indexing */
    dtype = PyArray_DESCR(self);
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype, 1, &size,
                                NULL, NULL, 0, NULL);
    if (ret == NULL) {
        return NULL;
    }

    itemsize = dtype->elsize;
    ret_data = PyArray_DATA(ret);

    /* Create an iterator for the data */
    if (size > 0) {
        NpyIter *iter;
        PyArrayObject *op[2] = {self, bmask};
        npy_uint32 flags, op_flags[2];
        npy_intp fixed_strides[3];
        PyArray_StridedUnaryOp *stransfer = NULL;
        NpyAuxData *transferdata = NULL;

        NpyIter_IterNextFunc *iternext;
        npy_intp innersize, *innerstrides;
        char **dataptrs;

        npy_intp self_stride, bmask_stride, subloopsize;
        char *self_data;
        char *bmask_data;
        NPY_BEGIN_THREADS_DEF;

        /* Set up the iterator */
        flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK;
        op_flags[0] = NPY_ITER_READONLY | NPY_ITER_NO_BROADCAST;
        op_flags[1] = NPY_ITER_READONLY;

        iter = NpyIter_MultiNew(2, op, flags, order, NPY_NO_CASTING,
                                op_flags, NULL);
        if (iter == NULL) {
            Py_DECREF(ret);
            return NULL;
        }

        /* Get a dtype transfer function */
        NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
        if (PyArray_GetDTypeTransferFunction(PyArray_ISALIGNED(self),
                        fixed_strides[0], itemsize,
                        dtype, dtype,
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
            Py_DECREF(ret);
            NpyIter_Deallocate(iter);
            return NULL;
        }

        /* Get the values needed for the inner loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            Py_DECREF(ret);
            NpyIter_Deallocate(iter);
            NPY_AUXDATA_FREE(transferdata);
            return NULL;
        }

        NPY_BEGIN_THREADS_NDITER(iter);

        innerstrides = NpyIter_GetInnerStrideArray(iter);
        dataptrs = NpyIter_GetDataPtrArray(iter);

        self_stride = innerstrides[0];
        bmask_stride = innerstrides[1];
        do {
            innersize = *NpyIter_GetInnerLoopSizePtr(iter);
            self_data = dataptrs[0];
            bmask_data = dataptrs[1];

            while (innersize > 0) {
                /* Skip masked values */
                bmask_data = npy_memchr(bmask_data, 0, bmask_stride,
                                        innersize, &subloopsize, 1);
                innersize -= subloopsize;
                self_data += subloopsize * self_stride;
                /* Process unmasked values */
                bmask_data = npy_memchr(bmask_data, 0, bmask_stride, innersize,
                                        &subloopsize, 0);
                stransfer(ret_data, itemsize, self_data, self_stride,
                            subloopsize, itemsize, transferdata);
                innersize -= subloopsize;
                self_data += subloopsize * self_stride;
                ret_data += subloopsize * itemsize;
            }
        } while (iternext(iter));

        NPY_END_THREADS;

        NpyIter_Deallocate(iter);
        NPY_AUXDATA_FREE(transferdata);
    }

    if (!PyArray_CheckExact(self)) {
        PyArrayObject *tmp = ret;

        Py_INCREF(dtype);
        ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self), dtype, 1,
                            &size, PyArray_STRIDES(ret), PyArray_BYTES(ret),
                            PyArray_FLAGS(self), (PyObject *)self);

        if (ret == NULL) {
            Py_DECREF(tmp);
            return NULL;
        }

        if (PyArray_SetBaseObject(ret, (PyObject *)tmp) < 0) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    return ret;
}

/*
 * Implements boolean indexing assignment. This takes the one-dimensional
 * array 'v' and assigns its values to all of the elements of 'self' for which
 * the corresponding element of 'op' is True.
 *
 * This operation is somewhat unfortunate, because to match up with
 * a one-dimensional output array, it has to choose a particular
 * iteration order, in the case of NumPy that is always C order even
 * though this function allows different choices.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_boolean_subscript(PyArrayObject *self,
                    PyArrayObject *bmask, PyArrayObject *v, NPY_ORDER order)
{
    npy_intp size, src_itemsize, v_stride;
    char *v_data;
    int needs_api = 0;
    npy_intp bmask_size;

    if (PyArray_DESCR(bmask)->type_num != NPY_BOOL) {
        PyErr_SetString(PyExc_TypeError,
                "NumPy boolean array indexing assignment "
                "requires a boolean index");
        return -1;
    }

    if (PyArray_NDIM(v) > 1) {
        PyErr_Format(PyExc_TypeError,
                "NumPy boolean array indexing assignment "
                "requires a 0 or 1-dimensional input, input "
                "has %d dimensions", PyArray_NDIM(v));
        return -1;
    }

    if (PyArray_NDIM(bmask) != PyArray_NDIM(self)) {
        PyErr_SetString(PyExc_ValueError,
                "The boolean mask assignment indexing array "
                "must have the same number of dimensions as "
                "the array being indexed");
        return -1;
    }

    size = count_boolean_trues(PyArray_NDIM(bmask), PyArray_DATA(bmask),
                                PyArray_DIMS(bmask), PyArray_STRIDES(bmask));
    /* Correction factor for broadcasting 'bmask' to 'self' */
    bmask_size = PyArray_SIZE(bmask);
    if (bmask_size > 0) {
        size *= PyArray_SIZE(self) / bmask_size;
    }

    /* Tweak the strides for 0-dim and broadcasting cases */
    if (PyArray_NDIM(v) > 0 && PyArray_DIMS(v)[0] != 1) {
        if (size != PyArray_DIMS(v)[0]) {
            PyErr_Format(PyExc_ValueError,
                    "NumPy boolean array indexing assignment "
                    "cannot assign %d input values to "
                    "the %d output values where the mask is true",
                    (int)PyArray_DIMS(v)[0], (int)size);
            return -1;
        }
        v_stride = PyArray_STRIDES(v)[0];
    }
    else {
        v_stride = 0;
    }

    src_itemsize = PyArray_DESCR(v)->elsize;
    v_data = PyArray_DATA(v);

    /* Create an iterator for the data */
    if (size > 0) {
        NpyIter *iter;
        PyArrayObject *op[2] = {self, bmask};
        npy_uint32 flags, op_flags[2];
        npy_intp fixed_strides[3];

        NpyIter_IterNextFunc *iternext;
        npy_intp innersize, *innerstrides;
        char **dataptrs;

        PyArray_StridedUnaryOp *stransfer = NULL;
        NpyAuxData *transferdata = NULL;
        npy_intp self_stride, bmask_stride, subloopsize;
        char *self_data;
        char *bmask_data;
        NPY_BEGIN_THREADS_DEF;

        /* Set up the iterator */
        flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK;
        op_flags[0] = NPY_ITER_WRITEONLY | NPY_ITER_NO_BROADCAST;
        op_flags[1] = NPY_ITER_READONLY;

        iter = NpyIter_MultiNew(2, op, flags, order, NPY_NO_CASTING,
                                op_flags, NULL);
        if (iter == NULL) {
            return -1;
        }

        /* Get the values needed for the inner loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }

        innerstrides = NpyIter_GetInnerStrideArray(iter);
        dataptrs = NpyIter_GetDataPtrArray(iter);

        self_stride = innerstrides[0];
        bmask_stride = innerstrides[1];

        /* Get a dtype transfer function */
        NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
        if (PyArray_GetDTypeTransferFunction(
                        PyArray_ISALIGNED(self) && PyArray_ISALIGNED(v),
                        v_stride, fixed_strides[0],
                        PyArray_DESCR(v), PyArray_DESCR(self),
                        0,
                        &stransfer, &transferdata,
                        &needs_api) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            return -1;
        }

        if (!needs_api) {
            NPY_BEGIN_THREADS_NDITER(iter);
        }

        do {
            innersize = *NpyIter_GetInnerLoopSizePtr(iter);
            self_data = dataptrs[0];
            bmask_data = dataptrs[1];

            while (innersize > 0) {
                /* Skip masked values */
                bmask_data = npy_memchr(bmask_data, 0, bmask_stride,
                                        innersize, &subloopsize, 1);
                innersize -= subloopsize;
                self_data += subloopsize * self_stride;
                /* Process unmasked values */
                bmask_data = npy_memchr(bmask_data, 0, bmask_stride, innersize,
                                        &subloopsize, 0);
                stransfer(self_data, self_stride, v_data, v_stride,
                            subloopsize, src_itemsize, transferdata);
                innersize -= subloopsize;
                self_data += subloopsize * self_stride;
                v_data += subloopsize * v_stride;
            }
        } while (iternext(iter));

        if (!needs_api) {
            NPY_END_THREADS;
        }

        NPY_AUXDATA_FREE(transferdata);
        NpyIter_Deallocate(iter);
    }

    if (needs_api) {
        /*
         * FIXME?: most assignment operations stop after the first occurrence
         * of an error. Boolean does not currently, but should at least
         * report the error. (This is only relevant for things like str->int
         * casts which call into python)
         */
        if (PyErr_Occurred()) {
            return -1;
        }
    }

    return 0;
}


/*
 * C-level integer indexing always returning an array and never a scalar.
 * Works also for subclasses, but it will not be called on one from the
 * Python API.
 *
 * This function does not accept negative indices because it is called by
 * PySequence_GetItem (through array_item) and that converts them to
 * positive indices.
 */
NPY_NO_EXPORT PyObject *
array_item_asarray(PyArrayObject *self, npy_intp i)
{
    npy_index_info indices[2];
    PyObject *result;

    if (PyArray_NDIM(self) == 0) {
        PyErr_SetString(PyExc_IndexError,
                        "too many indices for array");
        return NULL;
    }
    if (i < 0) {
        /* This is an error, but undo PySequence_GetItem fix for message */
        i -= PyArray_DIM(self, 0);
    }

    indices[0].value = i;
    indices[0].type = HAS_INTEGER;
    indices[1].value = PyArray_NDIM(self) - 1;
    indices[1].type = HAS_ELLIPSIS;
    if (get_view_from_index(self, (PyArrayObject **)&result,
                            indices, 2, 0) < 0) {
        return NULL;
    }
    return result;
}


/*
 * Python C-Api level item subscription (implementation for PySequence_GetItem)
 *
 * Negative indices are not accepted because PySequence_GetItem converts
 * them to positive indices before calling this.
 */
NPY_NO_EXPORT PyObject *
array_item(PyArrayObject *self, Py_ssize_t i)
{
    if (PyArray_NDIM(self) == 1) {
        char *item;
        npy_index_info index;

        if (i < 0) {
            /* This is an error, but undo PySequence_GetItem fix for message */
            i -= PyArray_DIM(self, 0);
        }

        index.value = i;
        index.type = HAS_INTEGER;
        if (get_item_pointer(self, &item, &index, 1) < 0) {
            return NULL;
        }
        return PyArray_Scalar(item, PyArray_DESCR(self), (PyObject *)self);
    }
    else {
        return array_item_asarray(self, i);
    }
}


/* make sure subscript always returns an array object */
NPY_NO_EXPORT PyObject *
array_subscript_asarray(PyArrayObject *self, PyObject *op)
{
    return PyArray_EnsureAnyArray(array_subscript(self, op));
}

/*
 * Attempts to subscript an array using a field name or list of field names.
 *
 * If an error occurred, return 0 and set view to NULL. If the subscript is not
 * a string or list of strings, return -1 and set view to NULL. Otherwise
 * return 0 and set view to point to a new view into arr for the given fields.
 */
NPY_NO_EXPORT int
_get_field_view(PyArrayObject *arr, PyObject *ind, PyArrayObject **view)
{
    *view = NULL;

    /* first check for a single field name */
#if defined(NPY_PY3K)
    if (PyUnicode_Check(ind)) {
#else
    if (PyString_Check(ind) || PyUnicode_Check(ind)) {
#endif
        PyObject *tup;
        PyArray_Descr *fieldtype;
        npy_intp offset;

        /* get the field offset and dtype */
        tup = PyDict_GetItem(PyArray_DESCR(arr)->fields, ind);
        if (tup == NULL){
            PyObject *errmsg = PyUString_FromString("no field of name ");
            PyUString_Concat(&errmsg, ind);
            PyErr_SetObject(PyExc_ValueError, errmsg);
            Py_DECREF(errmsg);
            return 0;
        }
        if (_unpack_field(tup, &fieldtype, &offset) < 0) {
            return 0;
        }

        /* view the array at the new offset+dtype */
        Py_INCREF(fieldtype);
        *view = (PyArrayObject*)PyArray_NewFromDescr(
                                    Py_TYPE(arr),
                                    fieldtype,
                                    PyArray_NDIM(arr),
                                    PyArray_SHAPE(arr),
                                    PyArray_STRIDES(arr),
                                    PyArray_BYTES(arr) + offset,
                                    PyArray_FLAGS(arr),
                                    (PyObject *)arr);
        if (*view == NULL) {
            return 0;
        }
        Py_INCREF(arr);
        if (PyArray_SetBaseObject(*view, (PyObject *)arr) < 0) {
            Py_DECREF(*view);
            *view = NULL;
        }
        return 0;
    }
    /* next check for a list of field names */
    else if (PySequence_Check(ind) && !PyTuple_Check(ind)) {
        int seqlen, i;
        PyObject *name = NULL, *tup;
        PyObject *fields, *names;
        PyArray_Descr *view_dtype;

        /* variables needed to make a copy, to remove in the future */
        static PyObject *copyfunc = NULL;
        PyObject *viewcopy;

        seqlen = PySequence_Size(ind);

        /* quit if have a 0-d array (seqlen==-1) or a 0-len array */
        if (seqlen == -1) {
            PyErr_Clear();
            return -1;
        }
        if (seqlen == 0) {
            return -1;
        }

        fields = PyDict_New();
        if (fields == NULL) {
            return 0;
        }
        names = PyTuple_New(seqlen);
        if (names == NULL) {
            Py_DECREF(fields);
            return 0;
        }

        for (i = 0; i < seqlen; i++) {
            name = PySequence_GetItem(ind, i);
            if (name == NULL) {
                /* only happens for strange sequence objects */
                PyErr_Clear();
                Py_DECREF(fields);
                Py_DECREF(names);
                return -1;
            }

#if defined(NPY_PY3K)
            if (!PyUnicode_Check(name)) {
#else
            if (!PyString_Check(name) && !PyUnicode_Check(name)) {
#endif
                Py_DECREF(name);
                Py_DECREF(fields);
                Py_DECREF(names);
                return -1;
            }

            tup = PyDict_GetItem(PyArray_DESCR(arr)->fields, name);
            if (tup == NULL){
                PyObject *errmsg = PyUString_FromString("no field of name ");
                PyUString_ConcatAndDel(&errmsg, name);
                PyErr_SetObject(PyExc_ValueError, errmsg);
                Py_DECREF(errmsg);
                Py_DECREF(fields);
                Py_DECREF(names);
                return 0;
            }
            if (PyDict_SetItem(fields, name, tup) < 0) {
                Py_DECREF(name);
                Py_DECREF(fields);
                Py_DECREF(names);
                return 0;
            }
            if (PyTuple_SetItem(names, i, name) < 0) {
                Py_DECREF(fields);
                Py_DECREF(names);
                return 0;
            }
        }

        view_dtype = PyArray_DescrNewFromType(NPY_VOID);
        if (view_dtype == NULL) {
            Py_DECREF(fields);
            Py_DECREF(names);
            return 0;
        }
        view_dtype->elsize = PyArray_DESCR(arr)->elsize;
        view_dtype->names = names;
        view_dtype->fields = fields;
        view_dtype->flags = PyArray_DESCR(arr)->flags;

        *view = (PyArrayObject*)PyArray_NewFromDescr(
                                    Py_TYPE(arr),
                                    view_dtype,
                                    PyArray_NDIM(arr),
                                    PyArray_SHAPE(arr),
                                    PyArray_STRIDES(arr),
                                    PyArray_DATA(arr),
                                    PyArray_FLAGS(arr),
                                    (PyObject *)arr);
        if (*view == NULL) {
            return 0;
        }
        Py_INCREF(arr);
        if (PyArray_SetBaseObject(*view, (PyObject *)arr) < 0) {
            Py_DECREF(*view);
            *view = NULL;
            return 0;
        }

        /*
         * Return copy for now (future plan to return the view above). All the
         * following code in this block can then be replaced by "return 0;"
         */
        npy_cache_import("numpy.core._internal", "_copy_fields", &copyfunc);
        if (copyfunc == NULL) {
            Py_DECREF(*view);
            *view = NULL;
            return 0;
        }

        viewcopy = PyObject_CallFunction(copyfunc, "O", *view);
        if (viewcopy == NULL) {
            Py_DECREF(*view);
            *view = NULL;
            return 0;
        }
        Py_DECREF(*view);
        *view = (PyArrayObject*)viewcopy;
        return 0;
    }
    return -1;
}

/*
 * General function for indexing a NumPy array with a Python object.
 */
NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op)
{
    int index_type;
    int index_num;
    int i, ndim, fancy_ndim;
    /*
     * Index info array. We can have twice as many indices as dimensions
     * (because of None). The + 1 is to not need to check as much.
     */
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];

    PyArrayObject *view = NULL;
    PyObject *result = NULL;

    PyArrayMapIterObject * mit = NULL;

    /* return fields if op is a string index */
    if (PyDataType_HASFIELDS(PyArray_DESCR(self))) {
        PyArrayObject *view;
        int ret = _get_field_view(self, op, &view);
        if (ret == 0){
            if (view == NULL) {
                return NULL;
            }

            /* warn if writing to a copy. copies will have no base */
            if (PyArray_BASE(view) == NULL) {
                PyArray_ENABLEFLAGS(view, NPY_ARRAY_WARN_ON_WRITE);
            }
            return (PyObject*)view;
        }
    }

    /* Prepare the indices */
    index_type = prepare_index(self, op, indices, &index_num,
                               &ndim, &fancy_ndim, 1);

    if (index_type < 0) {
        return NULL;
    }

    /* Full integer index */
    else if (index_type == HAS_INTEGER) {
        char *item;
        if (get_item_pointer(self, &item, indices, index_num) < 0) {
            goto finish;
        }
        result = (PyObject *) PyArray_Scalar(item, PyArray_DESCR(self),
                                             (PyObject *)self);
        /* Because the index is full integer, we do not need to decref */
        return result;
    }

    /* Single boolean array */
    else if (index_type == HAS_BOOL) {
        result = (PyObject *)array_boolean_subscript(self,
                                    (PyArrayObject *)indices[0].object,
                                    NPY_CORDER);
        goto finish;
    }

    /* If it is only a single ellipsis, just return a view */
    else if (index_type == HAS_ELLIPSIS) {
        /*
         * TODO: Should this be a view or not? The only reason not would be
         *       optimization (i.e. of array[...] += 1) I think.
         *       Before, it was just self for a single ellipsis.
         */
        result = PyArray_View(self, NULL, NULL);
        /* A single ellipsis, so no need to decref */
        return result;
    }

    /*
     * View based indexing.
     * There are two cases here. First we need to create a simple view,
     * second we need to create a (possibly invalid) view for the
     * subspace to the fancy index. This procedure is identical.
     */

    else if (index_type & (HAS_SLICE | HAS_NEWAXIS |
                           HAS_ELLIPSIS | HAS_INTEGER)) {
        if (get_view_from_index(self, &view, indices, index_num,
                                (index_type & HAS_FANCY)) < 0) {
            goto finish;
        }

        /*
         * There is a scalar array, so we need to force a copy to simulate
         * fancy indexing.
         */
        if (index_type & HAS_SCALAR_ARRAY) {
            result = PyArray_NewCopy(view, NPY_KEEPORDER);
            goto finish;
        }
    }

    /* If there is no fancy indexing, we have the result */
    if (!(index_type & HAS_FANCY)) {
        result = (PyObject *)view;
        Py_INCREF(result);
        goto finish;
    }

    /*
     * Special case for very simple 1-d fancy indexing, which however
     * is quite common. This saves not only a lot of setup time in the
     * iterator, but also is faster (must be exactly fancy because
     * we don't support 0-d booleans here)
     */
    if (index_type == HAS_FANCY && index_num == 1) {
        /* The array being indexed has one dimension and it is a fancy index */
        PyArrayObject *ind = (PyArrayObject*)indices[0].object;

        /* Check if the index is simple enough */
        if (PyArray_TRIVIALLY_ITERABLE(ind) &&
                /* Check if the type is equivalent to INTP */
                PyArray_ITEMSIZE(ind) == sizeof(npy_intp) &&
                PyArray_DESCR(ind)->kind == 'i' &&
                PyArray_ISALIGNED(ind) &&
                PyDataType_ISNOTSWAPPED(PyArray_DESCR(ind))) {

            Py_INCREF(PyArray_DESCR(self));
            result = PyArray_NewFromDescr(&PyArray_Type,
                                          PyArray_DESCR(self),
                                          PyArray_NDIM(ind),
                                          PyArray_SHAPE(ind),
                                          NULL, NULL,
                                          /* Same order as indices */
                                          PyArray_ISFORTRAN(ind) ?
                                              NPY_ARRAY_F_CONTIGUOUS : 0,
                                          NULL);
            if (result == NULL) {
                goto finish;
            }

            if (mapiter_trivial_get(self, ind, (PyArrayObject *)result) < 0) {
                Py_DECREF(result);
                result = NULL;
                goto finish;
            }

            goto wrap_out_array;
        }
    }

    /* fancy indexing has to be used. And view is the subspace. */
    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices, index_num,
                                                     index_type,
                                                     ndim, fancy_ndim,
                                                     self, view, 0,
                                                     NPY_ITER_READONLY,
                                                     NPY_ITER_WRITEONLY,
                                                     NULL, PyArray_DESCR(self));
    if (mit == NULL) {
        goto finish;
    }

    if (mit->numiter > 1) {
        /*
         * If it is one, the inner loop checks indices, otherwise
         * check indices beforehand, because it is much faster if
         * broadcasting occurs and most likely no big overhead
         */
        if (PyArray_MapIterCheckIndices(mit) < 0) {
            goto finish;
        }
    }

    /* Reset the outer iterator */
    if (NpyIter_Reset(mit->outer, NULL) < 0) {
        goto finish;
    }

    if (mapiter_get(mit) < 0) {
        goto finish;
    }

    result = (PyObject *)mit->extra_op;
    Py_INCREF(result);

    if (mit->consec) {
        PyArray_MapIterSwapAxes(mit, (PyArrayObject **)&result, 1);
    }

  wrap_out_array:
    if (!PyArray_CheckExact(self)) {
        /*
         * Need to create a new array as if the old one never existed.
         */
        PyArrayObject *tmp_arr = (PyArrayObject *)result;

        Py_INCREF(PyArray_DESCR(tmp_arr));
        result = PyArray_NewFromDescr(Py_TYPE(self),
                                      PyArray_DESCR(tmp_arr),
                                      PyArray_NDIM(tmp_arr),
                                      PyArray_SHAPE(tmp_arr),
                                      PyArray_STRIDES(tmp_arr),
                                      PyArray_BYTES(tmp_arr),
                                      PyArray_FLAGS(self),
                                      (PyObject *)self);

        if (result == NULL) {
            Py_DECREF(tmp_arr);
            goto finish;
        }

        if (PyArray_SetBaseObject((PyArrayObject *)result,
                                  (PyObject *)tmp_arr) < 0) {
            Py_DECREF(result);
            result = NULL;
            goto finish;
        }
    }

  finish:
    Py_XDECREF(mit);
    Py_XDECREF(view);
    /* Clean up indices */
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    return result;
}


/*
 * Python C-Api level item assignment (implementation for PySequence_SetItem)
 *
 * Negative indices are not accepted because PySequence_SetItem converts
 * them to positive indices before calling this.
 */
NPY_NO_EXPORT int
array_assign_item(PyArrayObject *self, Py_ssize_t i, PyObject *op)
{
    npy_index_info indices[2];

    if (op == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");
        return -1;
    }
    if (PyArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return -1;
    }
    if (PyArray_NDIM(self) == 0) {
        PyErr_SetString(PyExc_IndexError,
                        "too many indices for array");
        return -1;
    }

    if (i < 0) {
        /* This is an error, but undo PySequence_SetItem fix for message */
        i -= PyArray_DIM(self, 0);
    }

    indices[0].value = i;
    indices[0].type = HAS_INTEGER;
    if (PyArray_NDIM(self) == 1) {
        char *item;
        if (get_item_pointer(self, &item, indices, 1) < 0) {
            return -1;
        }
        if (PyArray_SETITEM(self, item, op) < 0) {
            return -1;
        }
    }
    else {
        PyArrayObject *view;

        indices[1].value = PyArray_NDIM(self) - 1;
        indices[1].type = HAS_ELLIPSIS;
        if (get_view_from_index(self, &view, indices, 2, 0) < 0) {
            return -1;
        }
        if (PyArray_CopyObject(view, op) < 0) {
            Py_DECREF(view);
            return -1;
        }
        Py_DECREF(view);
    }
    return 0;
}


/*
 * This fallback takes the old route of `arr.flat[index] = values`
 * for one dimensional `arr`. The route can sometimes fail slightly
 * differently (ValueError instead of IndexError), in which case we
 * warn users about the change. But since it does not actually care *at all*
 * about shapes, it should only fail for out of bound indexes or
 * casting errors.
 */
NPY_NO_EXPORT int
attempt_1d_fallback(PyArrayObject *self, PyObject *ind, PyObject *op)
{
    PyObject *err = PyErr_Occurred();
    PyArrayIterObject *self_iter = NULL;

    Py_INCREF(err);
    PyErr_Clear();

    self_iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
    if (self_iter == NULL) {
        goto fail;
    }
    if (iter_ass_subscript(self_iter, ind, op) < 0) {
        goto fail;
    }

    Py_XDECREF((PyObject *)self_iter);
    Py_DECREF(err);

    /* 2014-06-12, 1.9 */
    if (DEPRECATE(
            "assignment will raise an error in the future, most likely "
            "because your index result shape does not match the value array "
            "shape. You can use `arr.flat[index] = values` to keep the old "
            "behaviour.") < 0) {
        return -1;
    }
    return 0;

  fail:
    if (!PyErr_ExceptionMatches(err)) {
        PyObject *err, *val, *tb;
        PyErr_Fetch(&err, &val, &tb);
        /* 2014-06-12, 1.9 */
        DEPRECATE_FUTUREWARNING(
            "assignment exception type will change in the future");
        PyErr_Restore(err, val, tb);
    }

    Py_XDECREF((PyObject *)self_iter);
    Py_DECREF(err);
    return -1;
}


/*
 * General assignment with python indexing objects.
 */
static int
array_assign_subscript(PyArrayObject *self, PyObject *ind, PyObject *op)
{
    int index_type;
    int index_num;
    int i, ndim, fancy_ndim;
    PyArray_Descr *descr = PyArray_DESCR(self);
    PyArrayObject *view = NULL;
    PyArrayObject *tmp_arr = NULL;
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];

    PyArrayMapIterObject *mit = NULL;

    if (op == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");
        return -1;
    }
    if (PyArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return -1;
    }

    /* field access */
    if (PyDataType_HASFIELDS(PyArray_DESCR(self))){
        PyArrayObject *view;
        int ret = _get_field_view(self, ind, &view);
        if (ret == 0){

#if defined(NPY_PY3K)
            if (!PyUnicode_Check(ind)) {
#else
            if (!PyString_Check(ind) && !PyUnicode_Check(ind)) {
#endif
                PyErr_SetString(PyExc_ValueError,
                                "multi-field assignment is not supported");
                return -1;
            }

            if (view == NULL) {
                return -1;
            }
            if (PyArray_CopyObject(view, op) < 0) {
                Py_DECREF(view);
                return -1;
            }
            Py_DECREF(view);
            return 0;
        }
    }

    /* Prepare the indices */
    index_type = prepare_index(self, ind, indices, &index_num,
                               &ndim, &fancy_ndim, 1);

    if (index_type < 0) {
        return -1;
    }

    /* Full integer index */
    if (index_type == HAS_INTEGER) {
        char *item;
        if (get_item_pointer(self, &item, indices, index_num) < 0) {
            return -1;
        }
        if (PyArray_SETITEM(self, item, op) < 0) {
            return -1;
        }
        /* integers do not store objects in indices */
        return 0;
    }

    /* Single boolean array */
    if (index_type == HAS_BOOL) {
        if (!PyArray_Check(op)) {
            Py_INCREF(PyArray_DESCR(self));
            tmp_arr = (PyArrayObject *)PyArray_FromAny(op,
                                                   PyArray_DESCR(self), 0, 0,
                                                   NPY_ARRAY_FORCECAST, NULL);
            if (tmp_arr == NULL) {
                goto fail;
            }
        }
        else {
            Py_INCREF(op);
            tmp_arr = (PyArrayObject *)op;
        }

        if (array_assign_boolean_subscript(self,
                                           (PyArrayObject *)indices[0].object,
                                           tmp_arr, NPY_CORDER) < 0) {
            /*
             * Deprecated case. The old boolean indexing seemed to have some
             * check to allow wrong dimensional boolean arrays in all cases.
             */
            if (PyArray_NDIM(tmp_arr) > 1) {
                if (attempt_1d_fallback(self, indices[0].object,
                                        (PyObject*)tmp_arr) < 0) {
                    goto fail;
                }
                goto success;
            }
            goto fail;
        }
        goto success;
    }


    /*
     * Single ellipsis index, no need to create a new view.
     * Note that here, we do *not* go through self.__getitem__ for subclasses
     * (defchar array failed then, due to uninitialized values...)
     */
    else if (index_type == HAS_ELLIPSIS) {
        if ((PyObject *)self == op) {
            /*
             * CopyObject does not handle this case gracefully and
             * there is nothing to do. Removing the special case
             * will cause segfaults, though it is unclear what exactly
             * happens.
             */
            return 0;
        }
        /* we can just use self, but incref for error handling */
        Py_INCREF((PyObject *)self);
        view = self;
    }

    /*
     * WARNING: There is a huge special case here. If this is not a
     *          base class array, we have to get the view through its
     *          very own index machinery.
     *          Many subclasses should probably call __setitem__
     *          with a base class ndarray view to avoid this.
     */
    else if (!(index_type & (HAS_FANCY | HAS_SCALAR_ARRAY))
                && !PyArray_CheckExact(self)) {
        view = (PyArrayObject *)PyObject_GetItem((PyObject *)self, ind);
        if (view == NULL) {
            goto fail;
        }
        if (!PyArray_Check(view)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Getitem not returning array");
            goto fail;
        }
    }

    /*
     * View based indexing.
     * There are two cases here. First we need to create a simple view,
     * second we need to create a (possibly invalid) view for the
     * subspace to the fancy index. This procedure is identical.
     */
    else if (index_type & (HAS_SLICE | HAS_NEWAXIS |
                           HAS_ELLIPSIS | HAS_INTEGER)) {
        if (get_view_from_index(self, &view, indices, index_num,
                                (index_type & HAS_FANCY)) < 0) {
            goto fail;
        }
    }
    else {
        view = NULL;
    }

    /* If there is no fancy indexing, we have the array to assign to */
    if (!(index_type & HAS_FANCY)) {
        if (PyArray_CopyObject(view, op) < 0) {
            goto fail;
        }
        goto success;
    }

    if (!PyArray_Check(op)) {
        /*
         * If the array is of object converting the values to an array
         * might not be legal even though normal assignment works.
         * So allocate a temporary array of the right size and use the
         * normal assignment to handle this case.
         */
        if (PyDataType_REFCHK(descr) && PySequence_Check(op)) {
            tmp_arr = NULL;
        }
        else {
            /* There is nothing fancy possible, so just make an array */
            Py_INCREF(descr);
            tmp_arr = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0,
                                                    NPY_ARRAY_FORCECAST, NULL);
            if (tmp_arr == NULL) {
                goto fail;
            }
        }
    }
    else {
        Py_INCREF(op);
        tmp_arr = (PyArrayObject *)op;
    }

    /*
     * Special case for very simple 1-d fancy indexing, which however
     * is quite common. This saves not only a lot of setup time in the
     * iterator, but also is faster (must be exactly fancy because
     * we don't support 0-d booleans here)
     */
    if (index_type == HAS_FANCY &&
            index_num == 1 && tmp_arr) {
        /* The array being indexed has one dimension and it is a fancy index */
        PyArrayObject *ind = (PyArrayObject*)indices[0].object;

        /* Check if the type is equivalent */
        if (PyArray_EquivTypes(PyArray_DESCR(self),
                                   PyArray_DESCR(tmp_arr)) &&
                /*
                 * Either they are equivalent, or the values must
                 * be a scalar
                 */
                (PyArray_EQUIVALENTLY_ITERABLE(ind, tmp_arr) ||
                 (PyArray_NDIM(tmp_arr) == 0 &&
                        PyArray_TRIVIALLY_ITERABLE(tmp_arr))) &&
                /* Check if the type is equivalent to INTP */
                PyArray_ITEMSIZE(ind) == sizeof(npy_intp) &&
                PyArray_DESCR(ind)->kind == 'i' &&
                PyArray_ISALIGNED(ind) &&
                PyDataType_ISNOTSWAPPED(PyArray_DESCR(ind))) {

            /* trivial_set checks the index for us */
            if (mapiter_trivial_set(self, ind, tmp_arr) < 0) {
                goto fail;
            }
            goto success;
        }
    }

    /*
     * NOTE: If tmp_arr was not allocated yet, mit should
     *       handle the allocation.
     *       The NPY_ITER_READWRITE is necessary for automatic
     *       allocation. Readwrite would not allow broadcasting
     *       correctly, but such an operand always has the full
     *       size anyway.
     */
    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices,
                                             index_num, index_type,
                                             ndim, fancy_ndim, self,
                                             view, 0,
                                             NPY_ITER_WRITEONLY,
                                             ((tmp_arr == NULL) ?
                                                  NPY_ITER_READWRITE :
                                                  NPY_ITER_READONLY),
                                             tmp_arr, descr);

    if (mit == NULL) {
        /*
         * This is a deprecated special case to allow non-matching shapes
         * for the index and value arrays.
         */
        if (index_type != HAS_FANCY || index_num != 1) {
            /* This is not a "flat like" 1-d special case */
            goto fail;
        }
        if (attempt_1d_fallback(self, indices[0].object, op) < 0) {
            goto fail;
        }
        goto success;
    }

    if (tmp_arr == NULL) {
        /* Fill extra op, need to swap first */
        tmp_arr = mit->extra_op;
        Py_INCREF(tmp_arr);
        if (mit->consec) {
            PyArray_MapIterSwapAxes(mit, &tmp_arr, 1);
            if (tmp_arr == NULL) {
                goto fail;
            }
        }
        if (PyArray_CopyObject(tmp_arr, op) < 0) {
             /*
              * This is a deprecated special case to allow non-matching shapes
              * for the index and value arrays.
              */
              if (index_type != HAS_FANCY || index_num != 1) {
                 /* This is not a "flat like" 1-d special case */
                 goto fail;
             }
             if (attempt_1d_fallback(self, indices[0].object, op) < 0) {
                 goto fail;
             }
             goto success;
        }
    }

    /* Can now reset the outer iterator (delayed bufalloc) */
    if (NpyIter_Reset(mit->outer, NULL) < 0) {
        goto fail;
    }

    if (PyArray_MapIterCheckIndices(mit) < 0) {
        goto fail;
    }

    /*
     * Could add a casting check, but apparently most assignments do
     * not care about safe casting.
     */

    if (mapiter_set(mit) < 0) {
        goto fail;
    }

    Py_DECREF(mit);
    goto success;

    /* Clean up temporary variables and indices */
  fail:
    Py_XDECREF((PyObject *)view);
    Py_XDECREF((PyObject *)tmp_arr);
    Py_XDECREF((PyObject *)mit);
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    return -1;

  success:
    Py_XDECREF((PyObject *)view);
    Py_XDECREF((PyObject *)tmp_arr);
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    return 0;
}


NPY_NO_EXPORT PyMappingMethods array_as_mapping = {
    (lenfunc)array_length,              /*mp_length*/
    (binaryfunc)array_subscript,        /*mp_subscript*/
    (objobjargproc)array_assign_subscript,       /*mp_ass_subscript*/
};

/****************** End of Mapping Protocol ******************************/

/*********************** Subscript Array Iterator *************************
 *                                                                        *
 * This object handles subscript behavior for array objects.              *
 *  It is an iterator object with a next method                           *
 *  It abstracts the n-dimensional mapping behavior to make the looping   *
 *     code more understandable (maybe)                                   *
 *     and so that indexing can be set up ahead of time                   *
 */

/*
 * This function takes a Boolean array and constructs index objects and
 * iterators as if nonzero(Bool) had been called
 *
 * Must not be called on a 0-d array.
 */
static int
_nonzero_indices(PyObject *myBool, PyArrayObject **arrays)
{
    PyArray_Descr *typecode;
    PyArrayObject *ba = NULL, *new = NULL;
    int nd, j;
    npy_intp size, i, count;
    npy_bool *ptr;
    npy_intp coords[NPY_MAXDIMS], dims_m1[NPY_MAXDIMS];
    npy_intp *dptr[NPY_MAXDIMS];
    static npy_intp one = 1;
    NPY_BEGIN_THREADS_DEF;

    typecode=PyArray_DescrFromType(NPY_BOOL);
    ba = (PyArrayObject *)PyArray_FromAny(myBool, typecode, 0, 0,
                                          NPY_ARRAY_CARRAY, NULL);
    if (ba == NULL) {
        return -1;
    }
    nd = PyArray_NDIM(ba);

    for (j = 0; j < nd; j++) {
        arrays[j] = NULL;
    }
    size = PyArray_SIZE(ba);
    ptr = (npy_bool *)PyArray_DATA(ba);

    /*
     * pre-determine how many nonzero entries there are,
     * ignore dimensionality of input as its a CARRAY
     */
    count = count_boolean_trues(1, (char*)ptr, &size, &one);

    /* create count-sized index arrays for each dimension */
    for (j = 0; j < nd; j++) {
        new = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &count,
                                           NPY_INTP, NULL, NULL,
                                           0, 0, NULL);
        if (new == NULL) {
            goto fail;
        }
        arrays[j] = new;

        dptr[j] = (npy_intp *)PyArray_DATA(new);
        coords[j] = 0;
        dims_m1[j] = PyArray_DIMS(ba)[j]-1;
    }
    if (count == 0) {
        goto finish;
    }

    /*
     * Loop through the Boolean array  and copy coordinates
     * for non-zero entries
     */
    NPY_BEGIN_THREADS_THRESHOLDED(size);
    for (i = 0; i < size; i++) {
        if (*(ptr++)) {
            for (j = 0; j < nd; j++) {
                *(dptr[j]++) = coords[j];
            }
        }
        /* Borrowed from ITER_NEXT macro */
        for (j = nd - 1; j >= 0; j--) {
            if (coords[j] < dims_m1[j]) {
                coords[j]++;
                break;
            }
            else {
                coords[j] = 0;
            }
        }
    }
    NPY_END_THREADS;

 finish:
    Py_DECREF(ba);
    return nd;

 fail:
    for (j = 0; j < nd; j++) {
        Py_XDECREF(arrays[j]);
    }
    Py_XDECREF(ba);
    return -1;
}


/* Reset the map iterator to the beginning */
NPY_NO_EXPORT void
PyArray_MapIterReset(PyArrayMapIterObject *mit)
{
    npy_intp indval;
    char *baseptrs[2];
    int i;

    if (mit->size == 0) {
        return;
    }

    NpyIter_Reset(mit->outer, NULL);
    if (mit->extra_op_iter) {
        NpyIter_Reset(mit->extra_op_iter, NULL);

        baseptrs[1] = mit->extra_op_ptrs[0];
    }

    baseptrs[0] = mit->baseoffset;

    for (i = 0; i < mit->numiter; i++) {
        indval = *((npy_intp*)mit->outer_ptrs[i]);
        if (indval < 0) {
            indval += mit->fancy_dims[i];
        }
        baseptrs[0] += indval * mit->fancy_strides[i];
    }
    mit->dataptr = baseptrs[0];

    if (mit->subspace_iter) {
        NpyIter_ResetBasePointers(mit->subspace_iter, baseptrs, NULL);
        mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->subspace_iter);
    }
    else {
        mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->outer);
    }

    return;
}


/*NUMPY_API
 * This function needs to update the state of the map iterator
 * and point mit->dataptr to the memory-location of the next object
 *
 * Note that this function never handles an extra operand but provides
 * compatibility for an old (exposed) API.
 */
NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit)
{
    int i;
    char *baseptr;
    npy_intp indval;

    if (mit->subspace_iter) {
        if (--mit->iter_count > 0) {
            mit->subspace_ptrs[0] += mit->subspace_strides[0];
            mit->dataptr = mit->subspace_ptrs[0];
            return;
        }
        else if (mit->subspace_next(mit->subspace_iter)) {
            mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->subspace_iter);
            mit->dataptr = mit->subspace_ptrs[0];
        }
        else {
            if (!mit->outer_next(mit->outer)) {
                return;
            }

            baseptr = mit->baseoffset;

            for (i = 0; i < mit->numiter; i++) {
                indval = *((npy_intp*)mit->outer_ptrs[i]);
                if (indval < 0) {
                    indval += mit->fancy_dims[i];
                }
                baseptr += indval * mit->fancy_strides[i];
            }
            NpyIter_ResetBasePointers(mit->subspace_iter, &baseptr, NULL);
            mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->subspace_iter);

            mit->dataptr = mit->subspace_ptrs[0];
        }
    }
    else {
        if (--mit->iter_count > 0) {
            baseptr = mit->baseoffset;

            for (i = 0; i < mit->numiter; i++) {
                mit->outer_ptrs[i] += mit->outer_strides[i];

                indval = *((npy_intp*)mit->outer_ptrs[i]);
                if (indval < 0) {
                    indval += mit->fancy_dims[i];
                }
                baseptr += indval * mit->fancy_strides[i];
            }

            mit->dataptr = baseptr;
            return;
        }
        else {
            if (!mit->outer_next(mit->outer)) {
                return;
            }
            mit->iter_count = *NpyIter_GetInnerLoopSizePtr(mit->outer);
            baseptr = mit->baseoffset;

            for (i = 0; i < mit->numiter; i++) {
                indval = *((npy_intp*)mit->outer_ptrs[i]);
                if (indval < 0) {
                    indval += mit->fancy_dims[i];
                }
                baseptr += indval * mit->fancy_strides[i];
            }

            mit->dataptr = baseptr;
        }
    }
}


/**
 * Fill information about the iterator. The MapIterObject does not
 * need to have any information set for this function to work.
 * (PyArray_MapIterSwapAxes requires also nd and nd_fancy info)
 *
 * Sets the following information:
 *    * mit->consec: The axis where the fancy indices need transposing to.
 *    * mit->iteraxes: The axis which the fancy index corresponds to.
 *    * mit-> fancy_dims: the dimension of `arr` along the indexed dimension
 *          for each fancy index.
 *    * mit->fancy_strides: the strides for the dimension being indexed
 *          by each fancy index.
 *    * mit->dimensions: Broadcast dimension of the fancy indices and
 *          the subspace iteration dimension.
 *
 * @param MapIterObject
 * @param The parsed indices object
 * @param Number of indices
 * @param The array that is being iterated
 *
 * @return 0 on success -1 on failure
 */
static int
mapiter_fill_info(PyArrayMapIterObject *mit, npy_index_info *indices,
                  int index_num, PyArrayObject *arr)
{
    int j = 0, i;
    int curr_dim = 0;
     /* dimension of index result (up to first fancy index) */
    int result_dim = 0;
    /* -1 init; 0 found fancy; 1 fancy stopped; 2 found not consecutive fancy */
    int consec_status = -1;
    int axis, broadcast_axis;
    npy_intp dimension;
    PyObject *errmsg, *tmp;

    for (i = 0; i < mit->nd_fancy; i++) {
        mit->dimensions[i] = 1;
    }

    mit->consec = 0;
    for (i = 0; i < index_num; i++) {
        /* integer and fancy indexes are transposed together */
        if (indices[i].type & (HAS_FANCY | HAS_INTEGER)) {
            /* there was no previous fancy index, so set consec */
            if (consec_status == -1) {
                mit->consec = result_dim;
                consec_status = 0;
            }
            /* there was already a non-fancy index after a fancy one */
            else if (consec_status == 1) {
                consec_status = 2;
                mit->consec = 0;
            }
        }
        else {
            /* consec_status == 0 means there was a fancy index before */
            if (consec_status == 0) {
                consec_status = 1;
            }
        }

        /* (iterating) fancy index, store the iterator */
        if (indices[i].type == HAS_FANCY) {
            mit->fancy_strides[j] = PyArray_STRIDE(arr, curr_dim);
            mit->fancy_dims[j] = PyArray_DIM(arr, curr_dim);
            mit->iteraxes[j++] = curr_dim++;

            /* Check broadcasting */
            broadcast_axis = mit->nd_fancy;
            /* Fill from back, we know how many dims there are */
            for (axis = PyArray_NDIM((PyArrayObject *)indices[i].object) - 1;
                        axis >= 0; axis--) {
                broadcast_axis--;
                dimension = PyArray_DIM((PyArrayObject *)indices[i].object, axis);

                /* If it is 1, we can broadcast */
                if (dimension != 1) {
                    if (dimension != mit->dimensions[broadcast_axis]) {
                        if (mit->dimensions[broadcast_axis] != 1) {
                            goto broadcast_error;
                        }
                        mit->dimensions[broadcast_axis] = dimension;
                    }
                }
            }
        }
        else if (indices[i].type == HAS_0D_BOOL) {
            mit->fancy_strides[j] = 0;
            mit->fancy_dims[j] = 1;
            /* Does not exist */
            mit->iteraxes[j++] = -1;
        }

        /* advance curr_dim for non-fancy indices */
        else if (indices[i].type == HAS_ELLIPSIS) {
            curr_dim += (int)indices[i].value;
            result_dim += (int)indices[i].value;
        }
        else if (indices[i].type != HAS_NEWAXIS){
            curr_dim += 1;
            result_dim += 1;
        }
        else {
            result_dim += 1;
        }
    }

    /* Fill dimension of subspace */
    if (mit->subspace) {
        for (i = 0; i < PyArray_NDIM(mit->subspace); i++) {
            mit->dimensions[mit->nd_fancy + i] = PyArray_DIM(mit->subspace, i);
        }
    }

    return 0;

  broadcast_error:
    /*
     * Attempt to set a meaningful exception. Could also find out
     * if a boolean index was converted.
     */
    errmsg = PyUString_FromString("shape mismatch: indexing arrays could not "
                                  "be broadcast together with shapes ");
    if (errmsg == NULL) {
        return -1;
    }

    for (i = 0; i < index_num; i++) {
        if (indices[i].type != HAS_FANCY) {
            continue;
        }
        tmp = convert_shape_to_string(
                    PyArray_NDIM((PyArrayObject *)indices[i].object),
                    PyArray_SHAPE((PyArrayObject *)indices[i].object),
                    " ");
        if (tmp == NULL) {
            return -1;
        }
        PyUString_ConcatAndDel(&errmsg, tmp);
        if (errmsg == NULL) {
            return -1;
        }
    }

    PyErr_SetObject(PyExc_IndexError, errmsg);
    Py_DECREF(errmsg);
    return -1;
}


/*
 * Check whether the fancy indices are out of bounds.
 * Returns 0 on success and -1 on failure.
 * (Gets operands from the outer iterator, but iterates them independently)
 */
NPY_NO_EXPORT int
PyArray_MapIterCheckIndices(PyArrayMapIterObject *mit)
{
    PyArrayObject *op;
    NpyIter *op_iter;
    NpyIter_IterNextFunc *op_iternext;
    npy_intp outer_dim, indval;
    int outer_axis;
    npy_intp itersize, *iterstride;
    char **iterptr;
    PyArray_Descr *intp_type;
    int i;
    NPY_BEGIN_THREADS_DEF;

    if (mit->size == 0) {
        /* All indices got broadcast away, do *not* check as it always was */
        return 0;
    }

    intp_type = PyArray_DescrFromType(NPY_INTP);

    NPY_BEGIN_THREADS;

    for (i=0; i < mit->numiter; i++) {
        op = NpyIter_GetOperandArray(mit->outer)[i];

        outer_dim = mit->fancy_dims[i];
        outer_axis = mit->iteraxes[i];

        /* See if it is possible to just trivially iterate the array */
        if (PyArray_TRIVIALLY_ITERABLE(op) &&
                /* Check if the type is equivalent to INTP */
                PyArray_ITEMSIZE(op) == sizeof(npy_intp) &&
                PyArray_DESCR(op)->kind == 'i' &&
                PyArray_ISALIGNED(op) &&
                PyDataType_ISNOTSWAPPED(PyArray_DESCR(op))) {
            char *data;
            npy_intp stride;
            /* release GIL if it was taken by nditer below */
            if (_save == NULL) {
                NPY_BEGIN_THREADS;
            }

            PyArray_PREPARE_TRIVIAL_ITERATION(op, itersize, data, stride);

            while (itersize--) {
                indval = *((npy_intp*)data);
                if (check_and_adjust_index(&indval,
                                           outer_dim, outer_axis, _save) < 0) {
                    return -1;
                }
                data += stride;
            }
            /* GIL retake at end of function or if nditer path required */
            continue;
        }

        /* Use NpyIter if the trivial iteration is not possible */
        NPY_END_THREADS;
        op_iter = NpyIter_New(op,
                        NPY_ITER_BUFFERED | NPY_ITER_NBO | NPY_ITER_ALIGNED |
                        NPY_ITER_EXTERNAL_LOOP | NPY_ITER_GROWINNER |
                        NPY_ITER_READONLY,
                        NPY_KEEPORDER, NPY_SAME_KIND_CASTING, intp_type);

        if (op_iter == NULL) {
            Py_DECREF(intp_type);
            return -1;
        }

        op_iternext = NpyIter_GetIterNext(op_iter, NULL);
        if (op_iternext == NULL) {
            Py_DECREF(intp_type);
            NpyIter_Deallocate(op_iter);
            return -1;
        }

        NPY_BEGIN_THREADS_NDITER(op_iter);
        iterptr = NpyIter_GetDataPtrArray(op_iter);
        iterstride = NpyIter_GetInnerStrideArray(op_iter);
        do {
            itersize = *NpyIter_GetInnerLoopSizePtr(op_iter);
            while (itersize--) {
                indval = *((npy_intp*)*iterptr);
                if (check_and_adjust_index(&indval,
                                           outer_dim, outer_axis, _save) < 0) {
                    Py_DECREF(intp_type);
                    NpyIter_Deallocate(op_iter);
                    return -1;
                }
                *iterptr += *iterstride;
            }
        } while (op_iternext(op_iter));

        NPY_END_THREADS;
        NpyIter_Deallocate(op_iter);
    }

    NPY_END_THREADS;
    Py_DECREF(intp_type);
    return 0;
}


/*
 * Create new mapiter.
 *
 * NOTE: The outer iteration (and subspace if requested buffered) is
 *       created with DELAY_BUFALLOC. It must be reset before usage!
 *
 * @param Index information filled by prepare_index.
 * @param Number of indices (gotten through prepare_index).
 * @param Kind of index (gotten through preprare_index).
 * @param NpyIter flags for an extra array. If 0 assume that there is no
 *        extra operand. NPY_ITER_ALLOCATE can make sense here.
 * @param Array being indexed
 * @param subspace (result of getting view for the indices)
 * @param Subspace iterator flags can be used to enable buffering.
 *        NOTE: When no subspace is necessary, the extra operand will
 *              always be buffered! Buffering the subspace when not
 *              necessary is very slow when the subspace is small.
 * @param Subspace operand flags (should just be 0 normally)
 * @param Operand iteration flags for the extra operand, this must not be
 *        0 if an extra operand should be used, otherwise it must be 0.
 *        Should be at least READONLY, WRITEONLY or READWRITE.
 * @param Extra operand. For getmap, this would be the result, for setmap
 *        this would be the arrays to get from.
 *        Can be NULL, and will be allocated in that case. However,
 *        it matches the mapiter iteration, so you have to call
 *        MapIterSwapAxes(mit, &extra_op, 1) on it.
 *        The operand has no effect on the shape.
 * @param Dtype for the extra operand, borrows the reference and must not
 *        be NULL (if extra_op_flags is not 0).
 *
 * @return A new MapIter (PyObject *) or NULL.
 */
NPY_NO_EXPORT PyObject *
PyArray_MapIterNew(npy_index_info *indices , int index_num, int index_type,
                   int ndim, int fancy_ndim,
                   PyArrayObject *arr, PyArrayObject *subspace,
                   npy_uint32 subspace_iter_flags, npy_uint32 subspace_flags,
                   npy_uint32 extra_op_flags, PyArrayObject *extra_op,
                   PyArray_Descr *extra_op_dtype)
{
    PyObject *errmsg, *tmp;
    /* For shape reporting on error */
    PyArrayObject *original_extra_op = extra_op;

    PyArrayObject *index_arrays[NPY_MAXDIMS];
    PyArray_Descr *dtypes[NPY_MAXDIMS];

    npy_uint32 op_flags[NPY_MAXDIMS];
    npy_uint32 outer_flags;

    PyArrayMapIterObject *mit;

    int single_op_axis[NPY_MAXDIMS];
    int *op_axes[NPY_MAXDIMS] = {NULL};
    int i, j, dummy_array = 0;
    int nops;
    int uses_subspace;

    /* create new MapIter object */
    mit = (PyArrayMapIterObject *)PyArray_malloc(sizeof(PyArrayMapIterObject));
    if (mit == NULL) {
        return NULL;
    }
    /* set all attributes of mapiter to zero */
    memset(mit, 0, sizeof(PyArrayMapIterObject));
    PyObject_Init((PyObject *)mit, &PyArrayMapIter_Type);

    Py_INCREF(arr);
    mit->array = arr;
    Py_XINCREF(subspace);
    mit->subspace = subspace;

    /*
     * The subspace, the part of the array which is not indexed by
     * arrays, needs to be iterated when the size of the subspace
     * is larger than 1. If it is one, it has only an effect on the
     * result shape. (Optimizes for example np.newaxis usage)
     */
    if ((subspace == NULL) || PyArray_SIZE(subspace) == 1) {
        uses_subspace = 0;
    }
    else {
        uses_subspace = 1;
    }

    /* Fill basic information about the mapiter */
    mit->nd = ndim;
    mit->nd_fancy = fancy_ndim;
    if (mapiter_fill_info(mit, indices, index_num, arr) < 0) {
        Py_DECREF(mit);
        return NULL;
    }

    /*
     * Set iteration information of the indexing arrays.
     */
    for (i=0; i < index_num; i++) {
        if (indices[i].type & HAS_FANCY) {
            index_arrays[mit->numiter] = (PyArrayObject *)indices[i].object;
            dtypes[mit->numiter] = PyArray_DescrFromType(NPY_INTP);

            op_flags[mit->numiter] = (NPY_ITER_NBO |
                                      NPY_ITER_ALIGNED |
                                      NPY_ITER_READONLY);
            mit->numiter += 1;
        }
    }

    if (mit->numiter == 0) {
        /*
         * For MapIterArray, it is possible that there is no fancy index.
         * to support this case, add a a dummy iterator.
         * Since it is 0-d its transpose, etc. does not matter.
         */

        /* signal necessity to decref... */
        dummy_array = 1;

        index_arrays[0] = (PyArrayObject *)PyArray_Zeros(0, NULL,
                                        PyArray_DescrFromType(NPY_INTP), 0);
        if (index_arrays[0] == NULL) {
            Py_DECREF(mit);
            return NULL;
        }
        dtypes[0] = PyArray_DescrFromType(NPY_INTP);
        op_flags[0] = NPY_ITER_NBO | NPY_ITER_ALIGNED | NPY_ITER_READONLY;

        mit->fancy_dims[0] = 1;
        mit->numiter = 1;
    }

    /*
     * Now there are two general cases how extra_op is used:
     *   1. No subspace iteration is necessary, so the extra_op can
     *      be included into the index iterator (it will be buffered)
     *   2. Subspace iteration is necessary, so the extra op is iterated
     *      independently, and the iteration order is fixed at C (could
     *      also use Fortran order if the array is Fortran order).
     *      In this case the subspace iterator is not buffered.
     *
     * If subspace iteration is necessary and an extra_op was given,
     * it may also be necessary to transpose the extra_op (or signal
     * the transposing to the advanced iterator).
     */

    if (extra_op != NULL) {
        /*
         * If we have an extra_op given, need to prepare it.
         *   1. Subclasses might mess with the shape, so need a baseclass
         *   2. Need to make sure the shape is compatible
         *   3. May need to remove leading 1s and transpose dimensions.
         *      Normal assignments allows broadcasting away leading 1s, but
         *      the transposing code does not like this.
         */
        if (!PyArray_CheckExact(extra_op)) {
            extra_op = (PyArrayObject *)PyArray_View(extra_op, NULL,
                                                     &PyArray_Type);
            if (extra_op == NULL) {
                goto fail;
            }
        }
        else {
            Py_INCREF(extra_op);
        }

        if (PyArray_NDIM(extra_op) > mit->nd) {
            /*
             * Usual assignments allows removal of leading one dimensions.
             * (or equivalently adding of one dimensions to the array being
             * assigned to). To implement this, reshape the array.
             */
            PyArrayObject *tmp_arr;
            PyArray_Dims permute;

            permute.len = mit->nd;
            permute.ptr = &PyArray_DIMS(extra_op)[
                                            PyArray_NDIM(extra_op) - mit->nd];
            tmp_arr = (PyArrayObject*)PyArray_Newshape(extra_op, &permute,
                                                       NPY_CORDER);
            if (tmp_arr == NULL) {
                goto broadcast_error;
            }
            Py_DECREF(extra_op);
            extra_op = tmp_arr;
        }

        /*
         * If dimensions need to be prepended (and no swapaxis is needed),
         * use op_axes after extra_op is allocated for sure.
         */
        if (mit->consec) {
            PyArray_MapIterSwapAxes(mit, &extra_op, 0);
            if (extra_op == NULL) {
                goto fail;
            }
        }

        if (subspace && !uses_subspace) {
            /*
             * We are not using the subspace, so its size is 1.
             * All dimensions of the extra_op corresponding to the
             * subspace must be equal to 1.
             */
            if (PyArray_NDIM(subspace) <= PyArray_NDIM(extra_op)) {
                j = PyArray_NDIM(subspace);
            }
            else {
                j = PyArray_NDIM(extra_op);
            }
            for (i = 1; i < j + 1; i++) {
                if (PyArray_DIM(extra_op, PyArray_NDIM(extra_op) - i) != 1) {
                    goto broadcast_error;
                }
            }
        }
    }

    /*
     * If subspace is not NULL, NpyIter cannot allocate extra_op for us.
     * This is a bit of a kludge. A dummy iterator is created to find
     * the correct output shape and stride permutation.
     * TODO: This can at least partially be replaced, since the shape
     *       is found for broadcasting errors.
     */
    else if (extra_op_flags && (subspace != NULL)) {
        npy_uint32 tmp_op_flags[NPY_MAXDIMS];

        NpyIter *tmp_iter;
        npy_intp stride;
        npy_intp strides[NPY_MAXDIMS];
        npy_stride_sort_item strideperm[NPY_MAXDIMS];

        for (i=0; i < mit->numiter; i++) {
            tmp_op_flags[i] = NPY_ITER_READONLY;
        }

        Py_INCREF(extra_op_dtype);
        mit->extra_op_dtype = extra_op_dtype;

        /* Create an iterator, just to broadcast the arrays?! */
        tmp_iter = NpyIter_MultiNew(mit->numiter, index_arrays,
                                    NPY_ITER_ZEROSIZE_OK |
                                    NPY_ITER_REFS_OK |
                                    NPY_ITER_MULTI_INDEX |
                                    NPY_ITER_DONT_NEGATE_STRIDES,
                                    NPY_KEEPORDER,
                                    NPY_UNSAFE_CASTING,
                                    tmp_op_flags, NULL);
        if (tmp_iter == NULL) {
            goto fail;
        }

        if (PyArray_SIZE(subspace) == 1) {
            /*
             * nditer allows itemsize with npy_intp type, so it works
             * here, but it would *not* work directly, since elsize
             * is limited to int.
             */
            if (!NpyIter_CreateCompatibleStrides(tmp_iter,
                        extra_op_dtype->elsize * PyArray_SIZE(subspace),
                        strides)) {
                PyErr_SetString(PyExc_ValueError,
                        "internal error: failed to find output array strides");
                goto fail;
            }
        }
        else {
            /* Just use C-order strides (TODO: allow also F-order) */
            stride = extra_op_dtype->elsize * PyArray_SIZE(subspace);
            for (i=mit->nd_fancy - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= mit->dimensions[i];
            }
        }
        NpyIter_Deallocate(tmp_iter);

        /* shape is set, and strides is set up to mit->nd, set rest */
        PyArray_CreateSortedStridePerm(PyArray_NDIM(subspace),
                                PyArray_STRIDES(subspace), strideperm);
        stride = extra_op_dtype->elsize;
        for (i=PyArray_NDIM(subspace) - 1; i >= 0; i--) {
            strides[mit->nd_fancy + strideperm[i].perm] = stride;
            stride *= PyArray_DIM(subspace, (int)strideperm[i].perm);
        }

        /*
         * Allocate new array. Note: Always base class, because
         * subclasses might mess with the shape.
         */
        Py_INCREF(extra_op_dtype);
        extra_op = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                           extra_op_dtype,
                                           mit->nd_fancy + PyArray_NDIM(subspace),
                                           mit->dimensions, strides,
                                           NULL, 0, NULL);
        if (extra_op == NULL) {
            goto fail;
        }
    }

    /*
     * The extra op is now either allocated, can be allocated by
     * NpyIter (no subspace) or is not used at all.
     *
     * Need to set the axis remapping for the extra_op. This needs
     * to cause ignoring of subspace dimensions and prepending -1
     * for broadcasting.
     */
    if (extra_op) {
        for (j=0; j < mit->nd - PyArray_NDIM(extra_op); j++) {
            single_op_axis[j] = -1;
        }
        for (i=0; i < PyArray_NDIM(extra_op); i++) {
            /* (fills subspace dimensions too, but they are not unused) */
            single_op_axis[j++] = i;
        }
    }

    /*
     * NOTE: If for some reason someone wishes to use REDUCE_OK, be
     *       careful and fix the error message replacement at the end.
     */
    outer_flags = NPY_ITER_ZEROSIZE_OK |
                  NPY_ITER_REFS_OK |
                  NPY_ITER_BUFFERED |
                  NPY_ITER_DELAY_BUFALLOC |
                  NPY_ITER_GROWINNER;

    /*
     * For a single 1-d operand, guarantee iteration order
     * (scipy used this). Note that subspace may be used.
     */
    if ((mit->numiter == 1) && (PyArray_NDIM(index_arrays[0]) == 1)) {
        outer_flags |= NPY_ITER_DONT_NEGATE_STRIDES;
    }

    /* If external array is iterated, and no subspace is needed */
    nops = mit->numiter;
    if (extra_op_flags && !uses_subspace) {
        /*
         * NOTE: This small limitation should practically not matter.
         *       (replaces npyiter error)
         */
        if (mit->numiter > NPY_MAXDIMS - 1) {
            PyErr_Format(PyExc_IndexError,
                         "when no subspace is given, the number of index "
                         "arrays cannot be above %d, but %d index arrays found",
                         NPY_MAXDIMS - 1, mit->numiter);
            goto fail;
        }

        nops += 1;
        index_arrays[mit->numiter] = extra_op;

        Py_INCREF(extra_op_dtype);
        dtypes[mit->numiter] = extra_op_dtype;
        op_flags[mit->numiter] = (extra_op_flags |
                                  NPY_ITER_ALLOCATE |
                                  NPY_ITER_NO_SUBTYPE);

        if (extra_op) {
            /* Use the axis remapping */
            op_axes[mit->numiter] = single_op_axis;
            mit->outer = NpyIter_AdvancedNew(nops, index_arrays, outer_flags,
                             NPY_KEEPORDER, NPY_UNSAFE_CASTING, op_flags, dtypes,
                             mit->nd_fancy, op_axes, mit->dimensions, 0);
        }
        else {
            mit->outer = NpyIter_MultiNew(nops, index_arrays, outer_flags,
                             NPY_KEEPORDER, NPY_UNSAFE_CASTING, op_flags, dtypes);
        }

    }
    else {
        /* TODO: Maybe add test for the CORDER, and maybe also allow F */
        mit->outer = NpyIter_MultiNew(nops, index_arrays, outer_flags,
                         NPY_CORDER, NPY_UNSAFE_CASTING, op_flags, dtypes);
    }

    /* NpyIter cleanup and information: */
    for (i=0; i < nops; i++) {
        Py_DECREF(dtypes[i]);
    }
    if (dummy_array) {
        Py_DECREF(index_arrays[0]);
    }
    if (mit->outer == NULL) {
        goto fail;
    }
    if (!uses_subspace) {
        NpyIter_EnableExternalLoop(mit->outer);
    }

    mit->outer_next = NpyIter_GetIterNext(mit->outer, NULL);
    if (mit->outer_next == NULL) {
        goto fail;
    }
    mit->outer_ptrs = NpyIter_GetDataPtrArray(mit->outer);
    if (!uses_subspace) {
        mit->outer_strides = NpyIter_GetInnerStrideArray(mit->outer);
    }
    if (NpyIter_IterationNeedsAPI(mit->outer)) {
        mit->needs_api = 1;
        /* We may be doing a cast for the buffer, and that may have failed */
        if (PyErr_Occurred()) {
            goto fail;
        }
    }

    /* Get the allocated extra_op */
    if (extra_op_flags) {
        if (extra_op == NULL) {
            mit->extra_op = NpyIter_GetOperandArray(mit->outer)[mit->numiter];
        }
        else {
            mit->extra_op = extra_op;
        }
        Py_INCREF(mit->extra_op);
    }

    /*
     * If extra_op is being tracked but subspace is used, we need
     * to create a dedicated iterator for the outer iteration of
     * the extra operand.
     */
    if (extra_op_flags && uses_subspace) {
        op_axes[0] = single_op_axis;
        mit->extra_op_iter = NpyIter_AdvancedNew(1, &extra_op,
                                                 NPY_ITER_ZEROSIZE_OK |
                                                 NPY_ITER_REFS_OK |
                                                 NPY_ITER_GROWINNER,
                                                 NPY_CORDER,
                                                 NPY_NO_CASTING,
                                                 &extra_op_flags,
                                                 NULL,
                                                 mit->nd_fancy, op_axes,
                                                 mit->dimensions, 0);

        if (mit->extra_op_iter == NULL) {
            goto fail;
        }

        mit->extra_op_next = NpyIter_GetIterNext(mit->extra_op_iter, NULL);
        if (mit->extra_op_next == NULL) {
            goto fail;
        }
        mit->extra_op_ptrs = NpyIter_GetDataPtrArray(mit->extra_op_iter);
    }

    /* Get the full dimension information */
    if (subspace != NULL) {
        mit->baseoffset = PyArray_BYTES(subspace);
    }
    else {
        mit->baseoffset = PyArray_BYTES(arr);
    }

    /* Calculate total size of the MapIter */
    mit->size = PyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (mit->size < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "advanced indexing operation result is too large");
        goto fail;
    }

    /* Can now return early if no subspace is being used */
    if (!uses_subspace) {
        Py_XDECREF(extra_op);
        return (PyObject *)mit;
    }

    /* Fill in the last bit of mapiter information needed */

    /*
     * Now just need to create the correct subspace iterator.
     */
    index_arrays[0] = subspace;
    dtypes[0] = NULL;
    op_flags[0] = subspace_flags;
    op_axes[0] = NULL;

    if (extra_op_flags) {
        /* We should iterate the extra_op as well */
        nops = 2;
        index_arrays[1] = extra_op;

        op_axes[1] = &single_op_axis[mit->nd_fancy];

        /*
         * Buffering is never used here, but in case someone plugs it in
         * somewhere else, set the type correctly then.
         */
        if ((subspace_iter_flags & NPY_ITER_BUFFERED)) {
            dtypes[1] = extra_op_dtype;
        }
        else {
            dtypes[1] = NULL;
        }
        op_flags[1] = extra_op_flags;
    }
    else {
        nops = 1;
    }

    mit->subspace_iter = NpyIter_AdvancedNew(nops, index_arrays,
                                    NPY_ITER_ZEROSIZE_OK |
                                    NPY_ITER_REFS_OK |
                                    NPY_ITER_GROWINNER |
                                    NPY_ITER_EXTERNAL_LOOP |
                                    NPY_ITER_DELAY_BUFALLOC |
                                    subspace_iter_flags,
                                    (nops == 1 ? NPY_CORDER : NPY_KEEPORDER),
                                    NPY_UNSAFE_CASTING,
                                    op_flags, dtypes,
                                    PyArray_NDIM(subspace), op_axes,
                                    &mit->dimensions[mit->nd_fancy], 0);

    if (mit->subspace_iter == NULL) {
        goto fail;
    }

    mit->subspace_next = NpyIter_GetIterNext(mit->subspace_iter, NULL);
    if (mit->subspace_next == NULL) {
        goto fail;
    }
    mit->subspace_ptrs = NpyIter_GetDataPtrArray(mit->subspace_iter);
    mit->subspace_strides = NpyIter_GetInnerStrideArray(mit->subspace_iter);

    if (NpyIter_IterationNeedsAPI(mit->outer)) {
        mit->needs_api = 1;
        /*
         * NOTE: In this case, need to call PyErr_Occurred() after
         *       basepointer resetting (buffer allocation)
         */
    }

    Py_XDECREF(extra_op);
    return (PyObject *)mit;

  fail:
    /*
     * Check whether the operand could not be broadcast and replace the error
     * in that case. This should however normally be found early with a
     * direct goto to broadcast_error
     */
    if (extra_op == NULL) {
        goto finish;
    }

    j = mit->nd;
    for (i = PyArray_NDIM(extra_op) - 1; i >= 0; i--) {
        j--;
        if ((PyArray_DIM(extra_op, i) != 1) &&
                /* (j < 0 is currently impossible, extra_op is reshaped) */
                j >= 0 &&
                PyArray_DIM(extra_op, i) != mit->dimensions[j]) {
            /* extra_op cannot be broadcast to the indexing result */
            goto broadcast_error;
        }
    }
    goto finish;

  broadcast_error:
    errmsg = PyUString_FromString("shape mismatch: value array "
                    "of shape ");
    if (errmsg == NULL) {
        goto finish;
    }

    /* Report the shape of the original array if it exists */
    if (original_extra_op == NULL) {
        original_extra_op = extra_op;
    }

    tmp = convert_shape_to_string(PyArray_NDIM(original_extra_op),
                                  PyArray_DIMS(original_extra_op), " ");
    if (tmp == NULL) {
        goto finish;
    }
    PyUString_ConcatAndDel(&errmsg, tmp);
    if (errmsg == NULL) {
        goto finish;
    }

    tmp = PyUString_FromString("could not be broadcast to indexing "
                    "result of shape ");
    PyUString_ConcatAndDel(&errmsg, tmp);
    if (errmsg == NULL) {
        goto finish;
    }

    tmp = convert_shape_to_string(mit->nd, mit->dimensions, "");
    if (tmp == NULL) {
        goto finish;
    }
    PyUString_ConcatAndDel(&errmsg, tmp);
    if (errmsg == NULL) {
        goto finish;
    }

    PyErr_SetObject(PyExc_ValueError, errmsg);
    Py_DECREF(errmsg);

  finish:
    Py_XDECREF(extra_op);
    Py_DECREF(mit);
    return NULL;
}


/*NUMPY_API
 *
 * Use advanced indexing to iterate an array. Please note
 * that most of this public API is currently not guaranteed
 * to stay the same between versions. If you plan on using
 * it, please consider adding more utility functions here
 * to accommodate new features.
 */
NPY_NO_EXPORT PyObject *
PyArray_MapIterArray(PyArrayObject * a, PyObject * index)
{
    PyArrayMapIterObject * mit = NULL;
    PyArrayObject *subspace = NULL;
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];
    int i, index_num, ndim, fancy_ndim, index_type;

    index_type = prepare_index(a, index, indices, &index_num,
                               &ndim, &fancy_ndim, 0);

    if (index_type < 0) {
        return NULL;
    }

    /* If it is not a pure fancy index, need to get the subspace */
    if (index_type != HAS_FANCY) {
        if (get_view_from_index(a, &subspace, indices, index_num, 1) < 0) {
            goto fail;
        }
    }

    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices, index_num,
                                                     index_type, ndim,
                                                     fancy_ndim,
                                                     a, subspace, 0,
                                                     NPY_ITER_READWRITE,
                                                     0, NULL, NULL);
    if (mit == NULL) {
        goto fail;
    }

    /* Required for backward compatibility */
    mit->ait = (PyArrayIterObject *)PyArray_IterNew((PyObject *)a);
    if (mit->ait == NULL) {
        goto fail;
    }

    if (PyArray_MapIterCheckIndices(mit) < 0) {
        goto fail;
    }

    Py_XDECREF(subspace);
    PyArray_MapIterReset(mit);

    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }

    return (PyObject *)mit;

 fail:
    Py_XDECREF(subspace);
    Py_XDECREF((PyObject *)mit);
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    return NULL;
}


#undef HAS_INTEGER
#undef HAS_NEWAXIS
#undef HAS_SLICE
#undef HAS_ELLIPSIS
#undef HAS_FANCY
#undef HAS_BOOL
#undef HAS_SCALAR_ARRAY
#undef HAS_0D_BOOL


static void
arraymapiter_dealloc(PyArrayMapIterObject *mit)
{
    Py_XDECREF(mit->array);
    Py_XDECREF(mit->ait);
    Py_XDECREF(mit->subspace);
    Py_XDECREF(mit->extra_op);
    Py_XDECREF(mit->extra_op_dtype);
    if (mit->outer != NULL) {
        NpyIter_Deallocate(mit->outer);
    }
    if (mit->subspace_iter != NULL) {
        NpyIter_Deallocate(mit->subspace_iter);
    }
    if (mit->extra_op_iter != NULL) {
        NpyIter_Deallocate(mit->extra_op_iter);
    }
    PyArray_free(mit);
}

/*
 * The mapiter object must be created new each time.  It does not work
 * to bind to a new array, and continue.
 *
 * This was the orginal intention, but currently that does not work.
 * Do not expose the MapIter_Type to Python.
 *
 * The original mapiter(indexobj); mapiter.bind(a); idea is now fully
 * removed. This is not very useful anyway, since mapiter is equivalent
 * to a[indexobj].flat but the latter gets to use slice syntax.
 */
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.mapiter",                            /* tp_name */
    sizeof(PyArrayMapIterObject),               /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)arraymapiter_dealloc,           /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                         /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};
