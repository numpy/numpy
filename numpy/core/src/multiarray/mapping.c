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

/* Get array item as scalar type */
NPY_NO_EXPORT PyObject *
array_item_asscalar(PyArrayObject *self, npy_intp i)
{
    char *item;
    npy_intp dim0;

    /* Bounds check and get the data pointer */
    dim0 = PyArray_DIM(self, 0);
    if (i < 0) {
        i += dim0;
    }
    if (i < 0 || i >= dim0) {
        PyErr_SetString(PyExc_IndexError, "index out of bounds");
        return NULL;
    }
    item = PyArray_BYTES(self) + i * PyArray_STRIDE(self, 0);
    return PyArray_Scalar(item, PyArray_DESCR(self), (PyObject *)self);
}

/* Get array item as ndarray type */
NPY_NO_EXPORT PyObject *
array_item_asarray(PyArrayObject *self, npy_intp i)
{
    char *item;
    PyArrayObject *ret;
    npy_intp dim0;

    if(PyArray_NDIM(self) == 0) {
        PyErr_SetString(PyExc_IndexError,
                        "0-d arrays can't be indexed");
        return NULL;
    }

    /* Bounds check and get the data pointer */
    dim0 = PyArray_DIM(self, 0);
    if (check_and_adjust_index(&i, dim0, 0) < 0) {
        return NULL;
    }
    item = PyArray_BYTES(self) + i * PyArray_STRIDE(self, 0);

    /* Create the view array */
    Py_INCREF(PyArray_DESCR(self));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                                              PyArray_DESCR(self),
                                              PyArray_NDIM(self)-1,
                                              PyArray_DIMS(self)+1,
                                              PyArray_STRIDES(self)+1, item,
                                              PyArray_FLAGS(self),
                                              (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }

    /* Set the base object */
    Py_INCREF(self);
    if (PyArray_SetBaseObject(ret, (PyObject *)self) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    PyArray_UpdateFlags(ret, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    return (PyObject *)ret;
}

/* Get array item at given index */
NPY_NO_EXPORT PyObject *
array_item(PyArrayObject *self, Py_ssize_t _i)
{
    /* Workaround Python 2.4: Py_ssize_t not the same as npyint_p */
    npy_intp i = _i;

    if (PyArray_NDIM(self) == 1) {
        return array_item_asscalar(self, (npy_intp) i);
    }
    else {
        return array_item_asarray(self, (npy_intp) i);
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


#define SET_MIT_DATAPTR_CHECK_INDEX()                               \
    mit->dataptr = mit->baseoffset;                                 \
    for (j = 0; j < mit->numiter; j++) {                            \
        indval = *((npy_intp*)mit->iterptrs[j]);                    \
        if (check_and_adjust_index(&indval,                         \
                    mit->outer_dims[j], mit->iteraxes[j]) < 0) {    \
            goto fail;                                              \
        }                                                           \
        mit->iterptrs[j] += mit->iterstrides[j];                    \
        mit->dataptr += indval * mit->outer_strides[j];             \
    }                                                               \

#define SET_MIT_DATAPTR_1_NUMITER_CHECK_INDEX()                     \
    mit->dataptr = mit->baseoffset;                                 \
    indval = *((npy_intp*)mit->iterptrs[0]);                        \
    if (check_and_adjust_index(&indval,                             \
                mit->outer_dims[0], mit->iteraxes[0]) < 0) {        \
        goto fail;                                                  \
    }                                                               \
    mit->iterptrs[0] += mit->iterstrides[0];                        \
    mit->dataptr += indval * mit->outer_strides[0]                  \


static PyObject *
PyArray_GetMap(PyArrayMapIterObject *mit)
{
    PyArrayObject *ret, *temp;
    PyArrayIterObject *it;
    npy_intp counter;
    int swap, j;
    PyArray_CopySwapFunc *copyswap;

    npy_intp innersize, indval;

    /* Unbound map iterator --- Bind should have been called */
    if (mit->ait == NULL) {
        return NULL;
    }

    /* This relies on the map iterator object telling us the shape
       of the new array in nd and dimensions.
    */
    temp = mit->ait->ao;
    Py_INCREF(PyArray_DESCR(temp));
    ret = (PyArrayObject *)
        PyArray_NewFromDescr(Py_TYPE(temp),
                             PyArray_DESCR(temp),
                             mit->nd, mit->dimensions,
                             NULL, NULL,
                             PyArray_ISFORTRAN(temp),
                             (PyObject *)temp);
    if (ret == NULL) {
        return NULL;
    }

    if (mit->size == 0) {
        if (mit->consec) {
            PyArray_MapIterSwapAxes(mit, &ret, 1);
        }
        return ret;
    }

    /*
     * Make use of fast_take if possible!
     * 
     * Fast take could actually handle more cases, since it can take
     * along any axis, making the contiguity requirements slightly
     * less strict. But that would take a lot more to check for these.
     */
    if ((mit->numiter == 1) &&
            ((mit->subspace == NULL) ||
                    PyArray_IS_C_CONTIGUOUS(mit->subspace->ao)) &&
            PyArray_IS_C_CONTIGUOUS(ret) &&
            PyArray_IS_C_CONTIGUOUS(NpyIter_GetOperandArray(mit->outer)[0]) &&
            !NpyIter_RequiresBuffering(mit->outer) &&
            (PyArray_DESCR(temp)->f->fasttake != NULL)) {

        char *src, *dst;
        int err;
        npy_intp *indices;
        npy_intp n, m, nelem;

        /* Number of subspace elements (after index; n=1, so all are after) */
        if (mit->subspace == NULL) {
            nelem = 1;
        }
        else {
            nelem = PyArray_SIZE(mit->subspace->ao);
        }

        /*
         * We can only do this, if the strides add up to be "contiguous".
         */
        if (nelem * PyArray_ITEMSIZE(temp) == mit->outer_strides[0]) {
            src = mit->baseoffset;
            dst = PyArray_DATA(ret);
            indices = PyArray_DATA(NpyIter_GetOperandArray(mit->outer)[0]);

            /*
             * Always 1 since we would have axis=0 implicitely wit contiguous
             * subspace (i.e. the first axes are for the index, before transpose):
             */
            n = 1;
            /* Here, indices size: */
            m = PyArray_SIZE(NpyIter_GetOperandArray(mit->outer)[0]);

            err = PyArray_DESCR(temp)->f->fasttake(dst, src, indices,
                        mit->outer_dims[0], n, m, nelem, NPY_RAISE);

            if (err) {
                /*
                 * Error message won't include axis, but there is only one
                 * fancy index...
                 */
                goto fail;
            }

            /* check for consecutive axes */
            if (mit->consec) {
                PyArray_MapIterSwapAxes(mit, &ret, 1);
            }
            return (PyObject *)ret;
        }
    }

    /*
     * Now just iterate through the new array filling it in
     * with the next object from the original array as
     * defined by the mapping iterator
     */

    if ((it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)ret)) == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    swap = (PyArray_ISNOTSWAPPED(temp) != PyArray_ISNOTSWAPPED(ret));
    copyswap = PyArray_DESCR(ret)->f->copyswap;
    PyArray_MapIterReset(mit);

    /*
     * TODO: Could add special cases for iteration of return array
     *       as well as subspace iteration. Should add special case
     *       to make use of strided dtype transfer functions!
     */
    if ((mit->numiter == 1)) {
        if ((mit->subspace == NULL) || (PyArray_SIZE(mit->subspace->ao) == 1)) {
            do {
                innersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);

                while (innersize--) {
                    SET_MIT_DATAPTR_1_NUMITER_CHECK_INDEX();
                    copyswap(it->dataptr, mit->dataptr, swap, ret);
                    PyArray_ITER_NEXT(it);
                }
            } while (mit->iternext(mit->outer));
        }
        else {
            do {
                innersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);

                while (innersize--) {
                    SET_MIT_DATAPTR_1_NUMITER_CHECK_INDEX();
                    counter = mit->subspace->size;
                    PyArray_ITER_RESET(mit->subspace);
                    mit->subspace->dataptr = mit->dataptr;
                    while (counter--) {
                        copyswap(it->dataptr, mit->subspace->dataptr, swap, ret);
                        PyArray_ITER_NEXT(it);
                        PyArray_ITER_NEXT(mit->subspace);
                    }
                }
            } while (mit->iternext(mit->outer));
        }
    }
    else {
        if ((mit->subspace == NULL) || (PyArray_SIZE(mit->subspace->ao) == 1)) {
            do {
                innersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);

                while (innersize-- > 0) {
                    SET_MIT_DATAPTR_CHECK_INDEX();
                    copyswap(it->dataptr, mit->dataptr, swap, ret);
                    PyArray_ITER_NEXT(it);
                }
            } while (mit->iternext(mit->outer));
        }
        else {
            do {
                innersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);

                while (innersize-- > 0) {
                    SET_MIT_DATAPTR_CHECK_INDEX();
                    counter = mit->subspace->size;
                    PyArray_ITER_RESET(mit->subspace);
                    mit->subspace->dataptr = mit->dataptr;
                    while (counter--) {
                        copyswap(it->dataptr, mit->subspace->dataptr, swap, ret);
                        PyArray_ITER_NEXT(it);
                        PyArray_ITER_NEXT(mit->subspace);
                    }
                }
            } while (mit->iternext(mit->outer));
        }
    }

    Py_DECREF(it);

    /* check for consecutive axes */
    if (mit->consec) {
        PyArray_MapIterSwapAxes(mit, &ret, 1);
    }
    return (PyObject *)ret;
  fail:
    Py_DECREF(it);
    Py_DECREF(ret);
    return NULL;
}


static int
PyArray_SetMap(PyArrayMapIterObject *mit, PyObject *op)
{
    PyArrayObject *arr = NULL;
    PyArrayIterObject *it;
    npy_intp counter;
    int swap;
    PyArray_CopySwapFunc *copyswap;
    PyArray_Descr *descr;

    /* Unbound Map Iterator */
    if (mit->ait == NULL) {
        return -1;
    }
    descr = PyArray_DESCR(mit->ait->ao);
    Py_INCREF(descr);
    arr = (PyArrayObject *)PyArray_FromAny(op, descr,
                                0, 0, NPY_ARRAY_FORCECAST, NULL);
    if (arr == NULL) {
        return -1;
    }
    if ((mit->subspace != NULL) && (mit->consec)) {
        PyArray_MapIterSwapAxes(mit, &arr, 0);
        if (arr == NULL) {
            return -1;
        }
    }

    /* Be sure values array is "broadcastable"
       to shape of mit->dimensions, mit->nd */

    if ((it = (PyArrayIterObject *)\
         PyArray_BroadcastToShape((PyObject *)arr,
                                    mit->dimensions, mit->nd))==NULL) {
        Py_DECREF(arr);
        return -1;
    }

    counter = mit->size;
    swap = (PyArray_ISNOTSWAPPED(mit->ait->ao) !=
            (PyArray_ISNOTSWAPPED(arr)));
    copyswap = PyArray_DESCR(arr)->f->copyswap;
    PyArray_MapIterReset(mit);
    /* Need to decref arrays with objects in them */
    if (PyDataType_FLAGCHK(descr, NPY_ITEM_HASOBJECT)) {
        while (counter--) {
            PyArray_Item_INCREF(it->dataptr, PyArray_DESCR(arr));
            PyArray_Item_XDECREF(mit->dataptr, PyArray_DESCR(arr));
            memmove(mit->dataptr, it->dataptr, PyArray_ITEMSIZE(arr));
            /* ignored unless VOID array with object's */
            if (swap) {
                copyswap(mit->dataptr, NULL, swap, arr);
            }
            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(arr);
        Py_DECREF(it);
        return 0;
    }
    else {
        while(counter--) {
            memmove(mit->dataptr, it->dataptr, PyArray_ITEMSIZE(arr));
            if (swap) {
                copyswap(mit->dataptr, NULL, swap, arr);
            }
            PyArray_MapIterNext(mit);
            PyArray_ITER_NEXT(it);
        }
        Py_DECREF(arr);
        Py_DECREF(it);
        return 0;
    }
}


/*
 * This function handles all index preparations with the exception
 * of field access. It fills the array of index_info structs correctly.
 * It already handles the boolean array special case for fancy indexing,
 * i.e. if the index type is boolean, it is exactly one matching boolean
 * array. If the index type is fancy, the boolean array is already
 * converted to single arrays. There is (as before) no checking of the
 * boolean dimension. For this to be implemented, the index_info struct
 * would require a new field to save the original corresponding shape.
 *
 * Checks everything but the bounds.
 *
 * Returns the index_type or -1 on failure and fills the number of indices.
 */
NPY_NO_EXPORT int
prepare_index(PyArrayObject *self, PyObject *index,
              npy_index_info *indices,
              int *num, int *ndim, int allow_boolean) {
    int new_ndim, used_ndim, fancy_ndim, index_ndim;
    int curr_idx, get_idx;

    npy_intp i, n;

    npy_bool make_tuple = 0;
    PyObject *obj = NULL;
    PyArrayObject *arr;
    PyArrayObject *nonzero_result[NPY_MAXDIMS];

    int index_type = 0;
    int ellipsis_pos = -1;

    /*
     * The index might be a multi-dimensional index, but not yet a tuple
     * this makes it a tuple in that case.
     *
     * TODO: Refactor into its own function.
     */
    if (!PyTuple_CheckExact(index)
            /* Next three are just to avoid slow checks */
            && (!PyInt_CheckExact(index))
            && (index != Py_None)
            && (!PySlice_Check(index))
            && (!PyArray_Check(index))
            && (PySequence_Check(index))) {
        /*
         * Sequences < NPY_MAXDIMS with any slice objects
         * or newaxis, or Ellipsis is considered standard
         * as long as there are also no Arrays and or additional
         * sequences embedded.
         *
         * This check is historically as is.
         */

        if (PyTuple_Check(index)) {
            /* If it is already a tuple, make it an exact tuple anyway */
            n = 0;
            make_tuple = 1;
        }
        n = PySequence_Size(index);
        if (n < 0 || n >= NPY_MAXDIMS) {
            n = 0;
        }
        for (i = 0; i < n; i++) {
            PyObject *obj = PySequence_GetItem(index, i);
            if (obj == NULL) {
                make_tuple = 1;
                break;
            }
            if (PyArray_Check(obj) || PySequence_Check(obj)
                    || PySlice_Check(obj) || obj == Py_Ellipsis
                    || obj == Py_None) {
                make_tuple = 1;
                Py_DECREF(obj);
                break;
            }
            Py_DECREF(obj);
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
            get_idx += 1; /* only one loop */
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
                if (DEPRECATE(
                        "an index can only have a single Ellipsis (`...`); "
                        "replace all but one with slices (`:`).") < 0) {
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
            used_ndim += 0; /* We don't know yet */
            used_ndim += 0;
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
         * Single integer index, there are two cases here.
         * It could be an array, a 0-d array is handled
         * a bit weird however, so need to special case it.
         */
        else if (PyInt_CheckExact(obj) || !PyArray_Check(obj)) {
            i = PyArray_PyIntAsIntp(obj);
            if ((i == -1) && PyErr_Occurred()) {
                PyErr_Clear();
            }
            else {
                index_type |= HAS_INTEGER;
                indices[curr_idx].object = NULL;
                indices[curr_idx].value = i;
                indices[curr_idx].type = HAS_INTEGER;
                used_ndim += 1;
                new_ndim += 0;
                curr_idx += 1;
                continue;
            }
        }
        else if (PyArray_NDIM((PyArrayObject *)obj) == 0) {
             i = PyArray_PyIntAsIntp(obj);
             if ((i == -1) && PyErr_Occurred()) {
                 PyErr_Clear();
             }
             else {
                 index_type |= (HAS_INTEGER | HAS_SCALAR_ARRAY);
                 indices[curr_idx].object = NULL;
                 indices[curr_idx].value = i;
                 indices[curr_idx].type = HAS_INTEGER;
                 used_ndim += 1;
                 new_ndim += 0;
                 curr_idx += 1;
                 continue;
             }
        }

        /*
         * At this point, we must have an index array (or array-like).
         * It might still be a (purely) bool special case though.
         * (Or complete nonsense...)
         */
        index_type |= HAS_FANCY;
        indices[curr_idx].type = HAS_FANCY;
        indices[curr_idx].object = NULL;

        if (!PyArray_Check(obj)) {
            PyObject *tmp_arr;
            tmp_arr = PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
            if (tmp_arr == NULL) {
                /* TODO: Should maybe replace the error here? */
                goto failed_building_indices;
            }

            /*
             * For example an empty list can be cast to an integer array,
             * however it will default to a float one.
             */
            if (PyArray_SIZE((PyArrayObject *)tmp_arr) == 0) {
                PyArray_Descr *indtype = PyArray_DescrFromType(NPY_INTP);
                /* TODO: Is the force-cast the right way? */
                arr = (PyArrayObject *)PyArray_FromAny(tmp_arr, indtype, 0, 0,
                                        NPY_ARRAY_FORCECAST, NULL);
                if (arr == NULL) {
                    Py_DECREF(tmp_arr);
                    goto failed_building_indices;
                }
            }
            else {
                /*
                 * These Checks can be removed after deprecation, since
                 * they should then be either correct already or error out
                 * later just like a normal array.
                 */
                if (PyArray_ISBOOL((PyArrayObject *)tmp_arr)) {
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
                        goto failed_building_indices;
                    }
                }
                else if (!PyArray_ISINTEGER((PyArrayObject *)tmp_arr)) {
                    if (PyArray_NDIM(tmp_arr) == 0) {
                        /* match integer deprecation warning */
                        if (DEPRECATE(
                                    "using a non-integer number instead of an "
                                    "integer will result in an error in the "
                                    "future") < 0) {

                            /* The error message raised in the future */
                            PyErr_SetString(PyExc_IndexError,
                                "only integers, slices (`:`), ellipsis (`...`), "
                                "numpy.newaxis (`None`) and integer or boolean "
                                "arrays are valid indices");
                            Py_DECREF(tmp_arr);
                            goto failed_building_indices; 
                        }
                    }
                    else {
                        if (DEPRECATE(
                                    "non integer (and non boolean) array-likes "
                                    "will not be accepted as indices in the "
                                    "future") < 0) {

                            /* Error message to be raised in the future */
                            PyErr_SetString(PyExc_IndexError,
                                "non integer (and non boolean) array-likes will "
                                "not be accepted as indices in the future");
                            Py_DECREF(tmp_arr);
                            goto failed_building_indices;  
                        }
                    }   
                }

                PyArray_Descr *indtype = PyArray_DescrFromType(NPY_INTP);
                arr = (PyArrayObject *)PyArray_FromAny(tmp_arr, indtype, 0, 0,
                                        NPY_ARRAY_FORCECAST, NULL);

                if (arr == NULL) {
                    /* Since this will be removed, handle this later */
                    PyErr_Clear();
                    arr = tmp_arr;
                }
                else {
                    Py_DECREF(tmp_arr);
                }
            }
        }
        else {
            Py_INCREF(obj);
            arr = (PyArrayObject *)obj;
        }

        /* Check if the array is valid and fill the information */
        if PyArray_ISBOOL(arr) {
            /*
             * There are two types of boolean indices (which are equivalent,
             * for the most part though). A single boolean index of matching
             * dimensionality and size is a boolean index.
             * If this is not the case, it is instead expanded into (multiple)
             * integer array indices.
             */
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

                    /* keep track anyway, even if it is trivial */
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
                goto failed_building_indices;
                /*
                 * This can actually be well defined. A new axis is added,
                 * but at the same time no axis is "used". So if we have True,
                 * we add a new axis (a bit like with np.newaxis). If it is
                 * False, we add a new axis, but this axis has 0 entries.
                 */

                indices[curr_idx].type = HAS_0D_BOOL;

                /* TODO: This can't fail, right? Is there a faster way? */
                if (PyObject_IsTrue(arr)) {
                    n = 1;
                }
                else {
                    n = 0;
                }
                indices[curr_idx].object = PyArray_Zeros(1, &n,
                                            PyArray_DescrFromType(NPY_INTP), 0);
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
            n = _nonzero_indices(arr, nonzero_result);
            if (n < 0) {
                goto failed_building_indices;
            }

            /* Check that we will not run out of indices to store new ones */
            if (curr_idx + n >= NPY_MAXDIMS * 2) {
                PyErr_SetString(PyExc_IndexError,
                                "too many indices for array");
                goto failed_building_indices;
            }

            /* Add the arrays from the nonzero result to the index */
            for (i=0; i < n; i++) {
                indices[curr_idx].type = HAS_FANCY;
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
        else if PyArray_ISINTEGER(arr) {
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
        if (arr == obj) {
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
        goto failed_building_indices;
    }

    /*
     * Compare dimension of the index to the real dimension, this is
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
     * then is possible.
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
                         "indexed array has %d",
                         NPY_MAXDIMS, (new_ndim + fancy_ndim));
            goto failed_building_indices;
        }
    }

    *num = curr_idx;
    *ndim = new_ndim + fancy_ndim;

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


/*
 * For a purely integer index, set ptr to the memory address.
 * Returns 0 on success, -1 on failure.
 * The caller must ensure that the index is a full integer
 * one.
 */
static int
get_item_pointer(PyArrayObject *self, char **ptr,
                    npy_index_info *indices, int index_num) {
    int i;
    *ptr = PyArray_BYTES(self);
    for (i=0; i < index_num; i++) {
        if ((check_and_adjust_index(&(indices[i].value),
                               PyArray_DIMS(self)[i], i)) < 0) {
            return -1;
        }
        *ptr += PyArray_STRIDE(self, i) * indices[i].value;
    }
    return 0;
}

/*
 * For any index, get a view of the subspace into the original
 * array. If there are no fancy indices, this is the result of
 * the indexing operation.
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

    npy_intp start, stop, step, n_steps; /* for slice parsing */

    for (i=0; i < index_num; i++) {
        switch (indices[i].type) {
            case HAS_INTEGER:
                if ((check_and_adjust_index(&indices[i].value,
                                    PyArray_DIMS(self)[orig_dim], i)) < 0) {
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
                        PyErr_SetString(PyExc_IndexError,
                                        "invalid slice");
                    }
                    return -1;
                }
                if (n_steps <= 0) {
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
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self), dtype, 1, &size,
                                NULL, NULL, 0, (PyObject *)self);
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

        NpyIter_Deallocate(iter);
        NPY_AUXDATA_FREE(transferdata);
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
array_ass_boolean_subscript(PyArrayObject *self,
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

        NPY_AUXDATA_FREE(transferdata);
        NpyIter_Deallocate(iter);
    }

    return 0;
}


/* make sure subscript always returns an array object */
NPY_NO_EXPORT PyObject *
array_subscript_asarray(PyArrayObject *self, PyObject *op)
{
    return PyArray_EnsureAnyArray(array_subscript(self, op));
}

NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op)
{
    int index_type;
    int index_num;
    int i, ndim, fancy;
    /*
     * Index info array. We can have twice as many indices as dimensions
     * (because of None). The + 1 is to not need to check as much.
     */
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];

    PyArrayObject *view = NULL;
    PyObject *result = NULL;

    /* Check for multiple field access */
    if (PyDataType_HASFIELDS(PyArray_DESCR(self)))
        /* Check for single field access */
        /*
         * TODO: Moving this code block into the HASFIELDS, might have
         *       consequences for strings that can be interpreted as ints?
         */
        if (PyString_Check(op) || PyUnicode_Check(op)) {
            PyObject *temp, *obj;

            if (PyDataType_HASFIELDS(PyArray_DESCR(self))) {
                obj = PyDict_GetItem(PyArray_DESCR(self)->fields, op);
                if (obj != NULL) {
                    PyArray_Descr *descr;
                    int offset;
                    PyObject *title;

                    if (PyArg_ParseTuple(obj, "Oi|O", &descr, &offset, &title)) {
                        Py_INCREF(descr);
                        return PyArray_GetField(self, descr, offset);
                    }
                }
            }

            temp = op;
            if (PyUnicode_Check(op)) {
                temp = PyUnicode_AsUnicodeEscapeString(op);
            }
            PyErr_Format(PyExc_ValueError,
                         "field named %s not found",
                         PyBytes_AsString(temp));
            if (temp != op) {
                Py_DECREF(temp);
            }
            return NULL;
        }

        else if (PySequence_Check(op) && !PyTuple_Check(op)) {
            int seqlen, i;
            PyObject *obj;
            seqlen = PySequence_Size(op);
            for (i = 0; i < seqlen; i++) {
                obj = PySequence_GetItem(op, i);
                if (!PyString_Check(obj) && !PyUnicode_Check(obj)) {
                    Py_DECREF(obj);
                    break;
                }
                Py_DECREF(obj);
            }
            /*
             * Extract multiple fields if all elements in sequence
             * are either string or unicode (i.e. no break occurred).
             */
            fancy = ((seqlen > 0) && (i == seqlen));
            if (fancy) {
                PyObject *_numpy_internal;
                _numpy_internal = PyImport_ImportModule("numpy.core._internal");
                if (_numpy_internal == NULL) {
                    return NULL;
                }
                obj = PyObject_CallMethod(_numpy_internal,
                        "_index_fields", "OO", self, op);
                Py_DECREF(_numpy_internal);
                if (obj == NULL) {
                    return NULL;
                }
                PyArray_ENABLEFLAGS((PyArrayObject*)obj, NPY_ARRAY_WARN_ON_WRITE);
                return obj;
            }
    }

    /* Prepare the indices */
    //printf("preparing indices for getting\n");
    index_type = prepare_index(self, op, indices, &index_num, &ndim, 1);
    //PyObject_Print(op, stdout, 0);
    //printf("getting index_type %d, index_num %d, ndim %d\n", index_type, index_num, ndim);

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

    /* If it is only a single ellipsis, just return self */
    else if (index_type == HAS_ELLIPSIS) {
        /*
         * TODO: Should this be a view or not? The only reason not would be
         *       optimization (i.e. of array[...] += 1) I think.
         *       Before, it was just self for a single Ellipis.
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
            result = PyArray_NewCopy(view, NPY_ANYORDER);
            goto finish_view;
        }
    }

    /* If there is no fancy indexing, we have the result */
    if (!(index_type & HAS_FANCY)) {
        result = (PyObject *)view;
        goto finish;
    }

    /* fancy indexing has to be used. And view is the subspace. */
    /* TODO: Add 1-dim and 1 index special case */
    PyArrayMapIterObject * mit;
    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices,
                                                     index_num, index_type);
    if (mit == NULL) {
        goto finish_view;
    }

    /*
     * Bind mapiter. If no refcheck is needed, do not check the index
     * up front. TODO: otherwise check twice, to avoid having to do error
     * handling
     */
    if (PyArray_MapIterBind(mit, view, self, indices, index_num,
                (!PyDataType_REFCHK(PyArray_DESCR(self)) ? 1 : 0)) < 0) {
        Py_DECREF((PyObject *)mit);
        goto finish_view;
    }

    result = (PyObject *)PyArray_GetMap(mit);
    Py_DECREF(mit);
    if (result == NULL) {
        goto finish_view;
    }

  finish_view:
    Py_XDECREF(view);
    /* Clean up indices */
  finish:
    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }
    return result;
}


static int
array_ass_sub(PyArrayObject *self, PyObject *ind, PyObject *op)
{
    int index_type;
    int index_num;
    int i, ndim;
    PyArrayObject *view = NULL;
    PyArrayObject *tmp_arr = NULL;
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];

    if (op == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "cannot delete array elements");
        return -1;
    }
    if (PyArray_FailUnlessWriteable(self, "assignment destination") < 0) {
        return -1;
    }

    /* Single field access */
    if (PyString_Check(ind) || PyUnicode_Check(ind)) {
        if (PyDataType_HASFIELDS(PyArray_DESCR(self))) {
            PyObject *obj;

            obj = PyDict_GetItem(PyArray_DESCR(self)->fields, ind);
            if (obj != NULL) {
                PyArray_Descr *descr;
                int offset;
                PyObject *title;

                if (PyArg_ParseTuple(obj, "Oi|O", &descr, &offset, &title)) {
                    Py_INCREF(descr);
                    return PyArray_SetField(self, descr, offset, op);
                }
            }
        }
#if defined(NPY_PY3K)
        PyErr_Format(PyExc_ValueError,
                     "field named %S not found",
                     ind);
#else
        PyErr_Format(PyExc_ValueError,
                     "field named %s not found",
                     PyString_AsString(ind));
#endif
        return -1;
    }


    /* Prepare the indices */
    //printf("preparing indices\n");
    index_type = prepare_index(self, ind, indices, &index_num, &ndim, 1);
    //printf("setting index_type %d, index_num %d, ndim %d\n", index_type, index_num, ndim);

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
        if (array_ass_boolean_subscript(self,
                                        (PyArrayObject *)indices[0].object,
                                        tmp_arr, NPY_CORDER) < 0) {
            goto fail;
        }
        goto success;
    }

    /*
     * WARNING: There is a huge special case here. If this is not a
     *          base class array, we have to get the view through its
     *          very own index machinery.
     *          I find this weird, but not sure if there is a way to
     *          deprecate. (why does the class not implement getitem too?)
     */
    else if (!(index_type & HAS_FANCY) && !PyArray_CheckExact(self)) {
        view = PyObject_GetItem((PyObject *)self, ind);
        if (view == NULL) {
            goto fail;
        }
        if (!PyArray_Check(view)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Getitem not returning array");
            goto fail;
        }
    }

    /* Single ellipsis index, no need to create a new view */
    else if (index_type == HAS_ELLIPSIS) {
        if (self == op) {
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
        /*
         * CopyObject handles all weirdness for us, this however also
         * means that other array assignments which convert more strictly
         * do *not* handle all weirdnesses correctly.
         * TODO: To have other assignments handle them correctly, we
         *       should copy into a temporary array of the correct shape
         *       if it is not an array yet!
         * TODO: We could use PyArray_SETITEM if it is 0-d?
         */
        if (PyArray_CopyObject(view, op) < 0) {
            goto fail;
        }
        goto success;
    }

    /* fancy indexing has to be used. And view is the subspace. */
    /* TODO: Add 1-dim and 1 index special case */
    PyArrayMapIterObject * mit;
    mit = (PyArrayMapIterObject *)PyArray_MapIterNew(indices,
                                                     index_num, index_type);
    if (mit == NULL) {
        goto fail;
    }
    if (PyArray_MapIterBind(mit, view, self, indices, index_num, 0) < 0) {
        Py_DECREF((PyObject *)mit);
        goto fail;
    }
    if (PyArray_SetMap(mit, op) < 0) {
        Py_DECREF((PyObject *)mit);
        goto fail;
    }
    Py_DECREF(mit);
    goto success;

    /* Clean up temporary variables and indices */
    fail:
        Py_XDECREF((PyObject *)view);
        Py_XDECREF((PyObject *)tmp_arr);
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


NPY_NO_EXPORT int
array_ass_item(PyArrayObject *self, Py_ssize_t i, PyObject *v)
{
    PyObject * ind = PyLong_FromLong(i); /* Fix this */
    Py_INCREF(ind); /* TODO: Should not be needed! */
    return array_ass_sub(self, ind, v);
}


NPY_NO_EXPORT PyMappingMethods array_as_mapping = {
    (lenfunc)array_length,              /*mp_length*/
    (binaryfunc)array_subscript,        /*mp_subscript*/
    (objobjargproc)array_ass_sub,       /*mp_ass_subscript*/
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
    count = 0;

    /* pre-determine how many nonzero entries there are */
    for (i = 0; i < size; i++) {
        if (*(ptr++)) {
            count++;
        }
    }

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
    ptr = (npy_bool *)PyArray_DATA(ba);
    if (count == 0) {
        goto finish;
    }

    /*
     * Loop through the Boolean array  and copy coordinates
     * for non-zero entries
     */
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
    int i,j;
    npy_intp offset_add;
    mit->index = 0;

    NpyIter_Reset(mit->outer, NULL);
    mit->dataptr = mit->baseoffset;
    mit->itersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);

    for (j = 0; j < mit->numiter; j++) {
        offset_add = *((npy_intp*)mit->iterptrs[j]);
        if (offset_add < 0) {
            offset_add += mit->outer_dims[j];
        }
        mit->dataptr += offset_add * mit->outer_strides[j];
    }
    if (mit->subspace != NULL) {
        PyArray_ITER_RESET(mit->subspace);
        mit->subspace->dataptr = mit->dataptr;
    }

    return;
}


NPY_NO_EXPORT void
mapiter_outernext(PyArrayMapIterObject *mit)
{
    int j;
    npy_intp offset_add;
    mit->dataptr = mit->baseoffset;
    for (j = 0; j < mit->numiter; j++) {
        mit->iterptrs[j] += mit->iterstrides[j];
        offset_add = *((npy_intp*)mit->iterptrs[j]);
        if (offset_add < 0) {
            offset_add += mit->outer_dims[j];
        }
        mit->dataptr += offset_add * mit->outer_strides[j];
    }
    if ((--mit->itersize) == 0) {
        mit->iternext(mit->outer);
        mit->itersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);
    }
}


/*NUMPY_API
 * This function needs to update the state of the map iterator
 * and point mit->dataptr to the memory-location of the next object
 */
NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit)
{
    /* Sub-space iteration */
    if (mit->subspace != NULL) {
        PyArray_ITER_NEXT(mit->subspace);

        if (mit->subspace->index >= mit->subspace->size) {
            PyArray_ITER_RESET(mit->subspace);
            mapiter_outernext(mit);
            mit->subspace->dataptr = mit->dataptr;
        }
        mit->dataptr = mit->subspace->dataptr;
    }
    else {
        mapiter_outernext(mit);
    }
    return;
}

/*
 * Bind a mapiteration to a particular array and subspace.
 * Note that the subspace must be a base class array. Otherwise
 * there might be reshaping applied before the indexing is finished.
 *
 * Need to check for index-errors somewhere.
 *
 * Let's do it at bind time and also convert all <0 values to >0 here
 * as well.
 *
 * delayed_check causes the MapIterBind to not check the indices.
 * Since GetMap always has a fresh output array, it does not need to
 * check the indices beforehand.
 */
NPY_NO_EXPORT int
PyArray_MapIterBind(PyArrayMapIterObject *mit, PyArrayObject *subspace,
                    PyArrayObject *arr, npy_index_info *indices, int index_num,
                    int delayed_index_check)
{
    int subnd;
    int i, j, n, curr_dim, result_dim, consec_status;
    PyArrayIterObject *it;
    npy_intp *indptr;

    npy_intp indval;
    npy_intp *outer_dims = mit->outer_dims;
    int *outer_axis = mit->iteraxes;

    /* Note: ait is probably never really used. */
    mit->ait = (PyArrayIterObject *)PyArray_IterNew((PyObject *)arr);
    if (mit->ait == NULL) {
        return -1;
    }

    if (subspace != NULL) {
        /* Get subspace iterator */
        subnd = PyArray_NDIM(subspace);
        mit->baseoffset = PyArray_BYTES(subspace);
        mit->subspace = (PyArrayIterObject *)PyArray_IterNew((PyObject *)subspace);
        if (mit->subspace == NULL) {
            return -1;
        }

        /* Expand dimensions of result */
        n = PyArray_NDIM(mit->subspace->ao);
        for (i = 0; i < n; i++) {
            mit->dimensions[mit->nd+i] = PyArray_DIMS(mit->subspace->ao)[i];
        }
        mit->nd += n;
    }
    else {
        /*
         * FIXME: This means if subspace is a scalar, we still need to use it.
         *        It is however probably no worse then before when all scalars
         *        were promoted to iterator arrays of the outer iteration.
         */
        subnd = 0;
        mit->subspace == NULL;
        mit->baseoffset = PyArray_BYTES(arr);
    }

    /*
     * Now need to set mit->iteraxes and ..., also need
     * to find the correct mit->consec value.
     */
    j = 0;
    curr_dim = 0;
    result_dim = 0; /* dimension of index result (up to first fancy index) */
    mit->consec = 0;
    consec_status = -1; /* no fancy index yet */
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
            mit->outer_strides[j] = PyArray_STRIDE(arr, curr_dim);
            mit->outer_dims[j] = PyArray_DIM(arr, curr_dim);
            mit->iteraxes[j++] = curr_dim++;
        }
        else if (indices[i].type == HAS_0D_BOOL) {
            mit->outer_strides[j] = 0;
            mit->outer_dims[j] = 1;
            mit->iteraxes[j++] = -1; /* Does not exist */
        }

        /* advance curr_dim for non-fancy indices */
        else if (indices[i].type == HAS_ELLIPSIS) {
            curr_dim += indices[i].value;
            result_dim += indices[i].value;
        }
        else if (indices[i].type != HAS_NEWAXIS){
            curr_dim += 1;
            result_dim += 1;
        }
        else {
            result_dim += 1;
        }
    }

 finish:
    /* Here check the indexes (now that we have iteraxes) */
    mit->size = PyArray_OverflowMultiplyList(mit->dimensions, mit->nd);
    if (mit->size < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "dimensions too large in fancy indexing");
        return -1;
    }
    if (mit->ait->size == 0 && mit->size != 0) {
        PyErr_SetString(PyExc_IndexError,
                        "invalid index into a 0-size array");
        return -1;
    }

    if (delayed_index_check) {
        return 0;
    }

    PyArray_MapIterReset(mit); /* TODO: can probably remove this */

    if (NpyIter_GetIterSize(mit->outer) > 0) {
        if (mit->numiter == 1) {
            do {
                mit->itersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);
                while (mit->itersize--) {
                    indval = *((npy_intp*)mit->iterptrs[0]);
                    if (check_and_adjust_index(&indval,
                                *(outer_dims), *(outer_axis)) < 0) {
                        return -1;
                    }
                    mit->iterptrs[0] += mit->iterstrides[0];
                }
            } while (mit->iternext(mit->outer));
        }
        else {
            do {
                mit->itersize = *NpyIter_GetInnerLoopSizePtr(mit->outer);
                while (mit->itersize--) {
                    for (j=0; j<mit->numiter; j++) {
                        indval = *((npy_intp*)mit->iterptrs[j]);
                        if (check_and_adjust_index(&indval,
                                    *(outer_dims++), *(outer_axis++)) < 0) {
                            return -1;
                        }
                        mit->iterptrs[j] += mit->iterstrides[j];
                    }
                    outer_dims -= mit->numiter;
                    outer_axis -= mit->numiter;
                }
            } while (mit->iternext(mit->outer));
        }
    }
    return 0;
}


NPY_NO_EXPORT PyObject *
PyArray_MapIterNew(npy_index_info *indices , int index_num, int index_type)
{
    PyArrayObject *index_arrays[NPY_MAXDIMS];
    PyArray_Descr *dtypes[NPY_MAXDIMS];
    npy_uint32 op_flags[NPY_MAXDIMS];
    PyArrayMapIterObject *mit;
    int i, dummy_array=0;

    /* create new MapIter object */
    mit = (PyArrayMapIterObject *)PyArray_malloc(sizeof(PyArrayMapIterObject));
    /* set all attributes of mapiter to zero */
    memset(mit, 0, sizeof(PyArrayMapIterObject));
    PyObject_Init((PyObject *)mit, &PyArrayMapIter_Type);
    if (mit == NULL) {
        return NULL;
    }

    for (i=0; i < index_num; i++) {
        if (indices[i].type & HAS_FANCY) {
            index_arrays[mit->numiter] = (PyArrayObject *)indices[i].object;
            dtypes[mit->numiter] = PyArray_DescrFromType(NPY_INTP);

            // | NPY_ITER_COPY; Copying seems quite a bit faster then buffering...
            op_flags[mit->numiter] = NPY_ITER_NBO | NPY_ITER_ALIGNED | NPY_ITER_READONLY;
            mit->numiter += 1;
        }
    }

    if (mit->numiter == 0) {
        /*
         * For MapIterArray, it is possible that there is no fancy index.
         * to support this case, add a little cludgy a dummy iterator.
         * Since it is 0-d its transpose, etc. does not matter.
         */
        dummy_array = 1; /* signal necessity to decref... */
        index_arrays[0] = PyArray_Zeros(0, NULL,
                                        PyArray_DescrFromType(NPY_INTP), 0);
        if (index_arrays[0] == NULL) {
            return NULL;
        }
        dtypes[0] = PyArray_DescrFromType(NPY_INTP);
        op_flags[0] = NPY_ITER_NBO | NPY_ITER_ALIGNED | NPY_ITER_READONLY;

        mit->outer_strides[0] = 0;
        mit->outer_dims[0] = 1;
        mit->numiter = 1;
    }

    //printf("Creating MultiNew\n");
    mit->outer = NpyIter_MultiNew(mit->numiter,
                                  index_arrays,
                                  NPY_ITER_ZEROSIZE_OK |
                                  NPY_ITER_MULTI_INDEX | /* To force shape */
                                  NPY_ITER_BUFFERED,
                                  NPY_CORDER,
                                  NPY_SAME_KIND_CASTING,
                                  op_flags,
                                  dtypes);

    if (dummy_array) {
        Py_DECREF(index_arrays[0]);
    }

    if (mit->outer == NULL) {
        //printf("Allocating multiNew failed!\n");
        goto fail;
    }

    /* I doubt this can really fail, but should add error checking... */
    NpyIter_GetShape(mit->outer, mit->dimensions);
    mit->nd = NpyIter_GetNDim(mit->outer);
    mit->nd_fancy = mit->nd;
    NpyIter_RemoveMultiIndex(mit->outer);
    NpyIter_EnableExternalLoop(mit->outer);
    NpyIter_Reset(mit->outer, NULL);

    mit->iternext = NpyIter_GetIterNext(mit->outer, NULL);
    if (mit->iternext == NULL) {
        goto fail;
    }
    mit->iterptrs = NpyIter_GetDataPtrArray(mit->outer);
    mit->iterstrides = NpyIter_GetInnerStrideArray(mit->outer);

    //printf("Done mapiternew\n");
    return (PyObject *)mit;

 fail:
    Py_DECREF(mit);
    return NULL;
}

/*NUMPY_API
*/
NPY_NO_EXPORT PyObject *
PyArray_MapIterArray(PyArrayObject * a, PyObject * index)
{
    PyArrayMapIterObject * mit = NULL;
    PyArrayObject *subspace = NULL;
    npy_index_info indices[NPY_MAXDIMS * 2 + 1];
    int i, index_num, ndim, index_type;

    index_type = prepare_index(a, index, indices, &index_num, &ndim, 0);

    if (index_type < 0) {
        return NULL;
    }

    mit = (PyArrayMapIterObject *) PyArray_MapIterNew(indices, index_num,
                                                            index_type);
    if (mit == NULL) {
        goto fail;
    }

    /* If it is not a pure fancy index, need to get the subspace */
    if (index_type != HAS_FANCY) {
        if (get_view_from_index(a, &subspace, indices, index_num, 1) < 0) {
            Py_DECREF((PyObject *)mit);
            goto fail;
        }
    }

    if (PyArray_MapIterBind(mit, subspace, a, indices, index_num, 0) < 0) {
        Py_XDECREF(subspace);
        Py_DECREF((PyObject *)mit);
        goto fail;
    }
    Py_XDECREF(subspace);
    PyArray_MapIterReset(mit);

    for (i=0; i < index_num; i++) {
        Py_XDECREF(indices[i].object);
    }

    return (PyObject *)mit;

 fail:
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

#undef SET_MIT_DATAPTR_CHECK_INDEX
#undef SET_MIT_DATAPTR_0_NUMITER_CHECK_INDEX

static void
arraymapiter_dealloc(PyArrayMapIterObject *mit)
{
    int i;
    Py_XDECREF(mit->ait);
    Py_XDECREF(mit->subspace);
    if (mit->outer != NULL) {
        NpyIter_Deallocate(mit->outer);
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
 * It's not very useful anyway, since mapiter(indexobj); mapiter.bind(a);
 * mapiter is equivalent to a[indexobj].flat but the latter gets to use
 * slice syntax. (indexobj has been removed from it)
 */
NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.mapiter",                            /* tp_name */
    sizeof(PyArrayIterObject),                  /* tp_basicsize */
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

/** END of Subscript Iterator **/
