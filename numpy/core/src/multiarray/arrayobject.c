/*
  Provide multidimensional arrays as a basic object type in python.

  Based on Original Numeric implementation
  Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu

  with contributions from many Numeric Python developers 1995-2004

  Heavily modified in 2005 with inspiration from Numarray

  by

  Travis Oliphant,  oliphant@ee.byu.edu
  Brigham Young University


maintainer email:  oliphant.travis@ieee.org

  Numarray design (which provided guidance) by
  Space Science Telescope Institute
  (J. Todd Miller, Perry Greenfield, Rick White)
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "npy_config.h"

#include "npy_pycompat.h"

#include "common.h"

#include "number.h"
#include "usertypes.h"
#include "arraytypes.h"
#include "scalartypes.h"
#include "arrayobject.h"
#include "convert_datatype.h"
#include "conversion_utils.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "methods.h"
#include "descriptor.h"
#include "iterators.h"
#include "mapping.h"
#include "getset.h"
#include "sequence.h"
#include "npy_buffer.h"
#include "array_assign.h"
#include "alloc.h"
#include "mem_overlap.h"
#include "numpyos.h"
#include "refcount.h"
#include "strfuncs.h"

#include "binop_override.h"
#include "array_coercion.h"


NPY_NO_EXPORT npy_bool numpy_warn_if_no_mem_policy = 0;

/*NUMPY_API
  Compute the size of an array (in number of items)
*/
NPY_NO_EXPORT npy_intp
PyArray_Size(PyObject *op)
{
    if (PyArray_Check(op)) {
        return PyArray_SIZE((PyArrayObject *)op);
    }
    else {
        return 0;
    }
}

/*NUMPY_API */
NPY_NO_EXPORT int
PyArray_SetUpdateIfCopyBase(PyArrayObject *arr, PyArrayObject *base)
{
    /* 2021-Dec-15 1.23*/
    PyErr_SetString(PyExc_RuntimeError,
        "PyArray_SetUpdateIfCopyBase is disabled, use "
        "PyArray_SetWritebackIfCopyBase instead, and be sure to call "
        "PyArray_ResolveWritebackIfCopy before the array is deallocated, "
        "i.e. before the last call to Py_DECREF. If cleaning up from an "
        "error, PyArray_DiscardWritebackIfCopy may be called instead to "
        "throw away the scratch buffer.");
    return -1;
}

/*NUMPY_API
 *
 * Precondition: 'arr' is a copy of 'base' (though possibly with different
 * strides, ordering, etc.). This function sets the WRITEBACKIFCOPY flag and the
 * ->base pointer on 'arr', call PyArray_ResolveWritebackIfCopy to copy any
 * changes back to 'base' before deallocating the array.
 *
 * Steals a reference to 'base'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_SetWritebackIfCopyBase(PyArrayObject *arr, PyArrayObject *base)
{
    if (base == NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot WRITEBACKIFCOPY to NULL array");
        return -1;
    }
    if (PyArray_BASE(arr) != NULL) {
        PyErr_SetString(PyExc_ValueError,
                  "Cannot set array with existing base to WRITEBACKIFCOPY");
        goto fail;
    }
    if (PyArray_FailUnlessWriteable(base, "WRITEBACKIFCOPY base") < 0) {
        goto fail;
    }

    /*
     * Any writes to 'arr' will magically turn into writes to 'base', so we
     * should warn if necessary.
     */
    if (PyArray_FLAGS(base) & NPY_ARRAY_WARN_ON_WRITE) {
        PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
    }

    /*
     * Unlike PyArray_SetBaseObject, we do not compress the chain of base
     * references.
     */
    ((PyArrayObject_fields *)arr)->base = (PyObject *)base;
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WRITEBACKIFCOPY);
    PyArray_CLEARFLAGS(base, NPY_ARRAY_WRITEABLE);

    return 0;

  fail:
    Py_DECREF(base);
    return -1;
}

/*NUMPY_API
 * Sets the 'base' attribute of the array. This steals a reference
 * to 'obj'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_SetBaseObject(PyArrayObject *arr, PyObject *obj)
{
    if (obj == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency to NULL after initialization");
        return -1;
    }
    /*
     * Allow the base to be set only once. Once the object which
     * owns the data is set, it doesn't make sense to change it.
     */
    if (PyArray_BASE(arr) != NULL) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot set the NumPy array 'base' "
                "dependency more than once");
        return -1;
    }

    /*
     * Don't allow infinite chains of views, always set the base
     * to the first owner of the data.
     * That is, either the first object which isn't an array,
     * or the first object which owns its own data.
     */

    while (PyArray_Check(obj) && (PyObject *)arr != obj) {
        PyArrayObject *obj_arr = (PyArrayObject *)obj;
        PyObject *tmp;

        /* Propagate WARN_ON_WRITE through views. */
        if (PyArray_FLAGS(obj_arr) & NPY_ARRAY_WARN_ON_WRITE) {
            PyArray_ENABLEFLAGS(arr, NPY_ARRAY_WARN_ON_WRITE);
        }

        /* If this array owns its own data, stop collapsing */
        if (PyArray_CHKFLAGS(obj_arr, NPY_ARRAY_OWNDATA)) {
            break;
        }

        tmp = PyArray_BASE(obj_arr);
        /* If there's no base, stop collapsing */
        if (tmp == NULL) {
            break;
        }
        /* Stop the collapse new base when the would not be of the same
         * type (i.e. different subclass).
         */
        if (Py_TYPE(tmp) != Py_TYPE(arr)) {
            break;
        }


        Py_INCREF(tmp);
        Py_DECREF(obj);
        obj = tmp;
    }

    /* Disallow circular references */
    if ((PyObject *)arr == obj) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError,
                "Cannot create a circular NumPy array 'base' dependency");
        return -1;
    }

    ((PyArrayObject_fields *)arr)->base = obj;

    return 0;
}


/**
 * Assign an arbitrary object a NumPy array. This is largely basically
 * identical to PyArray_FromAny, but assigns directly to the output array.
 *
 * @param dest Array to be written to
 * @param src_object Object to be assigned, array-coercion rules apply.
 * @return 0 on success -1 on failures.
 */
/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_CopyObject(PyArrayObject *dest, PyObject *src_object)
{
    int ret = 0;
    PyArrayObject *view;
    PyArray_Descr *dtype = NULL;
    int ndim;
    npy_intp dims[NPY_MAXDIMS];
    coercion_cache_obj *cache = NULL;

    /*
     * We have to set the maximum number of dimensions here to support
     * sequences within object arrays.
     */
    ndim = PyArray_DiscoverDTypeAndShape(src_object,
            PyArray_NDIM(dest), dims, &cache,
            NPY_DTYPE(PyArray_DESCR(dest)), PyArray_DESCR(dest), &dtype, 0);
    if (ndim < 0) {
        return -1;
    }

    if (cache != NULL && !(cache->sequence)) {
        /* The input is an array or array object, so assign directly */
        assert(cache->converted_obj == src_object);
        view = (PyArrayObject *)cache->arr_or_sequence;
        Py_DECREF(dtype);
        ret = PyArray_AssignArray(dest, view, NULL, NPY_UNSAFE_CASTING);
        npy_free_coercion_cache(cache);
        return ret;
    }

    /*
     * We may need to broadcast, due to shape mismatches, in this case
     * create a temporary array first, and assign that after filling
     * it from the sequences/scalar.
     */
    if (ndim != PyArray_NDIM(dest) ||
            !PyArray_CompareLists(PyArray_DIMS(dest), dims, ndim)) {
        /*
         * Broadcasting may be necessary, so assign to a view first.
         * This branch could lead to a shape mismatch error later.
         */
        assert (ndim <= PyArray_NDIM(dest));  /* would error during discovery */
        view = (PyArrayObject *) PyArray_NewFromDescr(
                &PyArray_Type, dtype, ndim, dims, NULL, NULL,
                PyArray_FLAGS(dest) & NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (view == NULL) {
            npy_free_coercion_cache(cache);
            return -1;
        }
    }
    else {
        Py_DECREF(dtype);
        view = dest;
    }

    /* Assign the values to `view` (whichever array that is) */
    if (cache == NULL) {
        /* single (non-array) item, assign immediately */
        if (PyArray_Pack(
                PyArray_DESCR(view), PyArray_DATA(view), src_object) < 0) {
            goto fail;
        }
    }
    else {
        if (PyArray_AssignFromCache(view, cache) < 0) {
            goto fail;
        }
    }
    if (view == dest) {
        return 0;
    }
    ret = PyArray_AssignArray(dest, view, NULL, NPY_UNSAFE_CASTING);
    Py_DECREF(view);
    return ret;

  fail:
    if (view != dest) {
        Py_DECREF(view);
    }
    return -1;
}


/* returns an Array-Scalar Object of the type of arr
   from the given pointer to memory -- main Scalar creation function
   default new method calls this.
*/

/* Ideally, here the descriptor would contain all the information needed.
   So, that we simply need the data and the descriptor, and perhaps
   a flag
*/


/*
  Given a string return the type-number for
  the data-type with that string as the type-object name.
  Returns NPY_NOTYPE without setting an error if no type can be
  found.  Only works for user-defined data-types.
*/

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_TypeNumFromName(char const *str)
{
    int i;
    PyArray_Descr *descr;

    for (i = 0; i < NPY_NUMUSERTYPES; i++) {
        descr = userdescrs[i];
        if (strcmp(descr->typeobj->tp_name, str) == 0) {
            return descr->type_num;
        }
    }
    return NPY_NOTYPE;
}

/*NUMPY_API
 *
 * If WRITEBACKIFCOPY and self has data, reset the base WRITEABLE flag,
 * copy the local data to base, release the local data, and set flags
 * appropriately. Return 0 if not relevant, 1 if success, < 0 on failure
 */
NPY_NO_EXPORT int
PyArray_ResolveWritebackIfCopy(PyArrayObject * self)
{
    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;
    if (fa && fa->base) {
        if (fa->flags & NPY_ARRAY_WRITEBACKIFCOPY) {
            /*
             * WRITEBACKIFCOPY means that fa->base's data
             * should be updated with the contents
             * of self.
             * fa->base->flags is not WRITEABLE to protect the relationship
             * unlock it.
             */
            int retval = 0;
            PyArray_ENABLEFLAGS(((PyArrayObject *)fa->base),
                                                    NPY_ARRAY_WRITEABLE);
            PyArray_CLEARFLAGS(self, NPY_ARRAY_WRITEBACKIFCOPY);
            retval = PyArray_CopyAnyInto((PyArrayObject *)fa->base, self);
            Py_DECREF(fa->base);
            fa->base = NULL;
            if (retval < 0) {
                /* this should never happen, how did the two copies of data
                 * get out of sync?
                 */
                return retval;
            }
            return 1;
        }
    }
    return 0;
}

/*********************** end C-API functions **********************/


/* dealloc must not raise an error, best effort try to write
   to stderr and clear the error
*/

static inline void
WARN_IN_DEALLOC(PyObject* warning, const char * msg) {
    if (PyErr_WarnEx(warning, msg, 1) < 0) {
        PyObject * s;

        s = PyUnicode_FromString("array_dealloc");
        if (s) {
            PyErr_WriteUnraisable(s);
            Py_DECREF(s);
        }
        else {
            PyErr_WriteUnraisable(Py_None);
        }
    }
}

/* array object functions */

static void
array_dealloc(PyArrayObject *self)
{
    PyArrayObject_fields *fa = (PyArrayObject_fields *)self;

    if (_buffer_info_free(fa->_buffer_info, (PyObject *)self) < 0) {
        PyErr_WriteUnraisable(NULL);
    }

    if (fa->weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    if (fa->base) {
        int retval;
        if (PyArray_FLAGS(self) & NPY_ARRAY_WRITEBACKIFCOPY)
        {
            char const * msg = "WRITEBACKIFCOPY detected in array_dealloc. "
                " Required call to PyArray_ResolveWritebackIfCopy or "
                "PyArray_DiscardWritebackIfCopy is missing.";
            /*
             * prevent reaching 0 twice and thus recursing into dealloc.
             * Increasing sys.gettotalrefcount, but path should not be taken.
             */
            Py_INCREF(self);
            WARN_IN_DEALLOC(PyExc_RuntimeWarning, msg);
            retval = PyArray_ResolveWritebackIfCopy(self);
            if (retval < 0)
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        /*
         * If fa->base is non-NULL, it is something
         * to DECREF -- either a view or a buffer object
         */
        Py_XDECREF(fa->base);
    }

    if ((fa->flags & NPY_ARRAY_OWNDATA) && fa->data) {
        /* Free any internal references */
        if (PyDataType_REFCHK(fa->descr)) {
            if (PyArray_ClearArray(self) < 0) {
                PyErr_WriteUnraisable(NULL);
            }
        }
        if (fa->mem_handler == NULL) {
            if (numpy_warn_if_no_mem_policy) {
                char const *msg = "Trying to dealloc data, but a memory policy "
                    "is not set. If you take ownership of the data, you must "
                    "set a base owning the data (e.g. a PyCapsule).";
                WARN_IN_DEALLOC(PyExc_RuntimeWarning, msg);
            }
            // Guess at malloc/free ???
            free(fa->data);
        }
        else {
            size_t nbytes = PyArray_NBYTES(self);
            if (nbytes == 0) {
                nbytes = 1;
            }
            PyDataMem_UserFREE(fa->data, nbytes, fa->mem_handler);
            Py_DECREF(fa->mem_handler);
        }
    }

    /* must match allocation in PyArray_NewFromDescr */
    npy_free_cache_dim(fa->dimensions, 2 * fa->nd);
    Py_DECREF(fa->descr);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/*NUMPY_API
 * Prints the raw data of the ndarray in a form useful for debugging
 * low-level C issues.
 */
NPY_NO_EXPORT void
PyArray_DebugPrint(PyArrayObject *obj)
{
    int i;
    PyArrayObject_fields *fobj = (PyArrayObject_fields *)obj;

    printf("-------------------------------------------------------\n");
    printf(" Dump of NumPy ndarray at address %p\n", obj);
    if (obj == NULL) {
        printf(" It's NULL!\n");
        printf("-------------------------------------------------------\n");
        fflush(stdout);
        return;
    }
    printf(" ndim   : %d\n", fobj->nd);
    printf(" shape  :");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %" NPY_INTP_FMT, fobj->dimensions[i]);
    }
    printf("\n");

    printf(" dtype  : ");
    PyObject_Print((PyObject *)fobj->descr, stdout, 0);
    printf("\n");
    printf(" data   : %p\n", fobj->data);
    printf(" strides:");
    for (i = 0; i < fobj->nd; ++i) {
        printf(" %" NPY_INTP_FMT, fobj->strides[i]);
    }
    printf("\n");

    printf(" base   : %p\n", fobj->base);

    printf(" flags :");
    if (fobj->flags & NPY_ARRAY_C_CONTIGUOUS)
        printf(" NPY_C_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_F_CONTIGUOUS)
        printf(" NPY_F_CONTIGUOUS");
    if (fobj->flags & NPY_ARRAY_OWNDATA)
        printf(" NPY_OWNDATA");
    if (fobj->flags & NPY_ARRAY_ALIGNED)
        printf(" NPY_ALIGNED");
    if (fobj->flags & NPY_ARRAY_WRITEABLE)
        printf(" NPY_WRITEABLE");
    if (fobj->flags & NPY_ARRAY_WRITEBACKIFCOPY)
        printf(" NPY_WRITEBACKIFCOPY");
    printf("\n");

    if (fobj->base != NULL && PyArray_Check(fobj->base)) {
        printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
        printf("Dump of array's BASE:\n");
        PyArray_DebugPrint((PyArrayObject *)fobj->base);
        printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    }
    printf("-------------------------------------------------------\n");
    fflush(stdout);
}


/*NUMPY_API
 * This function is scheduled to be removed
 *
 * TO BE REMOVED - NOT USED INTERNALLY.
 */
NPY_NO_EXPORT void
PyArray_SetDatetimeParseFunction(PyObject *NPY_UNUSED(op))
{
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_CompareUCS4(npy_ucs4 const *s1, npy_ucs4 const *s2, size_t len)
{
    npy_ucs4 c1, c2;
    while(len-- > 0) {
        c1 = *s1++;
        c2 = *s2++;
        if (c1 != c2) {
            return (c1 < c2) ? -1 : 1;
        }
    }
    return 0;
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_CompareString(const char *s1, const char *s2, size_t len)
{
    const unsigned char *c1 = (unsigned char *)s1;
    const unsigned char *c2 = (unsigned char *)s2;
    size_t i;

    for(i = 0; i < len; ++i) {
        if (c1[i] != c2[i]) {
            return (c1[i] > c2[i]) ? 1 : -1;
        }
    }
    return 0;
}


/* Call this from contexts where an array might be written to, but we have no
 * way to tell. (E.g., when converting to a read-write buffer.)
 */
NPY_NO_EXPORT int
array_might_be_written(PyArrayObject *obj)
{
    const char *msg =
        "Numpy has detected that you (may be) writing to an array with\n"
        "overlapping memory from np.broadcast_arrays. If this is intentional\n"
        "set the WRITEABLE flag True or make a copy immediately before writing.";
    if (PyArray_FLAGS(obj) & NPY_ARRAY_WARN_ON_WRITE) {
        if (DEPRECATE(msg) < 0) {
            return -1;
        }
        /* Only warn once per array */
        while (1) {
            PyArray_CLEARFLAGS(obj, NPY_ARRAY_WARN_ON_WRITE);
            if (!PyArray_BASE(obj) || !PyArray_Check(PyArray_BASE(obj))) {
                break;
            }
            obj = (PyArrayObject *)PyArray_BASE(obj);
        }
    }
    return 0;
}

/*NUMPY_API
 *
 *  This function does nothing and returns 0 if *obj* is writeable.
 *  It raises an exception and returns -1 if *obj* is not writeable.
 *  It may also do other house-keeping, such as issuing warnings on
 *  arrays which are transitioning to become views. Always call this
 *  function at some point before writing to an array.
 *
 *  *name* is a name for the array, used to give better error messages.
 *  It can be something like "assignment destination", "output array",
 *  or even just "array".
 */
NPY_NO_EXPORT int
PyArray_FailUnlessWriteable(PyArrayObject *obj, const char *name)
{
    if (!PyArray_ISWRITEABLE(obj)) {
        PyErr_Format(PyExc_ValueError, "%s is read-only", name);
        return -1;
    }
    if (array_might_be_written(obj) < 0) {
        return -1;
    }
    return 0;
}


/* From umath/string_ufuncs.cpp/h */
NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip);

/*
 * VOID-type arrays can only be compared equal and not-equal
 * in which case the fields are all compared by extracting the fields
 * and testing one at a time...
 * equality testing is performed using logical_ands on all the fields.
 * in-equality testing is performed using logical_ors on all the fields.
 *
 * VOID-type arrays without fields are compared for equality by comparing their
 * memory at each location directly (using string-code).
 */
static PyObject *
_void_compare(PyArrayObject *self, PyArrayObject *other, int cmp_op)
{
    if (!(cmp_op == Py_EQ || cmp_op == Py_NE)) {
        PyErr_SetString(PyExc_TypeError,
                "Void-arrays can only be compared for equality.");
        return NULL;
    }
    if (PyArray_TYPE(other) != NPY_VOID) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot compare structured or void to non-void arrays.");
        return NULL;
    }
    if (PyArray_HASFIELDS(self) && PyArray_HASFIELDS(other)) {
        PyArray_Descr *self_descr = PyArray_DESCR(self);
        PyArray_Descr *other_descr = PyArray_DESCR(other);

        /* Use promotion to decide whether the comparison is valid */
        PyArray_Descr *promoted = PyArray_PromoteTypes(self_descr, other_descr);
        if (promoted == NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "Cannot compare structured arrays unless they have a "
                    "common dtype.  I.e. `np.result_type(arr1, arr2)` must "
                    "be defined.");
            return NULL;
        }
        Py_DECREF(promoted);

        npy_intp result_ndim = PyArray_NDIM(self) > PyArray_NDIM(other) ?
                            PyArray_NDIM(self) : PyArray_NDIM(other);

        int field_count = PyTuple_GET_SIZE(self_descr->names);
        if (field_count != PyTuple_GET_SIZE(other_descr->names)) {
            PyErr_SetString(PyExc_TypeError,
                    "Cannot compare structured dtypes with different number of "
                    "fields.  (unreachable error please report to NumPy devs)");
            return NULL;
        }

        PyObject *op = (cmp_op == Py_EQ ? n_ops.logical_and : n_ops.logical_or);
        PyObject *res = NULL;
        for (int i = 0; i < field_count; ++i) {
            PyObject *fieldname, *temp, *temp2;

            fieldname = PyTuple_GET_ITEM(self_descr->names, i);
            PyArrayObject *a = (PyArrayObject *)array_subscript_asarray(
                    self, fieldname);
            if (a == NULL) {
                Py_XDECREF(res);
                return NULL;
            }
            fieldname = PyTuple_GET_ITEM(other_descr->names, i);
            PyArrayObject *b = (PyArrayObject *)array_subscript_asarray(
                    other, fieldname);
            if (b == NULL) {
                Py_XDECREF(res);
                Py_DECREF(a);
                return NULL;
            }
            /*
             * If the fields were subarrays, the dimensions may have changed.
             * In that case, the new shape (subarray part) must match exactly.
             * (If this is 0, there is no subarray.)
             */
            int field_dims_a = PyArray_NDIM(a) - PyArray_NDIM(self);
            int field_dims_b = PyArray_NDIM(b) - PyArray_NDIM(other);
            if (field_dims_a != field_dims_b || (
                    field_dims_a != 0 &&  /* neither is subarray */
                    /* Compare only the added (subarray) dimensions: */
                    !PyArray_CompareLists(
                            PyArray_DIMS(a) + PyArray_NDIM(self),
                            PyArray_DIMS(b) + PyArray_NDIM(other),
                            field_dims_a))) {
                PyErr_SetString(PyExc_TypeError,
                        "Cannot compare subarrays with different shapes. "
                        "(unreachable error, please report to NumPy devs.)");
                Py_DECREF(a);
                Py_DECREF(b);
                Py_XDECREF(res);
                return NULL;
            }

            temp = array_richcompare(a, (PyObject *)b, cmp_op);
            Py_DECREF(a);
            Py_DECREF(b);
            if (temp == NULL) {
                Py_XDECREF(res);
                return NULL;
            }

            /*
             * If the field type has a non-trivial shape, additional
             * dimensions will have been appended to `a` and `b`.
             * In that case, reduce them using `op`.
             */
            if (PyArray_Check(temp) &&
                        PyArray_NDIM((PyArrayObject *)temp) > result_ndim) {
                /* If the type was multidimensional, collapse that part to 1-D
                 */
                if (PyArray_NDIM((PyArrayObject *)temp) != result_ndim+1) {
                    npy_intp dimensions[NPY_MAXDIMS];
                    PyArray_Dims newdims;

                    newdims.ptr = dimensions;
                    newdims.len = result_ndim+1;
                    if (result_ndim) {
                        memcpy(dimensions, PyArray_DIMS((PyArrayObject *)temp),
                               sizeof(npy_intp)*result_ndim);
                    }

                    /*
                     * Compute the new dimension size manually, as reshaping
                     * with -1 does not work on empty arrays.
                     */
                    dimensions[result_ndim] = PyArray_MultiplyList(
                        PyArray_DIMS((PyArrayObject *)temp) + result_ndim,
                        PyArray_NDIM((PyArrayObject *)temp) - result_ndim);

                    temp2 = PyArray_Newshape((PyArrayObject *)temp,
                                             &newdims, NPY_ANYORDER);
                    if (temp2 == NULL) {
                        Py_DECREF(temp);
                        Py_XDECREF(res);
                        return NULL;
                    }
                    Py_DECREF(temp);
                    temp = temp2;
                }
                /* Reduce the extra dimension of `temp` using `op` */
                temp2 = PyArray_GenericReduceFunction((PyArrayObject *)temp,
                                                      op, result_ndim,
                                                      NPY_BOOL, NULL);
                if (temp2 == NULL) {
                    Py_DECREF(temp);
                    Py_XDECREF(res);
                    return NULL;
                }
                Py_DECREF(temp);
                temp = temp2;
            }

            if (res == NULL) {
                res = temp;
            }
            else {
                temp2 = PyObject_CallFunction(op, "OO", res, temp);
                Py_DECREF(temp);
                Py_DECREF(res);
                if (temp2 == NULL) {
                    return NULL;
                }
                res = temp2;
            }
        }
        if (res == NULL && !PyErr_Occurred()) {
            /* these dtypes had no fields. Use a MultiIter to broadcast them
             * to an output array, and fill with True (for EQ)*/
            PyArrayMultiIterObject *mit = (PyArrayMultiIterObject *)
                                          PyArray_MultiIterNew(2, self, other);
            if (mit == NULL) {
                return NULL;
            }

            res = PyArray_NewFromDescr(&PyArray_Type,
                                       PyArray_DescrFromType(NPY_BOOL),
                                       mit->nd, mit->dimensions,
                                       NULL, NULL, 0, NULL);
            Py_DECREF(mit);
            if (res) {
                 PyArray_FILLWBYTE((PyArrayObject *)res,
                                   cmp_op == Py_EQ ? 1 : 0);
            }
        }
        return res;
    }
    else if (PyArray_HASFIELDS(self) || PyArray_HASFIELDS(other)) {
        PyErr_SetString(PyExc_TypeError,
                "Cannot compare structured with unstructured void arrays. "
                "(unreachable error, please report to NumPy devs.)");
        return NULL;
    }
    else {
        /*
         * Since arrays absorb subarray descriptors, this path can only be
         * reached when both arrays have unstructured voids "V<len>" dtypes.
         */
        if (PyArray_ITEMSIZE(self) != PyArray_ITEMSIZE(other)) {
            PyErr_SetString(PyExc_TypeError,
                    "cannot compare unstructured voids of different length. "
                    "Use bytes to compare. "
                    "(This may return array of False in the future.)");
            return NULL;
        }
        /* compare as a string. Assumes self and other have same descr->type */
        return _umath_strings_richcompare(self, other, cmp_op, 0);
    }
}

/*
 * Silence the current error and emit a deprecation warning instead.
 *
 * If warnings are raised as errors, this sets the warning __cause__ to the
 * silenced error.
 */
NPY_NO_EXPORT int
DEPRECATE_silence_error(const char *msg) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    if (DEPRECATE(msg) < 0) {
        npy_PyErr_ChainExceptionsCause(exc, val, tb);
        return -1;
    }
    Py_XDECREF(exc);
    Py_XDECREF(val);
    Py_XDECREF(tb);
    return 0;
}


NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op)
{
    PyArrayObject *array_other;
    PyObject *obj_self = (PyObject *)self;
    PyObject *result = NULL;

    switch (cmp_op) {
    case Py_LT:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.less);
        break;
    case Py_LE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.less_equal);
        break;
    case Py_EQ:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        /*
         * The ufunc does not support void/structured types, so these
         * need to be handled specifically. Only a few cases are supported.
         */

        if (PyArray_TYPE(self) == NPY_VOID) {
            array_other = (PyArrayObject *)PyArray_FROM_O(other);
            /*
             * If not successful, indicate that the items cannot be compared
             * this way.
             */
            if (array_other == NULL) {
                /* 2015-05-07, 1.10 */
                if (DEPRECATE_silence_error(
                        "elementwise == comparison failed and returning scalar "
                        "instead; this will raise an error in the future.") < 0) {
                    return NULL;
                }
                Py_INCREF(Py_NotImplemented);
                return Py_NotImplemented;
            }

            result = _void_compare(self, array_other, cmp_op);
            Py_DECREF(array_other);
            return result;
        }

        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, (PyObject *)other, n_ops.equal);
        break;
    case Py_NE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        /*
         * The ufunc does not support void/structured types, so these
         * need to be handled specifically. Only a few cases are supported.
         */

        if (PyArray_TYPE(self) == NPY_VOID) {
            array_other = (PyArrayObject *)PyArray_FROM_O(other);
            /*
             * If not successful, indicate that the items cannot be compared
             * this way.
            */
            if (array_other == NULL) {
                /* 2015-05-07, 1.10 */
                if (DEPRECATE_silence_error(
                        "elementwise != comparison failed and returning scalar "
                        "instead; this will raise an error in the future.") < 0) {
                    return NULL;
                }
                Py_INCREF(Py_NotImplemented);
                return Py_NotImplemented;
            }

            result = _void_compare(self, array_other, cmp_op);
            Py_DECREF(array_other);
            return result;
        }

        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, (PyObject *)other, n_ops.not_equal);
        break;
    case Py_GT:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.greater);
        break;
    case Py_GE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(
                (PyObject *)self, other, n_ops.greater_equal);
        break;
    default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    /*
     * At this point `self` can take control of the operation by converting
     * `other` to an array (it would have a chance to take control).
     * If we are not in `==` and `!=`, this is an error and we hope that
     * the existing error makes sense and derives from `TypeError` (which
     * python would raise for `NotImplemented`) when it should.
     *
     * However, if the issue is no matching loop for the given dtypes and
     * we are inside == and !=, then returning an array of True or False
     * makes sense (following Python behavior for `==` and `!=`).
     * Effectively: Both *dtypes* told us that they cannot be compared.
     *
     * In theory, the error could be raised from within an object loop, the
     * solution to that could be pushing this into the ufunc (where we can
     * distinguish the two easily).  In practice, it seems like it should not
     * but a huge problem:  The ufunc loop will itself call `==` which should
     * probably never raise a UFuncNoLoopError.
     *
     * TODO: If/once we correctly push structured comparisons into the ufunc
     *       we could consider pushing this path into the ufunc itself as a
     *       fallback loop (which ignores the input arrays).
     *       This would have the advantage that subclasses implementing
     *       `__array_ufunc__` do not explicitly need `__eq__` and `__ne__`.
     */
    if (result == NULL
            && (cmp_op == Py_EQ || cmp_op == Py_NE)
            && PyErr_ExceptionMatches(npy_UFuncNoLoopError)) {
        PyErr_Clear();

        PyArrayObject *array_other = (PyArrayObject *)PyArray_FROM_O(other);
        if (PyArray_TYPE(array_other) == NPY_VOID) {
            /*
            * Void arrays are currently not handled by ufuncs, so if the other
            * is a void array, we defer to it (will raise a TypeError).
            */
            Py_DECREF(array_other);
            Py_RETURN_NOTIMPLEMENTED;
        }

        if (PyArray_NDIM(self) == 0 && PyArray_NDIM(array_other) == 0) {
            /*
             * (seberg) not sure that this is best, but we preserve Python
             * bool result for "scalar" inputs for now by returning
             * `NotImplemented`.
             */
            Py_DECREF(array_other);
            Py_RETURN_NOTIMPLEMENTED;
        }

        /* Hack warning: using NpyIter to allocate broadcasted result. */
        PyArrayObject *ops[3] = {self, array_other, NULL};
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK | NPY_ITER_REFS_OK;
        npy_uint32 op_flags[3] = {
            NPY_ITER_READONLY, NPY_ITER_READONLY,
            NPY_ITER_ALLOCATE | NPY_ITER_WRITEONLY};

        PyArray_Descr *bool_descr = PyArray_DescrFromType(NPY_BOOL);
        PyArray_Descr *op_descrs[3] = {
            PyArray_DESCR(self), PyArray_DESCR(array_other), bool_descr};

        NpyIter *iter = NpyIter_MultiNew(
                    3, ops, flags, NPY_KEEPORDER, NPY_NO_CASTING,
                    op_flags, op_descrs);

        Py_CLEAR(bool_descr);
        Py_CLEAR(array_other);
        if (iter == NULL) {
            return NULL;
        }
        PyArrayObject *res = NpyIter_GetOperandArray(iter)[2];
        Py_INCREF(res);
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(res);
            return NULL;
        }

        /*
         * The array is guaranteed to be newly allocated and thus contiguous,
         * so simply fill it with 0 or 1.
         */
        memset(PyArray_BYTES(res), cmp_op == Py_EQ ? 0 : 1, PyArray_NBYTES(res));

        /* Ensure basic subclass support by wrapping: */
        if (!PyArray_CheckExact(self)) {
            /*
             * If other is also a subclass (with higher priority) we would
             * already have deferred.  So use `self` for wrapping.  If users
             * need more, they need to override `==` and `!=`.
             */
            Py_SETREF(res, PyArray_SubclassWrap(self, res));
        }
        return (PyObject *)res;
    }
    return result;
}

/*NUMPY_API
 */
NPY_NO_EXPORT int
PyArray_ElementStrides(PyObject *obj)
{
    PyArrayObject *arr;
    int itemsize;
    int i, ndim;
    npy_intp *strides;

    if (!PyArray_Check(obj)) {
        return 0;
    }

    arr = (PyArrayObject *)obj;

    itemsize = PyArray_ITEMSIZE(arr);
    ndim = PyArray_NDIM(arr);
    strides = PyArray_STRIDES(arr);

    for (i = 0; i < ndim; i++) {
        if ((strides[i] % itemsize) != 0) {
            return 0;
        }
    }
    return 1;
}

/*
 * This routine checks to see if newstrides (of length nd) will not
 * ever be able to walk outside of the memory implied numbytes and offset.
 *
 * The available memory is assumed to start at -offset and proceed
 * to numbytes-offset.  The strides are checked to ensure
 * that accessing memory using striding will not try to reach beyond
 * this memory for any of the axes.
 *
 * If numbytes is 0 it will be calculated using the dimensions and
 * element-size.
 *
 * This function checks for walking beyond the beginning and right-end
 * of the buffer and therefore works for any integer stride (positive
 * or negative).
 */

/*NUMPY_API*/
NPY_NO_EXPORT npy_bool
PyArray_CheckStrides(int elsize, int nd, npy_intp numbytes, npy_intp offset,
                     npy_intp const *dims, npy_intp const *newstrides)
{
    npy_intp begin, end;
    npy_intp lower_offset;
    npy_intp upper_offset;

    if (numbytes == 0) {
        numbytes = PyArray_MultiplyList(dims, nd) * elsize;
    }

    begin = -offset;
    end = numbytes - offset;

    offset_bounds_from_strides(elsize, nd, dims, newstrides,
                                        &lower_offset, &upper_offset);

    if ((upper_offset > end) || (lower_offset < begin)) {
        return NPY_FALSE;
    }
    return NPY_TRUE;
}


static PyObject *
array_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"shape", "dtype", "buffer", "offset", "strides",
                             "order", NULL};
    PyArray_Descr *descr = NULL;
    int itemsize;
    PyArray_Dims dims = {NULL, 0};
    PyArray_Dims strides = {NULL, -1};
    PyArray_Chunk buffer;
    npy_longlong offset = 0;
    NPY_ORDER order = NPY_CORDER;
    int is_f_order = 0;
    PyArrayObject *ret;

    buffer.ptr = NULL;
    /*
     * Usually called with shape and type but can also be called with buffer,
     * strides, and swapped info For now, let's just use this to create an
     * empty, contiguous array of a specific type and shape.
     */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&LO&O&:ndarray",
                                     kwlist, PyArray_IntpConverter,
                                     &dims,
                                     PyArray_DescrConverter,
                                     &descr,
                                     PyArray_BufferConverter,
                                     &buffer,
                                     &offset,
                                     &PyArray_OptionalIntpConverter,
                                     &strides,
                                     &PyArray_OrderConverter,
                                     &order)) {
        goto fail;
    }
    if (order == NPY_FORTRANORDER) {
        is_f_order = 1;
    }
    if (descr == NULL) {
        descr = PyArray_DescrFromType(NPY_DEFAULT_TYPE);
    }

    itemsize = descr->elsize;

    if (strides.len != -1) {
        npy_intp nb, off;
        if (strides.len != dims.len) {
            PyErr_SetString(PyExc_ValueError,
                            "strides, if given, must be "   \
                            "the same length as shape");
            goto fail;
        }

        if (buffer.ptr == NULL) {
            nb = 0;
            off = 0;
        }
        else {
            nb = buffer.len;
            off = (npy_intp) offset;
        }


        if (!PyArray_CheckStrides(itemsize, dims.len,
                                  nb, off,
                                  dims.ptr, strides.ptr)) {
            PyErr_SetString(PyExc_ValueError,
                            "strides is incompatible "      \
                            "with shape of requested "      \
                            "array and size of buffer");
            goto fail;
        }
    }

    if (buffer.ptr == NULL) {
        ret = (PyArrayObject *)
            PyArray_NewFromDescr_int(subtype, descr,
                                     (int)dims.len,
                                     dims.ptr,
                                     strides.ptr, NULL, is_f_order, NULL, NULL,
                                     _NPY_ARRAY_ALLOW_EMPTY_STRING);
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
        /* Logic shared by `empty`, `empty_like`, and `ndarray.__new__` */
        if (PyDataType_REFCHK(PyArray_DESCR(ret))) {
            /* place Py_None in object positions */
            PyArray_FillObjectArray(ret, Py_None);
            if (PyErr_Occurred()) {
                descr = NULL;
                goto fail;
            }
        }
    }
    else {
        /* buffer given -- use it */
        if (dims.len == 1 && dims.ptr[0] == -1) {
            dims.ptr[0] = (buffer.len-(npy_intp)offset) / itemsize;
        }
        else if ((strides.ptr == NULL) &&
                 (buffer.len < (offset + (((npy_intp)itemsize)*
                                          PyArray_MultiplyList(dims.ptr,
                                                               dims.len))))) {
            PyErr_SetString(PyExc_TypeError,
                            "buffer is too small for "      \
                            "requested array");
            goto fail;
        }
        /* get writeable and aligned */
        if (is_f_order) {
            buffer.flags |= NPY_ARRAY_F_CONTIGUOUS;
        }
        ret = (PyArrayObject *)PyArray_NewFromDescr_int(
                subtype, descr,
                dims.len, dims.ptr, strides.ptr, offset + (char *)buffer.ptr,
                buffer.flags, NULL, buffer.base,
                _NPY_ARRAY_ALLOW_EMPTY_STRING);
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
    }

    npy_free_cache_dim_obj(dims);
    npy_free_cache_dim_obj(strides);
    return (PyObject *)ret;

 fail:
    Py_XDECREF(descr);
    npy_free_cache_dim_obj(dims);
    npy_free_cache_dim_obj(strides);
    return NULL;
}


static PyObject *
array_iter(PyArrayObject *arr)
{
    if (PyArray_NDIM(arr) == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "iteration over a 0-d array");
        return NULL;
    }
    return PySeqIter_New((PyObject *)arr);
}


NPY_NO_EXPORT PyTypeObject PyArray_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.ndarray",
    .tp_basicsize = sizeof(PyArrayObject_fields),
    /* methods */
    .tp_dealloc = (destructor)array_dealloc,
    .tp_repr = (reprfunc)array_repr,
    .tp_as_number = &array_as_number,
    .tp_as_sequence = &array_as_sequence,
    .tp_as_mapping = &array_as_mapping,
    .tp_str = (reprfunc)array_str,
    .tp_as_buffer = &array_as_buffer,
    .tp_flags =(Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE),

    .tp_richcompare = (richcmpfunc)array_richcompare,
    .tp_weaklistoffset = offsetof(PyArrayObject_fields, weakreflist),
    .tp_iter = (getiterfunc)array_iter,
    .tp_methods = array_methods,
    .tp_getset = array_getsetlist,
    .tp_new = (newfunc)array_new,
};
