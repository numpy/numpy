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
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

/*#include <stdio.h>*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
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
#include "ctors.h"
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
#include "strfuncs.h"

#include "binop_override.h"

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

/*NUMPY_API
 *
 * Precondition: 'arr' is a copy of 'base' (though possibly with different
 * strides, ordering, etc.). This function sets the UPDATEIFCOPY flag and the
 * ->base pointer on 'arr', so that when 'arr' is destructed, it will copy any
 * changes back to 'base'. DEPRECATED, use PyArray_SetWritebackIfCopyBase
 *
 * Steals a reference to 'base'.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_SetUpdateIfCopyBase(PyArrayObject *arr, PyArrayObject *base)
{
    int ret;
    /* 2017-Nov  -10 1.14 (for PyPy only) */
    /* 2018-April-21 1.15 (all Python implementations) */
    if (DEPRECATE("PyArray_SetUpdateIfCopyBase is deprecated, use "
              "PyArray_SetWritebackIfCopyBase instead, and be sure to call "
              "PyArray_ResolveWritebackIfCopy before the array is deallocated, "
              "i.e. before the last call to Py_DECREF. If cleaning up from an "
              "error, PyArray_DiscardWritebackIfCopy may be called instead to "
              "throw away the scratch buffer.") < 0)
        return -1;
    ret = PyArray_SetWritebackIfCopyBase(arr, base);
    if (ret >=0) {
        PyArray_ENABLEFLAGS(arr, NPY_ARRAY_UPDATEIFCOPY);
        PyArray_CLEARFLAGS(arr, NPY_ARRAY_WRITEBACKIFCOPY);
    }
    return ret;
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


/*NUMPY_API*/
NPY_NO_EXPORT int
PyArray_CopyObject(PyArrayObject *dest, PyObject *src_object)
{
    int ret = 0;
    PyArrayObject *src;
    PyArray_Descr *dtype = NULL;
    int ndim = 0;
    npy_intp dims[NPY_MAXDIMS];

    Py_INCREF(src_object);
    /*
     * Special code to mimic Numeric behavior for
     * character arrays.
     */
    if (PyArray_DESCR(dest)->type == NPY_CHARLTR &&
                                PyArray_NDIM(dest) > 0 &&
                                PyString_Check(src_object)) {
        npy_intp n_new, n_old;
        char *new_string;
        PyObject *tmp;

        n_new = PyArray_DIMS(dest)[PyArray_NDIM(dest)-1];
        n_old = PyString_Size(src_object);
        if (n_new > n_old) {
            new_string = malloc(n_new);
            if (new_string == NULL) {
                Py_DECREF(src_object);
                PyErr_NoMemory();
                return -1;
            }
            memcpy(new_string, PyString_AS_STRING(src_object), n_old);
            memset(new_string + n_old, ' ', n_new - n_old);
            tmp = PyString_FromStringAndSize(new_string, n_new);
            free(new_string);
            Py_DECREF(src_object);
            src_object = tmp;
        }
    }

    /*
     * Get either an array object we can copy from, or its parameters
     * if there isn't a convenient array available.
     */
    if (PyArray_GetArrayParamsFromObject(src_object, PyArray_DESCR(dest),
                0, &dtype, &ndim, dims, &src, NULL) < 0) {
        Py_DECREF(src_object);
        return -1;
    }

    /* If it's not an array, either assign from a sequence or as a scalar */
    if (src == NULL) {
        /* If the input is scalar */
        if (ndim == 0) {
            /* If there's one dest element and src is a Python scalar */
            if (PyArray_IsScalar(src_object, Generic)) {
                char *value;
                int retcode;

                value = scalar_value(src_object, dtype);
                if (value == NULL) {
                    Py_DECREF(dtype);
                    Py_DECREF(src_object);
                    return -1;
                }

                /* TODO: switch to SAME_KIND casting */
                retcode = PyArray_AssignRawScalar(dest, dtype, value,
                                        NULL, NPY_UNSAFE_CASTING);
                Py_DECREF(dtype);
                Py_DECREF(src_object);
                return retcode;
            }
            /* Otherwise use the dtype's setitem function */
            else {
                if (PyArray_SIZE(dest) == 1) {
                    Py_DECREF(dtype);
                    Py_DECREF(src_object);
                    ret = PyArray_SETITEM(dest, PyArray_DATA(dest), src_object);
                    return ret;
                }
                else {
                    src = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                                        dtype, 0, NULL, NULL,
                                                        NULL, 0, NULL);
                    if (src == NULL) {
                        Py_DECREF(src_object);
                        return -1;
                    }
                    if (PyArray_SETITEM(src, PyArray_DATA(src), src_object) < 0) {
                        Py_DECREF(src_object);
                        Py_DECREF(src);
                        return -1;
                    }
                }
            }
        }
        else {
            /*
             * If there are more than enough dims, use AssignFromSequence
             * because it can handle this style of broadcasting.
             */
            if (ndim >= PyArray_NDIM(dest)) {
                int res;
                Py_DECREF(dtype);
                res = PyArray_AssignFromSequence(dest, src_object);
                Py_DECREF(src_object);
                return res;
            }
            /* Otherwise convert to an array and do an array-based copy */
            src = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                        dtype, ndim, dims, NULL, NULL,
                                        PyArray_ISFORTRAN(dest), NULL);
            if (src == NULL) {
                Py_DECREF(src_object);
                return -1;
            }
            if (PyArray_AssignFromSequence(src, src_object) < 0) {
                Py_DECREF(src);
                Py_DECREF(src_object);
                return -1;
            }
        }
    }

    /* If it's an array, do a move (handling possible overlapping data) */
    ret = PyArray_MoveInto(dest, src);
    Py_DECREF(src);
    Py_DECREF(src_object);
    return ret;
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
PyArray_TypeNumFromName(char *str)
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
        if ((fa->flags & NPY_ARRAY_UPDATEIFCOPY) || (fa->flags & NPY_ARRAY_WRITEBACKIFCOPY)) {
            /*
             * UPDATEIFCOPY or WRITEBACKIFCOPY means that fa->base's data
             * should be updated with the contents
             * of self.
             * fa->base->flags is not WRITEABLE to protect the relationship
             * unlock it.
             */
            int retval = 0;
            PyArray_ENABLEFLAGS(((PyArrayObject *)fa->base),
                                                    NPY_ARRAY_WRITEABLE);
            PyArray_CLEARFLAGS(self, NPY_ARRAY_UPDATEIFCOPY);
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

static NPY_INLINE void
WARN_IN_DEALLOC(PyObject* warning, const char * msg) {
    if (PyErr_WarnEx(warning, msg, 1) < 0) {
        PyObject * s;

        s = PyUString_FromString("array_dealloc");
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

    _dealloc_cached_buffer_info((PyObject*)self);

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
        if (PyArray_FLAGS(self) & NPY_ARRAY_UPDATEIFCOPY) {
            /* DEPRECATED, remove once the flag is removed */
            char const * msg = "UPDATEIFCOPY detected in array_dealloc. "
                " Required call to PyArray_ResolveWritebackIfCopy or "
                "PyArray_DiscardWritebackIfCopy is missing";
            /*
             * prevent reaching 0 twice and thus recursing into dealloc.
             * Increasing sys.gettotalrefcount, but path should not be taken.
             */
            Py_INCREF(self);
            /* 2017-Nov-10 1.14 */
            WARN_IN_DEALLOC(PyExc_DeprecationWarning, msg);
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
        /* Free internal references if an Object array */
        if (PyDataType_FLAGCHK(fa->descr, NPY_ITEM_REFCOUNT)) {
            PyArray_XDECREF(self);
        }
        npy_free_cache(fa->data, PyArray_NBYTES(self));
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
    if (fobj->flags & NPY_ARRAY_UPDATEIFCOPY)
        printf(" NPY_UPDATEIFCOPY");
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
PyArray_CompareUCS4(npy_ucs4 *s1, npy_ucs4 *s2, size_t len)
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
 * This function does nothing if obj is writeable, and raises an exception
 * (and returns -1) if obj is not writeable. It may also do other
 * house-keeping, such as issuing warnings on arrays which are transitioning
 * to become views. Always call this function at some point before writing to
 * an array.
 *
 * 'name' is a name for the array, used to give better error
 * messages. Something like "assignment destination", "output array", or even
 * just "array".
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

/* This also handles possibly mis-aligned data */
/* Compare s1 and s2 which are not necessarily NULL-terminated.
   s1 is of length len1
   s2 is of length len2
   If they are NULL terminated, then stop comparison.
*/
static int
_myunincmp(npy_ucs4 *s1, npy_ucs4 *s2, int len1, int len2)
{
    npy_ucs4 *sptr;
    npy_ucs4 *s1t=s1, *s2t=s2;
    int val;
    npy_intp size;
    int diff;

    if ((npy_intp)s1 % sizeof(npy_ucs4) != 0) {
        size = len1*sizeof(npy_ucs4);
        s1t = malloc(size);
        memcpy(s1t, s1, size);
    }
    if ((npy_intp)s2 % sizeof(npy_ucs4) != 0) {
        size = len2*sizeof(npy_ucs4);
        s2t = malloc(size);
        memcpy(s2t, s2, size);
    }
    val = PyArray_CompareUCS4(s1t, s2t, PyArray_MIN(len1,len2));
    if ((val != 0) || (len1 == len2)) {
        goto finish;
    }
    if (len2 > len1) {
        sptr = s2t+len1;
        val = -1;
        diff = len2-len1;
    }
    else {
        sptr = s1t+len2;
        val = 1;
        diff=len1-len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            goto finish;
        }
        sptr++;
    }
    val = 0;

 finish:
    if (s1t != s1) {
        free(s1t);
    }
    if (s2t != s2) {
        free(s2t);
    }
    return val;
}




/*
 * Compare s1 and s2 which are not necessarily NULL-terminated.
 * s1 is of length len1
 * s2 is of length len2
 * If they are NULL terminated, then stop comparison.
 */
static int
_mystrncmp(char *s1, char *s2, int len1, int len2)
{
    char *sptr;
    int val;
    int diff;

    val = memcmp(s1, s2, PyArray_MIN(len1, len2));
    if ((val != 0) || (len1 == len2)) {
        return val;
    }
    if (len2 > len1) {
        sptr = s2 + len1;
        val = -1;
        diff = len2 - len1;
    }
    else {
        sptr = s1 + len2;
        val = 1;
        diff = len1 - len2;
    }
    while (diff--) {
        if (*sptr != 0) {
            return val;
        }
        sptr++;
    }
    return 0; /* Only happens if NULLs are everywhere */
}

/* Borrowed from Numarray */

#define SMALL_STRING 2048

static void _rstripw(char *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        int c = s[i];

        if (!c || NumPyOS_ascii_isspace((int)c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}

static void _unistripw(npy_ucs4 *s, int n)
{
    int i;
    for (i = n - 1; i >= 1; i--) { /* Never strip to length 0. */
        npy_ucs4 c = s[i];
        if (!c || NumPyOS_ascii_isspace((int)c)) {
            s[i] = 0;
        }
        else {
            break;
        }
    }
}


static char *
_char_copy_n_strip(char *original, char *temp, int nc)
{
    if (nc > SMALL_STRING) {
        temp = malloc(nc);
        if (!temp) {
            PyErr_NoMemory();
            return NULL;
        }
    }
    memcpy(temp, original, nc);
    _rstripw(temp, nc);
    return temp;
}

static void
_char_release(char *ptr, int nc)
{
    if (nc > SMALL_STRING) {
        free(ptr);
    }
}

static char *
_uni_copy_n_strip(char *original, char *temp, int nc)
{
    if (nc*sizeof(npy_ucs4) > SMALL_STRING) {
        temp = malloc(nc*sizeof(npy_ucs4));
        if (!temp) {
            PyErr_NoMemory();
            return NULL;
        }
    }
    memcpy(temp, original, nc*sizeof(npy_ucs4));
    _unistripw((npy_ucs4 *)temp, nc);
    return temp;
}

static void
_uni_release(char *ptr, int nc)
{
    if (nc*sizeof(npy_ucs4) > SMALL_STRING) {
        free(ptr);
    }
}


/* End borrowed from numarray */

#define _rstrip_loop(CMP) {                                     \
        void *aptr, *bptr;                                      \
        char atemp[SMALL_STRING], btemp[SMALL_STRING];          \
        while(size--) {                                         \
            aptr = stripfunc(iself->dataptr, atemp, N1);        \
            if (!aptr) return -1;                               \
            bptr = stripfunc(iother->dataptr, btemp, N2);       \
            if (!bptr) {                                        \
                relfunc(aptr, N1);                              \
                return -1;                                      \
            }                                                   \
            val = compfunc(aptr, bptr, N1, N2);                  \
            *dptr = (val CMP 0);                                \
            PyArray_ITER_NEXT(iself);                           \
            PyArray_ITER_NEXT(iother);                          \
            dptr += 1;                                          \
            relfunc(aptr, N1);                                  \
            relfunc(bptr, N2);                                  \
        }                                                       \
    }

#define _reg_loop(CMP) {                                \
        while(size--) {                                 \
            val = compfunc((void *)iself->dataptr,       \
                          (void *)iother->dataptr,      \
                          N1, N2);                      \
            *dptr = (val CMP 0);                        \
            PyArray_ITER_NEXT(iself);                   \
            PyArray_ITER_NEXT(iother);                  \
            dptr += 1;                                  \
        }                                               \
    }

static int
_compare_strings(PyArrayObject *result, PyArrayMultiIterObject *multi,
                 int cmp_op, void *func, int rstrip)
{
    PyArrayIterObject *iself, *iother;
    npy_bool *dptr;
    npy_intp size;
    int val;
    int N1, N2;
    int (*compfunc)(void *, void *, int, int);
    void (*relfunc)(char *, int);
    char* (*stripfunc)(char *, char *, int);

    compfunc = func;
    dptr = (npy_bool *)PyArray_DATA(result);
    iself = multi->iters[0];
    iother = multi->iters[1];
    size = multi->size;
    N1 = PyArray_DESCR(iself->ao)->elsize;
    N2 = PyArray_DESCR(iother->ao)->elsize;
    if ((void *)compfunc == (void *)_myunincmp) {
        N1 >>= 2;
        N2 >>= 2;
        stripfunc = _uni_copy_n_strip;
        relfunc = _uni_release;
    }
    else {
        stripfunc = _char_copy_n_strip;
        relfunc = _char_release;
    }
    switch (cmp_op) {
    case Py_EQ:
        if (rstrip) {
            _rstrip_loop(==);
        } else {
            _reg_loop(==);
        }
        break;
    case Py_NE:
        if (rstrip) {
            _rstrip_loop(!=);
        } else {
            _reg_loop(!=);
        }
        break;
    case Py_LT:
        if (rstrip) {
            _rstrip_loop(<);
        } else {
            _reg_loop(<);
        }
        break;
    case Py_LE:
        if (rstrip) {
            _rstrip_loop(<=);
        } else {
            _reg_loop(<=);
        }
        break;
    case Py_GT:
        if (rstrip) {
            _rstrip_loop(>);
        } else {
            _reg_loop(>);
        }
        break;
    case Py_GE:
        if (rstrip) {
            _rstrip_loop(>=);
        } else {
            _reg_loop(>=);
        }
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "bad comparison operator");
        return -1;
    }
    return 0;
}

#undef _reg_loop
#undef _rstrip_loop
#undef SMALL_STRING

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip)
{
    PyArrayObject *result;
    PyArrayMultiIterObject *mit;
    int val, cast = 0;

    /* Cast arrays to a common type */
    if (PyArray_TYPE(self) != PyArray_DESCR(other)->type_num) {
#if defined(NPY_PY3K)
        /*
         * Comparison between Bytes and Unicode is not defined in Py3K;
         * we follow.
         */
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
#else
        cast = 1;
#endif  /* define(NPY_PY3K) */
    }
    if (cast || (PyArray_ISNOTSWAPPED(self) != PyArray_ISNOTSWAPPED(other))) {
        PyObject *new;
        if (PyArray_TYPE(self) == NPY_STRING &&
                PyArray_DESCR(other)->type_num == NPY_UNICODE) {
            PyArray_Descr* unicode = PyArray_DescrNew(PyArray_DESCR(other));
            unicode->elsize = PyArray_DESCR(self)->elsize << 2;
            new = PyArray_FromAny((PyObject *)self, unicode,
                                  0, 0, 0, NULL);
            if (new == NULL) {
                return NULL;
            }
            Py_INCREF(other);
            self = (PyArrayObject *)new;
        }
        else if ((PyArray_TYPE(self) == NPY_UNICODE) &&
                 ((PyArray_DESCR(other)->type_num == NPY_STRING) ||
                 (PyArray_ISNOTSWAPPED(self) != PyArray_ISNOTSWAPPED(other)))) {
            PyArray_Descr* unicode = PyArray_DescrNew(PyArray_DESCR(self));

            if (PyArray_DESCR(other)->type_num == NPY_STRING) {
                unicode->elsize = PyArray_DESCR(other)->elsize << 2;
            }
            else {
                unicode->elsize = PyArray_DESCR(other)->elsize;
            }
            new = PyArray_FromAny((PyObject *)other, unicode,
                                  0, 0, 0, NULL);
            if (new == NULL) {
                return NULL;
            }
            Py_INCREF(self);
            other = (PyArrayObject *)new;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "invalid string data-types "
                            "in comparison");
            return NULL;
        }
    }
    else {
        Py_INCREF(self);
        Py_INCREF(other);
    }

    /* Broad-cast the arrays to a common shape */
    mit = (PyArrayMultiIterObject *)PyArray_MultiIterNew(2, self, other);
    Py_DECREF(self);
    Py_DECREF(other);
    if (mit == NULL) {
        return NULL;
    }

    result = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                  PyArray_DescrFromType(NPY_BOOL),
                                  mit->nd,
                                  mit->dimensions,
                                  NULL, NULL, 0,
                                  NULL);
    if (result == NULL) {
        goto finish;
    }

    if (PyArray_TYPE(self) == NPY_UNICODE) {
        val = _compare_strings(result, mit, cmp_op, _myunincmp, rstrip);
    }
    else {
        val = _compare_strings(result, mit, cmp_op, _mystrncmp, rstrip);
    }

    if (val < 0) {
        Py_DECREF(result);
        result = NULL;
    }

 finish:
    Py_DECREF(mit);
    return (PyObject *)result;
}

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
        PyErr_SetString(PyExc_ValueError,
                "Void-arrays can only be compared for equality.");
        return NULL;
    }
    if (PyArray_HASFIELDS(self)) {
        PyObject *res = NULL, *temp, *a, *b;
        PyObject *key, *value, *temp2;
        PyObject *op;
        Py_ssize_t pos = 0;
        npy_intp result_ndim = PyArray_NDIM(self) > PyArray_NDIM(other) ?
                            PyArray_NDIM(self) : PyArray_NDIM(other);

        op = (cmp_op == Py_EQ ? n_ops.logical_and : n_ops.logical_or);
        while (PyDict_Next(PyArray_DESCR(self)->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            a = array_subscript_asarray(self, key);
            if (a == NULL) {
                Py_XDECREF(res);
                return NULL;
            }
            b = array_subscript_asarray(other, key);
            if (b == NULL) {
                Py_XDECREF(res);
                Py_DECREF(a);
                return NULL;
            }
            temp = array_richcompare((PyArrayObject *)a,b,cmp_op);
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
                    dimensions[result_ndim] = -1;
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
    else {
        /* compare as a string. Assumes self and other have same descr->type */
        return _strings_richcompare(self, other, cmp_op, 0);
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

/*
 * Comparisons can fail, but we do not always want to pass on the exception
 * (see comment in array_richcompare below), but rather return NotImplemented.
 * Here, an exception should be set on entrance.
 * Returns either NotImplemented with the exception cleared, or NULL
 * with the exception set.
 * Raises deprecation warnings for cases where behaviour is meant to change
 * (2015-05-14, 1.10)
 */

NPY_NO_EXPORT PyObject *
_failed_comparison_workaround(PyArrayObject *self, PyObject *other, int cmp_op)
{
    PyObject *exc, *val, *tb;
    PyArrayObject *array_other;
    int other_is_flexible, ndim_other;
    int self_is_flexible = PyTypeNum_ISFLEXIBLE(PyArray_DESCR(self)->type_num);

    PyErr_Fetch(&exc, &val, &tb);
    /*
     * Determine whether other has a flexible dtype; here, inconvertible
     * is counted as inflexible.  (This repeats work done in the ufunc,
     * but OK to waste some time in an unlikely path.)
     */
    array_other = (PyArrayObject *)PyArray_FROM_O(other);
    if (array_other) {
        other_is_flexible = PyTypeNum_ISFLEXIBLE(
            PyArray_DESCR(array_other)->type_num);
        ndim_other = PyArray_NDIM(array_other);
        Py_DECREF(array_other);
    }
    else {
        PyErr_Clear(); /* we restore the original error if needed */
        other_is_flexible = 0;
        ndim_other = 0;
    }
    if (cmp_op == Py_EQ || cmp_op == Py_NE) {
        /*
         * note: for == and !=, a structured dtype self cannot get here,
         * but a string can. Other can be string or structured.
         */
        if (other_is_flexible || self_is_flexible) {
            /*
             * For scalars, returning NotImplemented is correct.
             * For arrays, we emit a future deprecation warning.
             * When this warning is removed, a correctly shaped
             * array of bool should be returned.
             */
            if (ndim_other != 0 || PyArray_NDIM(self) != 0) {
                /* 2015-05-14, 1.10 */
                if (DEPRECATE_FUTUREWARNING(
                        "elementwise comparison failed; returning scalar "
                        "instead, but in the future will perform "
                        "elementwise comparison") < 0) {
                    goto fail;
                }
            }
        }
        else {
            /*
             * If neither self nor other had a flexible dtype, the error cannot
             * have been caused by a lack of implementation in the ufunc.
             *
             * 2015-05-14, 1.10
             */
            if (DEPRECATE(
                    "elementwise comparison failed; "
                    "this will raise an error in the future.") < 0) {
                goto fail;
            }
        }
        Py_XDECREF(exc);
        Py_XDECREF(val);
        Py_XDECREF(tb);
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    else if (other_is_flexible || self_is_flexible) {
        /*
         * For LE, LT, GT, GE and a flexible self or other, we return
         * NotImplemented, which is the correct answer since the ufuncs do
         * not in fact implement loops for those.  On python 3 this will
         * get us the desired TypeError, but on python 2, one gets strange
         * ordering, so we emit a warning.
         */
#if !defined(NPY_PY3K)
        /* 2015-05-14, 1.10 */
        if (DEPRECATE(
                "unorderable dtypes; returning scalar but in "
                "the future this will be an error") < 0) {
            goto fail;
        }
#endif
        Py_XDECREF(exc);
        Py_XDECREF(val);
        Py_XDECREF(tb);
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    else {
        /* LE, LT, GT, or GE with non-flexible other; just pass on error */
        goto fail;
    }

fail:
    /*
     * Reraise the original exception, possibly chaining with a new one.
     */
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
    return NULL;
}

NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op)
{
    PyArrayObject *array_other;
    PyObject *obj_self = (PyObject *)self;
    PyObject *result = NULL;

    /* Special case for string arrays (which don't and currently can't have
     * ufunc loops defined, so there's no point in trying).
     */
    if (PyArray_ISSTRING(self)) {
        array_other = (PyArrayObject *)PyArray_FromObject(other,
                                                          NPY_NOTYPE, 0, 0);
        if (array_other == NULL) {
            PyErr_Clear();
            /* Never mind, carry on, see what happens */
        }
        else if (!PyArray_ISSTRING(array_other)) {
            Py_DECREF(array_other);
            /* Never mind, carry on, see what happens */
        }
        else {
            result = _strings_richcompare(self, array_other, cmp_op, 0);
            Py_DECREF(array_other);
            return result;
        }
        /* If we reach this point, it means that we are not comparing
         * string-to-string. It's possible that this will still work out,
         * e.g. if the other array is an object array, then both will be cast
         * to object or something? I don't know how that works actually, but
         * it does, b/c this works:
         *   l = ["a", "b"]
         *   assert np.array(l, dtype="S1") == np.array(l, dtype="O")
         * So we fall through and see what happens.
         */
    }

    switch (cmp_op) {
    case Py_LT:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other, n_ops.less);
        break;
    case Py_LE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other, n_ops.less_equal);
        break;
    case Py_EQ:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        /*
         * The ufunc does not support void/structured types, so these
         * need to be handled specifically. Only a few cases are supported.
         */

        if (PyArray_TYPE(self) == NPY_VOID) {
            int _res;

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

            _res = PyArray_CanCastTypeTo(PyArray_DESCR(self),
                                         PyArray_DESCR(array_other),
                                         NPY_EQUIV_CASTING);
            if (_res == 0) {
                /* 2015-05-07, 1.10 */
                Py_DECREF(array_other);
                if (DEPRECATE_FUTUREWARNING(
                        "elementwise == comparison failed and returning scalar "
                        "instead; this will raise an error or perform "
                        "elementwise comparison in the future.") < 0) {
                    return NULL;
                }
                Py_INCREF(Py_False);
                return Py_False;
            }
            else {
                result = _void_compare(self, array_other, cmp_op);
            }
            Py_DECREF(array_other);
            return result;
        }

        result = PyArray_GenericBinaryFunction(self,
                (PyObject *)other,
                n_ops.equal);
        break;
    case Py_NE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        /*
         * The ufunc does not support void/structured types, so these
         * need to be handled specifically. Only a few cases are supported.
         */

        if (PyArray_TYPE(self) == NPY_VOID) {
            int _res;

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

            _res = PyArray_CanCastTypeTo(PyArray_DESCR(self),
                                         PyArray_DESCR(array_other),
                                         NPY_EQUIV_CASTING);
            if (_res == 0) {
                /* 2015-05-07, 1.10 */
                Py_DECREF(array_other);
                if (DEPRECATE_FUTUREWARNING(
                        "elementwise != comparison failed and returning scalar "
                        "instead; this will raise an error or perform "
                        "elementwise comparison in the future.") < 0) {
                    return NULL;
                }
                Py_INCREF(Py_True);
                return Py_True;
            }
            else {
                result = _void_compare(self, array_other, cmp_op);
                Py_DECREF(array_other);
            }
            return result;
        }

        result = PyArray_GenericBinaryFunction(self, (PyObject *)other,
                n_ops.not_equal);
        break;
    case Py_GT:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.greater);
        break;
    case Py_GE:
        RICHCMP_GIVE_UP_IF_NEEDED(obj_self, other);
        result = PyArray_GenericBinaryFunction(self, other,
                n_ops.greater_equal);
        break;
    default:
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    if (result == NULL) {
        /*
         * 2015-05-14, 1.10; updated 2018-06-18, 1.16.
         *
         * Comparisons can raise errors when element-wise comparison is not
         * possible. Some of these, though, should not be passed on.
         * In particular, the ufuncs do not have loops for flexible dtype,
         * so those should be treated separately.  Furthermore, for EQ and NE,
         * we should never fail.
         *
         * Our ideal behaviour would be:
         *
         * 1. For EQ and NE:
         *   - If self and other are scalars, return NotImplemented,
         *     so that python can assign True of False as appropriate.
         *   - If either is an array, return an array of False or True.
         *
         * 2. For LT, LE, GE, GT:
         *   - If self or other was flexible, return NotImplemented
         *     (as is in fact the case), so python can raise a TypeError.
         *   - If other is not convertible to an array, pass on the error
         *     (MHvK, 2018-06-18: not sure about this, but it's what we have).
         *
         * However, for backwards compatibilty, we cannot yet return arrays,
         * so we raise warnings instead.  Furthermore, we warn on python2
         * for LT, LE, GE, GT, since fall-back behaviour is poorly defined.
         */
        result = _failed_comparison_workaround(self, other, cmp_op);
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
                     npy_intp *dims, npy_intp *newstrides)
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
    PyArray_Dims strides = {NULL, 0};
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
                                     &PyArray_IntpConverter,
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

    if (strides.ptr != NULL) {
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
                                     0, 1);
        if (ret == NULL) {
            descr = NULL;
            goto fail;
        }
        if (PyDataType_FLAGCHK(descr, NPY_ITEM_HASOBJECT)) {
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
                0, 1);
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

static PyObject *
array_alloc(PyTypeObject *type, Py_ssize_t NPY_UNUSED(nitems))
{
    /* nitems will always be 0 */
    PyObject *obj = PyObject_Malloc(type->tp_basicsize);
    PyObject_Init(obj, type);
    return obj;
}

static void
array_free(PyObject * v)
{
    /* avoid same deallocator as PyBaseObject, see gentype_free */
    PyObject_Free(v);
}


NPY_NO_EXPORT PyTypeObject PyArray_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.ndarray",                            /* tp_name */
    NPY_SIZEOF_PYARRAYOBJECT,                   /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)array_dealloc,                  /* tp_dealloc */
    (printfunc)NULL,                            /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc)array_repr,                       /* tp_repr */
    &array_as_number,                           /* tp_as_number */
    &array_as_sequence,                         /* tp_as_sequence */
    &array_as_mapping,                          /* tp_as_mapping */
    /*
     * The tp_hash slot will be set PyObject_HashNotImplemented when the
     * module is loaded.
     */
    (hashfunc)0,                                /* tp_hash */
    (ternaryfunc)0,                             /* tp_call */
    (reprfunc)array_str,                        /* tp_str */
    (getattrofunc)0,                            /* tp_getattro */
    (setattrofunc)0,                            /* tp_setattro */
    &array_as_buffer,                           /* tp_as_buffer */
    (Py_TPFLAGS_DEFAULT
#if !defined(NPY_PY3K)
     | Py_TPFLAGS_CHECKTYPES
     | Py_TPFLAGS_HAVE_NEWBUFFER
#endif
     | Py_TPFLAGS_BASETYPE),                    /* tp_flags */
    0,                                          /* tp_doc */

    (traverseproc)0,                            /* tp_traverse */
    (inquiry)0,                                 /* tp_clear */
    (richcmpfunc)array_richcompare,             /* tp_richcompare */
    offsetof(PyArrayObject_fields, weakreflist), /* tp_weaklistoffset */
    (getiterfunc)array_iter,                    /* tp_iter */
    (iternextfunc)0,                            /* tp_iternext */
    array_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    array_getsetlist,                           /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)0,                                /* tp_init */
    (allocfunc)array_alloc,                     /* tp_alloc */
    (newfunc)array_new,                         /* tp_new */
    (freefunc)array_free,                       /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
    0,                                          /* tp_version_tag */
};
