#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API
#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"

#include "numpy/npy_3kcompat.h"

#include "ctors.h"

#include "shape.h"

#define PyArrayObject PyArrayObject

static int
_check_ones(PyArrayObject *self, int newnd,
                npy_intp* newdims, npy_intp *strides, npy_intp *masknastrides);

static int
_fix_unknown_dimension(PyArray_Dims *newshape, npy_intp s_original);

static int
_attempt_nocopy_reshape(PyArrayObject *self, int newnd, npy_intp* newdims,
                        npy_intp *newstrides, npy_intp *newmasknastrides,
                        int is_f_order);

static void
_putzero(char *optr, PyObject *zero, PyArray_Descr *dtype);

/*NUMPY_API
 * Resize (reallocate data).  Only works if nothing else is referencing this
 * array and it is contiguous.  If refcheck is 0, then the reference count is
 * not checked and assumed to be 1.  You still must own this data and have no
 * weak-references and no base object.
 */
NPY_NO_EXPORT PyObject *
PyArray_Resize(PyArrayObject *self, PyArray_Dims *newshape, int refcheck,
               NPY_ORDER order)
{
    npy_intp oldsize, newsize;
    int new_nd=newshape->len, k, n, elsize;
    int refcnt;
    npy_intp* new_dimensions=newshape->ptr;
    npy_intp new_strides[NPY_MAXDIMS];
    size_t sd;
    npy_intp *dimptr;
    char *new_data;
    npy_intp largest;

    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError,
                "resize only works on single-segment arrays");
        return NULL;
    }

    if (PyArray_DESCR(self)->elsize == 0) {
        PyErr_SetString(PyExc_ValueError,
                "Bad data-type size.");
        return NULL;
    }
    newsize = 1;
    largest = MAX_INTP / PyArray_DESCR(self)->elsize;
    for(k = 0; k < new_nd; k++) {
        if (new_dimensions[k] == 0) {
            break;
        }
        if (new_dimensions[k] < 0) {
            PyErr_SetString(PyExc_ValueError,
                    "negative dimensions not allowed");
            return NULL;
        }
        newsize *= new_dimensions[k];
        if (newsize <= 0 || newsize > largest) {
            return PyErr_NoMemory();
        }
    }
    oldsize = PyArray_SIZE(self);

    if (oldsize != newsize) {
        if (!(PyArray_FLAGS(self) & NPY_ARRAY_OWNDATA)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize this array: it does not own its data");
            return NULL;
        }

        if (refcheck) {
            refcnt = REFCOUNT(self);
        }
        else {
            refcnt = 1;
        }
        if ((refcnt > 2)
                || (PyArray_BASE(self) != NULL)
                || (((PyArrayObject_fieldaccess *)self)->weakreflist != NULL)) {
            PyErr_SetString(PyExc_ValueError,
                    "cannot resize an array references or is referenced\n"\
                    "by another array in this way.  Use the resize function");
            return NULL;
        }

        if (newsize == 0) {
            sd = PyArray_DESCR(self)->elsize;
        }
        else {
            sd = newsize*PyArray_DESCR(self)->elsize;
        }
        /* Reallocate space if needed */
        new_data = PyDataMem_RENEW(PyArray_DATA(self), sd);
        if (new_data == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate memory for array");
            return NULL;
        }
        ((PyArrayObject_fieldaccess *)self)->data = new_data;
    }

    if ((newsize > oldsize) && PyArray_ISWRITEABLE(self)) {
        /* Fill new memory with zeros */
        elsize = PyArray_DESCR(self)->elsize;
        if (PyDataType_FLAGCHK(PyArray_DESCR(self), NPY_ITEM_REFCOUNT)) {
            PyObject *zero = PyInt_FromLong(0);
            char *optr;
            optr = PyArray_DATA(self) + oldsize*elsize;
            n = newsize - oldsize;
            for (k = 0; k < n; k++) {
                _putzero((char *)optr, zero, PyArray_DESCR(self));
                optr += elsize;
            }
            Py_DECREF(zero);
        }
        else{
            memset(PyArray_DATA(self)+oldsize*elsize, 0, (newsize-oldsize)*elsize);
        }
    }

    if (PyArray_NDIM(self) != new_nd) {
        /* Different number of dimensions. */
        ((PyArrayObject_fieldaccess *)self)->nd = new_nd;
        /* Need new dimensions and strides arrays */
        dimptr = PyDimMem_RENEW(PyArray_DIMS(self), 3*new_nd);
        if (dimptr == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                    "cannot allocate memory for array");
            return NULL;
        }
        ((PyArrayObject_fieldaccess *)self)->dimensions = dimptr;
        ((PyArrayObject_fieldaccess *)self)->strides = dimptr + new_nd;
        ((PyArrayObject_fieldaccess *)self)->maskna_strides = dimptr + 2*new_nd;
    }

    /* make new_strides variable */
    sd = (size_t) PyArray_DESCR(self)->elsize;
    sd = (size_t) _array_fill_strides(new_strides, new_dimensions, new_nd, sd,
            PyArray_FLAGS(self), &(((PyArrayObject_fieldaccess *)self)->flags));
    memmove(PyArray_DIMS(self), new_dimensions, new_nd*sizeof(npy_intp));
    memmove(PyArray_STRIDES(self), new_strides, new_nd*sizeof(npy_intp));
    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * Returns a new array
 * with the new shape from the data
 * in the old array --- order-perspective depends on order argument.
 * copy-only-if-necessary
 */

/*NUMPY_API
 * New shape for an array
 */
NPY_NO_EXPORT PyObject *
PyArray_Newshape(PyArrayObject *self, PyArray_Dims *newdims,
                 NPY_ORDER order)
{
    npy_intp i;
    npy_intp *dimensions = newdims->ptr;
    PyArrayObject *ret;
    int ndim = newdims->len;
    npy_bool same, incref = TRUE;
    npy_intp *strides = NULL;
    npy_intp newstrides[NPY_MAXDIMS];
    npy_intp newmasknastrides[NPY_MAXDIMS];
    int flags, build_maskna_strides = 0;

    printf("in newshape\n"); fflush(stdout);

    if (order == NPY_ANYORDER) {
        order = PyArray_ISFORTRAN(self);
    }
    /*  Quick check to make sure anything actually needs to be done */
    if (ndim == PyArray_NDIM(self)) {
        same = TRUE;
        i = 0;
        while (same && i < ndim) {
            if (PyArray_DIM(self,i) != dimensions[i]) {
                same=FALSE;
            }
            i++;
        }
        if (same) {
            printf("returning view\n"); fflush(stdout);
            return PyArray_View(self, NULL, NULL);
        }
    }

    /*
     * Returns a pointer to an appropriate strides array
     * if all we are doing is inserting ones into the shape,
     * or removing ones from the shape
     * or doing a combination of the two
     * In this case we don't need to do anything but update strides and
     * dimensions.  So, we can handle non single-segment cases.
     */
    i = _check_ones(self, ndim, dimensions, newstrides, newmasknastrides);
    if (i == 0) {
        printf("setting strides to newstrides\n"); fflush(stdout);
        strides = newstrides;
    }
    flags = PyArray_FLAGS(self) & ~(NPY_ARRAY_OWNMASKNA |
                                    NPY_ARRAY_MASKNA);

    if (strides == NULL) {
        printf("strides are null\n"); fflush(stdout);
        /*
         * we are really re-shaping not just adding ones to the shape somewhere
         * fix any -1 dimensions and check new-dimensions against old size
         */
        if (_fix_unknown_dimension(newdims, PyArray_SIZE(self)) < 0) {
            return NULL;
        }
        /*
         * sometimes we have to create a new copy of the array
         * in order to get the right orientation and
         * because we can't just re-use the buffer with the
         * data in the order it is in.
         */
        if (!(PyArray_ISONESEGMENT(self)) ||
            (((PyArray_CHKFLAGS(self, NPY_ARRAY_C_CONTIGUOUS) &&
               order == NPY_FORTRANORDER) ||
              (PyArray_CHKFLAGS(self, NPY_ARRAY_F_CONTIGUOUS) &&
                  order == NPY_CORDER)) && (PyArray_NDIM(self) > 1))) {
            int success = 0;
            success = _attempt_nocopy_reshape(self, ndim, dimensions,
                                          newstrides, newmasknastrides, order);
            if (success) {
                printf("nocopy reshape succeeded\n"); fflush(stdout);
                /* no need to copy the array after all */
                strides = newstrides;
            }
            else {
                printf("nocopy reshape failed\n"); fflush(stdout);
                PyObject *newcopy;
                newcopy = PyArray_NewCopy(self, order);
                if (newcopy == NULL) {
                    return NULL;
                }
                incref = FALSE;
                self = (PyArrayObject *)newcopy;
                build_maskna_strides = 1;
            }
            flags = PyArray_FLAGS(self) & ~(NPY_ARRAY_OWNMASKNA |
                                            NPY_ARRAY_MASKNA);
        }
        else {
            build_maskna_strides = 1;
        }

        /* We always have to interpret the contiguous buffer correctly */

        /* Make sure the flags argument is set. */
        if (ndim > 1) {
            if (order == NPY_FORTRANORDER) {
                flags &= ~NPY_ARRAY_C_CONTIGUOUS;
                flags |= NPY_ARRAY_F_CONTIGUOUS;
            }
            else {
                flags &= ~NPY_ARRAY_F_CONTIGUOUS;
                flags |= NPY_ARRAY_C_CONTIGUOUS;
            }
        }
    }
    else if (ndim > 0) {
        /*
         * replace any 0-valued strides with
         * appropriate value to preserve contiguousness
         */
        if (order == NPY_FORTRANORDER) {
            if (strides[0] == 0) {
                strides[0] = PyArray_DESCR(self)->elsize;
            }
            for (i = 1; i < ndim; i++) {
                if (strides[i] == 0) {
                    strides[i] = strides[i-1] * dimensions[i-1];
                }
            }
        }
        else {
            if (strides[ndim-1] == 0) {
                strides[ndim-1] = PyArray_DESCR(self)->elsize;
            }
            for (i = ndim - 2; i > -1; i--) {
                if (strides[i] == 0) {
                    strides[i] = strides[i+1] * dimensions[i+1];
                }
            }
        }
    }

    Py_INCREF(PyArray_DESCR(self));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                                       PyArray_DESCR(self),
                                       ndim, dimensions,
                                       strides,
                                       PyArray_DATA(self),
                                       flags, (PyObject *)self);

    if (ret == NULL) {
        goto fail;
    }

    if (incref) {
        Py_INCREF(self);
    }
    if (PyArray_SetBaseObject(ret, (PyObject *)self)) {
        Py_DECREF(ret);
        return NULL;
    }

    /* If there's an NA mask, make sure to view it too */
    if (PyArray_HASMASKNA(self)) {
        PyArrayObject_fieldaccess *fa = (PyArrayObject_fieldaccess *)ret;
        fa->maskna_dtype = PyArray_MASKNA_DTYPE(self);
        Py_INCREF(fa->maskna_dtype);
        fa->maskna_data = PyArray_MASKNA_DATA(self);
        if (build_maskna_strides) {
            npy_intp stride = 1;
            if (order == NPY_FORTRANORDER) {
                printf("building fortran strides\n"); fflush(stdout);
                for (i = 0; i < ndim; ++i) {
                    fa->maskna_strides[i] = stride;
                    printf("stride %d\n", (int)stride);
                    stride *= fa->dimensions[i];
                }
            }
            else {
                printf("building C strides\n"); fflush(stdout);
                for (i = ndim; i >= 0; --i) {
                    fa->maskna_strides[i] = stride;
                    stride *= fa->dimensions[i];
                }
            }
        }
        else {
            memcpy(fa->maskna_strides, newmasknastrides,
                                fa->nd * sizeof(npy_intp));
        }
        fa->flags |= NPY_ARRAY_MASKNA;
    }

    PyArray_UpdateFlags(ret, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    return (PyObject *)ret;

 fail:
    if (!incref) {
        Py_DECREF(self);
    }
    return NULL;
}



/* For back-ward compatability -- Not recommended */

/*NUMPY_API
 * Reshape
 */
NPY_NO_EXPORT PyObject *
PyArray_Reshape(PyArrayObject *self, PyObject *shape)
{
    PyObject *ret;
    PyArray_Dims newdims;

    if (!PyArray_IntpConverter(shape, &newdims)) {
        return NULL;
    }
    ret = PyArray_Newshape(self, &newdims, PyArray_CORDER);
    PyDimMem_FREE(newdims.ptr);
    return ret;
}

/* inserts 0 for strides where dimension will be 1 */
static int
_check_ones(PyArrayObject *self, int newnd,
                npy_intp* newdims, npy_intp *strides, npy_intp *masknastrides)
{
    int nd;
    npy_intp *dims;
    npy_bool done=FALSE;
    int j, k;
    int has_maskna = PyArray_HASMASKNA(self);

    nd = PyArray_NDIM(self);
    dims = PyArray_DIMS(self);

    for (k = 0, j = 0; !done && (j < nd || k < newnd);) {
        if ((j<nd) && (k<newnd) && (newdims[k] == dims[j])) {
            strides[k] = PyArray_STRIDES(self)[j];
            if (has_maskna) {
                masknastrides[k] = PyArray_MASKNA_STRIDES(self)[j];
            }
            j++;
            k++;
        }
        else if ((k < newnd) && (newdims[k] == 1)) {
            strides[k] = 0;
            if (has_maskna) {
                masknastrides[k] = 0;
            }
            k++;
        }
        else if ((j<nd) && (dims[j] == 1)) {
            j++;
        }
        else {
            done = TRUE;
        }
    }
    if (done) {
        return -1;
    }
    return 0;
}

static void
_putzero(char *optr, PyObject *zero, PyArray_Descr *dtype)
{
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)) {
        memset(optr, 0, dtype->elsize);
    }
    else if (PyDescr_HASFIELDS(dtype)) {
        PyObject *key, *value, *title = NULL;
        PyArray_Descr *new;
        int offset;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
            if NPY_TITLE_KEY(key, value) {
                continue;
            }
            if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset, &title)) {
                return;
            }
            _putzero(optr + offset, zero, new);
        }
    }
    else {
        Py_INCREF(zero);
        NPY_COPY_PYOBJECT_PTR(optr, &zero);
    }
    return;
}


/*
 * attempt to reshape an array without copying data
 *
 * This function should correctly handle all reshapes, including
 * axes of length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills newstrides
 *     with appropriate strides
 *
 * The "is_f_order" argument describes how the array should be viewed
 * during the reshape, not how it is stored in memory (that
 * information is in PyArray_STRIDES(self)).
 *
 * If some output dimensions have length 1, the strides assigned to
 * them are arbitrary. In the current implementation, they are the
 * stride of the next-fastest index.
 */
static int
_attempt_nocopy_reshape(PyArrayObject *self, int newnd, npy_intp* newdims,
                        npy_intp *newstrides, npy_intp *newmasknastrides,
                        int is_f_order)
{
    int oldnd;
    npy_intp olddims[NPY_MAXDIMS];
    npy_intp oldstrides[NPY_MAXDIMS], oldmasknastrides[NPY_MAXDIMS];
    int oi, oj, ok, ni, nj, nk;
    int np, op;
    int has_maskna = PyArray_HASMASKNA(self);

    oldnd = 0;
    for (oi = 0; oi < PyArray_NDIM(self); oi++) {
        if (PyArray_DIMS(self)[oi]!= 1) {
            olddims[oldnd] = PyArray_DIMS(self)[oi];
            oldstrides[oldnd] = PyArray_STRIDES(self)[oi];
            if (has_maskna) {
                oldmasknastrides[oldnd] = PyArray_MASKNA_STRIDES(self)[oi];
            }
            oldnd++;
        }
    }

    /*
      fprintf(stderr, "_attempt_nocopy_reshape( (");
      for (oi=0; oi<oldnd; oi++)
      fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
      fprintf(stderr, ") -> (");
      for (ni=0; ni<newnd; ni++)
      fprintf(stderr, "(%d,*), ", newdims[ni]);
      fprintf(stderr, "), is_f_order=%d)\n", is_f_order);
    */


    np = 1;
    for (ni = 0; ni < newnd; ni++) {
        np *= newdims[ni];
    }
    op = 1;
    for (oi = 0; oi < oldnd; oi++) {
        op *= olddims[oi];
    }
    if (np != op) {
        /* different total sizes; no hope */
        return 0;
    }
    /* the current code does not handle 0-sized arrays, so give up */
    if (np == 0) {
        return 0;
    }

    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while(ni < newnd && oi < oldnd) {
        np = newdims[ni];
        op = olddims[oi];

        while (np != op) {
            if (np < op) {
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        for (ok = oi; ok < oj - 1; ok++) {
            if (is_f_order) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok] ||
                       (has_maskna && oldmasknastrides[ok+1] !=
                                        olddims[ok]*oldmasknastrides[ok])) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1] ||
                        (has_maskna && oldmasknastrides[ok] !=
                                    olddims[ok+1]*oldmasknastrides[ok+1])) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        if (is_f_order) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
            if (has_maskna) {
                newmasknastrides[ni] = oldmasknastrides[oi];
                for (nk = ni + 1; nk < nj; nk++) {
                    newmasknastrides[nk] =
                                    newmasknastrides[nk - 1]*newdims[nk - 1];
                }
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
            if (has_maskna) {
                newmasknastrides[nj - 1] = oldmasknastrides[oj - 1];
                for (nk = nj - 1; nk > ni; nk--) {
                    newmasknastrides[nk - 1] =
                                    newmasknastrides[nk]*newdims[nk];
                }
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
      fprintf(stderr, "success: _attempt_nocopy_reshape (");
      for (oi=0; oi<oldnd; oi++)
      fprintf(stderr, "(%d,%d), ", olddims[oi], oldstrides[oi]);
      fprintf(stderr, ") -> (");
      for (ni=0; ni<newnd; ni++)
      fprintf(stderr, "(%d,%d), ", newdims[ni], newstrides[ni]);
      fprintf(stderr, ")\n");
    */

    return 1;
}

static int
_fix_unknown_dimension(PyArray_Dims *newshape, npy_intp s_original)
{
    npy_intp *dimensions;
    npy_intp i_unknown, s_known;
    int i, n;
    static char msg[] = "total size of new array must be unchanged";

    dimensions = newshape->ptr;
    n = newshape->len;
    s_known = 1;
    i_unknown = -1;

    for (i = 0; i < n; i++) {
        if (dimensions[i] < 0) {
            if (i_unknown == -1) {
                i_unknown = i;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                                "can only specify one"  \
                                " unknown dimension");
                return -1;
            }
        }
        else {
            s_known *= dimensions[i];
        }
    }

    if (i_unknown >= 0) {
        if ((s_known == 0) || (s_original % s_known != 0)) {
            PyErr_SetString(PyExc_ValueError, msg);
            return -1;
        }
        dimensions[i_unknown] = s_original/s_known;
    }
    else {
        if (s_original != s_known) {
            PyErr_SetString(PyExc_ValueError, msg);
            return -1;
        }
    }
    return 0;
}

/*NUMPY_API
 *
 * return a new view of the array object with all of its unit-length
 * dimensions squeezed out if needed, otherwise
 * return the same array.
 */
NPY_NO_EXPORT PyObject *
PyArray_Squeeze(PyArrayObject *self)
{
    int nd = PyArray_NDIM(self);
    int newnd = nd;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];
    int i, j;
    PyArrayObject *ret;
    PyArray_Descr *dtype;

    if (nd == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    for (j = 0, i = 0; i < nd; i++) {
        if (PyArray_DIMS(self)[i] == 1) {
            newnd -= 1;
        }
        else {
            dimensions[j] = PyArray_DIMS(self)[i];
            strides[j++] = PyArray_STRIDES(self)[i];
        }
    }

    dtype = PyArray_DESCR(self);
    Py_INCREF(dtype);
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(self),
                               dtype,
                               newnd, dimensions,
                               strides, PyArray_DATA(self),
                               PyArray_FLAGS(self),
                               (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    PyArray_CLEARFLAGS(ret, NPY_ARRAY_OWNDATA);
    Py_INCREF(self);
    if (PyArray_SetBaseObject(ret, (PyObject *)self) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}

/*NUMPY_API
 * SwapAxes
 */
NPY_NO_EXPORT PyObject *
PyArray_SwapAxes(PyArrayObject *ap, int a1, int a2)
{
    PyArray_Dims new_axes;
    npy_intp dims[NPY_MAXDIMS];
    int n, i, val;
    PyObject *ret;

    if (a1 == a2) {
        Py_INCREF(ap);
        return (PyObject *)ap;
    }

    n = PyArray_NDIM(ap);
    if (n <= 1) {
        Py_INCREF(ap);
        return (PyObject *)ap;
    }

    if (a1 < 0) {
        a1 += n;
    }
    if (a2 < 0) {
        a2 += n;
    }
    if ((a1 < 0) || (a1 >= n)) {
        PyErr_SetString(PyExc_ValueError,
                        "bad axis1 argument to swapaxes");
        return NULL;
    }
    if ((a2 < 0) || (a2 >= n)) {
        PyErr_SetString(PyExc_ValueError,
                        "bad axis2 argument to swapaxes");
        return NULL;
    }
    new_axes.ptr = dims;
    new_axes.len = n;

    for (i = 0; i < n; i++) {
        if (i == a1) {
            val = a2;
        }
        else if (i == a2) {
            val = a1;
        }
        else {
            val = i;
        }
        new_axes.ptr[i] = val;
    }
    ret = PyArray_Transpose(ap, &new_axes);
    return ret;
}

/*NUMPY_API
 * Return Transpose.
 */
NPY_NO_EXPORT PyObject *
PyArray_Transpose(PyArrayObject *ap, PyArray_Dims *permute)
{
    npy_intp *axes, axis;
    npy_intp i, n;
    npy_intp permutation[NPY_MAXDIMS], reverse_permutation[NPY_MAXDIMS];
    PyArrayObject *ret = NULL;

    if (permute == NULL) {
        n = PyArray_NDIM(ap);
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    else {
        n = permute->len;
        axes = permute->ptr;
        if (n != PyArray_NDIM(ap)) {
            PyErr_SetString(PyExc_ValueError,
                            "axes don't match array");
            return NULL;
        }
        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }
        for (i = 0; i < n; i++) {
            axis = axes[i];
            if (axis < 0) {
                axis = PyArray_NDIM(ap) + axis;
            }
            if (axis < 0 || axis >= PyArray_NDIM(ap)) {
                PyErr_SetString(PyExc_ValueError,
                                "invalid axis for this array");
                return NULL;
            }
            if (reverse_permutation[axis] != -1) {
                PyErr_SetString(PyExc_ValueError,
                                "repeated axis in transpose");
                return NULL;
            }
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
        for (i = 0; i < n; i++) {
        }
    }

    /*
     * this allocates memory for dimensions and strides (but fills them
     * incorrectly), sets up descr, and points data at PyArray_DATA(ap).
     */
    Py_INCREF(PyArray_DESCR(ap));
    ret = (PyArrayObject *)\
        PyArray_NewFromDescr(Py_TYPE(ap),
                             PyArray_DESCR(ap),
                             n, PyArray_DIMS(ap),
                             NULL, PyArray_DATA(ap), PyArray_FLAGS(ap),
                             (PyObject *)ap);
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    Py_INCREF(ap);
    if (PyArray_SetBaseObject(ret, (PyObject *)ap) < 0) {
        Py_DECREF(ret);
        return NULL;
    }

    /* fix the dimensions and strides of the return-array */
    for (i = 0; i < n; i++) {
        PyArray_DIMS(ret)[i] = PyArray_DIMS(ap)[permutation[i]];
        PyArray_STRIDES(ret)[i] = PyArray_STRIDES(ap)[permutation[i]];
    }
    PyArray_UpdateFlags(ret, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    return (PyObject *)ret;
}

/*
 * Sorts items so stride is descending, because C-order
 * is the default in the face of ambiguity.
 */
int _npy_stride_sort_item_comparator(const void *a, const void *b)
{
    npy_intp astride = ((npy_stride_sort_item *)a)->stride,
            bstride = ((npy_stride_sort_item *)b)->stride;

    /* Sort the absolute value of the strides */
    if (astride < 0) {
        astride = -astride;
    }
    if (bstride < 0) {
        bstride = -bstride;
    }

    if (astride > bstride) {
        return -1;
    }
    else if (astride == bstride) {
        /*
         * Make the qsort stable by next comparing the perm order.
         * (Note that two perm entries will never be equal)
         */
        npy_intp aperm = ((npy_stride_sort_item *)a)->perm,
                bperm = ((npy_stride_sort_item *)b)->perm;
        return (aperm < bperm) ? -1 : 1;
    }
    else {
        return 1;
    }
}

/*NUMPY_API
 *
 * This function populates the first ndim elements
 * of strideperm with sorted descending by their absolute values.
 * For example, the stride array (4, -2, 12) becomes
 * [(2, 12), (0, 4), (1, -2)].
 */
NPY_NO_EXPORT void
PyArray_CreateSortedStridePerm(int ndim, npy_intp * strides,
                           npy_stride_sort_item *strideperm)
{
    int i;

    /* Set up the strideperm values */
    for (i = 0; i < ndim; ++i) {
        strideperm[i].perm = i;
        strideperm[i].stride = strides[i];
    }

    /* Sort them */
    qsort(strideperm, ndim, sizeof(npy_stride_sort_item),
                                    &_npy_stride_sort_item_comparator);
}

/*NUMPY_API
 * Ravel
 * Returns a contiguous array
 */
NPY_NO_EXPORT PyObject *
PyArray_Ravel(PyArrayObject *a, NPY_ORDER order)
{
    PyArray_Dims newdim = {NULL,1};
    npy_intp val[1] = {-1};

    newdim.ptr = val;

    if (order == NPY_ANYORDER) {
        order = PyArray_ISFORTRAN(a) ? NPY_FORTRANORDER : NPY_CORDER;
    }
    else if (order == NPY_KEEPORDER) {
        if (PyArray_IS_C_CONTIGUOUS(a)) {
            order = NPY_CORDER;
        }
        else if (PyArray_IS_F_CONTIGUOUS(a)) {
            order = NPY_FORTRANORDER;
        }
    }

    if (order == NPY_CORDER && PyArray_ISCONTIGUOUS(a)) {
        return PyArray_Newshape(a, &newdim, NPY_CORDER);
    }
    else if (order == NPY_FORTRANORDER && PyArray_ISFORTRAN(a)) {
        return PyArray_Newshape(a, &newdim, NPY_FORTRANORDER);
    }
    /* For KEEPORDER, check if we can make a flattened view */
    else if (order == NPY_KEEPORDER) {
        npy_stride_sort_item strideperm[NPY_MAXDIMS];
        npy_intp stride;
        int i, ndim = PyArray_NDIM(a);

        PyArray_CreateSortedStridePerm(PyArray_NDIM(a), PyArray_STRIDES(a),
                                        strideperm);

        stride = PyArray_DESCR(a)->elsize;
        for (i = ndim-1; i >= 0; --i) {
            if (strideperm[i].stride != stride) {
                break;
            }
            stride *= PyArray_DIM(a, strideperm[i].perm);
        }

        /* If all the strides matched a contiguous layout, return a view */
        if (i < 0) {
            PyArrayObject *ret;
            npy_intp stride = PyArray_DESCR(a)->elsize;

            val[0] = PyArray_SIZE(a);

            Py_INCREF(PyArray_DESCR(a));
            ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(a),
                               PyArray_DESCR(a),
                               1, val,
                               &stride,
                               PyArray_BYTES(a),
                               PyArray_FLAGS(a),
                               (PyObject *)a);

            if (ret != NULL) {
                PyArray_UpdateFlags(ret,
                            NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS);
                Py_INCREF(a);
                if (PyArray_SetBaseObject(ret, (PyObject *)a) < 0) {
                    Py_DECREF(ret);
                    ret = NULL;
                }
            }
            return (PyObject *)ret;
        }

    }

    return PyArray_Flatten(a, order);
}

/*NUMPY_API
 * Flatten
 */
NPY_NO_EXPORT PyObject *
PyArray_Flatten(PyArrayObject *a, NPY_ORDER order)
{
    PyArrayObject *ret;
    npy_intp size;

    if (order == NPY_ANYORDER) {
        order = PyArray_ISFORTRAN(a) ? NPY_FORTRANORDER : NPY_CORDER;
    }

    size = PyArray_SIZE(a);
    Py_INCREF(PyArray_DESCR(a));
    ret = (PyArrayObject *)PyArray_NewFromDescr(Py_TYPE(a),
                               PyArray_DESCR(a),
                               1, &size,
                               NULL,
                               NULL,
                               0, (PyObject *)a);

    if (ret == NULL) {
        return NULL;
    }
    if (PyArray_CopyAsFlat(ret, a, order) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return (PyObject *)ret;
}
