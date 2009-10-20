#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define _MULTIARRAYMODULE
#define NPY_NO_PREFIX
#include "numpy/arrayobject.h"
#include "numpy/arrayscalars.h"

#include "numpy/npy_math.h"

#include "npy_config.h"

#include "ctors.h"

#include "shape.h"

#define PyAO PyArrayObject

static int
_check_ones(PyArrayObject *self, int newnd, intp* newdims, intp *strides);

static int
_fix_unknown_dimension(PyArray_Dims *newshape, intp s_original);

static int
_attempt_nocopy_reshape(PyArrayObject *self, int newnd, intp* newdims,
                        intp *newstrides, int fortran);

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
               NPY_ORDER fortran)
{
    intp oldsize, newsize;
    int new_nd=newshape->len, k, n, elsize;
    int refcnt;
    intp* new_dimensions=newshape->ptr;
    intp new_strides[MAX_DIMS];
    size_t sd;
    intp *dimptr;
    char *new_data;
    intp largest;

    if (!PyArray_ISONESEGMENT(self)) {
        PyErr_SetString(PyExc_ValueError,
                        "resize only works on single-segment arrays");
        return NULL;
    }

    if (fortran == PyArray_ANYORDER) {
        fortran = PyArray_CORDER;
    }
    if (self->descr->elsize == 0) {
        PyErr_SetString(PyExc_ValueError, "Bad data-type size.");
        return NULL;
    }
    newsize = 1;
    largest = MAX_INTP / self->descr->elsize;
    for(k=0; k<new_nd; k++) {
        if (new_dimensions[k]==0) {
            break;
        }
        if (new_dimensions[k] < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "negative dimensions not allowed");
            return NULL;
        }
        newsize *= new_dimensions[k];
        if (newsize <=0 || newsize > largest) {
            return PyErr_NoMemory();
        }
    }
    oldsize = PyArray_SIZE(self);

    if (oldsize != newsize) {
        if (!(self->flags & OWNDATA)) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot resize this array:  "   \
                            "it does not own its data");
            return NULL;
        }

        if (refcheck) {
            refcnt = REFCOUNT(self);
        }
        else {
            refcnt = 1;
        }
        if ((refcnt > 2) || (self->base != NULL) ||
            (self->weakreflist != NULL)) {
            PyErr_SetString(PyExc_ValueError,
                            "cannot resize an array that has "\
                            "been referenced or is referencing\n"\
                            "another array in this way.  Use the "\
                            "resize function");
            return NULL;
        }

        if (newsize == 0) {
            sd = self->descr->elsize;
        }
        else {
            sd = newsize*self->descr->elsize;
        }
        /* Reallocate space if needed */
        new_data = PyDataMem_RENEW(self->data, sd);
        if (new_data == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                            "cannot allocate memory for array");
            return NULL;
        }
        self->data = new_data;
    }

    if ((newsize > oldsize) && PyArray_ISWRITEABLE(self)) {
        /* Fill new memory with zeros */
        elsize = self->descr->elsize;
        if (PyDataType_FLAGCHK(self->descr, NPY_ITEM_REFCOUNT)) {
            PyObject *zero = PyInt_FromLong(0);
            char *optr;
            optr = self->data + oldsize*elsize;
            n = newsize - oldsize;
            for (k = 0; k < n; k++) {
                _putzero((char *)optr, zero, self->descr);
                optr += elsize;
            }
            Py_DECREF(zero);
        }
        else{
            memset(self->data+oldsize*elsize, 0, (newsize-oldsize)*elsize);
        }
    }

    if (self->nd != new_nd) {
        /* Different number of dimensions. */
        self->nd = new_nd;
        /* Need new dimensions and strides arrays */
        dimptr = PyDimMem_RENEW(self->dimensions, 2*new_nd);
        if (dimptr == NULL) {
            PyErr_SetString(PyExc_MemoryError,
                            "cannot allocate memory for array " \
                            "(array may be corrupted)");
            return NULL;
        }
        self->dimensions = dimptr;
        self->strides = dimptr + new_nd;
    }

    /* make new_strides variable */
    sd = (size_t) self->descr->elsize;
    sd = (size_t) _array_fill_strides(new_strides, new_dimensions, new_nd, sd,
                                      self->flags, &(self->flags));
    memmove(self->dimensions, new_dimensions, new_nd*sizeof(intp));
    memmove(self->strides, new_strides, new_nd*sizeof(intp));
    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * Returns a new array
 * with the new shape from the data
 * in the old array --- order-perspective depends on fortran argument.
 * copy-only-if-necessary
 */

/*NUMPY_API
 * New shape for an array
 */
NPY_NO_EXPORT PyObject *
PyArray_Newshape(PyArrayObject *self, PyArray_Dims *newdims,
                 NPY_ORDER fortran)
{
    intp i;
    intp *dimensions = newdims->ptr;
    PyArrayObject *ret;
    int n = newdims->len;
    Bool same, incref = TRUE;
    intp *strides = NULL;
    intp newstrides[MAX_DIMS];
    int flags;

    if (fortran == PyArray_ANYORDER) {
        fortran = PyArray_ISFORTRAN(self);
    }
    /*  Quick check to make sure anything actually needs to be done */
    if (n == self->nd) {
        same = TRUE;
        i = 0;
        while (same && i < n) {
            if (PyArray_DIM(self,i) != dimensions[i]) {
                same=FALSE;
            }
            i++;
        }
        if (same) {
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
    i = _check_ones(self, n, dimensions, newstrides);
    if (i == 0) {
        strides = newstrides;
    }
    flags = self->flags;

    if (strides == NULL) {
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
            (((PyArray_CHKFLAGS(self, NPY_CONTIGUOUS) &&
               fortran == NPY_FORTRANORDER) ||
              (PyArray_CHKFLAGS(self, NPY_FORTRAN) &&
                  fortran == NPY_CORDER)) && (self->nd > 1))) {
            int success = 0;
            success = _attempt_nocopy_reshape(self,n,dimensions,
                                              newstrides,fortran);
            if (success) {
                /* no need to copy the array after all */
                strides = newstrides;
                flags = self->flags;
            }
            else {
                PyObject *new;
                new = PyArray_NewCopy(self, fortran);
                if (new == NULL) {
                    return NULL;
                }
                incref = FALSE;
                self = (PyArrayObject *)new;
                flags = self->flags;
            }
        }

        /* We always have to interpret the contiguous buffer correctly */

        /* Make sure the flags argument is set. */
        if (n > 1) {
            if (fortran == NPY_FORTRANORDER) {
                flags &= ~NPY_CONTIGUOUS;
                flags |= NPY_FORTRAN;
            }
            else {
                flags &= ~NPY_FORTRAN;
                flags |= NPY_CONTIGUOUS;
            }
        }
    }
    else if (n > 0) {
        /*
         * replace any 0-valued strides with
         * appropriate value to preserve contiguousness
         */
        if (fortran == PyArray_FORTRANORDER) {
            if (strides[0] == 0) {
                strides[0] = self->descr->elsize;
            }
            for (i = 1; i < n; i++) {
                if (strides[i] == 0) {
                    strides[i] = strides[i-1] * dimensions[i-1];
                }
            }
        }
        else {
            if (strides[n-1] == 0) {
                strides[n-1] = self->descr->elsize;
            }
            for (i = n - 2; i > -1; i--) {
                if (strides[i] == 0) {
                    strides[i] = strides[i+1] * dimensions[i+1];
                }
            }
        }
    }

    Py_INCREF(self->descr);
    ret = (PyAO *)PyArray_NewFromDescr(self->ob_type,
                                       self->descr,
                                       n, dimensions,
                                       strides,
                                       self->data,
                                       flags, (PyObject *)self);

    if (ret == NULL) {
        goto fail;
    }
    if (incref) {
        Py_INCREF(self);
    }
    ret->base = (PyObject *)self;
    PyArray_UpdateFlags(ret, CONTIGUOUS | FORTRAN);
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
_check_ones(PyArrayObject *self, int newnd, intp* newdims, intp *strides)
{
    int nd;
    intp *dims;
    Bool done=FALSE;
    int j, k;

    nd = self->nd;
    dims = self->dimensions;

    for (k = 0, j = 0; !done && (j < nd || k < newnd);) {
        if ((j<nd) && (k<newnd) && (newdims[k] == dims[j])) {
            strides[k] = self->strides[j];
            j++;
            k++;
        }
        else if ((k < newnd) && (newdims[k] == 1)) {
            strides[k] = 0;
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
 * The "fortran" argument describes how the array should be viewed
 * during the reshape, not how it is stored in memory (that
 * information is in self->strides).
 *
 * If some output dimensions have length 1, the strides assigned to
 * them are arbitrary. In the current implementation, they are the
 * stride of the next-fastest index.
 */
static int
_attempt_nocopy_reshape(PyArrayObject *self, int newnd, intp* newdims,
                        intp *newstrides, int fortran)
{
    int oldnd;
    intp olddims[MAX_DIMS];
    intp oldstrides[MAX_DIMS];
    int oi, oj, ok, ni, nj, nk;
    int np, op;

    oldnd = 0;
    for (oi = 0; oi < self->nd; oi++) {
        if (self->dimensions[oi]!= 1) {
            olddims[oldnd] = self->dimensions[oi];
            oldstrides[oldnd] = self->strides[oi];
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
      fprintf(stderr, "), fortran=%d)\n", fortran);
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
            if (fortran) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                     /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        if (fortran) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
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
_fix_unknown_dimension(PyArray_Dims *newshape, intp s_original)
{
    intp *dimensions;
    intp i_unknown, s_known;
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
    int nd = self->nd;
    int newnd = nd;
    intp dimensions[MAX_DIMS];
    intp strides[MAX_DIMS];
    int i, j;
    PyObject *ret;

    if (nd == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    for (j = 0, i = 0; i < nd; i++) {
        if (self->dimensions[i] == 1) {
            newnd -= 1;
        }
        else {
            dimensions[j] = self->dimensions[i];
            strides[j++] = self->strides[i];
        }
    }

    Py_INCREF(self->descr);
    ret = PyArray_NewFromDescr(self->ob_type,
                               self->descr,
                               newnd, dimensions,
                               strides, self->data,
                               self->flags,
                               (PyObject *)self);
    if (ret == NULL) {
        return NULL;
    }
    PyArray_FLAGS(ret) &= ~OWNDATA;
    PyArray_BASE(ret) = (PyObject *)self;
    Py_INCREF(self);
    return (PyObject *)ret;
}

/*NUMPY_API
 * SwapAxes
 */
NPY_NO_EXPORT PyObject *
PyArray_SwapAxes(PyArrayObject *ap, int a1, int a2)
{
    PyArray_Dims new_axes;
    intp dims[MAX_DIMS];
    int n, i, val;
    PyObject *ret;

    if (a1 == a2) {
        Py_INCREF(ap);
        return (PyObject *)ap;
    }

    n = ap->nd;
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
    intp *axes, axis;
    intp i, n;
    intp permutation[MAX_DIMS], reverse_permutation[MAX_DIMS];
    PyArrayObject *ret = NULL;

    if (permute == NULL) {
        n = ap->nd;
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    else {
        n = permute->len;
        axes = permute->ptr;
        if (n != ap->nd) {
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
                axis = ap->nd + axis;
            }
            if (axis < 0 || axis >= ap->nd) {
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
     * incorrectly), sets up descr, and points data at ap->data.
     */
    Py_INCREF(ap->descr);
    ret = (PyArrayObject *)\
        PyArray_NewFromDescr(ap->ob_type,
                             ap->descr,
                             n, ap->dimensions,
                             NULL, ap->data, ap->flags,
                             (PyObject *)ap);
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    ret->base = (PyObject *)ap;
    Py_INCREF(ap);

    /* fix the dimensions and strides of the return-array */
    for (i = 0; i < n; i++) {
        ret->dimensions[i] = ap->dimensions[permutation[i]];
        ret->strides[i] = ap->strides[permutation[i]];
    }
    PyArray_UpdateFlags(ret, CONTIGUOUS | FORTRAN);
    return (PyObject *)ret;
}

/*NUMPY_API
 * Ravel
 * Returns a contiguous array
 */
NPY_NO_EXPORT PyObject *
PyArray_Ravel(PyArrayObject *a, NPY_ORDER fortran)
{
    PyArray_Dims newdim = {NULL,1};
    intp val[1] = {-1};

    if (fortran == PyArray_ANYORDER) {
        fortran = PyArray_ISFORTRAN(a);
    }
    newdim.ptr = val;
    if (!fortran && PyArray_ISCONTIGUOUS(a)) {
        return PyArray_Newshape(a, &newdim, PyArray_CORDER);
    }
    else if (fortran && PyArray_ISFORTRAN(a)) {
        return PyArray_Newshape(a, &newdim, PyArray_FORTRANORDER);
    }
    else {
        return PyArray_Flatten(a, fortran);
    }
}

/*NUMPY_API
 * Flatten
 */
NPY_NO_EXPORT PyObject *
PyArray_Flatten(PyArrayObject *a, NPY_ORDER order)
{
    PyObject *ret;
    intp size;

    if (order == PyArray_ANYORDER) {
        order = PyArray_ISFORTRAN(a);
    }
    size = PyArray_SIZE(a);
    Py_INCREF(a->descr);
    ret = PyArray_NewFromDescr(a->ob_type,
                               a->descr,
                               1, &size,
                               NULL,
                               NULL,
                               0, (PyObject *)a);

    if (ret == NULL) {
        return NULL;
    }
    if (_flat_copyinto(ret, (PyObject *)a, order) < 0) {
        Py_DECREF(ret);
        return NULL;
    }
    return ret;
}


