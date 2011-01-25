/*
 * Python Universal Functions Object -- Math for all types, plus fast
 * arrays math
 *
 * Full description
 *
 * This supports mathematical (and Boolean) functions on arrays and other python
 * objects.  Math on large arrays of basic C types is rather efficient.
 *
 * Travis E. Oliphant  2005, 2006 oliphant@ee.byu.edu (oliphant.travis@ieee.org)
 * Brigham Young University
 *
 * based on the
 *
 * Original Implementation:
 * Copyright (c) 1995, 1996, 1997 Jim Hugunin, hugunin@mit.edu
 *
 * with inspiration and code from
 * Numarray
 * Space Science Telescope Institute
 * J. Todd Miller
 * Perry Greenfield
 * Rick White
 *
 */
#define _UMATHMODULE

#include "Python.h"

#include "npy_config.h"
#ifdef ENABLE_SEPARATE_COMPILATION
#define PY_ARRAY_UNIQUE_SYMBOL _npy_umathmodule_ARRAY_API
#define NO_IMPORT_ARRAY
#endif

#include "numpy/npy_3kcompat.h"

#include "numpy/noprefix.h"
#include "numpy/ufuncobject.h"
#include "lowlevel_strided_loops.h"

#include "ufunc_object.h"

/********** PRINTF DEBUG TRACING **************/
#define NPY_UF_DBG_TRACING 0

#if NPY_UF_DBG_TRACING
#define NPY_UF_DBG_PRINTF(...) printf(__VA_ARGS__)
#else
#define NPY_UF_DBG_PRINTF(...)
#endif
/**********************************************/


/********************/
#define USE_USE_DEFAULTS 1
#define USE_NEW_ITERATOR_GENFUNC 1
/********************/

/* ---------------------------------------------------------------- */

static int
_does_loop_use_arrays(void *data);

/*
 * fpstatus is the ufunc_formatted hardware status
 * errmask is the handling mask specified by the user.
 * errobj is a Python object with (string, callable object or None)
 * or NULL
 */

/*
 * 2. for each of the flags
 * determine whether to ignore, warn, raise error, or call Python function.
 * If ignore, do nothing
 * If warn, print a warning and continue
 * If raise return an error
 * If call, call a user-defined function with string
 */

static int
_error_handler(int method, PyObject *errobj, char *errtype, int retstatus, int *first)
{
    PyObject *pyfunc, *ret, *args;
    char *name = PyBytes_AS_STRING(PyTuple_GET_ITEM(errobj,0));
    char msg[100];
    ALLOW_C_API_DEF;

    ALLOW_C_API;
    switch(method) {
    case UFUNC_ERR_WARN:
        PyOS_snprintf(msg, sizeof(msg), "%s encountered in %s", errtype, name);
        if (PyErr_Warn(PyExc_RuntimeWarning, msg) < 0) {
            goto fail;
        }
        break;
    case UFUNC_ERR_RAISE:
        PyErr_Format(PyExc_FloatingPointError, "%s encountered in %s",
                errtype, name);
        goto fail;
    case UFUNC_ERR_CALL:
        pyfunc = PyTuple_GET_ITEM(errobj, 1);
        if (pyfunc == Py_None) {
            PyErr_Format(PyExc_NameError,
                    "python callback specified for %s (in " \
                    " %s) but no function found.",
                    errtype, name);
            goto fail;
        }
        args = Py_BuildValue("NN", PyUString_FromString(errtype),
                PyInt_FromLong((long) retstatus));
        if (args == NULL) {
            goto fail;
        }
        ret = PyObject_CallObject(pyfunc, args);
        Py_DECREF(args);
        if (ret == NULL) {
            goto fail;
        }
        Py_DECREF(ret);
        break;
    case UFUNC_ERR_PRINT:
        if (*first) {
            fprintf(stderr, "Warning: %s encountered in %s\n", errtype, name);
            *first = 0;
        }
        break;
    case UFUNC_ERR_LOG:
        if (first) {
            *first = 0;
            pyfunc = PyTuple_GET_ITEM(errobj, 1);
            if (pyfunc == Py_None) {
                PyErr_Format(PyExc_NameError,
                        "log specified for %s (in %s) but no " \
                        "object with write method found.",
                        errtype, name);
                goto fail;
            }
            PyOS_snprintf(msg, sizeof(msg),
                    "Warning: %s encountered in %s\n", errtype, name);
            ret = PyObject_CallMethod(pyfunc, "write", "s", msg);
            if (ret == NULL) {
                goto fail;
            }
            Py_DECREF(ret);
        }
        break;
    }
    DISABLE_C_API;
    return 0;

fail:
    DISABLE_C_API;
    return -1;
}


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_getfperr(void)
{
    int retstatus;
    UFUNC_CHECK_STATUS(retstatus);
    return retstatus;
}

#define HANDLEIT(NAME, str) {if (retstatus & UFUNC_FPE_##NAME) {        \
            handle = errmask & UFUNC_MASK_##NAME;                       \
            if (handle &&                                               \
                _error_handler(handle >> UFUNC_SHIFT_##NAME,            \
                               errobj, str, retstatus, first) < 0)      \
                return -1;                                              \
        }}

/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_handlefperr(int errmask, PyObject *errobj, int retstatus, int *first)
{
    int handle;
    if (errmask && retstatus) {
        HANDLEIT(DIVIDEBYZERO, "divide by zero");
        HANDLEIT(OVERFLOW, "overflow");
        HANDLEIT(UNDERFLOW, "underflow");
        HANDLEIT(INVALID, "invalid value");
    }
    return 0;
}

#undef HANDLEIT


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_checkfperr(int errmask, PyObject *errobj, int *first)
{
    int retstatus;

    /* 1. check hardware flag --- this is platform dependent code */
    retstatus = PyUFunc_getfperr();
    return PyUFunc_handlefperr(errmask, errobj, retstatus, first);
}


/* Checking the status flag clears it */
/*UFUNC_API*/
NPY_NO_EXPORT void
PyUFunc_clearfperr()
{
    PyUFunc_getfperr();
}


#define NO_UFUNCLOOP        0
#define ZERO_EL_REDUCELOOP  0
#define ONE_UFUNCLOOP       1
#define ONE_EL_REDUCELOOP   1
#define NOBUFFER_UFUNCLOOP  2
#define NOBUFFER_REDUCELOOP 2
#define BUFFER_UFUNCLOOP    3
#define BUFFER_REDUCELOOP   3
#define SIGNATURE_NOBUFFER_UFUNCLOOP 4


static char
_lowest_type(char intype)
{
    switch(intype) {
    /* case PyArray_BYTE */
    case PyArray_SHORT:
    case PyArray_INT:
    case PyArray_LONG:
    case PyArray_LONGLONG:
    case PyArray_DATETIME:
    case PyArray_TIMEDELTA:
        return PyArray_BYTE;
    /* case PyArray_UBYTE */
    case PyArray_USHORT:
    case PyArray_UINT:
    case PyArray_ULONG:
    case PyArray_ULONGLONG:
        return PyArray_UBYTE;
    /* case PyArray_HALF: */
    case PyArray_FLOAT:
    case PyArray_DOUBLE:
    case PyArray_LONGDOUBLE:
        return PyArray_HALF;
    /* case PyArray_CFLOAT:*/
    case PyArray_CDOUBLE:
    case PyArray_CLONGDOUBLE:
        return PyArray_CFLOAT;
    default:
        return intype;
    }
}

static char *_types_msg =  "function not supported for these types, "   \
    "and can't coerce safely to supported types";

/*
 * This function analyzes the input arguments
 * and determines an appropriate __array_prepare__ function to call
 * for the outputs.
 *
 * If an output argument is provided, then it is prepped
 * with its own __array_prepare__ not with the one determined by
 * the input arguments.
 *
 * if the provided output argument is already an ndarray,
 * the prepping function is None (which means no prepping will
 * be done --- not even PyArray_Return).
 *
 * A NULL is placed in output_prep for outputs that
 * should just have PyArray_Return called.
 */
static void
_find_array_prepare(PyObject *args, PyObject *kwds,
                    PyObject **output_prep, int nin, int nout)
{
    Py_ssize_t nargs;
    int i;
    int np = 0;
    PyObject *with_prep[NPY_MAXARGS], *preps[NPY_MAXARGS];
    PyObject *obj, *prep = NULL;

    nargs = PyTuple_GET_SIZE(args);
    for (i = 0; i < nin; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        prep = PyObject_GetAttrString(obj, "__array_prepare__");
        if (prep) {
            if (PyCallable_Check(prep)) {
                with_prep[np] = obj;
                preps[np] = prep;
                ++np;
            }
            else {
                Py_DECREF(prep);
                prep = NULL;
            }
        }
        else {
            PyErr_Clear();
        }
    }
    if (np > 0) {
        /* If we have some preps defined, find the one of highest priority */
        prep = preps[0];
        if (np > 1) {
            double maxpriority = PyArray_GetPriority(with_prep[0],
                        PyArray_SUBTYPE_PRIORITY);
            for (i = 1; i < np; ++i) {
                double priority = PyArray_GetPriority(with_prep[i],
                            PyArray_SUBTYPE_PRIORITY);
                if (priority > maxpriority) {
                    maxpriority = priority;
                    Py_DECREF(prep);
                    prep = preps[i];
                }
                else {
                    Py_DECREF(preps[i]);
                }
            }
        }
    }

    /*
     * Here prep is the prepping function determined from the
     * input arrays (could be NULL).
     *
     * For all the output arrays decide what to do.
     *
     * 1) Use the prep function determined from the input arrays
     * This is the default if the output array is not
     * passed in.
     *
     * 2) Use the __array_prepare__ method of the output object.
     * This is special cased for
     * exact ndarray so that no PyArray_Return is
     * done in that case.
     */
    for (i = 0; i < nout; i++) {
        int j = nin + i;
        int incref = 1;
        output_prep[i] = prep;
        obj = NULL;
        if (j < nargs) {
            obj = PyTuple_GET_ITEM(args, j);
            /* Output argument one may also be in a keyword argument */
            if (i == 0 && obj == Py_None && kwds != NULL) {
                obj = PyDict_GetItemString(kwds, "out");
            }
        }
        /* Output argument one may also be in a keyword argument */
        else if (i == 0 && kwds != NULL) {
            obj = PyDict_GetItemString(kwds, "out");
        }

        if (obj != Py_None && obj != NULL) {
            if (PyArray_CheckExact(obj)) {
                /* None signals to not call any wrapping */
                output_prep[i] = Py_None;
            }
            else {
                PyObject *oprep = PyObject_GetAttrString(obj,
                            "__array_prepare__");
                incref = 0;
                if (!(oprep) || !(PyCallable_Check(oprep))) {
                    Py_XDECREF(oprep);
                    oprep = prep;
                    incref = 1;
                    PyErr_Clear();
                }
                output_prep[i] = oprep;
            }
        }

        if (incref) {
            Py_XINCREF(output_prep[i]);
        }
    }
    Py_XDECREF(prep);
    return;
}

/*
 * Called for non-NULL user-defined functions.
 * The object should be a CObject pointing to a linked-list of functions
 * storing the function, data, and signature of all user-defined functions.
 * There must be a match with the input argument types or an error
 * will occur.
 */
static int
_find_matching_userloop(PyObject *obj, int *arg_types,
                        PyArray_SCALARKIND *scalars,
                        PyUFuncGenericFunction *function, void **data,
                        int nargs, int nin)
{
    PyUFunc_Loop1d *funcdata;
    int i;

    funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
    while (funcdata != NULL) {
        for (i = 0; i < nin; i++) {
            if (!PyArray_CanCoerceScalar(arg_types[i],
                                         funcdata->arg_types[i],
                                         scalars[i]))
                break;
        }
        if (i == nin) {
            /* match found */
            *function = funcdata->func;
            *data = funcdata->data;
            /* Make sure actual arg_types supported by the loop are used */
            for (i = 0; i < nargs; i++) {
                arg_types[i] = funcdata->arg_types[i];
            }
            return 0;
        }
        funcdata = funcdata->next;
    }
    return -1;
}

/*
 * if only one type is specified then it is the "first" output data-type
 * and the first signature matching this output data-type is returned.
 *
 * if a tuple of types is specified then an exact match to the signature
 * is searched and it much match exactly or an error occurs
 */
static int
extract_specified_loop(PyUFuncObject *self, int *arg_types,
                       PyUFuncGenericFunction *function, void **data,
                       PyObject *type_tup, int userdef)
{
    Py_ssize_t n = 1;
    int *rtypenums;
    static char msg[] = "loop written to specified type(s) not found";
    PyArray_Descr *dtype;
    int nargs;
    int i, j;
    int strtype = 0;

    nargs = self->nargs;
    if (PyTuple_Check(type_tup)) {
        n = PyTuple_GET_SIZE(type_tup);
        if (n != 1 && n != nargs) {
            PyErr_Format(PyExc_ValueError,
                         "a type-tuple must be specified " \
                         "of length 1 or %d for %s", nargs,
                         self->name ? self->name : "(unknown)");
            return -1;
        }
    }
    else if (PyString_Check(type_tup)) {
            Py_ssize_t slen;
            char *thestr;

            slen = PyString_GET_SIZE(type_tup);
            thestr = PyString_AS_STRING(type_tup);
            for (i = 0; i < slen - 2; i++) {
                if (thestr[i] == '-' && thestr[i+1] == '>') {
                    break;
                }
            }
            if (i < slen-2) {
                strtype = 1;
                n = slen - 2;
                if (i != self->nin
                    || slen - 2 - i != self->nout) {
                    PyErr_Format(PyExc_ValueError,
                                 "a type-string for %s, "   \
                                 "requires %d typecode(s) before " \
                                 "and %d after the -> sign",
                                 self->name ? self->name : "(unknown)",
                                 self->nin, self->nout);
                    return -1;
                }
            }
        }
    rtypenums = (int *)_pya_malloc(n*sizeof(int));
    if (rtypenums == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    if (strtype) {
        char *ptr;
        ptr = PyString_AS_STRING(type_tup);
        i = 0;
        while (i < n) {
            if (*ptr == '-' || *ptr == '>') {
                ptr++;
                continue;
            }
            dtype = PyArray_DescrFromType((int) *ptr);
            if (dtype == NULL) {
                goto fail;
            }
            rtypenums[i] = dtype->type_num;
            Py_DECREF(dtype);
            ptr++;
            i++;
        }
    }
    else if (PyTuple_Check(type_tup)) {
        for (i = 0; i < n; i++) {
            if (PyArray_DescrConverter(PyTuple_GET_ITEM(type_tup, i),
                                       &dtype) == NPY_FAIL) {
                goto fail;
            }
            rtypenums[i] = dtype->type_num;
            Py_DECREF(dtype);
        }
    }
    else {
        if (PyArray_DescrConverter(type_tup, &dtype) == NPY_FAIL) {
            goto fail;
        }
        rtypenums[0] = dtype->type_num;
        Py_DECREF(dtype);
    }

    if (userdef > 0) {
        /* search in the user-defined functions */
        PyObject *key, *obj;
        PyUFunc_Loop1d *funcdata;

        obj = NULL;
        key = PyInt_FromLong((long) userdef);
        if (key == NULL) {
            goto fail;
        }
        obj = PyDict_GetItem(self->userloops, key);
        Py_DECREF(key);
        if (obj == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "user-defined type used in ufunc" \
                            " with no registered loops");
            goto fail;
        }
        /*
         * extract the correct function
         * data and argtypes
         */
        funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
        while (funcdata != NULL) {
            if (n != 1) {
                for (i = 0; i < nargs; i++) {
                    if (rtypenums[i] != funcdata->arg_types[i]) {
                        break;
                    }
                }
            }
            else if (rtypenums[0] == funcdata->arg_types[self->nin]) {
                i = nargs;
            }
            else {
                i = -1;
            }
            if (i == nargs) {
                *function = funcdata->func;
                *data = funcdata->data;
                for(i = 0; i < nargs; i++) {
                    arg_types[i] = funcdata->arg_types[i];
                }
                Py_DECREF(obj);
                goto finish;
            }
            funcdata = funcdata->next;
        }
        PyErr_SetString(PyExc_TypeError, msg);
        goto fail;
    }

    /* look for match in self->functions */
    for (j = 0; j < self->ntypes; j++) {
        if (n != 1) {
            for(i = 0; i < nargs; i++) {
                if (rtypenums[i] != self->types[j*nargs + i]) {
                    break;
                }
            }
        }
        else if (rtypenums[0] == self->types[j*nargs+self->nin]) {
            i = nargs;
        }
        else {
            i = -1;
        }
        if (i == nargs) {
            *function = self->functions[j];
            *data = self->data[j];
            for (i = 0; i < nargs; i++) {
                arg_types[i] = self->types[j*nargs+i];
            }
            goto finish;
        }
    }
    PyErr_SetString(PyExc_TypeError, msg);

 fail:
    _pya_free(rtypenums);
    return -1;

 finish:
    _pya_free(rtypenums);
    return 0;
}


/*
 * Called to determine coercion
 * Can change arg_types.
 */
static int
select_types(PyUFuncObject *self, int *arg_types,
             PyUFuncGenericFunction *function, void **data,
             PyArray_SCALARKIND *scalars,
             PyObject *typetup)
{
    int i, j;
    char start_type;
    int userdef = -1;
    int userdef_ind = -1;

    if (self->userloops) {
        for(i = 0; i < self->nin; i++) {
            if (PyTypeNum_ISUSERDEF(arg_types[i])) {
                userdef = arg_types[i];
                userdef_ind = i;
                break;
            }
        }
    }

    if (typetup != NULL)
        return extract_specified_loop(self, arg_types, function, data,
                                      typetup, userdef);

    if (userdef > 0) {
        PyObject *key, *obj;
        int ret = -1;
        obj = NULL;

        /*
         * Look through all the registered loops for all the user-defined
         * types to find a match.
         */
        while (ret == -1) {
            if (userdef_ind >= self->nin) {
                break;
            }
            userdef = arg_types[userdef_ind++];
            if (!(PyTypeNum_ISUSERDEF(userdef))) {
                continue;
            }
            key = PyInt_FromLong((long) userdef);
            if (key == NULL) {
                return -1;
            }
            obj = PyDict_GetItem(self->userloops, key);
            Py_DECREF(key);
            if (obj == NULL) {
                continue;
            }
            /*
             * extract the correct function
             * data and argtypes for this user-defined type.
             */
            ret = _find_matching_userloop(obj, arg_types, scalars,
                                          function, data, self->nargs,
                                          self->nin);
        }
        if (ret == 0) {
            return ret;
        }
        PyErr_SetString(PyExc_TypeError, _types_msg);
        return ret;
    }

    start_type = arg_types[0];
    /*
     * If the first argument is a scalar we need to place
     * the start type as the lowest type in the class
     */
    if (scalars[0] != PyArray_NOSCALAR) {
        start_type = _lowest_type(start_type);
    }

    i = 0;
    while (i < self->ntypes && start_type > self->types[i*self->nargs]) {
        i++;
    }
    for (; i < self->ntypes; i++) {
        for (j = 0; j < self->nin; j++) {
            if (!PyArray_CanCoerceScalar(arg_types[j],
                                         self->types[i*self->nargs + j],
                                         scalars[j]))
                break;
        }
        if (j == self->nin) {
            break;
        }
    }
    if (i >= self->ntypes) {
        PyErr_SetString(PyExc_TypeError, _types_msg);
        return -1;
    }
    for (j = 0; j < self->nargs; j++) {
        arg_types[j] = self->types[i*self->nargs+j];
    }
    if (self->data) {
        *data = self->data[i];
    }
    else {
        *data = NULL;
    }
    *function = self->functions[i];

    return 0;
}

#if USE_USE_DEFAULTS==1
static int PyUFunc_NUM_NODEFAULTS = 0;
#endif
static PyObject *PyUFunc_PYVALS_NAME = NULL;


/*
 * Extracts some values from the global pyvals tuple.
 * ref - should hold the global tuple
 * name - is the name of the ufunc (ufuncobj->name)
 * bufsize - receives the buffer size to use
 * errmask - receives the bitmask for error handling
 * errobj - receives the python object to call with the error,
 *          if an error handling method is 'call'
 */
static int
_extract_pyvals(PyObject *ref, char *name, int *bufsize,
                int *errmask, PyObject **errobj)
{
    PyObject *retval;

    *errobj = NULL;
    if (!PyList_Check(ref) || (PyList_GET_SIZE(ref)!=3)) {
        PyErr_Format(PyExc_TypeError, "%s must be a length 3 list.",
                     UFUNC_PYVALS_NAME);
        return -1;
    }

    *bufsize = PyInt_AsLong(PyList_GET_ITEM(ref, 0));
    if ((*bufsize == -1) && PyErr_Occurred()) {
        return -1;
    }
    if ((*bufsize < PyArray_MIN_BUFSIZE)
        || (*bufsize > PyArray_MAX_BUFSIZE)
        || (*bufsize % 16 != 0)) {
        PyErr_Format(PyExc_ValueError,
                     "buffer size (%d) is not in range "
                     "(%"INTP_FMT" - %"INTP_FMT") or not a multiple of 16",
                     *bufsize, (intp) PyArray_MIN_BUFSIZE,
                     (intp) PyArray_MAX_BUFSIZE);
        return -1;
    }

    *errmask = PyInt_AsLong(PyList_GET_ITEM(ref, 1));
    if (*errmask < 0) {
        if (PyErr_Occurred()) {
            return -1;
        }
        PyErr_Format(PyExc_ValueError,
                     "invalid error mask (%d)",
                     *errmask);
        return -1;
    }

    retval = PyList_GET_ITEM(ref, 2);
    if (retval != Py_None && !PyCallable_Check(retval)) {
        PyObject *temp;
        temp = PyObject_GetAttrString(retval, "write");
        if (temp == NULL || !PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError,
                            "python object must be callable or have " \
                            "a callable write method");
            Py_XDECREF(temp);
            return -1;
        }
        Py_DECREF(temp);
    }

    *errobj = Py_BuildValue("NO", PyBytes_FromString(name), retval);
    if (*errobj == NULL) {
        return -1;
    }
    return 0;
}



/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_GetPyValues(char *name, int *bufsize, int *errmask, PyObject **errobj)
{
    PyObject *thedict;
    PyObject *ref = NULL;

#if USE_USE_DEFAULTS==1
    if (PyUFunc_NUM_NODEFAULTS != 0) {
#endif
        if (PyUFunc_PYVALS_NAME == NULL) {
            PyUFunc_PYVALS_NAME = PyUString_InternFromString(UFUNC_PYVALS_NAME);
        }
        thedict = PyThreadState_GetDict();
        if (thedict == NULL) {
            thedict = PyEval_GetBuiltins();
        }
        ref = PyDict_GetItem(thedict, PyUFunc_PYVALS_NAME);
#if USE_USE_DEFAULTS==1
    }
#endif
    if (ref == NULL) {
        *errmask = UFUNC_ERR_DEFAULT;
        *errobj = Py_BuildValue("NO", PyBytes_FromString(name), Py_None);
        *bufsize = PyArray_BUFSIZE;
        return 0;
    }
    return _extract_pyvals(ref, name, bufsize, errmask, errobj);
}

/*
 * Create copies for any arrays that are less than loop->bufsize
 * in total size (or core_enabled) and are mis-behaved or in need
 * of casting.
 */
static int
_create_copies(PyUFuncLoopObject *loop, int *arg_types, PyArrayObject **mps)
{
    int nin = loop->ufunc->nin;
    int i;
    intp size;
    PyObject *new;
    PyArray_Descr *ntype;
    PyArray_Descr *atype;

    for (i = 0; i < nin; i++) {
        size = PyArray_SIZE(mps[i]);
        /*
         * if the type of mps[i] is equivalent to arg_types[i]
         * then set arg_types[i] equal to type of mps[i] for later checking....
         */
        if (PyArray_TYPE(mps[i]) != arg_types[i]) {
            ntype = mps[i]->descr;
            atype = PyArray_DescrFromType(arg_types[i]);
            if (PyArray_EquivTypes(atype, ntype)) {
                arg_types[i] = ntype->type_num;
            }
            Py_DECREF(atype);
        }
        if (size < loop->bufsize || loop->ufunc->core_enabled) {
            if (!(PyArray_ISBEHAVED_RO(mps[i]))
                || PyArray_TYPE(mps[i]) != arg_types[i]) {
                ntype = PyArray_DescrFromType(arg_types[i]);
                new = PyArray_FromAny((PyObject *)mps[i],
                                      ntype, 0, 0,
                                      FORCECAST | ALIGNED, NULL);
                if (new == NULL) {
                        return -1;
                }
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)new;
            }
        }
    }
    return 0;
}

#define _GETATTR_(str, rstr) do {if (strcmp(name, #str) == 0)     \
        return PyObject_HasAttrString(op, "__" #rstr "__");} while (0);

static int
_has_reflected_op(PyObject *op, char *name)
{
    _GETATTR_(add, radd);
    _GETATTR_(subtract, rsub);
    _GETATTR_(multiply, rmul);
    _GETATTR_(divide, rdiv);
    _GETATTR_(true_divide, rtruediv);
    _GETATTR_(floor_divide, rfloordiv);
    _GETATTR_(remainder, rmod);
    _GETATTR_(power, rpow);
    _GETATTR_(left_shift, rlshift);
    _GETATTR_(right_shift, rrshift);
    _GETATTR_(bitwise_and, rand);
    _GETATTR_(bitwise_xor, rxor);
    _GETATTR_(bitwise_or, ror);
    return 0;
}

#undef _GETATTR_


/* Return the position of next non-white-space char in the string */
static int
_next_non_white_space(const char* str, int offset)
{
    int ret = offset;
    while (str[ret] == ' ' || str[ret] == '\t') {
        ret++;
    }
    return ret;
}

static int
_is_alpha_underscore(char ch)
{
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
}

static int
_is_alnum_underscore(char ch)
{
    return _is_alpha_underscore(ch) || (ch >= '0' && ch <= '9');
}

/*
 * Return the ending position of a variable name
 */
static int
_get_end_of_name(const char* str, int offset)
{
    int ret = offset;
    while (_is_alnum_underscore(str[ret])) {
        ret++;
    }
    return ret;
}

/*
 * Returns 1 if the dimension names pointed by s1 and s2 are the same,
 * otherwise returns 0.
 */
static int
_is_same_name(const char* s1, const char* s2)
{
    while (_is_alnum_underscore(*s1) && _is_alnum_underscore(*s2)) {
        if (*s1 != *s2) {
            return 0;
        }
        s1++;
        s2++;
    }
    return !_is_alnum_underscore(*s1) && !_is_alnum_underscore(*s2);
}

/*
 * Sets core_num_dim_ix, core_num_dims, core_dim_ixs, core_offsets,
 * and core_signature in PyUFuncObject "self".  Returns 0 unless an
 * error occured.
 */
static int
_parse_signature(PyUFuncObject *self, const char *signature)
{
    size_t len;
    char const **var_names;
    int nd = 0;             /* number of dimension of the current argument */
    int cur_arg = 0;        /* index into core_num_dims&core_offsets */
    int cur_core_dim = 0;   /* index into core_dim_ixs */
    int i = 0;
    char *parse_error = NULL;

    if (signature == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_parse_signature with NULL signature");
        return -1;
    }

    len = strlen(signature);
    self->core_signature = _pya_malloc(sizeof(char) * (len+1));
    if (self->core_signature) {
        strcpy(self->core_signature, signature);
    }
    /* Allocate sufficient memory to store pointers to all dimension names */
    var_names = _pya_malloc(sizeof(char const*) * len);
    if (var_names == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    self->core_enabled = 1;
    self->core_num_dim_ix = 0;
    self->core_num_dims = _pya_malloc(sizeof(int) * self->nargs);
    self->core_dim_ixs = _pya_malloc(sizeof(int) * len); /* shrink this later */
    self->core_offsets = _pya_malloc(sizeof(int) * self->nargs);
    if (self->core_num_dims == NULL || self->core_dim_ixs == NULL
        || self->core_offsets == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    i = _next_non_white_space(signature, 0);
    while (signature[i] != '\0') {
        /* loop over input/output arguments */
        if (cur_arg == self->nin) {
            /* expect "->" */
            if (signature[i] != '-' || signature[i+1] != '>') {
                parse_error = "expect '->'";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 2);
        }

        /*
         * parse core dimensions of one argument,
         * e.g. "()", "(i)", or "(i,j)"
         */
        if (signature[i] != '(') {
            parse_error = "expect '('";
            goto fail;
        }
        i = _next_non_white_space(signature, i + 1);
        while (signature[i] != ')') {
            /* loop over core dimensions */
            int j = 0;
            if (!_is_alpha_underscore(signature[i])) {
                parse_error = "expect dimension name";
                goto fail;
            }
            while (j < self->core_num_dim_ix) {
                if (_is_same_name(signature+i, var_names[j])) {
                    break;
                }
                j++;
            }
            if (j >= self->core_num_dim_ix) {
                var_names[j] = signature+i;
                self->core_num_dim_ix++;
            }
            self->core_dim_ixs[cur_core_dim] = j;
            cur_core_dim++;
            nd++;
            i = _get_end_of_name(signature, i);
            i = _next_non_white_space(signature, i);
            if (signature[i] != ',' && signature[i] != ')') {
                parse_error = "expect ',' or ')'";
                goto fail;
            }
            if (signature[i] == ',')
            {
                i = _next_non_white_space(signature, i + 1);
                if (signature[i] == ')') {
                    parse_error = "',' must not be followed by ')'";
                    goto fail;
                }
            }
        }
        self->core_num_dims[cur_arg] = nd;
        self->core_offsets[cur_arg] = cur_core_dim-nd;
        cur_arg++;
        nd = 0;

        i = _next_non_white_space(signature, i + 1);
        if (cur_arg != self->nin && cur_arg != self->nargs) {
            /*
             * The list of input arguments (or output arguments) was
             * only read partially
             */
            if (signature[i] != ',') {
                parse_error = "expect ','";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 1);
        }
    }
    if (cur_arg != self->nargs) {
        parse_error = "incomplete signature: not all arguments found";
        goto fail;
    }
    self->core_dim_ixs = _pya_realloc(self->core_dim_ixs,
            sizeof(int)*cur_core_dim);
    /* check for trivial core-signature, e.g. "(),()->()" */
    if (cur_core_dim == 0) {
        self->core_enabled = 0;
    }
    _pya_free((void*)var_names);
    return 0;

fail:
    _pya_free((void*)var_names);
    if (parse_error) {
        char *buf = _pya_malloc(sizeof(char) * (len + 200));
        if (buf) {
            sprintf(buf, "%s at position %d in \"%s\"",
                    parse_error, i, signature);
            PyErr_SetString(PyExc_ValueError, signature);
            _pya_free(buf);
        }
        else {
            PyErr_NoMemory();
        }
    }
    return -1;
}

/*
 * Concatenate the loop and core dimensions of
 * PyArrayMultiIterObject's iarg-th argument, to recover a full
 * dimension array (used for output arguments).
 */
static npy_intp*
_compute_output_dims(PyUFuncLoopObject *loop, int iarg,
                     int *out_nd, npy_intp *tmp_dims)
{
    int i;
    PyUFuncObject *ufunc = loop->ufunc;
    if (ufunc->core_enabled == 0) {
        /* case of ufunc with trivial core-signature */
        *out_nd = loop->nd;
        return loop->dimensions;
    }

    *out_nd = loop->nd + ufunc->core_num_dims[iarg];
    if (*out_nd > NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                        "dimension of output variable exceeds limit");
        return NULL;
    }

    /* copy loop dimensions */
    memcpy(tmp_dims, loop->dimensions, sizeof(npy_intp) * loop->nd);

    /* copy core dimension */
    for (i = 0; i < ufunc->core_num_dims[iarg]; i++) {
        tmp_dims[loop->nd + i] = loop->core_dim_sizes[1 +
            ufunc->core_dim_ixs[ufunc->core_offsets[iarg] + i]];
    }
    return tmp_dims;
}

/* Check and set core_dim_sizes and core_strides for the i-th argument. */
static int
_compute_dimension_size(PyUFuncLoopObject *loop, PyArrayObject **mps, int i)
{
    PyUFuncObject *ufunc = loop->ufunc;
    int j = ufunc->core_offsets[i];
    int k = PyArray_NDIM(mps[i]) - ufunc->core_num_dims[i];
    int ind;
    for (ind = 0; ind < ufunc->core_num_dims[i]; ind++, j++, k++) {
        npy_intp dim = k < 0 ? 1 : PyArray_DIM(mps[i], k);
        /* First element of core_dim_sizes will be used for looping */
        int dim_ix = ufunc->core_dim_ixs[j] + 1;
        if (loop->core_dim_sizes[dim_ix] == 1) {
            /* broadcast core dimension  */
            loop->core_dim_sizes[dim_ix] = dim;
        }
        else if (dim != 1 && dim != loop->core_dim_sizes[dim_ix]) {
            PyErr_SetString(PyExc_ValueError, "core dimensions mismatch");
            return -1;
        }
        /* First ufunc->nargs elements will be used for looping */
        loop->core_strides[ufunc->nargs + j] =
            dim == 1 ? 0 : PyArray_STRIDE(mps[i], k);
    }
    return 0;
}

/* Return a view of array "ap" with "core_nd" dimensions cut from tail. */
static PyArrayObject *
_trunc_coredim(PyArrayObject *ap, int core_nd)
{
    PyArrayObject *ret;
    int nd = ap->nd - core_nd;

    if (nd < 0) {
        nd = 0;
    }
    /* The following code is basically taken from PyArray_Transpose */
    /* NewFromDescr will steal this reference */
    Py_INCREF(ap->descr);
    ret = (PyArrayObject *)
        PyArray_NewFromDescr(Py_TYPE(ap), ap->descr,
                             nd, ap->dimensions,
                             ap->strides, ap->data, ap->flags,
                             (PyObject *)ap);
    if (ret == NULL) {
        return NULL;
    }
    /* point at true owner of memory: */
    ret->base = (PyObject *)ap;
    Py_INCREF(ap);
    PyArray_UpdateFlags(ret, CONTIGUOUS | FORTRAN);
    return ret;
}

static Py_ssize_t
construct_arrays(PyUFuncLoopObject *loop, PyObject *args, PyArrayObject **mps,
                 PyObject *typetup)
{
    Py_ssize_t nargs;
    int i;
    int arg_types[NPY_MAXARGS];
    PyArray_SCALARKIND scalars[NPY_MAXARGS];
    PyArray_SCALARKIND maxarrkind, maxsckind, new;
    PyUFuncObject *self = loop->ufunc;
    Bool allscalars = TRUE;
    PyTypeObject *subtype = &PyArray_Type;
    PyObject *context = NULL;
    PyObject *obj;
    int flexible = 0;
    int object = 0;

    npy_intp temp_dims[NPY_MAXDIMS];
    npy_intp *out_dims;
    int out_nd;
    PyObject *wraparr[NPY_MAXARGS];

    /* Check number of arguments */
    nargs = PyTuple_Size(args);
    if ((nargs < self->nin) || (nargs > self->nargs)) {
        PyErr_SetString(PyExc_ValueError, "invalid number of arguments");
        return -1;
    }

    /* Get each input argument */
    maxarrkind = PyArray_NOSCALAR;
    maxsckind = PyArray_NOSCALAR;
    for(i = 0; i < self->nin; i++) {
        obj = PyTuple_GET_ITEM(args,i);
        if (!PyArray_Check(obj) && !PyArray_IsScalar(obj, Generic)) {
            context = Py_BuildValue("OOi", self, args, i);
        }
        else {
            context = NULL;
        }
        mps[i] = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, context);
        Py_XDECREF(context);
        if (mps[i] == NULL) {
            return -1;
        }
        arg_types[i] = PyArray_TYPE(mps[i]);
        if (!flexible && PyTypeNum_ISFLEXIBLE(arg_types[i])) {
            flexible = 1;
        }
        if (!object && PyTypeNum_ISOBJECT(arg_types[i])) {
            object = 1;
        }
        /*
         * debug
         * fprintf(stderr, "array %d has reference %d\n", i,
         * (mps[i])->ob_refcnt);
         */

        /*
         * Scalars are 0-dimensional arrays at this point
         */

        /*
         * We need to keep track of whether or not scalars
         * are mixed with arrays of different kinds.
         */

        if (mps[i]->nd > 0) {
            scalars[i] = PyArray_NOSCALAR;
            allscalars = FALSE;
            new = PyArray_ScalarKind(arg_types[i], NULL);
            maxarrkind = NPY_MAX(new, maxarrkind);
        }
        else {
            scalars[i] = PyArray_ScalarKind(arg_types[i], &(mps[i]));
            maxsckind = NPY_MAX(scalars[i], maxsckind);
        }
    }

    /* We don't do strings */
    if (flexible && !object) {
        loop->notimplemented = 1;
        return nargs;
    }

    /*
     * If everything is a scalar, or scalars mixed with arrays of
     * different kinds of lesser kinds then use normal coercion rules
     */
    if (allscalars || (maxsckind > maxarrkind)) {
        for (i = 0; i < self->nin; i++) {
            scalars[i] = PyArray_NOSCALAR;
        }
    }

    /* Select an appropriate function for these argument types. */
    if (select_types(loop->ufunc, arg_types, &(loop->function),
                     &(loop->funcdata), scalars, typetup) == -1) {
        return -1;
    }
    /*
     * FAIL with NotImplemented if the other object has
     * the __r<op>__ method and has __array_priority__ as
     * an attribute (signalling it can handle ndarray's)
     * and is not already an ndarray or a subtype of the same type.
     */
    if ((arg_types[1] == PyArray_OBJECT)
        && (loop->ufunc->nin==2) && (loop->ufunc->nout == 1)) {
        PyObject *_obj = PyTuple_GET_ITEM(args, 1);
        if (!PyArray_CheckExact(_obj)
            /* If both are same subtype of object arrays, then proceed */
            && !(Py_TYPE(_obj) == Py_TYPE(PyTuple_GET_ITEM(args, 0)))
            && PyObject_HasAttrString(_obj, "__array_priority__")
            && _has_reflected_op(_obj, loop->ufunc->name)) {
            loop->notimplemented = 1;
            return nargs;
        }
    }

    /*
     * Create copies for some of the arrays if they are small
     * enough and not already contiguous
     */
    if (_create_copies(loop, arg_types, mps) < 0) {
        return -1;
    }

    /*
     * Only use loop dimensions when constructing Iterator:
     * temporarily replace mps[i] (will be recovered below).
     */
    if (self->core_enabled) {
        for (i = 0; i < self->nin; i++) {
            PyArrayObject *ao;

            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
            ao = _trunc_coredim(mps[i], self->core_num_dims[i]);
            if (ao == NULL) {
                return -1;
            }
            mps[i] = ao;
        }
    }

    /* Create Iterators for the Inputs */
    for (i = 0; i < self->nin; i++) {
        loop->iters[i] = (PyArrayIterObject *)
            PyArray_IterNew((PyObject *)mps[i]);
        if (loop->iters[i] == NULL) {
            return -1;
        }
    }

    /* Recover mps[i]. */
    if (self->core_enabled) {
        for (i = 0; i < self->nin; i++) {
            PyArrayObject *ao = mps[i];
            mps[i] = (PyArrayObject *)mps[i]->base;
            Py_DECREF(ao);
        }
    }

    /* Broadcast the result */
    loop->numiter = self->nin;
    if (PyArray_Broadcast((PyArrayMultiIterObject *)loop) < 0) {
        return -1;
    }

    /* Get any return arguments */
    for (i = self->nin; i < nargs; i++) {
        mps[i] = (PyArrayObject *)PyTuple_GET_ITEM(args, i);
        if (((PyObject *)mps[i])==Py_None) {
            mps[i] = NULL;
            continue;
        }
        Py_INCREF(mps[i]);
        if (!PyArray_Check((PyObject *)mps[i])) {
            PyObject *new;
            if (PyArrayIter_Check(mps[i])) {
                new = PyObject_CallMethod((PyObject *)mps[i],
                                          "__array__", NULL);
                Py_DECREF(mps[i]);
                mps[i] = (PyArrayObject *)new;
            }
            else {
                PyErr_SetString(PyExc_TypeError,
                                "return arrays must be "\
                                "of ArrayType");
                Py_DECREF(mps[i]);
                mps[i] = NULL;
                return -1;
            }
        }

        if (self->core_enabled) {
            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
        }
        out_dims = _compute_output_dims(loop, i, &out_nd, temp_dims);
        if (!out_dims) {
            return -1;
        }
        if (mps[i]->nd != out_nd
            || !PyArray_CompareLists(mps[i]->dimensions, out_dims, out_nd)) {
            PyErr_SetString(PyExc_ValueError, "invalid return array shape");
            Py_DECREF(mps[i]);
            mps[i] = NULL;
            return -1;
        }
        if (!PyArray_ISWRITEABLE(mps[i])) {
            PyErr_SetString(PyExc_ValueError, "return array is not writeable");
            Py_DECREF(mps[i]);
            mps[i] = NULL;
            return -1;
        }
    }

    /* construct any missing return arrays and make output iterators */
    for(i = self->nin; i < self->nargs; i++) {
        PyArray_Descr *ntype;

        if (mps[i] == NULL) {
            out_dims = _compute_output_dims(loop, i, &out_nd, temp_dims);
            if (!out_dims) {
                return -1;
            }
            mps[i] = (PyArrayObject *)PyArray_New(subtype,
                                                  out_nd,
                                                  out_dims,
                                                  arg_types[i],
                                                  NULL, NULL,
                                                  0, 0, NULL);
            if (mps[i] == NULL) {
                return -1;
            }
        }

        /*
         * reset types for outputs that are equivalent
         * -- no sense casting uselessly
         */
        else {
            if (mps[i]->descr->type_num != arg_types[i]) {
                PyArray_Descr *atype;
                ntype = mps[i]->descr;
                atype = PyArray_DescrFromType(arg_types[i]);
                if (PyArray_EquivTypes(atype, ntype)) {
                    arg_types[i] = ntype->type_num;
                }
                Py_DECREF(atype);
            }

            /* still not the same -- or will we have to use buffers?*/
            if (mps[i]->descr->type_num != arg_types[i]
                || !PyArray_ISBEHAVED_RO(mps[i])) {
                if (loop->size < loop->bufsize || self->core_enabled) {
                    PyObject *new;
                    /*
                     * Copy the array to a temporary copy
                     * and set the UPDATEIFCOPY flag
                     */
                    ntype = PyArray_DescrFromType(arg_types[i]);
                    new = PyArray_FromAny((PyObject *)mps[i],
                                          ntype, 0, 0,
                                          FORCECAST | ALIGNED |
                                          UPDATEIFCOPY, NULL);
                    if (new == NULL) {
                        return -1;
                    }
                    Py_DECREF(mps[i]);
                    mps[i] = (PyArrayObject *)new;
                }
            }
        }

        if (self->core_enabled) {
            PyArrayObject *ao;

            /* computer for all output arguments, and set strides in "loop" */
            if (_compute_dimension_size(loop, mps, i) < 0) {
                return -1;
            }
            ao = _trunc_coredim(mps[i], self->core_num_dims[i]);
            if (ao == NULL) {
                return -1;
            }
            /* Temporarily modify mps[i] for constructing iterator. */
            mps[i] = ao;
        }

        loop->iters[i] = (PyArrayIterObject *)
            PyArray_IterNew((PyObject *)mps[i]);
        if (loop->iters[i] == NULL) {
            return -1;
        }

        /* Recover mps[i]. */
        if (self->core_enabled) {
            PyArrayObject *ao = mps[i];
            mps[i] = (PyArrayObject *)mps[i]->base;
            Py_DECREF(ao);
        }

    }

    /*
     * Use __array_prepare__ on all outputs
     * if present on one of the input arguments.
     * If present for multiple inputs:
     * use __array_prepare__ of input object with largest
     * __array_priority__ (default = 0.0)
     *
     * Exception:  we should not wrap outputs for items already
     * passed in as output-arguments.  These items should either
     * be left unwrapped or wrapped by calling their own __array_prepare__
     * routine.
     *
     * For each output argument, wrap will be either
     * NULL --- call PyArray_Return() -- default if no output arguments given
     * None --- array-object passed in don't call PyArray_Return
     * method --- the __array_prepare__ method to call.
     */
    _find_array_prepare(args, NULL, wraparr,
                        loop->ufunc->nin, loop->ufunc->nout);

    /* wrap outputs */
    for (i = 0; i < loop->ufunc->nout; i++) {
        int j = loop->ufunc->nin+i;
        PyObject *wrap;
        PyObject *res;
        wrap = wraparr[i];
        if (wrap != NULL) {
            if (wrap == Py_None) {
                Py_DECREF(wrap);
                continue;
            }
            res = PyObject_CallFunction(wrap, "O(OOi)",
                        mps[j], loop->ufunc, args, i);
            Py_DECREF(wrap);
            if ((res == NULL) || (res == Py_None)) {
                if (!PyErr_Occurred()){
                    PyErr_SetString(PyExc_TypeError,
                            "__array_prepare__ must return an ndarray or subclass thereof");
                }
                return -1;
            }
            Py_DECREF(mps[j]);
            mps[j] = (PyArrayObject *)res;
        }
    }

    /*
     * If any of different type, or misaligned or swapped
     * then must use buffers
     */
    loop->bufcnt = 0;
    loop->obj = 0;
    /* Determine looping method needed */
    loop->meth = NO_UFUNCLOOP;
    if (loop->size == 0) {
        return nargs;
    }
    if (self->core_enabled) {
        loop->meth = SIGNATURE_NOBUFFER_UFUNCLOOP;
    }
    for (i = 0; i < self->nargs; i++) {
        loop->needbuffer[i] = 0;
        if (arg_types[i] != mps[i]->descr->type_num
            || !PyArray_ISBEHAVED_RO(mps[i])) {
            if (self->core_enabled) {
                PyErr_SetString(PyExc_RuntimeError,
                                "never reached; copy should have been made");
                return -1;
            }
            loop->meth = BUFFER_UFUNCLOOP;
            loop->needbuffer[i] = 1;
        }
        if (!(loop->obj & UFUNC_OBJ_ISOBJECT)
                && ((mps[i]->descr->type_num == PyArray_OBJECT)
                    || (arg_types[i] == PyArray_OBJECT))) {
            loop->obj = UFUNC_OBJ_ISOBJECT|UFUNC_OBJ_NEEDS_API;
        }
        if (!(loop->obj & UFUNC_OBJ_NEEDS_API)
                && ((mps[i]->descr->type_num == PyArray_DATETIME)
                    || (mps[i]->descr->type_num == PyArray_TIMEDELTA)
                    || (arg_types[i] == PyArray_DATETIME)
                    || (arg_types[i] == PyArray_TIMEDELTA))) {
            loop->obj = UFUNC_OBJ_NEEDS_API;
        }
    }

    if (self->core_enabled && (loop->obj & UFUNC_OBJ_ISOBJECT)) {
        PyErr_SetString(PyExc_TypeError,
                        "Object type not allowed in ufunc with signature");
        return -1;
    }
    if (loop->meth == NO_UFUNCLOOP) {
        loop->meth = ONE_UFUNCLOOP;

        /* All correct type and BEHAVED */
        /* Check for non-uniform stridedness */
        for (i = 0; i < self->nargs; i++) {
            if (!(loop->iters[i]->contiguous)) {
                /*
                 * May still have uniform stride
                 * if (broadcast result) <= 1-d
                 */
                if (mps[i]->nd != 0 &&                  \
                    (loop->iters[i]->nd_m1 > 0)) {
                    loop->meth = NOBUFFER_UFUNCLOOP;
                    break;
                }
            }
        }
        if (loop->meth == ONE_UFUNCLOOP) {
            for (i = 0; i < self->nargs; i++) {
                loop->bufptr[i] = mps[i]->data;
            }
        }
    }

    loop->numiter = self->nargs;

    /* Fill in steps  */
    if (loop->meth == SIGNATURE_NOBUFFER_UFUNCLOOP && loop->nd == 0) {
        /* Use default core_strides */
    }
    else if (loop->meth != ONE_UFUNCLOOP) {
        int ldim;
        intp minsum;
        intp maxdim;
        PyArrayIterObject *it;
        intp stride_sum[NPY_MAXDIMS];
        int j;

        /* Fix iterators */

        /*
         * Optimize axis the iteration takes place over
         *
         * The first thought was to have the loop go
         * over the largest dimension to minimize the number of loops
         *
         * However, on processors with slow memory bus and cache,
         * the slowest loops occur when the memory access occurs for
         * large strides.
         *
         * Thus, choose the axis for which strides of the last iterator is
         * smallest but non-zero.
         */
        for (i = 0; i < loop->nd; i++) {
            stride_sum[i] = 0;
            for (j = 0; j < loop->numiter; j++) {
                stride_sum[i] += loop->iters[j]->strides[i];
            }
        }

        ldim = loop->nd - 1;
        minsum = stride_sum[loop->nd - 1];
        for (i = loop->nd - 2; i >= 0; i--) {
            if (stride_sum[i] < minsum ) {
                ldim = i;
                minsum = stride_sum[i];
            }
        }
        maxdim = loop->dimensions[ldim];
        loop->size /= maxdim;
        loop->bufcnt = maxdim;
        loop->lastdim = ldim;

        /*
         * Fix the iterators so the inner loop occurs over the
         * largest dimensions -- This can be done by
         * setting the size to 1 in that dimension
         * (just in the iterators)
         */
        for (i = 0; i < loop->numiter; i++) {
            it = loop->iters[i];
            it->contiguous = 0;
            it->size /= (it->dims_m1[ldim] + 1);
            it->dims_m1[ldim] = 0;
            it->backstrides[ldim] = 0;

            /*
             * (won't fix factors because we
             * don't use PyArray_ITER_GOTO1D
             * so don't change them)
             *
             * Set the steps to the strides in that dimension
             */
            loop->steps[i] = it->strides[ldim];
        }

        /*
         * Set looping part of core_dim_sizes and core_strides.
         */
        if (loop->meth == SIGNATURE_NOBUFFER_UFUNCLOOP) {
            loop->core_dim_sizes[0] = maxdim;
            for (i = 0; i < self->nargs; i++) {
                loop->core_strides[i] = loop->steps[i];
            }
        }

        /*
         * fix up steps where we will be copying data to
         * buffers and calculate the ninnerloops and leftover
         * values -- if step size is already zero that is not changed...
         */
        if (loop->meth == BUFFER_UFUNCLOOP) {
            loop->leftover = maxdim % loop->bufsize;
            loop->ninnerloops = (maxdim / loop->bufsize) + 1;
            for (i = 0; i < self->nargs; i++) {
                if (loop->needbuffer[i] && loop->steps[i]) {
                    loop->steps[i] = mps[i]->descr->elsize;
                }
                /* These are changed later if casting is needed */
            }
        }
    }
    else if (loop->meth == ONE_UFUNCLOOP) {
        /* uniformly-strided case */
        for (i = 0; i < self->nargs; i++) {
            if (PyArray_SIZE(mps[i]) == 1) {
                loop->steps[i] = 0;
            }
            else {
                loop->steps[i] = mps[i]->strides[mps[i]->nd - 1];
            }
        }
    }


    /* Finally, create memory for buffers if we need them */

    /*
     * Buffers for scalars are specially made small -- scalars are
     * not copied multiple times
     */
    if (loop->meth == BUFFER_UFUNCLOOP) {
        int cnt = 0, cntcast = 0;
        int scnt = 0, scntcast = 0;
        char *castptr;
        char *bufptr;
        int last_was_scalar = 0;
        int last_cast_was_scalar = 0;
        int oldbufsize = 0;
        int oldsize = 0;
        int scbufsize = 4*sizeof(double);
        int memsize;
        PyArray_Descr *descr;

        /* compute the element size */
        for (i = 0; i < self->nargs; i++) {
            if (!loop->needbuffer[i]) {
                continue;
            }
            if (arg_types[i] != mps[i]->descr->type_num) {
                descr = PyArray_DescrFromType(arg_types[i]);
                if (loop->steps[i]) {
                    cntcast += descr->elsize;
                }
                else {
                    scntcast += descr->elsize;
                }
                if (i < self->nin) {
                    loop->cast[i] = PyArray_GetCastFunc(mps[i]->descr,
                                            arg_types[i]);
                }
                else {
                    loop->cast[i] = PyArray_GetCastFunc \
                        (descr, mps[i]->descr->type_num);
                }
                Py_DECREF(descr);
                if (!loop->cast[i]) {
                    return -1;
                }
            }
            loop->swap[i] = !(PyArray_ISNOTSWAPPED(mps[i]));
            if (loop->steps[i]) {
                cnt += mps[i]->descr->elsize;
            }
            else {
                scnt += mps[i]->descr->elsize;
            }
        }
        memsize = loop->bufsize*(cnt+cntcast) + scbufsize*(scnt+scntcast);
        loop->buffer[0] = PyDataMem_NEW(memsize);

        /*
         * debug
         * fprintf(stderr, "Allocated buffer at %p of size %d, cnt=%d, cntcast=%d\n",
         *               loop->buffer[0], loop->bufsize * (cnt + cntcast), cnt, cntcast);
         */
        if (loop->buffer[0] == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        if (loop->obj & UFUNC_OBJ_ISOBJECT) {
            memset(loop->buffer[0], 0, memsize);
        }
        castptr = loop->buffer[0] + loop->bufsize*cnt + scbufsize*scnt;
        bufptr = loop->buffer[0];
        loop->objfunc = 0;
        for (i = 0; i < self->nargs; i++) {
            if (!loop->needbuffer[i]) {
                continue;
            }
            loop->buffer[i] = bufptr + (last_was_scalar ? scbufsize :
                                        loop->bufsize)*oldbufsize;
            last_was_scalar = (loop->steps[i] == 0);
            bufptr = loop->buffer[i];
            oldbufsize = mps[i]->descr->elsize;
            /* fprintf(stderr, "buffer[%d] = %p\n", i, loop->buffer[i]); */
            if (loop->cast[i]) {
                PyArray_Descr *descr;
                loop->castbuf[i] = castptr + (last_cast_was_scalar ? scbufsize :
                                              loop->bufsize)*oldsize;
                last_cast_was_scalar = last_was_scalar;
                /* fprintf(stderr, "castbuf[%d] = %p\n", i, loop->castbuf[i]); */
                descr = PyArray_DescrFromType(arg_types[i]);
                oldsize = descr->elsize;
                Py_DECREF(descr);
                loop->bufptr[i] = loop->castbuf[i];
                castptr = loop->castbuf[i];
                if (loop->steps[i]) {
                    loop->steps[i] = oldsize;
                }
            }
            else {
                loop->bufptr[i] = loop->buffer[i];
            }
            if (!loop->objfunc && (loop->obj & UFUNC_OBJ_ISOBJECT)) {
                if (arg_types[i] == PyArray_OBJECT) {
                    loop->objfunc = 1;
                }
            }
        }
    }

    if (_does_loop_use_arrays(loop->funcdata)) {
        loop->funcdata = (void*)mps;
    }

    return nargs;
}

static void
ufuncreduce_dealloc(PyUFuncReduceObject *self)
{
    if (self->ufunc) {
        Py_XDECREF(self->it);
        Py_XDECREF(self->rit);
        Py_XDECREF(self->ret);
        Py_XDECREF(self->errobj);
        Py_XDECREF(self->decref);
        if (self->buffer) {
            PyDataMem_FREE(self->buffer);
        }
        Py_DECREF(self->ufunc);
    }
    _pya_free(self);
}

static void
ufuncloop_dealloc(PyUFuncLoopObject *self)
{
    int i;

    if (self->ufunc != NULL) {
        if (self->core_dim_sizes) {
            _pya_free(self->core_dim_sizes);
        }
        if (self->core_strides) {
            _pya_free(self->core_strides);
        }
        for (i = 0; i < self->ufunc->nargs; i++) {
            Py_XDECREF(self->iters[i]);
        }
        if (self->buffer[0]) {
            PyDataMem_FREE(self->buffer[0]);
        }
        Py_XDECREF(self->errobj);
        Py_DECREF(self->ufunc);
    }
    _pya_free(self);
}

static PyUFuncLoopObject *
construct_loop(PyUFuncObject *self, PyObject *args, PyObject *kwds, PyArrayObject **mps)
{
    PyUFuncLoopObject *loop;
    int i;
    PyObject *typetup = NULL;
    PyObject *extobj = NULL;
    char *name;

    if (self == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return NULL;
    }
    if ((loop = _pya_malloc(sizeof(PyUFuncLoopObject))) == NULL) {
        PyErr_NoMemory();
        return loop;
    }

    loop->index = 0;
    loop->ufunc = self;
    Py_INCREF(self);
    loop->buffer[0] = NULL;
    for (i = 0; i < self->nargs; i++) {
        loop->iters[i] = NULL;
        loop->cast[i] = NULL;
    }
    loop->errobj = NULL;
    loop->notimplemented = 0;
    loop->first = 1;
    loop->core_dim_sizes = NULL;
    loop->core_strides = NULL;

    if (self->core_enabled) {
        int num_dim_ix = 1 + self->core_num_dim_ix;
        int nstrides = self->nargs + self->core_offsets[self->nargs - 1]
                        + self->core_num_dims[self->nargs - 1];
        loop->core_dim_sizes = _pya_malloc(sizeof(npy_intp)*num_dim_ix);
        loop->core_strides = _pya_malloc(sizeof(npy_intp)*nstrides);
        if (loop->core_dim_sizes == NULL || loop->core_strides == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        memset(loop->core_strides, 0, sizeof(npy_intp) * nstrides);
        for (i = 0; i < num_dim_ix; i++) {
            loop->core_dim_sizes[i] = 1;
        }
    }
    name = self->name ? self->name : "";

    /*
     * Extract sig= keyword and extobj= keyword if present.
     * Raise an error if anything else is present in the
     * keyword dictionary
     */
    if (kwds != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            char *keystring = PyString_AsString(key);

            if (keystring == NULL) {
                PyErr_Clear();
                PyErr_SetString(PyExc_TypeError, "invalid keyword");
                goto fail;
            }
            if (strncmp(keystring,"extobj",6) == 0) {
                extobj = value;
            }
            else if (strncmp(keystring,"sig",3) == 0) {
                typetup = value;
            }
            else {
                char *format = "'%s' is an invalid keyword to %s";
                PyErr_Format(PyExc_TypeError,format,keystring, name);
                goto fail;
            }
        }
    }

    if (extobj == NULL) {
        if (PyUFunc_GetPyValues(name,
                                &(loop->bufsize), &(loop->errormask),
                                &(loop->errobj)) < 0) {
            goto fail;
        }
    }
    else {
        if (_extract_pyvals(extobj, name,
                            &(loop->bufsize), &(loop->errormask),
                            &(loop->errobj)) < 0) {
            goto fail;
        }
    }

    /* Setup the arrays */
    if (construct_arrays(loop, args, mps, typetup) < 0) {
        goto fail;
    }
    PyUFunc_clearfperr();
    return loop;

fail:
    ufuncloop_dealloc(loop);
    return NULL;
}


/*
  static void
  _printbytebuf(PyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing byte buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %d\n", *(((byte *)(loop->buffer[bufnum]))+i));
  }
  }

  static void
  _printlongbuf(PyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing long buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %ld\n", *(((long *)(loop->buffer[bufnum]))+i));
  }
  }

  static void
  _printlongbufptr(PyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing long buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %ld\n", *(((long *)(loop->bufptr[bufnum]))+i));
  }
  }



  static void
  _printcastbuf(PyUFuncLoopObject *loop, int bufnum)
  {
  int i;

  fprintf(stderr, "Printing long buffer %d\n", bufnum);
  for(i=0; i<loop->bufcnt; i++) {
  fprintf(stderr, "  %ld\n", *(((long *)(loop->castbuf[bufnum]))+i));
  }
  }

*/




/*
 * currently generic ufuncs cannot be built for use on flexible arrays.
 *
 * The cast functions in the generic loop would need to be fixed to pass
 * in something besides NULL, NULL.
 *
 * Also the underlying ufunc loops would not know the element-size unless
 * that was passed in as data (which could be arranged).
 *
 */

NPY_NO_EXPORT int
PyUFunc_GenericFunction_Old(PyUFuncObject *self, PyObject *args, PyObject *kwds,
                        PyArrayObject **mps)
{
    PyUFuncLoopObject *loop;
    int i;
    NPY_BEGIN_THREADS_DEF;

    if (!(loop = construct_loop(self, args, kwds, mps))) {
        return -1;
    }
    if (loop->notimplemented) {
        ufuncloop_dealloc(loop);
        return -2;
    }
    if (self->core_enabled && loop->meth != SIGNATURE_NOBUFFER_UFUNCLOOP) {
        PyErr_SetString(PyExc_RuntimeError,
                        "illegal loop method for ufunc with signature");
        goto fail;
    }

    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
    case ONE_UFUNCLOOP:
        /*
         * Everything is contiguous, notswapped, aligned,
         * and of the right type.  -- Fastest.
         * Or if not contiguous, then a single-stride
         * increment moves through the entire array.
         */
        /*fprintf(stderr, "ONE...%d\n", loop->size);*/
        loop->function((char **)loop->bufptr, &(loop->size),
                loop->steps, loop->funcdata);
        UFUNC_CHECK_ERROR(loop);
        break;
    case NOBUFFER_UFUNCLOOP:
        /*
         * Everything is notswapped, aligned and of the
         * right type but not contiguous. -- Almost as fast.
         */
        /*fprintf(stderr, "NOBUFFER...%d\n", loop->size);*/
        while (loop->index < loop->size) {
            for (i = 0; i < self->nargs; i++) {
                loop->bufptr[i] = loop->iters[i]->dataptr;
            }
            loop->function((char **)loop->bufptr, &(loop->bufcnt),
                    loop->steps, loop->funcdata);
            UFUNC_CHECK_ERROR(loop);

            /* Adjust loop pointers */
            for (i = 0; i < self->nargs; i++) {
                PyArray_ITER_NEXT(loop->iters[i]);
            }
            loop->index++;
        }
        break;
    case SIGNATURE_NOBUFFER_UFUNCLOOP:
        while (loop->index < loop->size) {
            for (i = 0; i < self->nargs; i++) {
                loop->bufptr[i] = loop->iters[i]->dataptr;
            }
            loop->function((char **)loop->bufptr, loop->core_dim_sizes,
                    loop->core_strides, loop->funcdata);
            UFUNC_CHECK_ERROR(loop);

            /* Adjust loop pointers */
            for (i = 0; i < self->nargs; i++) {
                PyArray_ITER_NEXT(loop->iters[i]);
            }
            loop->index++;
        }
        break;
    case BUFFER_UFUNCLOOP: {
        /* This should be a function */
        PyArray_CopySwapNFunc *copyswapn[NPY_MAXARGS];
        PyArrayIterObject **iters=loop->iters;
        int *swap=loop->swap;
        char **dptr=loop->dptr;
        int mpselsize[NPY_MAXARGS];
        intp laststrides[NPY_MAXARGS];
        int fastmemcpy[NPY_MAXARGS];
        int *needbuffer = loop->needbuffer;
        intp index=loop->index, size=loop->size;
        int bufsize;
        intp bufcnt;
        int copysizes[NPY_MAXARGS];
        char **bufptr = loop->bufptr;
        char **buffer = loop->buffer;
        char **castbuf = loop->castbuf;
        intp *steps = loop->steps;
        char *tptr[NPY_MAXARGS];
        int ninnerloops = loop->ninnerloops;
        Bool pyobject[NPY_MAXARGS];
        int datasize[NPY_MAXARGS];
        int j, k, stopcondition;
        char *myptr1, *myptr2;

        for (i = 0; i <self->nargs; i++) {
            copyswapn[i] = mps[i]->descr->f->copyswapn;
            mpselsize[i] = mps[i]->descr->elsize;
            pyobject[i] = ((loop->obj & UFUNC_OBJ_ISOBJECT)
                    && (mps[i]->descr->type_num == PyArray_OBJECT));
            laststrides[i] = iters[i]->strides[loop->lastdim];
            if (steps[i] && laststrides[i] != mpselsize[i]) {
                fastmemcpy[i] = 0;
            }
            else {
                fastmemcpy[i] = 1;
            }
        }
        /* Do generic buffered looping here (works for any kind of
         * arrays -- some need buffers, some don't.
         *
         *
         * New algorithm: N is the largest dimension.  B is the buffer-size.
         * quotient is loop->ninnerloops-1
         * remainder is loop->leftover
         *
         * Compute N = quotient * B + remainder.
         * quotient = N / B  # integer math
         * (store quotient + 1) as the number of innerloops
         * remainder = N % B # integer remainder
         *
         * On the inner-dimension we will have (quotient + 1) loops where
         * the size of the inner function is B for all but the last when the niter size is
         * remainder.
         *
         * So, the code looks very similar to NOBUFFER_LOOP except the inner-most loop is
         * replaced with...
         *
         * for(i=0; i<quotient+1; i++) {
         * if (i==quotient+1) make itersize remainder size
         * copy only needed items to buffer.
         * swap input buffers if needed
         * cast input buffers if needed
         * call loop_function()
         * cast outputs in buffers if needed
         * swap outputs in buffers if needed
         * copy only needed items back to output arrays.
         * update all data-pointers by strides*niter
         * }
         */


        /*
         * fprintf(stderr, "BUFFER...%d,%d,%d\n", loop->size,
         * loop->ninnerloops, loop->leftover);
         */
        /*
         * for(i=0; i<self->nargs; i++) {
         * fprintf(stderr, "iters[%d]->dataptr = %p, %p of size %d\n", i,
         * iters[i], iters[i]->ao->data, PyArray_NBYTES(iters[i]->ao));
         * }
         */
        stopcondition = ninnerloops;
        if (loop->leftover == 0) {
            stopcondition--;
        }
        while (index < size) {
            bufsize=loop->bufsize;
            for(i = 0; i<self->nargs; i++) {
                tptr[i] = loop->iters[i]->dataptr;
                if (needbuffer[i]) {
                    dptr[i] = bufptr[i];
                    datasize[i] = (steps[i] ? bufsize : 1);
                    copysizes[i] = datasize[i] * mpselsize[i];
                }
                else {
                    dptr[i] = tptr[i];
                }
            }

            /* This is the inner function over the last dimension */
            for (k = 1; k<=stopcondition; k++) {
                if (k == ninnerloops) {
                    bufsize = loop->leftover;
                    for (i=0; i<self->nargs;i++) {
                        if (!needbuffer[i]) {
                            continue;
                        }
                        datasize[i] = (steps[i] ? bufsize : 1);
                        copysizes[i] = datasize[i] * mpselsize[i];
                    }
                }
                for (i = 0; i < self->nin; i++) {
                    if (!needbuffer[i]) {
                        continue;
                    }
                    if (fastmemcpy[i]) {
                        memcpy(buffer[i], tptr[i], copysizes[i]);
                    }
                    else {
                        myptr1 = buffer[i];
                        myptr2 = tptr[i];
                        for (j = 0; j < bufsize; j++) {
                            memcpy(myptr1, myptr2, mpselsize[i]);
                            myptr1 += mpselsize[i];
                            myptr2 += laststrides[i];
                        }
                    }

                    /* swap the buffer if necessary */
                    if (swap[i]) {
                        /* fprintf(stderr, "swapping...\n");*/
                        copyswapn[i](buffer[i], mpselsize[i], NULL, -1,
                                (intp) datasize[i], 1,
                                mps[i]);
                    }
                    /* cast to the other buffer if necessary */
                    if (loop->cast[i]) {
                        /* fprintf(stderr, "casting... %d, %p %p\n", i, buffer[i]); */
                        loop->cast[i](buffer[i], castbuf[i],
                                (intp) datasize[i],
                                NULL, NULL);
                    }
                }

                bufcnt = (intp) bufsize;
                loop->function((char **)dptr, &bufcnt, steps, loop->funcdata);
                UFUNC_CHECK_ERROR(loop);

                for (i = self->nin; i < self->nargs; i++) {
                    if (!needbuffer[i]) {
                        continue;
                    }
                    if (loop->cast[i]) {
                        /* fprintf(stderr, "casting back... %d, %p", i, castbuf[i]); */
                        loop->cast[i](castbuf[i],
                                buffer[i],
                                (intp) datasize[i],
                                NULL, NULL);
                    }
                    if (swap[i]) {
                        copyswapn[i](buffer[i], mpselsize[i], NULL, -1,
                                (intp) datasize[i], 1,
                                mps[i]);
                    }
                    /*
                     * copy back to output arrays
                     * decref what's already there for object arrays
                     */
                    if (pyobject[i]) {
                        myptr1 = tptr[i];
                        for (j = 0; j < datasize[i]; j++) {
                            Py_XDECREF(*((PyObject **)myptr1));
                            myptr1 += laststrides[i];
                        }
                    }
                    if (fastmemcpy[i]) {
                        memcpy(tptr[i], buffer[i], copysizes[i]);
                    }
                    else {
                        myptr2 = buffer[i];
                        myptr1 = tptr[i];
                        for (j = 0; j < bufsize; j++) {
                            memcpy(myptr1, myptr2, mpselsize[i]);
                            myptr1 += laststrides[i];
                            myptr2 += mpselsize[i];
                        }
                    }
                }
                if (k == stopcondition) {
                    continue;
                }
                for (i = 0; i < self->nargs; i++) {
                    tptr[i] += bufsize * laststrides[i];
                    if (!needbuffer[i]) {
                        dptr[i] = tptr[i];
                    }
                }
            }
            /* end inner function over last dimension */

            if (loop->objfunc) {
                /*
                 * DECREF castbuf when underlying function used
                 * object arrays and casting was needed to get
                 * to object arrays
                 */
                for (i = 0; i < self->nargs; i++) {
                    if (loop->cast[i]) {
                        if (steps[i] == 0) {
                            Py_XDECREF(*((PyObject **)castbuf[i]));
                        }
                        else {
                            int size = loop->bufsize;

                            PyObject **objptr = (PyObject **)castbuf[i];
                            /*
                             * size is loop->bufsize unless there
                             * was only one loop
                             */
                            if (ninnerloops == 1) {
                                size = loop->leftover;
                            }
                            for (j = 0; j < size; j++) {
                                Py_XDECREF(*objptr);
                                *objptr = NULL;
                                objptr += 1;
                            }
                        }
                    }
                }
            }
            /* fixme -- probably not needed here*/
            UFUNC_CHECK_ERROR(loop);

            for (i = 0; i < self->nargs; i++) {
                PyArray_ITER_NEXT(loop->iters[i]);
            }
            index++;
        }
    } /* end of last case statement */
    }

    NPY_LOOP_END_THREADS;
    ufuncloop_dealloc(loop);
    return 0;

fail:
    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncloop_dealloc(loop);
    }
    return -1;
}


/********* GENERIC UFUNC USING ITERATOR *********/

/*
 * Parses the positional and keyword arguments for a generic ufunc call.
 *
 * Note that if an error is returned, the caller must free the
 * non-zero references in out_op.  This
 * function does not do its own clean-up.
 */
static int get_ufunc_arguments(PyUFuncObject *self,
                PyObject *args, PyObject *kwds,
                PyArrayObject **out_op,
                NPY_ORDER *out_order,
                NPY_CASTING *out_casting,
                PyObject **out_extobj,
                PyObject **out_typetup,
                int *out_any_object)
{
    npy_intp i, nargs, nin = self->nin;
    PyObject *obj, *context;
    char *ufunc_name;

    int any_flexible = 0, any_object = 0;

    ufunc_name = self->name ? self->name : "<unnamed ufunc>";

    /* Check number of arguments */
    nargs = PyTuple_Size(args);
    if ((nargs < nin) || (nargs > self->nargs)) {
        PyErr_SetString(PyExc_ValueError, "invalid number of arguments");
        return -1;
    }

    /* Get input arguments */
    for(i = 0; i < nin; ++i) {
        obj = PyTuple_GET_ITEM(args, i);
        if (!PyArray_Check(obj) && !PyArray_IsScalar(obj, Generic)) {
            /*
             * TODO: There should be a comment here explaining what
             *       context does.
             */
            context = Py_BuildValue("OOi", self, args, i);
            if (context == NULL) {
                return -1;
            }
        }
        else {
            context = NULL;
        }
        out_op[i] = (PyArrayObject *)PyArray_FromAny(obj,
                                        NULL, 0, 0, 0, context);
        Py_XDECREF(context);
        if (out_op[i] == NULL) {
            return -1;
        }
        if (!any_flexible &&
                PyTypeNum_ISFLEXIBLE(PyArray_DESCR(out_op[i])->type_num)) {
            any_flexible = 1;
        }
        if (!any_object &&
                PyTypeNum_ISOBJECT(PyArray_DESCR(out_op[i])->type_num)) {
            any_object = 1;
        }
    }

    /*
     * Indicate not implemented if there are flexible objects (structured
     * type or string) but no object types.
     *
     * Not sure - adding this increased to 246 errors, 150 failures.
     */
    if (any_flexible && !any_object) {
        return -2;

    }

    /* Get positional output arguments */
    for (i = nin; i < nargs; ++i) {
        obj = PyTuple_GET_ITEM(args, i);
        /* Translate None to NULL */
        if (obj == Py_None) {
            continue;
        }
        /* If it's an array, can use it */
        if (PyArray_Check(obj)) {
            if (!PyArray_ISWRITEABLE(obj)) {
                PyErr_SetString(PyExc_ValueError,
                                "return array is not writeable");
                return -1;
            }
            Py_INCREF(obj);
            out_op[i] = (PyArrayObject *)obj;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "return arrays must be "
                            "of ArrayType");
            return -1;
        }
    }

    /*
     * Get keyword output and other arguments.
     * Raise an error if anything else is present in the
     * keyword dictionary.
     */
    if (kwds != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            Py_ssize_t length = 0;
            char *str = NULL;
            int bad_arg = 1;
            
            if (PyString_AsStringAndSize(key, &str, &length) == -1) {
                PyErr_SetString(PyExc_TypeError, "invalid keyword argument");
                return -1;
            }

            switch (str[0]) {
                case 'c':
                    /* Provides a policy for allowed casting */
                    if (strncmp(str,"casting",7) == 0) {
                        if (!PyArray_CastingConverter(value, out_casting)) {
                            return -1;
                        }
                        bad_arg = 0;
                    }
                    break;
                case 'e':
                    /*
                     * Overrides the global parameters buffer size,
                     * error mask, and error object
                     */
                    if (strncmp(str,"extobj",6) == 0) {
                        *out_extobj = value;
                        bad_arg = 0;
                    }
                    break;
                case 'o':
                    /* First output may be specified as a keyword parameter */
                    if (strncmp(str,"out",3) == 0) {
                        if (out_op[nin] != NULL) {
                            PyErr_SetString(PyExc_ValueError,
                                    "cannot specify 'out' as both a "
                                    "positional and keyword argument");
                            return -1;
                        }

                        if (PyArray_Check(value)) {
                            if (!PyArray_ISWRITEABLE(value)) {
                                PyErr_SetString(PyExc_ValueError,
                                        "return array is not writeable");
                                return -1;
                            }
                            Py_INCREF(value);
                            out_op[nin] = (PyArrayObject *)value;
                        }
                        else {
                            PyErr_SetString(PyExc_TypeError,
                                            "return arrays must be "
                                            "of ArrayType");
                            return -1;
                        }
                        bad_arg = 0;
                    }
                    /* Allows the default output layout to be overridden */
                    else if (strncmp(str,"order",5) == 0) {
                        if (!PyArray_OrderConverter(value, out_order)) {
                            return -1;
                        }
                        bad_arg = 0;
                    }
                    break;
                case 's':
                    /* Allows a specific function inner loop to be selected */
                    if (strncmp(str,"sig",3) == 0) {
                        *out_typetup = value;
                        bad_arg = 0;
                    }
                    break;
            }

            if (bad_arg) {
                char *format = "'%s' is an invalid keyword to ufunc '%s'";
                PyErr_Format(PyExc_TypeError, format, str, ufunc_name);
                return -1;
            }
        }
    }

    *out_any_object = any_object;

    return 0;
}

static const char *
_casting_to_string(NPY_CASTING casting)
{
    switch (casting) {
        case NPY_NO_CASTING:
            return "no";
        case NPY_EQUIV_CASTING:
            return "equiv";
        case NPY_SAFE_CASTING:
            return "safe";
        case NPY_SAME_KIND_CASTING:
            return "same_kind";
        case NPY_UNSAFE_CASTING:
            return "unsafe";
        default:
            return "<unknown>";
    }
}


static int
ufunc_loop_matches(PyUFuncObject *self,
                    PyArrayObject **op,
                    NPY_CASTING input_casting,
                    NPY_CASTING output_casting,
                    int any_object,
                    int all_inputs_scalar,
                    int *types,
                    int *out_no_castable_output,
                    char *out_err_src_typecode,
                    char *out_err_dst_typecode)
{
    npy_intp i, nin = self->nin, niter = nin + self->nout;

    /*
     * First check if all the inputs can be safely cast
     * to the types for this function
     */
    for (i = 0; i < nin; ++i) {
        PyArray_Descr *tmp = PyArray_DescrFromType(types[i]);
        if (tmp == NULL) {
            return -1;
        }

        /*
         * If no inputs are objects and there are more than one
         * loop, don't allow conversion to object.  The rationale
         * behind this is mostly performance.  Except for custom
         * ufuncs built with just one object-parametered inner loop,
         * only the types that are supported are implemented.  Trying
         * the object version of logical_or on float arguments doesn't
         * seem right.
         */
        if (types[i] == NPY_OBJECT && !any_object && self->ntypes > 1) {
            return 0;
        }
#if NPY_UF_DBG_TRACING
        printf("Checking type for op %d, type: ", (int)i);
        PyObject_Print((PyObject *)tmp, stdout, 0);
        printf(", operand type: ");
        PyObject_Print((PyObject *)PyArray_DESCR(op[i]), stdout, 0);
        printf("\n");
#endif
        /*
         * If all the inputs are scalars, use the regular
         * promotion rules, not the special value-checking ones.
         */
        if (all_inputs_scalar) {
            if (!PyArray_CanCastTypeTo(PyArray_DESCR(op[i]), tmp,
                                                    input_casting)) {
                Py_DECREF(tmp);
                return 0;
            }
        }
        else {
            if (!PyArray_CanCastArrayTo(op[i], tmp, input_casting)) {
                Py_DECREF(tmp);
                return 0;
            }
        }
        Py_DECREF(tmp);
    }
    NPY_UF_DBG_PRINTF("The inputs all worked\n");

    /*
     * If all the inputs were ok, then check casting back to the
     * outputs.
     */
    for (i = nin; i < niter; ++i) {
        if (op[i] != NULL) {
            PyArray_Descr *tmp = PyArray_DescrFromType(types[i]);
            if (tmp == NULL) {
                return -1;
            }
            if (!PyArray_CanCastTypeTo(tmp, PyArray_DESCR(op[i]),
                                                        output_casting)) {
                Py_DECREF(tmp);
                if (!(*out_no_castable_output)) {
                    *out_no_castable_output = 1;
                    *out_err_src_typecode = tmp->type;
                    *out_err_dst_typecode = PyArray_DESCR(op[i])->type;
                }
                return 0;
            }
            Py_DECREF(tmp);
        }
    }
    NPY_UF_DBG_PRINTF("The outputs all worked\n");

    return 1;
}

static int
set_ufunc_loop_data_types(PyUFuncObject *self, PyArrayObject **op,
                    PyArray_Descr **out_dtype,
                    int *types,
                    npy_intp buffersize, int *out_trivial_loop_ok)
{
    npy_intp i, nin = self->nin, niter = nin + self->nout;

    *out_trivial_loop_ok = 1;
    /* Fill the dtypes array */
    for (i = 0; i < niter; ++i) {
        out_dtype[i] = PyArray_DescrFromType(types[i]);
        if (out_dtype[i] == NULL) {
            return -1;
        }
        /*
         * If the dtype doesn't match, or the array isn't aligned,
         * indicate that the trivial loop can't be done.
         */
        if (*out_trivial_loop_ok && op[i] != NULL &&
                (!PyArray_ISALIGNED(op[i]) ||
                !PyArray_EquivTypes(out_dtype[i], PyArray_DESCR(op[i]))
                                        )) {
            /*
             * If op[j] is a scalar or small one dimensional
             * array input, make a copy to keep the opportunity
             * for a trivial loop.
             */
            if (i < nin && (PyArray_NDIM(op[i]) == 0 ||
                    (PyArray_NDIM(op[i]) == 1 &&
                     PyArray_DIM(op[i],0) <= buffersize))) {
                PyArrayObject *tmp;
                Py_INCREF(out_dtype[i]);
                tmp = (PyArrayObject *)
                            PyArray_CastToType(op[i], out_dtype[i], 0);
                if (tmp == NULL) {
                    return -1;
                }
                Py_DECREF(op[i]);
                op[i] = tmp;
            }
            else {
                *out_trivial_loop_ok = 0;
            }
        }
    }

    return 0;
}

/*
 * Does a search through the arguments and the loops
 */
static int
find_ufunc_matching_userloop(PyUFuncObject *self,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING output_casting,
                        npy_intp buffersize,
                        int any_object,
                        int all_inputs_scalar,
                        PyArray_Descr **out_dtype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata,
                        int *out_trivial_loop_ok,
                        int *out_no_castable_output,
                        char *out_err_src_typecode,
                        char *out_err_dst_typecode)
{
    npy_intp i, nin = self->nin;
    PyUFunc_Loop1d *funcdata;

    /* Use this to try to avoid repeating the same userdef loop search */
    int last_userdef = -1;

    for (i = 0; i < nin; ++i) {
        int type_num = PyArray_DESCR(op[i])->type_num;
        if (type_num != last_userdef && PyTypeNum_ISUSERDEF(type_num)) {
            PyObject *key, *obj;

            last_userdef = type_num;

            key = PyInt_FromLong(type_num);
            if (key == NULL) {
                return -1;
            }
            obj = PyDict_GetItem(self->userloops, key);
            Py_DECREF(key);
            if (obj == NULL) {
                continue;
            }
            funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
            while (funcdata != NULL) {
                int *types = funcdata->arg_types;
                switch (ufunc_loop_matches(self, op,
                            input_casting, output_casting,
                            any_object, all_inputs_scalar,
                            types,
                            out_no_castable_output, out_err_src_typecode,
                            out_err_dst_typecode)) {
                    /* Error */
                    case -1:
                        return -1;
                    /* Found a match */
                    case 1:
                        set_ufunc_loop_data_types(self, op, out_dtype, types,
                                            buffersize, out_trivial_loop_ok);

                        /* Save the inner loop and its data */
                        *out_innerloop = funcdata->func;
                        *out_innerloopdata = funcdata->data;

                        NPY_UF_DBG_PRINTF("Returning userdef inner "
                                                "loop successfully\n");

                        return 0;
                }

                funcdata = funcdata->next;
            }
        }
    }

    /* Didn't find a match */
    return 0;
}

/*
 * Does a search through the arguments and the loops
 */
static int
find_ufunc_specified_userloop(PyUFuncObject *self,
                        int n_specified,
                        int *specified_types,
                        PyArrayObject **op,
                        NPY_CASTING casting,
                        npy_intp buffersize,
                        int any_object,
                        int all_inputs_scalar,
                        PyArray_Descr **out_dtype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata,
                        int *out_trivial_loop_ok)
{
    npy_intp i, j, nin = self->nin, niter = nin + self->nout;
    PyUFunc_Loop1d *funcdata;

    /* Use this to try to avoid repeating the same userdef loop search */
    int last_userdef = -1;

    int no_castable_output = 0;
    char err_src_typecode = '-', err_dst_typecode = '-';

    for (i = 0; i < nin; ++i) {
        int type_num = PyArray_DESCR(op[i])->type_num;
        if (type_num != last_userdef && PyTypeNum_ISUSERDEF(type_num)) {
            PyObject *key, *obj;

            last_userdef = type_num;

            key = PyInt_FromLong(type_num);
            if (key == NULL) {
                return -1;
            }
            obj = PyDict_GetItem(self->userloops, key);
            Py_DECREF(key);
            if (obj == NULL) {
                continue;
            }
            funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
            while (funcdata != NULL) {
                int *types = funcdata->arg_types;
                int matched = 1;

                if (n_specified == niter) {
                    for (j = 0; j < niter; ++j) {
                        if (types[j] != specified_types[j]) {
                            matched = 0;
                            break;
                        }
                    }
                } else {
                    if (types[nin] != specified_types[0]) {
                        matched = 0;
                    }
                }
                if (!matched) {
                    continue;
                }

                switch (ufunc_loop_matches(self, op,
                            casting, casting,
                            any_object, all_inputs_scalar,
                            types,
                            &no_castable_output, &err_src_typecode,
                            &err_dst_typecode)) {
                    /* It works */
                    case 1:
                        set_ufunc_loop_data_types(self, op, out_dtype, types,
                                            buffersize, out_trivial_loop_ok);

                        /* Save the inner loop and its data */
                        *out_innerloop = funcdata->func;
                        *out_innerloopdata = funcdata->data;

                        NPY_UF_DBG_PRINTF("Returning userdef inner "
                                                "loop successfully\n");

                        return 0;
                    /* Didn't match */
                    case 0:
                        PyErr_Format(PyExc_TypeError,
                             "found a user loop for ufunc '%s' "
                             "matching the type-tuple, "
                             "but the inputs and/or outputs could not be "
                             "cast according to the casting rule",
                             self->name ? self->name : "(unknown)");
                        return -1;
                    /* Error */
                    case -1:
                        return -1;
                }

                funcdata = funcdata->next;
            }
        }
    }

    /* Didn't find a match */
    return 0;
}

/*
 * Does a linear search for the best inner loop of the ufunc.
 * When op[i] is a scalar or a one dimensional array smaller than
 * the buffersize, and needs a dtype conversion, this function
 * may substitute op[i] with a version cast to the correct type.  This way,
 * the later trivial loop detection has a higher chance of being triggered.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
static int
find_best_ufunc_inner_loop(PyUFuncObject *self,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING output_casting,
                        npy_intp buffersize,
                        int any_object,
                        PyArray_Descr **out_dtype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata,
                        int *out_trivial_loop_ok)
{
    npy_intp i, j, nin = self->nin, niter = nin + self->nout;
    int types[NPY_MAXARGS];
    char *ufunc_name;
    int no_castable_output, all_inputs_scalar;

    /* For making a better error message on coercion error */
    char err_dst_typecode = '-', err_src_typecode = '-';

    ufunc_name = self->name ? self->name : "(unknown)";

    /* Check whether all the inputs are scalar */
    all_inputs_scalar = 1;
    for(i = 0; i < nin; ++i) {
        if (PyArray_NDIM(op[i]) > 0) {
            all_inputs_scalar = 0;
        }
    }

    /* If the ufunc has userloops, search for them. */
    if (self->userloops) {
        switch (find_ufunc_matching_userloop(self, op,
                                input_casting, output_casting,
                                buffersize, any_object, all_inputs_scalar,
                                out_dtype, out_innerloop, out_innerloopdata,
                                out_trivial_loop_ok,
                                &no_castable_output, &err_src_typecode,
                                &err_dst_typecode)) {
            /* Error */
            case -1:
                return -1;
            /* A loop was found */
            case 1:
                return 0;
        }
    }

    /*
     * Determine the UFunc loop.  This could in general be *much* faster,
     * and a better way to implement it might be for the ufunc to
     * provide a function which gives back the result type and inner
     * loop function.
     *
     * A default fast mechanism could be provided for functions which
     * follow the most typical pattern, when all functions have signatures
     * "xx...x -> x" for some built-in data type x, as follows.
     *  - Use PyArray_ResultType to get the output type
     *  - Look up the inner loop in a table based on the output type_num
     *
     * The method for finding the loop in the previous code did not
     * appear consistent (as noted by some asymmetry in the generated
     * coercion tables for np.add).
     */
    no_castable_output = 0;
    for (i = 0; i < self->ntypes; ++i) {
        char *orig_types = self->types + i*self->nargs;

        /* Copy the types into an int array for matching */
        for (j = 0; j < niter; ++j) {
            types[j] = orig_types[j];
        }

        NPY_UF_DBG_PRINTF("Trying function loop %d\n", (int)i);
        switch (ufunc_loop_matches(self, op,
                    input_casting, output_casting,
                    any_object, all_inputs_scalar,
                    types,
                    &no_castable_output, &err_src_typecode,
                    &err_dst_typecode)) {
            /* Error */
            case -1:
                return -1;
            /* Found a match */
            case 1:
                set_ufunc_loop_data_types(self, op, out_dtype, types,
                                    buffersize, out_trivial_loop_ok);

                /* Save the inner loop and its data */
                *out_innerloop = self->functions[i];
                *out_innerloopdata = self->data[i];

                NPY_UF_DBG_PRINTF("Returning inner loop successfully\n");

                return 0;
        }

    }

    /* If no function was found, throw an error */
    NPY_UF_DBG_PRINTF("No loop was found\n");
    if (no_castable_output) {
        PyErr_Format(PyExc_TypeError,
                "ufunc '%s' output (typecode '%c') could not be coerced to "
                "provided output parameter (typecode '%c') according "
                "to the casting rule '%s'",
                ufunc_name, err_src_typecode, err_dst_typecode,
                _casting_to_string(output_casting));
    }
    else {
        /*
         * TODO: We should try again if the casting rule is same_kind
         *       or unsafe, and look for a function more liberally.
         */
        PyErr_Format(PyExc_TypeError,
                "ufunc '%s' not supported for the input types, and the "
                "inputs could not be safely coerced to any supported "
                "types according to the casting rule '%s'",
                ufunc_name,
                _casting_to_string(input_casting));
    }

    return -1;
}

/*
 * Does a linear search for the inner loop of the ufunc specified by type_tup.
 * When op[i] is a scalar or a one dimensional array smaller than
 * the buffersize, and needs a dtype conversion, this function
 * may substitute op[i] with a version cast to the correct type.  This way,
 * the later trivial loop detection has a higher chance of being triggered.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
static int
find_specified_ufunc_inner_loop(PyUFuncObject *self,
                        PyObject *type_tup,
                        PyArrayObject **op,
                        NPY_CASTING casting,
                        npy_intp buffersize,
                        int any_object,
                        PyArray_Descr **out_dtype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata,
                        int *out_trivial_loop_ok)
{
    npy_intp i, j, n, nin = self->nin, niter = nin + self->nout;
    int n_specified = 0;
    int specified_types[NPY_MAXARGS], types[NPY_MAXARGS];
    char *ufunc_name;
    int no_castable_output, all_inputs_scalar;

    /* For making a better error message on coercion error */
    char err_dst_typecode = '-', err_src_typecode = '-';

    ufunc_name = self->name ? self->name : "(unknown)";

    /* Check whether all the inputs are scalar */
    all_inputs_scalar = 1;
    for(i = 0; i < nin; ++i) {
        if (PyArray_NDIM(op[i]) > 0) {
            all_inputs_scalar = 0;
        }
    }

    /* Fill in specified_types from the tuple or string */
    if (PyTuple_Check(type_tup)) {
        n = PyTuple_GET_SIZE(type_tup);
        if (n != 1 && n != niter) {
            PyErr_Format(PyExc_ValueError,
                         "a type-tuple must be specified " \
                         "of length 1 or %d for ufunc '%s'", (int)niter,
                         self->name ? self->name : "(unknown)");
            return -1;
        }

        for (i = 0; i < n; ++i) {
            PyArray_Descr *dtype = NULL;
            if (!PyArray_DescrConverter2(PyTuple_GET_ITEM(type_tup, i),
                                                                &dtype)) {
                return -1;
            }
            specified_types[i] = dtype->type_num;
            Py_DECREF(dtype);
        }

        n_specified = n;
    }
    else if (PyString_Check(type_tup)) {
        Py_ssize_t length;
        char *str;

        if (!PyString_AsStringAndSize(type_tup, &str, &length) < 0) {
            return -1;
        }
        if (length != 1 && (length != niter + 2 ||
                                str[nin] != '-' || str[nin+1] != '>')) {
            PyErr_Format(PyExc_ValueError,
                                 "a type-string for %s, "   \
                                 "requires 1 typecode, or "
                                 "%d typecode(s) before " \
                                 "and %d after the -> sign",
                                 self->name ? self->name : "(unknown)",
                                 self->nin, self->nout);
            return -1;
        }
        if (length == 1) {
            PyArray_Descr *dtype;
            n_specified = 1;
            dtype = PyArray_DescrFromType(str[0]);
            if (dtype == NULL) {
                return -1;
            }
            NPY_UF_DBG_PRINTF("signature character '%c', type num %d\n",
                                str[0], dtype->type_num);
            specified_types[0] = dtype->type_num;
            Py_DECREF(dtype);
        }
        else {
            PyArray_Descr *dtype;
            n_specified = (int)niter;

            for (i = 0; i < niter; ++i) {
                npy_intp istr = i < nin ? i : i+2;

                dtype = PyArray_DescrFromType(str[istr]);
                if (dtype == NULL) {
                    return -1;
                }
                NPY_UF_DBG_PRINTF("signature character '%c', type num %d\n",
                                    str[istr], dtype->type_num);
                specified_types[i] = dtype->type_num;
                Py_DECREF(dtype);
            }
        }
    }

    /* If the ufunc has userloops, search for them. */
    if (self->userloops) {
        NPY_UF_DBG_PRINTF("Searching user loops for specified sig\n");
        switch (find_ufunc_specified_userloop(self,
                        n_specified, specified_types,
                        op, casting,
                        buffersize, any_object, all_inputs_scalar,
                        out_dtype, out_innerloop, out_innerloopdata,
                        out_trivial_loop_ok)) {
            /* Error */
            case -1:
                return -1;
            /* Found matching loop */
            case 1:
                return 0;
        }
    }
 
    NPY_UF_DBG_PRINTF("Searching loops for specified sig\n");
    for (i = 0; i < self->ntypes; ++i) {
        char *orig_types = self->types + i*self->nargs;
        int matched = 1;

        NPY_UF_DBG_PRINTF("Trying function loop %d\n", (int)i);

        /* Copy the types into an int array for matching */
        for (j = 0; j < niter; ++j) {
            types[j] = orig_types[j];
        }

        if (n_specified == niter) {
            for (j = 0; j < niter; ++j) {
                if (types[j] != specified_types[j]) {
                    matched = 0;
                    break;
                }
            }
        } else {
            NPY_UF_DBG_PRINTF("Specified type: %d, first output type: %d\n",
                                        specified_types[0], types[nin]);
            if (types[nin] != specified_types[0]) {
                matched = 0;
            }
        }
        if (!matched) {
            continue;
        }

        NPY_UF_DBG_PRINTF("It matches, confirming type casting\n");
        switch (ufunc_loop_matches(self, op,
                    casting, casting,
                    any_object, all_inputs_scalar,
                    types,
                    &no_castable_output, &err_src_typecode,
                    &err_dst_typecode)) {
            /* Error */
            case -1:
                return -1;
            /* It worked */
            case 1:
                set_ufunc_loop_data_types(self, op, out_dtype, types,
                                    buffersize, out_trivial_loop_ok);

                /* Save the inner loop and its data */
                *out_innerloop = self->functions[i];
                *out_innerloopdata = self->data[i];

                NPY_UF_DBG_PRINTF("Returning specified inner loop successfully\n");

                return 0;
            /* Didn't work */
            case 0:
                PyErr_Format(PyExc_TypeError,
                     "found a loop for ufunc '%s' "
                     "matching the type-tuple, "
                     "but the inputs and/or outputs could not be "
                     "cast according to the casting rule",
                     ufunc_name);
                return -1;
        }

    }

    /* If no function was found, throw an error */
    NPY_UF_DBG_PRINTF("No specified loop was found\n");

    PyErr_Format(PyExc_TypeError,
            "No loop matching the specified signature was found "
            "for ufunc %s", ufunc_name);

    return -1;
}

static void
trivial_two_operand_loop(PyArrayObject **op,
                    PyUFuncGenericFunction innerloop,
                    void *innerloopdata)
{
    char *data[2];
    npy_intp count[2], stride[2];
    int needs_api;
    NPY_BEGIN_THREADS_DEF;

    needs_api = PyDataType_REFCHK(PyArray_DESCR(op[0])) ||
                PyDataType_REFCHK(PyArray_DESCR(op[1]));

    PyArray_PREPARE_TRIVIAL_PAIR_ITERATION(op[0], op[1],
                                            count[0],
                                            data[0], data[1],
                                            stride[0], stride[1]);
    count[1] = count[0];
    NPY_UF_DBG_PRINTF("two operand loop count %d\n", (int)count[0]);

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    innerloop(data, count, stride, innerloopdata);

    if (!needs_api) {
        NPY_END_THREADS;
    }
}

static void
trivial_three_operand_loop(PyArrayObject **op,
                    PyUFuncGenericFunction innerloop,
                    void *innerloopdata)
{
    char *data[3];
    npy_intp count[3], stride[3];
    int needs_api;
    NPY_BEGIN_THREADS_DEF;

    needs_api = PyDataType_REFCHK(PyArray_DESCR(op[0])) ||
                PyDataType_REFCHK(PyArray_DESCR(op[1])) ||
                PyDataType_REFCHK(PyArray_DESCR(op[2]));

    PyArray_PREPARE_TRIVIAL_TRIPLE_ITERATION(op[0], op[1], op[2],
                                            count[0],
                                            data[0], data[1], data[2],
                                            stride[0], stride[1], stride[2]);
    count[1] = count[0];
    count[2] = count[0];
    NPY_UF_DBG_PRINTF("three operand loop count %d\n", (int)count[0]);

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    innerloop(data, count, stride, innerloopdata);

    if (!needs_api) {
        NPY_END_THREADS;
    }
}

/*
 * Calls the given __array_prepare__ function on the operand *op,
 * substituting it in place if a new array is returned and matches
 * the old one.
 *
 * This requires that the dimensions, strides and data type remain
 * exactly the same, which may be more strict than before.
 */
static int
prepare_ufunc_output(PyUFuncObject *self,
                    PyArrayObject **op,
                    PyObject *arr_prep,
                    PyObject *arr_prep_args,
                    int i)
{
    if (arr_prep != NULL && arr_prep != Py_None) {
        PyObject *res;

        res = PyObject_CallFunction(arr_prep, "O(OOi)",
                    *op, self, arr_prep_args, i);
        if ((res == NULL) || (res == Py_None) || !PyArray_Check(res)) {
            if (!PyErr_Occurred()){
                PyErr_SetString(PyExc_TypeError,
                        "__array_prepare__ must return an "
                        "ndarray or subclass thereof");
            }
            Py_XDECREF(res);
            return -1;
        }

        /* If the same object was returned, nothing to do */
        if (res == (PyObject *)*op) {
            Py_DECREF(res);
        }
        /* If the result doesn't match, throw an error */
        else if (PyArray_NDIM(res) != PyArray_NDIM(*op) ||
                !PyArray_CompareLists(PyArray_DIMS(res),
                                      PyArray_DIMS(*op),
                                      PyArray_NDIM(res)) ||
                !PyArray_CompareLists(PyArray_STRIDES(res),
                                      PyArray_STRIDES(*op),
                                      PyArray_NDIM(res)) ||
                !PyArray_EquivTypes(PyArray_DESCR(res),
                                    PyArray_DESCR(*op))) {
            PyErr_SetString(PyExc_TypeError,
                    "__array_prepare__ must return an "
                    "ndarray or subclass thereof which is "
                    "otherwise identical to its input");
            Py_DECREF(res);
            return -1;
        }
        /* Replace the op value */
        else {
            Py_DECREF(*op);
            *op = (PyArrayObject *)res;
        }
    }

    return 0;
}

static int
iterator_loop(PyUFuncObject *self,
                    PyArrayObject **op,
                    PyArray_Descr **dtype,
                    NPY_ORDER order,
                    npy_intp buffersize,
                    PyObject **arr_prep,
                    PyObject *arr_prep_args,
                    PyUFuncGenericFunction innerloop,
                    void *innerloopdata)
{
    npy_intp i, nin = self->nin, nout = self->nout;
    npy_intp niter = nin + nout;
    npy_uint32 op_flags[NPY_MAXARGS];
    NpyIter *iter;
    char *baseptrs[NPY_MAXARGS];
    int needs_api;
    NPY_BEGIN_THREADS_DEF;

    NpyIter_IterNext_Fn iternext;
    char **dataptr;
    npy_intp *stride;
    npy_intp *count_ptr;

    PyArrayObject **op_it;

    /* Set up the flags */
    for (i = 0; i < nin; ++i) {
        op_flags[i] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
    }
    for (i = nin; i < niter; ++i) {
        op_flags[i] = NPY_ITER_WRITEONLY|
                      NPY_ITER_ALIGNED|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_NO_SUBTYPE;
    }

    /*
     * Allocate the iterator.  Because the types of the inputs
     * were already checked, we use the casting rule 'unsafe' which
     * is faster to calculate.
     */
    iter = NpyIter_MultiNew(niter, op,
                        NPY_ITER_NO_INNER_ITERATION|
                        NPY_ITER_REFS_OK|
                        NPY_ITER_ZEROSIZE_OK|
                        NPY_ITER_BUFFERED|
                        NPY_ITER_GROWINNER|
                        NPY_ITER_DELAY_BUFALLOC,
                        order, NPY_UNSAFE_CASTING,
                        op_flags, dtype,
                        0, NULL, buffersize);
    if (iter == NULL) {
        return -1;
    }

    needs_api = NpyIter_IterationNeedsAPI(iter);

    /* Copy any allocated outputs */
    op_it = NpyIter_GetOperandArray(iter);
    for (i = nin; i < niter; ++i) {
        if (op[i] == NULL) {
            op[i] = op_it[i];
            Py_INCREF(op[i]);
        }
    }

    /* Call the __array_prepare__ functions where necessary */
    for (i = 0; i < nout; ++i) {
        if (prepare_ufunc_output(self, &op[nin+i],
                            arr_prep[i], arr_prep_args, i) < 0) {
            NpyIter_Deallocate(iter);
            return -1;
        }
    }

    /* Only do the loop if the iteration size is non-zero */
    if (NpyIter_GetIterSize(iter) != 0) {

        /* Reset the iterator with the base pointers from the wrapped outputs */
        for (i = 0; i < niter; ++i) {
            baseptrs[i] = PyArray_BYTES(op[i]);
        }
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            return -1;
        }

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        stride = NpyIter_GetInnerStrideArray(iter);
        count_ptr = NpyIter_GetInnerLoopSizePtr(iter);

        if (!needs_api) {
            NPY_BEGIN_THREADS;
        }

        /* Execute the loop */
        do {
            NPY_UF_DBG_PRINTF("iterator loop count %d\n", (int)*count_ptr);
            innerloop(dataptr, count_ptr, stride, innerloopdata);
        } while (iternext(iter));

        if (!needs_api) {
            NPY_END_THREADS;
        }
    }

    NpyIter_Deallocate(iter);
    return 0;
}

/*
 * trivial_loop_ok - 1 if no alignment, data conversion, etc required
 * nin             - number of inputs
 * nout            - number of outputs
 * op              - the operands (nin + nout of them)
 * order           - the loop execution order/output memory order
 * buffersize      - how big of a buffer to use
 * arr_prep        - the __array_prepare__ functions for the outputs
 * innerloop       - the inner loop function
 * innerloopdata   - data to pass to the inner loop
 */
static int
execute_ufunc_loop(PyUFuncObject *self,
                    int trivial_loop_ok,
                    PyArrayObject **op,
                    PyArray_Descr **dtype,
                    NPY_ORDER order,
                    npy_intp buffersize,
                    PyObject **arr_prep,
                    PyObject *arr_prep_args,
                    PyUFuncGenericFunction innerloop,
                    void *innerloopdata)
{
    npy_intp nin = self->nin, nout = self->nout;

    /* First check for the trivial cases that don't need an iterator */
    if (trivial_loop_ok) {
        if (nin == 1 && nout == 1) {
            if (op[1] == NULL &&
                        (order == NPY_ANYORDER || order == NPY_KEEPORDER) &&
                        PyArray_TRIVIALLY_ITERABLE(op[0])) {
                Py_INCREF(dtype[1]);
                op[1] = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                             dtype[1],
                             PyArray_NDIM(op[0]),
                             PyArray_DIMS(op[0]),
                             NULL, NULL,
                             PyArray_ISFORTRAN(op[0]) ? NPY_F_CONTIGUOUS : 0,
                             NULL);

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(self, &op[1],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINTF("trivial 1 input with allocated output\n");
                trivial_two_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
            else if (op[1] != NULL &&
                        PyArray_NDIM(op[1]) >= PyArray_NDIM(op[0]) &&
                        PyArray_TRIVIALLY_ITERABLE_PAIR(op[0], op[1])) {

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(self, &op[1],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINTF("trivial 1 input\n");
                trivial_two_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
        }
        else if (nin == 2 && nout == 1) {
            if (op[2] == NULL &&
                        (order == NPY_ANYORDER || order == NPY_KEEPORDER) &&
                        PyArray_TRIVIALLY_ITERABLE_PAIR(op[0], op[1])) {
                PyArrayObject *tmp;
                /*
                 * Have to choose the input with more dimensions to clone, as
                 * one of them could be a scalar.
                 */
                if (PyArray_NDIM(op[0]) >= PyArray_NDIM(op[1])) {
                    tmp = op[0];
                }
                else {
                    tmp = op[1];
                }
                Py_INCREF(dtype[2]);
                op[2] = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                 dtype[2],
                                 PyArray_NDIM(tmp),
                                 PyArray_DIMS(tmp),
                                 NULL, NULL,
                                 PyArray_ISFORTRAN(tmp) ? NPY_F_CONTIGUOUS : 0,
                                 NULL);

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(self, &op[2],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINTF("trivial 2 input with allocated output\n");
                trivial_three_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
            else if (op[2] != NULL &&
                    PyArray_NDIM(op[2]) >= PyArray_NDIM(op[0]) &&
                    PyArray_NDIM(op[2]) >= PyArray_NDIM(op[1]) &&
                    PyArray_TRIVIALLY_ITERABLE_TRIPLE(op[0], op[1], op[2])) {

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(self, &op[2],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINTF("trivial 2 input\n");
                trivial_three_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
        }
    }

    /*
     * If no trivial loop matched, an iterator is required to
     * resolve broadcasting, etc
     */

    NPY_UF_DBG_PRINTF("iterator loop\n");
    if (iterator_loop(self, op, dtype, order,
                    buffersize, arr_prep, arr_prep_args,
                    innerloop, innerloopdata) < 0) {
        return -1;
    }

    return 0;
}

static PyObject *
make_arr_prep_args(npy_intp nin, PyObject *args, PyObject *kwds)
{
    PyObject *out = kwds ? PyDict_GetItemString(kwds, "out") : NULL;
    PyObject *arr_prep_args;

    if (out == NULL) {
        Py_INCREF(args);
        return args;
    }
    else {
        npy_intp i, nargs = PyTuple_GET_SIZE(args), n;
        n = nargs;
        if (n < nin + 1) {
            n = nin + 1;
        }
        arr_prep_args = PyTuple_New(n);
        if (arr_prep_args == NULL) {
            return NULL;
        }
        /* Copy the tuple, but set the nin-th item to the keyword arg */
        for (i = 0; i < nin; ++i) {
            PyObject *item = PyTuple_GET_ITEM(args, i);
            Py_INCREF(item);
            PyTuple_SET_ITEM(arr_prep_args, i, item);
        }
        Py_INCREF(out);
        PyTuple_SET_ITEM(arr_prep_args, nin, out);
        for (i = nin+1; i < n; ++i) {
            PyObject *item = PyTuple_GET_ITEM(args, i);
            Py_INCREF(item);
            PyTuple_SET_ITEM(arr_prep_args, i, item);
        }

        return arr_prep_args;
    }
}

/*UFUNC_API
 *
 * This generic function is called with the ufunc object, the arguments to it,
 * and an array of (pointers to) PyArrayObjects which are NULL.
 */
NPY_NO_EXPORT int
PyUFunc_GenericFunction(PyUFuncObject *self,
                        PyObject *args, PyObject *kwds,
                        PyArrayObject **op)
{
    npy_intp nin, nout;
    npy_intp i, niter;
    char *ufunc_name;
    int retval = -1, any_object = 0;
    NPY_CASTING input_casting;

    PyArray_Descr *dtype[NPY_MAXARGS];

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;
    PyObject *errobj = NULL;
    int first_error = 1;

    /* The selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;

    /* The __array_prepare__ function to call for each output */
    PyObject *arr_prep[NPY_MAXARGS];
    /*
     * This is either args, or args with the out= parameter from
     * kwds added appropriately.
     */
    PyObject *arr_prep_args = NULL;

    int trivial_loop_ok = 0;

    /* TODO: For 1.6, the default should probably be NPY_CORDER */
    NPY_ORDER order = NPY_KEEPORDER;
    /*
     * Many things in NumPy do unsafe casting (doing int += float, etc).
     * The strictness should probably become a state parameter, similar
     * to the seterr/geterr.
     */
    NPY_CASTING casting = NPY_UNSAFE_CASTING;
    /* When provided, extobj and typetup contain borrowed references */
    PyObject *extobj = NULL, *type_tup = NULL;

    if (self == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return -1;
    }

    nin = self->nin;
    nout = self->nout;
    niter = nin + nout;

    /* TODO: support generalized ufunc */
    if (self->core_enabled) {
        PyErr_SetString(PyExc_RuntimeError,
                    "core_enabled (generalized ufunc) not supported yet");
        return -1;
    }

    ufunc_name = self->name ? self->name : "<unnamed ufunc>";

    NPY_UF_DBG_PRINTF("\nEvaluating ufunc %s\n", ufunc_name);

    /* Initialize all the operands and dtypes to NULL */
    for (i = 0; i < niter; ++i) {
        op[i] = NULL;
        dtype[i] = NULL;
        arr_prep[i] = NULL;
    }

    NPY_UF_DBG_PRINTF("Getting arguments\n");

    /* Get all the arguments */
    retval = get_ufunc_arguments(self, args, kwds,
                op, &order, &casting, &extobj, &type_tup, &any_object);
    if (retval < 0) {
        goto fail;
    }

    /* Get the buffersize, errormask, and error object globals */
    if (extobj == NULL) {
        if (PyUFunc_GetPyValues(ufunc_name,
                                &buffersize, &errormask, &errobj) < 0) {
            retval = -1;
            goto fail;
        }
    }
    else {
        if (_extract_pyvals(extobj, ufunc_name,
                                &buffersize, &errormask, &errobj) < 0) {
            retval = -1;
            goto fail;
        }
    }

    NPY_UF_DBG_PRINTF("Finding inner loop\n");

    /*
     * Decide the casting rules for inputs and outputs.  We want
     * NPY_SAFE_CASTING or stricter, so that the loop selection code
     * doesn't choose an integer loop for float inputs, for example.
     */
    input_casting = (casting > NPY_SAFE_CASTING) ? NPY_SAFE_CASTING : casting;

    if (type_tup == NULL) {
        /* Find the best ufunc inner loop, and fill in the dtypes */
        retval = find_best_ufunc_inner_loop(self, op, input_casting, casting,
                        buffersize, any_object, dtype,
                        &innerloop, &innerloopdata, &trivial_loop_ok);
    } else {
        /* Find the specified ufunc inner loop, and fill in the dtypes */
        retval = find_specified_ufunc_inner_loop(self, type_tup,
                        op, casting,
                        buffersize, any_object, dtype,
                        &innerloop, &innerloopdata, &trivial_loop_ok);
    }
    if (retval < 0) {
        goto fail;
    }

    /*
     * FAIL with NotImplemented if the other object has
     * the __r<op>__ method and has __array_priority__ as
     * an attribute (signalling it can handle ndarray's)
     * and is not already an ndarray or a subtype of the same type.
    */
    if (nin == 2 && nout == 1 && dtype[1]->type_num == NPY_OBJECT) {
        PyObject *_obj = PyTuple_GET_ITEM(args, 1);
        if (!PyArray_CheckExact(_obj)
               /* If both are same subtype of object arrays, then proceed */
                && !(Py_TYPE(_obj) == Py_TYPE(PyTuple_GET_ITEM(args, 0)))
                && PyObject_HasAttrString(_obj, "__array_priority__")
                && _has_reflected_op(_obj, ufunc_name)) {
            retval = -2;
            goto fail;
        }
    }

#if NPY_UF_DBG_TRACING
    printf("input types:\n");
    for (i = 0; i < nin; ++i) {
        PyObject_Print((PyObject *)dtype[i], stdout, 0);
        printf(" ");
    }
    printf("\noutput types:\n");
    for (i = nin; i < niter; ++i) {
        PyObject_Print((PyObject *)dtype[i], stdout, 0);
        printf(" ");
    }
    printf("\n");
#endif

    /*
     * Get the appropriate __array_prepare__ function to call
     * for each output
     */
    _find_array_prepare(args, kwds, arr_prep, nin, nout);

    /* Set up arr_prep_args if a prep function was needed */
    for (i = 0; i < nout; ++i) {
        if (arr_prep[i] != NULL && arr_prep[i] != Py_None) {
            arr_prep_args = make_arr_prep_args(nin, args, kwds);
            break;
        }
    }

    /* If the loop wants the arrays, provide them */
    if (_does_loop_use_arrays(innerloopdata)) {
        innerloopdata = (void*)op;
    }

    /* Start with the floating-point exception flags cleared */
    PyUFunc_clearfperr();

    NPY_UF_DBG_PRINTF("Executing inner loop\n");

    /* Do the ufunc loop */
    retval = execute_ufunc_loop(self, trivial_loop_ok, op, dtype, order,
                        buffersize, arr_prep, arr_prep_args,
                        innerloop, innerloopdata);
    if (retval < 0) {
        goto fail;
    }

    /* Check whether any errors occurred during the loop */
    if (PyErr_Occurred() || (errormask &&
            PyUFunc_checkfperr(errormask, errobj, &first_error))) {
        retval = -1;
        goto fail;
    }

    /* The caller takes ownership of all the references in op */
    for (i = 0; i < niter; ++i) {
        Py_XDECREF(dtype[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(errobj);
    Py_XDECREF(arr_prep_args);

    NPY_UF_DBG_PRINTF("Returning Success\n");

    return 0;

fail:
    NPY_UF_DBG_PRINTF("Returning failure code %d\n", retval);
    for (i = 0; i < niter; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
        Py_XDECREF(dtype[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(errobj);
    Py_XDECREF(arr_prep_args);

    return retval;
}

static PyArrayObject *
_getidentity(PyUFuncObject *self, int otype, char *str)
{
    PyObject *obj, *arr;
    PyArray_Descr *typecode;

    if (self->identity == PyUFunc_None) {
        PyErr_Format(PyExc_ValueError,
                     "zero-size array to ufunc.%s "      \
                     "without identity", str);
        return NULL;
    }
    if (self->identity == PyUFunc_One) {
        obj = PyInt_FromLong((long) 1);
    } else {
        obj = PyInt_FromLong((long) 0);
    }

    typecode = PyArray_DescrFromType(otype);
    arr = PyArray_FromAny(obj, typecode, 0, 0, CARRAY, NULL);
    Py_DECREF(obj);
    return (PyArrayObject *)arr;
}

static int
_create_reduce_copy(PyUFuncReduceObject *loop, PyArrayObject **arr, int rtype)
{
    intp maxsize;
    PyObject *new;
    PyArray_Descr *ntype;

    maxsize = PyArray_SIZE(*arr);

    if (maxsize < loop->bufsize) {
        if (!(PyArray_ISBEHAVED_RO(*arr))
            || PyArray_TYPE(*arr) != rtype) {
            ntype = PyArray_DescrFromType(rtype);
            new = PyArray_FromAny((PyObject *)(*arr),
                                  ntype, 0, 0,
                                  FORCECAST | ALIGNED, NULL);
            if (new == NULL) {
                return -1;
            }
            *arr = (PyArrayObject *)new;
            loop->decref = new;
        }
    }

    /*
     * Don't decref *arr before re-assigning
     * because it was not going to be DECREF'd anyway.
     *
     * If a copy is made, then the copy will be removed
     * on deallocation of the loop structure by setting
     * loop->decref.
     */
    return 0;
}

static PyUFuncReduceObject *
construct_reduce(PyUFuncObject *self, PyArrayObject **arr, PyArrayObject *out,
                 int axis, int otype, int operation, intp ind_size, char *str)
{
    PyUFuncReduceObject *loop;
    PyArrayObject *idarr;
    PyArrayObject *aar;
    intp loop_i[MAX_DIMS], outsize = 0;
    int arg_types[3];
    PyArray_SCALARKIND scalars[3] = {PyArray_NOSCALAR, PyArray_NOSCALAR,
                                     PyArray_NOSCALAR};
    int i, j, nd;
    int flags;

    /* Reduce type is the type requested of the input during reduction */
    if (self->core_enabled) {
        PyErr_Format(PyExc_RuntimeError,
                     "construct_reduce not allowed on ufunc with signature");
        return NULL;
    }
    nd = (*arr)->nd;
    arg_types[0] = otype;
    arg_types[1] = otype;
    arg_types[2] = otype;
    if ((loop = _pya_malloc(sizeof(PyUFuncReduceObject))) == NULL) {
        PyErr_NoMemory();
        return loop;
    }

    loop->retbase = 0;
    loop->swap = 0;
    loop->index = 0;
    loop->ufunc = self;
    Py_INCREF(self);
    loop->cast = NULL;
    loop->buffer = NULL;
    loop->ret = NULL;
    loop->it = NULL;
    loop->rit = NULL;
    loop->errobj = NULL;
    loop->first = 1;
    loop->decref = NULL;
    loop->N = (*arr)->dimensions[axis];
    loop->instrides = (*arr)->strides[axis];
    if (select_types(loop->ufunc, arg_types, &(loop->function),
                     &(loop->funcdata), scalars, NULL) == -1) {
        goto fail;
    }
    /*
     * output type may change -- if it does
     * reduction is forced into that type
     * and we need to select the reduction function again
     */
    if (otype != arg_types[2]) {
        otype = arg_types[2];
        arg_types[0] = otype;
        arg_types[1] = otype;
        if (select_types(loop->ufunc, arg_types, &(loop->function),
                         &(loop->funcdata), scalars, NULL) == -1) {
            goto fail;
        }
    }

    /* get looping parameters from Python */
    if (PyUFunc_GetPyValues(str, &(loop->bufsize), &(loop->errormask),
                            &(loop->errobj)) < 0) {
        goto fail;
    }
    /* Make copy if misbehaved or not otype for small arrays */
    if (_create_reduce_copy(loop, arr, otype) < 0) {
        goto fail;
    }
    aar = *arr;

    if (loop->N == 0) {
        loop->meth = ZERO_EL_REDUCELOOP;
    }
    else if (PyArray_ISBEHAVED_RO(aar) && (otype == (aar)->descr->type_num)) {
        if (loop->N == 1) {
            loop->meth = ONE_EL_REDUCELOOP;
        }
        else {
            loop->meth = NOBUFFER_UFUNCLOOP;
            loop->steps[1] = (aar)->strides[axis];
            loop->N -= 1;
        }
    }
    else {
        loop->meth = BUFFER_UFUNCLOOP;
        loop->swap = !(PyArray_ISNOTSWAPPED(aar));
    }

    /* Determine if object arrays are involved */
    if (otype == PyArray_OBJECT || aar->descr->type_num == PyArray_OBJECT) {
        loop->obj = UFUNC_OBJ_ISOBJECT | UFUNC_OBJ_NEEDS_API;
    }
    else if ((otype == PyArray_DATETIME)
            || (aar->descr->type_num == PyArray_DATETIME)
            || (otype == PyArray_TIMEDELTA)
            || (aar->descr->type_num == PyArray_TIMEDELTA))
    {
        loop->obj = UFUNC_OBJ_NEEDS_API;
    }
    else {
        loop->obj = 0;
    }
    if ((loop->meth == ZERO_EL_REDUCELOOP)
            || ((operation == UFUNC_REDUCEAT)
                && (loop->meth == BUFFER_UFUNCLOOP))) {
        idarr = _getidentity(self, otype, str);
        if (idarr == NULL) {
            goto fail;
        }
        if (idarr->descr->elsize > UFUNC_MAXIDENTITY) {
            PyErr_Format(PyExc_RuntimeError,
                    "UFUNC_MAXIDENTITY (%d) is too small"\
                    "(needs to be at least %d)",
                    UFUNC_MAXIDENTITY, idarr->descr->elsize);
            Py_DECREF(idarr);
            goto fail;
        }
        memcpy(loop->idptr, idarr->data, idarr->descr->elsize);
        Py_DECREF(idarr);
    }

    /* Construct return array */
    flags = NPY_CARRAY | NPY_UPDATEIFCOPY | NPY_FORCECAST;
    switch(operation) {
    case UFUNC_REDUCE:
        for (j = 0, i = 0; i < nd; i++) {
            if (i != axis) {
                loop_i[j++] = (aar)->dimensions[i];
            }
        }
        if (out == NULL) {
            loop->ret = (PyArrayObject *)
                PyArray_New(Py_TYPE(aar), aar->nd-1, loop_i,
                            otype, NULL, NULL, 0, 0,
                            (PyObject *)aar);
        }
        else {
            outsize = PyArray_MultiplyList(loop_i, aar->nd - 1);
        }
        break;
    case UFUNC_ACCUMULATE:
        if (out == NULL) {
            loop->ret = (PyArrayObject *)
                PyArray_New(Py_TYPE(aar), aar->nd, aar->dimensions,
                        otype, NULL, NULL, 0, 0, (PyObject *)aar);
        }
        else {
            outsize = PyArray_MultiplyList(aar->dimensions, aar->nd);
        }
        break;
    case UFUNC_REDUCEAT:
        memcpy(loop_i, aar->dimensions, nd*sizeof(intp));
        /* Index is 1-d array */
        loop_i[axis] = ind_size;
        if (out == NULL) {
            loop->ret = (PyArrayObject *)
                PyArray_New(Py_TYPE(aar), aar->nd, loop_i, otype,
                        NULL, NULL, 0, 0, (PyObject *)aar);
        }
        else {
            outsize = PyArray_MultiplyList(loop_i, aar->nd);
        }
        if (ind_size == 0) {
            loop->meth = ZERO_EL_REDUCELOOP;
            return loop;
        }
        if (loop->meth == ONE_EL_REDUCELOOP) {
            loop->meth = NOBUFFER_REDUCELOOP;
        }
        break;
    }
    if (out) {
        if (PyArray_SIZE(out) != outsize) {
            PyErr_SetString(PyExc_ValueError,
                    "wrong shape for output");
            goto fail;
        }
        loop->ret = (PyArrayObject *)
            PyArray_FromArray(out, PyArray_DescrFromType(otype), flags);
        if (loop->ret && loop->ret != out) {
            loop->retbase = 1;
        }
    }
    if (loop->ret == NULL) {
        goto fail;
    }
    loop->insize = aar->descr->elsize;
    loop->outsize = loop->ret->descr->elsize;
    loop->bufptr[0] = loop->ret->data;

    if (loop->meth == ZERO_EL_REDUCELOOP) {
        loop->size = PyArray_SIZE(loop->ret);
        return loop;
    }

    loop->it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)aar);
    if (loop->it == NULL) {
        return NULL;
    }
    if (loop->meth == ONE_EL_REDUCELOOP) {
        loop->size = loop->it->size;
        return loop;
    }

    /*
     * Fix iterator to loop over correct dimension
     * Set size in axis dimension to 1
     */
    loop->it->contiguous = 0;
    loop->it->size /= (loop->it->dims_m1[axis]+1);
    loop->it->dims_m1[axis] = 0;
    loop->it->backstrides[axis] = 0;
    loop->size = loop->it->size;
    if (operation == UFUNC_REDUCE) {
        loop->steps[0] = 0;
    }
    else {
        loop->rit = (PyArrayIterObject *)                       \
            PyArray_IterNew((PyObject *)(loop->ret));
        if (loop->rit == NULL) {
            return NULL;
        }
        /*
         * Fix iterator to loop over correct dimension
         * Set size in axis dimension to 1
         */
        loop->rit->contiguous = 0;
        loop->rit->size /= (loop->rit->dims_m1[axis] + 1);
        loop->rit->dims_m1[axis] = 0;
        loop->rit->backstrides[axis] = 0;

        if (operation == UFUNC_ACCUMULATE) {
            loop->steps[0] = loop->ret->strides[axis];
        }
        else {
            loop->steps[0] = 0;
        }
    }
    loop->steps[2] = loop->steps[0];
    loop->bufptr[2] = loop->bufptr[0] + loop->steps[2];
    if (loop->meth == BUFFER_UFUNCLOOP) {
        int _size;

        loop->steps[1] = loop->outsize;
        if (otype != aar->descr->type_num) {
            _size=loop->bufsize*(loop->outsize + aar->descr->elsize);
            loop->buffer = PyDataMem_NEW(_size);
            if (loop->buffer == NULL) {
                goto fail;
            }
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                memset(loop->buffer, 0, _size);
            }
            loop->castbuf = loop->buffer + loop->bufsize*aar->descr->elsize;
            loop->bufptr[1] = loop->castbuf;
            loop->cast = PyArray_GetCastFunc(aar->descr, otype);
            if (loop->cast == NULL) {
                goto fail;
            }
        }
        else {
            _size = loop->bufsize * loop->outsize;
            loop->buffer = PyDataMem_NEW(_size);
            if (loop->buffer == NULL) {
                goto fail;
            }
            if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                memset(loop->buffer, 0, _size);
            }
            loop->bufptr[1] = loop->buffer;
        }
    }
    PyUFunc_clearfperr();
    return loop;

 fail:
    ufuncreduce_dealloc(loop);
    return NULL;
}


/*
 * Given the output type, finds the specified binary op.  The
 * ufunc must have nin==2 and nout==1.  The function may modify
 * otype if the given type isn't found.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
get_binary_op_function(PyUFuncObject *self, int *otype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata)
{
    int i;
    PyUFunc_Loop1d *funcdata;

    /* If the type is custom and there are userloops, search for it here */
    if (self->userloops != NULL && PyTypeNum_ISUSERDEF(*otype)) {
        PyObject *key, *obj;
        key = PyInt_FromLong(*otype);
        if (key == NULL) {
            return -1;
        }
        obj = PyDict_GetItem(self->userloops, key);
        Py_DECREF(key);
        if (obj != NULL) {
            funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
            while (funcdata != NULL) {
                int *types = funcdata->arg_types;

                if (types[0] == *otype && types[1] == *otype &&
                                                types[2] == *otype) {
                    *out_innerloop = funcdata->func;
                    *out_innerloopdata = funcdata->data;
                    return 0;
                }

                funcdata = funcdata->next;
            }
        }
    }

    /* Search for a function with compatible inputs */
    for (i = 0; i < self->ntypes; ++i) {
        char *types = self->types + i*self->nargs;

        if (PyArray_CanCastSafely(*otype, types[0]) &&
                    types[0] == types[1] &&
                    (*otype == NPY_OBJECT || types[0] != NPY_OBJECT)) {
            /* If the signature is "xx->x", we found the loop */
            if (types[2] == types[0]) {
                *out_innerloop = self->functions[i];
                *out_innerloopdata = self->data[i];
                *otype = types[0];
                return 0;
            }
            /*
             * Otherwise, we found the natural type of the reduction,
             * replace otype and search again
             */
            else {
                *otype = types[2];
                break;
            }
        }
    }

    /* Search for the exact function */
    for (i = 0; i < self->ntypes; ++i) {
        char *types = self->types + i*self->nargs;

        if (PyArray_CanCastSafely(*otype, types[0]) &&
                    types[0] == types[1] &&
                    types[1] == types[2] &&
                    (*otype == NPY_OBJECT || types[0] != NPY_OBJECT)) {
            /* Since the signature is "xx->x", we found the loop */
            *out_innerloop = self->functions[i];
            *out_innerloopdata = self->data[i];
            *otype = types[0];
            return 0;
        }
    }

    return -1;
}

static PyObject *
PyUFunc_ReductionOp(PyUFuncObject *self, PyArrayObject *arr,
                    PyArrayObject *out, int axis, int otype,
                    int operation, char *opname)
{
    PyArrayObject *op[2];
    PyArray_Descr *op_dtypes[2] = {NULL, NULL};
    npy_intp op_axes_arrays[2][NPY_MAXDIMS];
    npy_intp *op_axes[2] = {op_axes_arrays[0], op_axes_arrays[1]};
    npy_uint32 op_flags[2];
    int i, idim, ndim, otype_final;
    int needs_api, need_outer_iterator;
    NPY_BEGIN_THREADS_DEF;

    NpyIter *iter = NULL, *iter_inner = NULL;

    /* The selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;

    char *ufunc_name = self->name ? self->name : "(unknown)";

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;
    PyObject *errobj = NULL;

    NPY_UF_DBG_PRINTF("\nEvaluating ufunc %s.%s\n", ufunc_name, opname);

#if 0
    printf("Doing %s.%s on array with dtype :  ", ufunc_name, opname);
    PyObject_Print((PyObject *)PyArray_DESCR(arr), stdout, 0);
    printf("\n");
#endif

    if (PyUFunc_GetPyValues(opname, &buffersize, &errormask, &errobj) < 0) {
        return NULL;
    }

    /* Take a reference to out for later returning */
    Py_XINCREF(out);

    otype_final = otype;
    if (get_binary_op_function(self, &otype_final,
                                &innerloop, &innerloopdata) < 0) {
        PyArray_Descr *dtype = PyArray_DescrFromType(otype);
        PyErr_Format(PyExc_ValueError,
                     "could not find a matching type for %s.%s, "
                     "requested type has type code '%c'",
                            ufunc_name, opname, dtype ? dtype->type : '-');
        Py_XDECREF(dtype);
        goto fail;
    }

    ndim = PyArray_NDIM(arr);

    /* Set up the output data type */
    op_dtypes[0] = PyArray_DescrFromType(otype_final);
    if (op_dtypes[0] == NULL) {
        goto fail;
    }

#if 0
    printf("Found %s.%s inner loop with dtype :  ", ufunc_name, opname);
    PyObject_Print((PyObject *)op_dtypes[0], stdout, 0);
    printf("\n");
#endif

    /* Set up the op_axes for the outer loop */
    if (operation == UFUNC_REDUCE) {
        for (i = 0, idim = 0; idim < ndim; ++idim) {
            if (idim != axis) {
                op_axes_arrays[0][i] = i;
                op_axes_arrays[1][i] = idim;
                i++;
            }
        }
    }
    else if (operation == UFUNC_ACCUMULATE) {
        for (idim = 0; idim < ndim; ++idim) {
            op_axes_arrays[0][idim] = idim;
            op_axes_arrays[1][idim] = idim;
        }
    }
    else {
        PyErr_Format(PyExc_RuntimeError,
                    "invalid reduction operation %s.%s", ufunc_name, opname);
        goto fail;
    }

    /* The per-operand flags for the outer loop */
    op_flags[0] = NPY_ITER_READWRITE|
                  NPY_ITER_NO_BROADCAST|
                  NPY_ITER_ALLOCATE|
                  NPY_ITER_NO_SUBTYPE;
    op_flags[1] = NPY_ITER_READONLY;

    op[0] = out;
    op[1] = arr;

    need_outer_iterator = (ndim > 1);
    if (operation == UFUNC_ACCUMULATE) {
        /* This is because we can't buffer, so must do UPDATEIFCOPY */
        if (!PyArray_ISALIGNED(arr) || (out && !PyArray_ISALIGNED(out)) ||
                !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(arr)) ||
                (out &&
                 !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(out)))) {
            need_outer_iterator = 1;
        }
    }

    if (need_outer_iterator) {
        npy_intp ndim_iter = 0;
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK|
                           NPY_ITER_REFS_OK;
        PyArray_Descr **op_dtypes_param = NULL;

        if (operation == UFUNC_REDUCE) {
            ndim_iter = ndim - 1;
            if (out == NULL) {
                op_dtypes_param = op_dtypes;
            }
        }
        else if (operation == UFUNC_ACCUMULATE) {
            /*
             * The way accumulate is set up, we can't do buffering,
             * so make a copy instead when necessary.
             */
            ndim_iter = ndim;
            flags |= NPY_ITER_COORDS;
            /* Add some more flags */
            op_flags[0] |= NPY_ITER_UPDATEIFCOPY|NPY_ITER_ALIGNED;
            op_flags[1] |= NPY_ITER_COPY|NPY_ITER_ALIGNED;
            op_dtypes_param = op_dtypes;
            op_dtypes[1] = op_dtypes[0];
        }
        NPY_UF_DBG_PRINTF("Allocating outer iterator\n");
        iter = NpyIter_MultiNew(2, op, flags,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                   op_flags,
                                   op_dtypes_param,
                                   ndim_iter, op_axes, 0);
        if (iter == NULL) {
            goto fail;
        }

        if (operation == UFUNC_ACCUMULATE) {
            if (NpyIter_RemoveAxis(iter, axis) != NPY_SUCCEED) {
                goto fail;
            }
            if (NpyIter_RemoveCoords(iter) != NPY_SUCCEED) {
                goto fail;
            }

            op[0] = NpyIter_GetOperandArray(iter)[0];
            op[1] = NpyIter_GetOperandArray(iter)[1];
        }
    }

    /* Get the output */
    if (!out) {
        if (iter) {
            op[0] = out = NpyIter_GetOperandArray(iter)[0];
            Py_INCREF(out);
        }
        else {
            PyArray_Descr *dtype = op_dtypes[0];
            Py_INCREF(dtype);
            if (operation == UFUNC_REDUCE) {
                op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                        &PyArray_Type, dtype,
                                        0, NULL, NULL, NULL,
                                        0, NULL);
            }
            else if (operation == UFUNC_ACCUMULATE) {
                op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                        &PyArray_Type, dtype,
                                        ndim, PyArray_DIMS(op[1]), NULL, NULL,
                                        0, NULL);
            }
            if (out == NULL) {
                goto fail;
            }
        }
    }

    /*
     * If the reduction unit has size zero, either return the reduction
     * unit for UFUNC_REDUCE, or return the zero-sized output array
     * for UFUNC_ACCUMULATE.
     */
    if (PyArray_DIM(op[1], axis) == 0) {
        if (operation == UFUNC_REDUCE) {
            if (self->identity == PyUFunc_None) {
                PyErr_Format(PyExc_ValueError,
                             "zero-size array to %s.%s "
                             "without identity", ufunc_name, opname);
                goto fail;
            }
            if (self->identity == PyUFunc_One) {
                PyObject *obj = PyInt_FromLong((long) 1);
                if (obj == NULL) {
                    goto fail;
                }
                PyArray_FillWithScalar(op[0], obj);
                Py_DECREF(obj);
            } else {
                PyObject *obj = PyInt_FromLong((long) 0);
                if (obj == NULL) {
                    goto fail;
                }
                PyArray_FillWithScalar(op[0], obj);
                Py_DECREF(obj);
            }
        }

        goto finish;
    }

    /* Only allocate an inner iterator if it's necessary */
    if (!PyArray_ISALIGNED(op[1]) || !PyArray_ISALIGNED(op[0]) ||
                !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(op[1])) ||
                !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(op[0]))) {
        /* Also set the dtype for buffering arr */
        op_dtypes[1] = op_dtypes[0];

        NPY_UF_DBG_PRINTF("Allocating inner iterator\n");
        if (operation == UFUNC_REDUCE) {
            /* The per-operand flags for the inner loop */
            op_flags[0] = NPY_ITER_READWRITE|
                          NPY_ITER_ALIGNED;
            op_flags[1] = NPY_ITER_READONLY|
                          NPY_ITER_ALIGNED;

            op_axes[0][0] = -1;
            op_axes[1][0] = axis;

            iter_inner = NpyIter_MultiNew(2, op, NPY_ITER_NO_INNER_ITERATION|
                                       NPY_ITER_BUFFERED|
                                       NPY_ITER_DELAY_BUFALLOC|
                                       NPY_ITER_GROWINNER|
                                       NPY_ITER_REDUCE_OK|
                                       NPY_ITER_REFS_OK,
                                       NPY_CORDER, NPY_UNSAFE_CASTING,
                                       op_flags, op_dtypes,
                                       1, op_axes, buffersize);
        }
        else if (operation == UFUNC_ACCUMULATE) {
            /* The per-operand flags for the inner loop */
            op_flags[0] = NPY_ITER_READWRITE|
                          NPY_ITER_ALIGNED;
            op_flags[1] = NPY_ITER_READONLY|
                          NPY_ITER_ALIGNED;

            op_axes[0][0] = axis;
            op_axes[1][0] = axis;

            iter_inner = NpyIter_MultiNew(2, op, NPY_ITER_NO_INNER_ITERATION|
                                       NPY_ITER_REFS_OK,
                                       NPY_CORDER, NPY_UNSAFE_CASTING,
                                       op_flags, op_dtypes,
                                       1, op_axes, buffersize);
        }
        if (iter_inner == NULL) {
            goto fail;
        }
    }

    if (iter && NpyIter_GetIterSize(iter) != 0) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];

        NpyIter_IterNext_Fn iternext;
        char **dataptr;
        npy_intp *stride;
        npy_intp *count_ptr;

        int itemsize = op_dtypes[0]->elsize;

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        stride = NpyIter_GetInnerStrideArray(iter);
        count_ptr = NpyIter_GetInnerLoopSizePtr(iter);

        
        /* Execute the loop with two nested iterators */
        if (iter_inner) {
            NpyIter_IterNext_Fn iternext_inner;
            char **dataptr_inner;
            npy_intp *stride_inner;
            npy_intp *count_ptr_inner;

            NPY_UF_DBG_PRINTF("UFunc: Reduce loop with two nested iterators\n");
            iternext_inner = NpyIter_GetIterNext(iter_inner, NULL);
            if (iternext_inner == NULL) {
                goto fail;
            }
            dataptr_inner = NpyIter_GetDataPtrArray(iter_inner);
            stride_inner = NpyIter_GetInnerStrideArray(iter_inner);
            count_ptr_inner = NpyIter_GetInnerLoopSizePtr(iter_inner);

            needs_api = NpyIter_IterationNeedsAPI(iter) ||
                        NpyIter_IterationNeedsAPI(iter_inner);

            if (!needs_api) {
                NPY_BEGIN_THREADS;
            }

            do {
                /* Reset the inner iterator to the outer's data */
                if (NpyIter_ResetBasePointers(iter_inner, dataptr, NULL)
                                                != NPY_SUCCEED) {
                    goto fail;
                }

                do {
                    npy_intp count = *count_ptr_inner;

                    /* Copy the first element to start the reduction */
                    if (otype == NPY_OBJECT) {
                        Py_XDECREF(*(PyObject **)dataptr_inner[0]);
                        *(PyObject **)dataptr_inner[0] =
                                            *(PyObject **)dataptr_inner[1];
                        Py_XINCREF(*(PyObject **)dataptr_inner[0]);
                    }
                    else {
                        memcpy(dataptr_inner[0], dataptr_inner[1], itemsize);
                    }
                    
                    if (count > 1) {
                        --count;
                        /* Turn the two items into three for the inner loop */
                        if (operation == UFUNC_REDUCE) {
                            dataptr_copy[0] = dataptr_inner[0];
                            dataptr_copy[1] = dataptr_inner[1] +
                                              stride_inner[1];
                            dataptr_copy[2] = dataptr_inner[0];
                        }
                        else if (operation == UFUNC_ACCUMULATE) {
                            dataptr_copy[0] = dataptr_inner[0];
                            dataptr_copy[1] = dataptr_inner[1] +
                                              stride_inner[1];
                            dataptr_copy[2] = dataptr_inner[0] +
                                              stride_inner[0];
                        }
                        stride_copy[0] = stride_inner[0];
                        stride_copy[1] = stride_inner[1];
                        stride_copy[2] = stride_inner[0];
                        NPY_UF_DBG_PRINTF("iterator loop count %d\n",
                                                    (int)count);
                        innerloop(dataptr_copy, &count,
                                    stride_copy, innerloopdata);
                    }
                } while(iternext_inner(iter_inner));
            } while (iternext(iter));

            if (!needs_api) {
                NPY_END_THREADS;
            }
        }
        /* Execute the loop with just the outer iterator */
        else {
            npy_intp count_m1 = PyArray_DIM(op[1], axis)-1;
            npy_intp stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

            NPY_UF_DBG_PRINTF("UFunc: Reduce loop with just outer iterator\n");

            if (operation == UFUNC_ACCUMULATE) {
                stride0 = PyArray_STRIDE(op[0], axis);
            }

            stride_copy[0] = stride0;
            stride_copy[1] = stride1;
            stride_copy[2] = stride0;

            needs_api = NpyIter_IterationNeedsAPI(iter);

            if (!needs_api) {
                NPY_BEGIN_THREADS;
            }

            do {
                
                /* Copy the first element to start the reduction */
                if (otype == NPY_OBJECT) {
                    Py_XDECREF(*(PyObject **)dataptr[0]);
                    *(PyObject **)dataptr[0] =
                                        *(PyObject **)dataptr[1];
                    Py_XINCREF(*(PyObject **)dataptr[0]);
                }
                else {
                    memcpy(dataptr[0], dataptr[1], itemsize);
                }

                if (count_m1 > 0) {
                    /* Turn the two items into three for the inner loop */
                    if (operation == UFUNC_REDUCE) {
                        dataptr_copy[0] = dataptr[0];
                        dataptr_copy[1] = dataptr[1] + stride1;
                        dataptr_copy[2] = dataptr[0];
                    }
                    else if (operation == UFUNC_ACCUMULATE) {
                        dataptr_copy[0] = dataptr[0];
                        dataptr_copy[1] = dataptr[1] + stride1;
                        dataptr_copy[2] = dataptr[0] + stride0;
                    }
                    NPY_UF_DBG_PRINTF("iterator loop count %d\n",
                                                    (int)count_m1);
                    innerloop(dataptr_copy, &count_m1,
                                stride_copy, innerloopdata);
                }
            } while (iternext(iter));

            if (!needs_api) {
                NPY_END_THREADS;
            }
        }
    }
    else if (iter == NULL) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];

        int itemsize = op_dtypes[0]->elsize;

        /* Execute the loop with just the inner iterator */
        if (iter_inner) {
            NpyIter_IterNext_Fn iternext_inner;
            char **dataptr_inner;
            npy_intp *stride_inner;
            npy_intp *count_ptr_inner;

            NPY_UF_DBG_PRINTF("UFunc: Reduce loop with just inner iterator\n");

            iternext_inner = NpyIter_GetIterNext(iter_inner, NULL);
            if (iternext_inner == NULL) {
                goto fail;
            }
            dataptr_inner = NpyIter_GetDataPtrArray(iter_inner);
            stride_inner = NpyIter_GetInnerStrideArray(iter_inner);
            count_ptr_inner = NpyIter_GetInnerLoopSizePtr(iter_inner);

            /* Reset the inner iterator to prepare the buffers */
            if (NpyIter_Reset(iter_inner, NULL) != NPY_SUCCEED) {
                goto fail;
            }

            needs_api = NpyIter_IterationNeedsAPI(iter_inner);

            if (!needs_api) {
                NPY_BEGIN_THREADS;
            }

            do {
                npy_intp count = *count_ptr_inner;

                /* Copy the first element to start the reduction */
                if (otype == NPY_OBJECT) {
                    Py_XDECREF(*(PyObject **)dataptr_inner[0]);
                    *(PyObject **)dataptr_inner[0] =
                                        *(PyObject **)dataptr_inner[1];
                    Py_XINCREF(*(PyObject **)dataptr_inner[0]);
                }
                else {
                    memcpy(dataptr_inner[0], dataptr_inner[1], itemsize);
                }
                
                if (count > 1) {
                    --count;
                    /* Turn the two items into three for the inner loop */
                    if (operation == UFUNC_REDUCE) {
                        dataptr_copy[0] = dataptr_inner[0];
                        dataptr_copy[1] = dataptr_inner[1] + stride_inner[1];
                        dataptr_copy[2] = dataptr_inner[0];
                    }
                    else if (operation == UFUNC_ACCUMULATE) {
                        dataptr_copy[0] = dataptr_inner[0];
                        dataptr_copy[1] = dataptr_inner[1] + stride_inner[1];
                        dataptr_copy[2] = dataptr_inner[0] + stride_inner[0];
                    }
                    stride_copy[0] = stride_inner[0];
                    stride_copy[1] = stride_inner[1];
                    stride_copy[2] = stride_inner[0];
                    NPY_UF_DBG_PRINTF("iterator loop count %d\n", (int)count);
                    innerloop(dataptr_copy, &count,
                                stride_copy, innerloopdata);
                }
            } while(iternext_inner(iter_inner));

            if (!needs_api) {
                NPY_END_THREADS;
            }
        }
        /* Execute the loop with no iterators */
        else {
            npy_intp count = PyArray_DIM(op[1], axis);
            npy_intp stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

            NPY_UF_DBG_PRINTF("UFunc: Reduce loop with no iterators\n");

            if (operation == UFUNC_REDUCE) {
                if (PyArray_NDIM(op[0]) != 0) {
                    PyErr_SetString(PyExc_ValueError,
                            "provided out is the wrong size "
                            "for the reduction");
                    goto fail;
                }
            }
            else if (operation == UFUNC_ACCUMULATE) {
                if (PyArray_NDIM(op[0]) != PyArray_NDIM(op[1]) ||
                        !PyArray_CompareLists(PyArray_DIMS(op[0]),
                                              PyArray_DIMS(op[1]),
                                              PyArray_NDIM(op[0]))) {
                    PyErr_SetString(PyExc_ValueError,
                            "provided out is the wrong size "
                            "for the reduction");
                    goto fail;
                }
                stride0 = PyArray_STRIDE(op[0], axis);
            }

            stride_copy[0] = stride0;
            stride_copy[1] = stride1;
            stride_copy[2] = stride0;

            /* Turn the two items into three for the inner loop */
            dataptr_copy[0] = PyArray_BYTES(op[0]);
            dataptr_copy[1] = PyArray_BYTES(op[1]);
            dataptr_copy[2] = PyArray_BYTES(op[0]);

            /* Copy the first element to start the reduction */
            if (otype == NPY_OBJECT) {
                Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                *(PyObject **)dataptr_copy[0] =
                                    *(PyObject **)dataptr_copy[1];
                Py_XINCREF(*(PyObject **)dataptr_copy[0]);
            }
            else {
                memcpy(dataptr_copy[0], dataptr_copy[1], itemsize);
            }

            if (count > 1) {
                --count;
                if (operation == UFUNC_REDUCE) {
                    dataptr_copy[1] += stride1;
                }
                else if (operation == UFUNC_ACCUMULATE) {
                    dataptr_copy[1] += stride1;
                    dataptr_copy[2] += stride0;
                }

                NPY_UF_DBG_PRINTF("iterator loop count %d\n", (int)count);

                needs_api = PyDataType_REFCHK(op_dtypes[0]);

                if (!needs_api) {
                    NPY_BEGIN_THREADS;
                }

                innerloop(dataptr_copy, &count,
                            stride_copy, innerloopdata);

                if (!needs_api) {
                    NPY_END_THREADS;
                }
            }
        }
    }

finish:
    Py_XDECREF(op_dtypes[0]);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }
    if (iter_inner != NULL) {
        NpyIter_Deallocate(iter_inner);
    }
    return (PyObject *)out;

fail:
    Py_XDECREF(out);
    Py_XDECREF(op_dtypes[0]);

    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }

    Py_XDECREF(errobj);

    return NULL;
}

/*
 * We have two basic kinds of loops. One is used when arr is not-swapped
 * and aligned and output type is the same as input type.  The other uses
 * buffers when one of these is not satisfied.
 *
 *  Zero-length and one-length axes-to-be-reduced are handled separately.
 */
static PyObject *
PyUFunc_Reduce(PyUFuncObject *self, PyArrayObject *arr, PyArrayObject *out,
        int axis, int otype)
{
    return PyUFunc_ReductionOp(self, arr, out, axis, otype,
                                UFUNC_REDUCE, "reduce");
}


static PyObject *
PyUFunc_Accumulate(PyUFuncObject *self, PyArrayObject *arr, PyArrayObject *out,
                   int axis, int otype)
{
    return PyUFunc_ReductionOp(self, arr, out, axis, otype,
                                UFUNC_ACCUMULATE, "accumulate");
}

/*
 * Reduceat performs a reduce over an axis using the indices as a guide
 *
 * op.reduceat(array,indices)  computes
 * op.reduce(array[indices[i]:indices[i+1]]
 * for i=0..end with an implicit indices[i+1]=len(array)
 * assumed when i=end-1
 *
 * if indices[i+1] <= indices[i]+1
 * then the result is array[indices[i]] for that value
 *
 * op.accumulate(array) is the same as
 * op.reduceat(array,indices)[::2]
 * where indices is range(len(array)-1) with a zero placed in every other sample
 * indices = zeros(len(array)*2-1)
 * indices[1::2] = range(1,len(array))
 *
 * output shape is based on the size of indices
 */
static PyObject *
PyUFunc_Reduceat(PyUFuncObject *self, PyArrayObject *arr, PyArrayObject *ind,
                 PyArrayObject *out, int axis, int otype)
{
    PyArrayObject *ret;
    PyUFuncReduceObject *loop;
    intp *ptr = (intp *)ind->data;
    intp nn = ind->dimensions[0];
    intp mm = arr->dimensions[axis] - 1;
    intp n, i, j;
    char *dptr;
    NPY_BEGIN_THREADS_DEF;

    /* Check for out-of-bounds values in indices array */
    for (i = 0; i<nn; i++) {
        if ((*ptr < 0) || (*ptr > mm)) {
            PyErr_Format(PyExc_IndexError,
                    "index out-of-bounds (0, %d)", (int) mm);
            return NULL;
        }
        ptr++;
    }

    ptr = (intp *)ind->data;
    /* Construct loop object */
    loop = construct_reduce(self, &arr, out, axis, otype,
            UFUNC_REDUCEAT, nn, "reduceat");
    if (!loop) {
        return NULL;
    }

    NPY_LOOP_BEGIN_THREADS;
    switch(loop->meth) {
    case ZERO_EL_REDUCELOOP:
        /* zero-length index -- return array immediately */
        /* fprintf(stderr, "ZERO..\n"); */
        break;
    case NOBUFFER_UFUNCLOOP:
        /* Reduceat
         * NOBUFFER -- behaved array and same type
         */
        /* fprintf(stderr, "NOBUFFER..%d\n", loop->size); */
        while (loop->index < loop->size) {
            ptr = (intp *)ind->data;
            for (i = 0; i < nn; i++) {
                loop->bufptr[1] = loop->it->dataptr + (*ptr)*loop->steps[1];
                if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                    Py_XINCREF(*((PyObject **)loop->bufptr[1]));
                }
                memcpy(loop->bufptr[0], loop->bufptr[1], loop->outsize);
                mm = (i == nn - 1 ? arr->dimensions[axis] - *ptr :
                        *(ptr + 1) - *ptr) - 1;
                if (mm > 0) {
                    loop->bufptr[1] += loop->steps[1];
                    loop->bufptr[2] = loop->bufptr[0];
                    loop->function((char **)loop->bufptr, &mm,
                            loop->steps, loop->funcdata);
                    UFUNC_CHECK_ERROR(loop);
                }
                loop->bufptr[0] += loop->ret->strides[axis];
                ptr++;
            }
            PyArray_ITER_NEXT(loop->it);
            PyArray_ITER_NEXT(loop->rit);
            loop->bufptr[0] = loop->rit->dataptr;
            loop->index++;
        }
        break;

    case BUFFER_UFUNCLOOP:
        /* Reduceat
         * BUFFER -- misbehaved array or different types
         */
        /* fprintf(stderr, "BUFFERED..%d\n", loop->size); */
        while (loop->index < loop->size) {
            ptr = (intp *)ind->data;
            for (i = 0; i < nn; i++) {
                if (loop->obj & UFUNC_OBJ_ISOBJECT) {
                    Py_XINCREF(*((PyObject **)loop->idptr));
                }
                memcpy(loop->bufptr[0], loop->idptr, loop->outsize);
                n = 0;
                mm = (i == nn - 1 ? arr->dimensions[axis] - *ptr :
                        *(ptr + 1) - *ptr);
                if (mm < 1) {
                    mm = 1;
                }
                loop->inptr = loop->it->dataptr + (*ptr)*loop->instrides;
                while (n < mm) {
                    /* Copy up to loop->bufsize elements to buffer */
                    dptr = loop->buffer;
                    for (j = 0; j < loop->bufsize; j++, n++) {
                        if (n == mm) {
                            break;
                        }
                        arr->descr->f->copyswap(dptr, loop->inptr,
                             loop->swap, NULL);
                        loop->inptr += loop->instrides;
                        dptr += loop->insize;
                    }
                    if (loop->cast) {
                        loop->cast(loop->buffer, loop->castbuf, j, NULL, NULL);
                    }
                    loop->bufptr[2] = loop->bufptr[0];
                    loop->function((char **)loop->bufptr, &j,
                            loop->steps, loop->funcdata);
                    UFUNC_CHECK_ERROR(loop);
                    loop->bufptr[0] += j*loop->steps[0];
                }
                loop->bufptr[0] += loop->ret->strides[axis];
                ptr++;
            }
            PyArray_ITER_NEXT(loop->it);
            PyArray_ITER_NEXT(loop->rit);
            loop->bufptr[0] = loop->rit->dataptr;
            loop->index++;
        }

        /*
         * DECREF left-over objects if buffering was used.
         * It is needed when casting created new objects in
         * castbuf.  Intermediate copying into castbuf (via
         * loop->function) decref'd what was already there.

         * It's the final copy into the castbuf that needs a DECREF.
         */

        /* Only when casting needed and it is from a non-object array */
        if ((loop->obj & UFUNC_OBJ_ISOBJECT) && loop->cast &&
            (!PyArray_ISOBJECT(arr))) {
            for (i=0; i<loop->bufsize; i++) {
                Py_CLEAR(((PyObject **)loop->castbuf)[i]);
            }
        }

        break;
    }
    NPY_LOOP_END_THREADS;
    /* Hang on to this reference -- will be decref'd with loop */
    if (loop->retbase) {
        ret = (PyArrayObject *)loop->ret->base;
    }
    else {
        ret = loop->ret;
    }
    Py_INCREF(ret);
    ufuncreduce_dealloc(loop);
    return (PyObject *)ret;

fail:
    NPY_LOOP_END_THREADS;
    if (loop) {
        ufuncreduce_dealloc(loop);
    }
    return NULL;
}


/*
 * This code handles reduce, reduceat, and accumulate
 * (accumulate and reduce are special cases of the more general reduceat
 * but they are handled separately for speed)
 */
static PyObject *
PyUFunc_GenericReduction(PyUFuncObject *self, PyObject *args,
                         PyObject *kwds, int operation)
{
    int axis=0;
    PyArrayObject *mp, *ret = NULL;
    PyObject *op, *res = NULL;
    PyObject *obj_ind, *context;
    PyArrayObject *indices = NULL;
    PyArray_Descr *otype = NULL;
    PyArrayObject *out = NULL;
    static char *kwlist1[] = {"array", "axis", "dtype", "out", NULL};
    static char *kwlist2[] = {"array", "indices", "axis", "dtype", "out", NULL};
    static char *_reduce_type[] = {"reduce", "accumulate", "reduceat", NULL};

    if (self == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return NULL;
    }
    if (self->core_enabled) {
        PyErr_Format(PyExc_RuntimeError,
                     "Reduction not defined on ufunc with signature");
        return NULL;
    }
    if (self->nin != 2) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for binary functions",
                     _reduce_type[operation]);
        return NULL;
    }
    if (self->nout != 1) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for functions " \
                     "returning a single value",
                     _reduce_type[operation]);
        return NULL;
    }

    if (operation == UFUNC_REDUCEAT) {
        PyArray_Descr *indtype;
        indtype = PyArray_DescrFromType(PyArray_INTP);
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iO&O&", kwlist2,
                                        &op, &obj_ind, &axis,
                                        PyArray_DescrConverter2, &otype,
                                        PyArray_OutputConverter, &out)) {
            Py_XDECREF(otype);
            return NULL;
        }
        indices = (PyArrayObject *)PyArray_FromAny(obj_ind, indtype,
                                                   1, 1, CARRAY, NULL);
        if (indices == NULL) {
            Py_XDECREF(otype);
            return NULL;
        }
    }
    else {
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO&O&", kwlist1,
                                        &op, &axis,
                                        PyArray_DescrConverter2, &otype,
                                        PyArray_OutputConverter, &out)) {
            Py_XDECREF(otype);
            return NULL;
        }
    }
    /* Ensure input is an array */
    if (!PyArray_Check(op) && !PyArray_IsScalar(op, Generic)) {
        context = Py_BuildValue("O(O)i", self, op, 0);
    }
    else {
        context = NULL;
    }
    mp = (PyArrayObject *)PyArray_FromAny(op, NULL, 0, 0, 0, context);
    Py_XDECREF(context);
    if (mp == NULL) {
        return NULL;
    }
    /* Check to see if input is zero-dimensional */
    if (mp->nd == 0) {
        PyErr_Format(PyExc_TypeError, "cannot %s on a scalar",
                     _reduce_type[operation]);
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }
    /* Check to see that type (and otype) is not FLEXIBLE */
    if (PyArray_ISFLEXIBLE(mp) ||
        (otype && PyTypeNum_ISFLEXIBLE(otype->type_num))) {
        PyErr_Format(PyExc_TypeError,
                     "cannot perform %s with flexible type",
                     _reduce_type[operation]);
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }

    if (axis < 0) {
        axis += mp->nd;
    }
    if (axis < 0 || axis >= mp->nd) {
        PyErr_SetString(PyExc_ValueError, "axis not in array");
        Py_XDECREF(otype);
        Py_DECREF(mp);
        return NULL;
    }
     /*
      * If out is specified it determines otype
      * unless otype already specified.
      */
    if (otype == NULL && out != NULL) {
        otype = out->descr;
        Py_INCREF(otype);
    }
    if (otype == NULL) {
        /*
         * For integer types --- make sure at least a long
         * is used for add and multiply reduction to avoid overflow
         */
        int typenum = PyArray_TYPE(mp);
        if ((typenum < NPY_HALF)
            && ((strcmp(self->name,"add") == 0)
                || (strcmp(self->name,"multiply") == 0))) {
            if (PyTypeNum_ISBOOL(typenum)) {
                typenum = PyArray_LONG;
            }
            else if ((size_t)mp->descr->elsize < sizeof(long)) {
                if (PyTypeNum_ISUNSIGNED(typenum)) {
                    typenum = PyArray_ULONG;
                }
                else {
                    typenum = PyArray_LONG;
                }
            }
        }
        otype = PyArray_DescrFromType(typenum);
    }


    switch(operation) {
    case UFUNC_REDUCE:
        ret = (PyArrayObject *)PyUFunc_Reduce(self, mp, out, axis,
                                              otype->type_num);
        break;
    case UFUNC_ACCUMULATE:
        ret = (PyArrayObject *)PyUFunc_Accumulate(self, mp, out, axis,
                                                  otype->type_num);
        break;
    case UFUNC_REDUCEAT:
        ret = (PyArrayObject *)PyUFunc_Reduceat(self, mp, indices, out,
                                                axis, otype->type_num);
        Py_DECREF(indices);
        break;
    }
    Py_DECREF(mp);
    Py_DECREF(otype);
    if (ret == NULL) {
        return NULL;
    }
    if (Py_TYPE(op) != Py_TYPE(ret)) {
        res = PyObject_CallMethod(op, "__array_wrap__", "O", ret);
        if (res == NULL) {
            PyErr_Clear();
        }
        else if (res == Py_None) {
            Py_DECREF(res);
        }
        else {
            Py_DECREF(ret);
            return res;
        }
    }
    return PyArray_Return(ret);
}

/*
 * This function analyzes the input arguments
 * and determines an appropriate __array_wrap__ function to call
 * for the outputs.
 *
 * If an output argument is provided, then it is wrapped
 * with its own __array_wrap__ not with the one determined by
 * the input arguments.
 *
 * if the provided output argument is already an array,
 * the wrapping function is None (which means no wrapping will
 * be done --- not even PyArray_Return).
 *
 * A NULL is placed in output_wrap for outputs that
 * should just have PyArray_Return called.
 */
static void
_find_array_wrap(PyObject *args, PyObject *kwds,
                PyObject **output_wrap, int nin, int nout)
{
    Py_ssize_t nargs;
    int i;
    int np = 0;
    PyObject *with_wrap[NPY_MAXARGS], *wraps[NPY_MAXARGS];
    PyObject *obj, *wrap = NULL;

    nargs = PyTuple_GET_SIZE(args);
    for (i = 0; i < nin; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        wrap = PyObject_GetAttrString(obj, "__array_wrap__");
        if (wrap) {
            if (PyCallable_Check(wrap)) {
                with_wrap[np] = obj;
                wraps[np] = wrap;
                ++np;
            }
            else {
                Py_DECREF(wrap);
                wrap = NULL;
            }
        }
        else {
            PyErr_Clear();
        }
    }
    if (np > 0) {
        /* If we have some wraps defined, find the one of highest priority */
        wrap = wraps[0];
        if (np > 1) {
            double maxpriority = PyArray_GetPriority(with_wrap[0],
                        PyArray_SUBTYPE_PRIORITY);
            for (i = 1; i < np; ++i) {
                double priority = PyArray_GetPriority(with_wrap[i],
                            PyArray_SUBTYPE_PRIORITY);
                if (priority > maxpriority) {
                    maxpriority = priority;
                    Py_DECREF(wrap);
                    wrap = wraps[i];
                }
                else {
                    Py_DECREF(wraps[i]);
                }
            }
        }
    }

    /*
     * Here wrap is the wrapping function determined from the
     * input arrays (could be NULL).
     *
     * For all the output arrays decide what to do.
     *
     * 1) Use the wrap function determined from the input arrays
     * This is the default if the output array is not
     * passed in.
     *
     * 2) Use the __array_wrap__ method of the output object
     * passed in. -- this is special cased for
     * exact ndarray so that no PyArray_Return is
     * done in that case.
     */
    for (i = 0; i < nout; i++) {
        int j = nin + i;
        int incref = 1;
        output_wrap[i] = wrap;
        obj = NULL;
        if (j < nargs) {
            obj = PyTuple_GET_ITEM(args, j);
            /* Output argument one may also be in a keyword argument */
            if (i == 0 && obj == Py_None && kwds != NULL) {
                obj = PyDict_GetItemString(kwds, "out");
            }
        }
        /* Output argument one may also be in a keyword argument */
        else if (i == 0 && kwds != NULL) {
            obj = PyDict_GetItemString(kwds, "out");
        }

        if (obj != Py_None && obj != NULL) {
            if (PyArray_CheckExact(obj)) {
                /* None signals to not call any wrapping */
                output_wrap[i] = Py_None;
            }
            else {
                PyObject *owrap = PyObject_GetAttrString(obj,"__array_wrap__");
                incref = 0;
                if (!(owrap) || !(PyCallable_Check(owrap))) {
                    Py_XDECREF(owrap);
                    owrap = wrap;
                    incref = 1;
                    PyErr_Clear();
                }
                output_wrap[i] = owrap;
            }
        }

        if (incref) {
            Py_XINCREF(output_wrap[i]);
        }
    }
    Py_XDECREF(wrap);
    return;
}


static PyObject *
ufunc_generic_call_iter(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    int i;
    PyTupleObject *ret;
    PyArrayObject *mps[NPY_MAXARGS];
    PyObject *retobj[NPY_MAXARGS];
    PyObject *wraparr[NPY_MAXARGS];
    PyObject *res;
    int errval;

    /*
     * Initialize all array objects to NULL to make cleanup easier
     * if something goes wrong.
     */
    for(i = 0; i < self->nargs; i++) {
        mps[i] = NULL;
    }
    errval = PyUFunc_GenericFunction(self, args, kwds, mps);
    if (errval < 0) {
        for (i = 0; i < self->nargs; i++) {
            PyArray_XDECREF_ERR(mps[i]);
        }
        if (errval == -1)
            return NULL;
        else if (self->nin == 2 && self->nout == 1) {
          /* To allow the other argument to be given a chance
           */
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
        else {
            PyErr_SetString(PyExc_NotImplementedError,
                                        "Not implemented for this type");
            return NULL;
        }
    }

    /* Free the input references */
    for (i = 0; i < self->nin; i++) {
        Py_DECREF(mps[i]);
    }

    /*
     * Use __array_wrap__ on all outputs
     * if present on one of the input arguments.
     * If present for multiple inputs:
     * use __array_wrap__ of input object with largest
     * __array_priority__ (default = 0.0)
     *
     * Exception:  we should not wrap outputs for items already
     * passed in as output-arguments.  These items should either
     * be left unwrapped or wrapped by calling their own __array_wrap__
     * routine.
     *
     * For each output argument, wrap will be either
     * NULL --- call PyArray_Return() -- default if no output arguments given
     * None --- array-object passed in don't call PyArray_Return
     * method --- the __array_wrap__ method to call.
     */
    _find_array_wrap(args, kwds, wraparr, self->nin, self->nout);

    /* wrap outputs */
    for (i = 0; i < self->nout; i++) {
        int j = self->nin+i;
        PyObject *wrap = wraparr[i];

        if (wrap != NULL) {
            if (wrap == Py_None) {
                Py_DECREF(wrap);
                retobj[i] = (PyObject *)mps[j];
                continue;
            }
            res = PyObject_CallFunction(wrap, "O(OOi)", mps[j], self, args, i);
            if (res == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                res = PyObject_CallFunctionObjArgs(wrap, mps[j], NULL);
            }
            Py_DECREF(wrap);
            if (res == NULL) {
                goto fail;
            }
            else if (res == Py_None) {
                Py_DECREF(res);
            }
            else {
                Py_DECREF(mps[j]);
                retobj[i] = res;
                continue;
            }
        }
        /* default behavior */
        retobj[i] = PyArray_Return(mps[j]);
    }

    if (self->nout == 1) {
        return retobj[0];
    }
    else {
        ret = (PyTupleObject *)PyTuple_New(self->nout);
        for (i = 0; i < self->nout; i++) {
            PyTuple_SET_ITEM(ret, i, retobj[i]);
        }
        return (PyObject *)ret;
    }

fail:
    for (i = self->nin; i < self->nargs; i++) {
        Py_XDECREF(mps[i]);
    }
    return NULL;
}

static PyObject *
ufunc_generic_call(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    int i;
    PyTupleObject *ret;
    PyArrayObject *mps[NPY_MAXARGS];
    PyObject *retobj[NPY_MAXARGS];
    PyObject *wraparr[NPY_MAXARGS];
    PyObject *res;
    int errval;

    /*
     * Initialize all array objects to NULL to make cleanup easier
     * if something goes wrong.
     */
    for(i = 0; i < self->nargs; i++) {
        mps[i] = NULL;
    }
    errval = PyUFunc_GenericFunction_Old(self, args, kwds, mps);
    if (errval < 0) {
        for (i = 0; i < self->nargs; i++) {
            PyArray_XDECREF_ERR(mps[i]);
        }
        if (errval == -1)
            return NULL;
        else if (self->nin == 2 && self->nout == 1) {
          /* To allow the other argument to be given a chance
           */
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
        else {
            PyErr_SetString(PyExc_NotImplementedError,
                                        "Not implemented for this type");
            return NULL;
        }
    }
    for (i = 0; i < self->nin; i++) {
        Py_DECREF(mps[i]);
    }
    /*
     * Use __array_wrap__ on all outputs
     * if present on one of the input arguments.
     * If present for multiple inputs:
     * use __array_wrap__ of input object with largest
     * __array_priority__ (default = 0.0)
     *
     * Exception:  we should not wrap outputs for items already
     * passed in as output-arguments.  These items should either
     * be left unwrapped or wrapped by calling their own __array_wrap__
     * routine.
     *
     * For each output argument, wrap will be either
     * NULL --- call PyArray_Return() -- default if no output arguments given
     * None --- array-object passed in don't call PyArray_Return
     * method --- the __array_wrap__ method to call.
     */
    _find_array_wrap(args, kwds, wraparr, self->nin, self->nout);

    /* wrap outputs */
    for (i = 0; i < self->nout; i++) {
        int j = self->nin+i;
        PyObject *wrap;
        /*
         * check to see if any UPDATEIFCOPY flags are set
         * which meant that a temporary output was generated
         */
        if (mps[j]->flags & UPDATEIFCOPY) {
            PyObject *old = mps[j]->base;
            /* we want to hang on to this */
            Py_INCREF(old);
            /* should trigger the copyback into old */
            Py_DECREF(mps[j]);
            mps[j] = (PyArrayObject *)old;
        }
        wrap = wraparr[i];
        if (wrap != NULL) {
            if (wrap == Py_None) {
                Py_DECREF(wrap);
                retobj[i] = (PyObject *)mps[j];
                continue;
            }
            res = PyObject_CallFunction(wrap, "O(OOi)", mps[j], self, args, i);
            if (res == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                res = PyObject_CallFunctionObjArgs(wrap, mps[j], NULL);
            }
            Py_DECREF(wrap);
            if (res == NULL) {
                goto fail;
            }
            else if (res == Py_None) {
                Py_DECREF(res);
            }
            else {
                Py_DECREF(mps[j]);
                retobj[i] = res;
                continue;
            }
        }
        /* default behavior */
        retobj[i] = PyArray_Return(mps[j]);
    }

    if (self->nout == 1) {
        return retobj[0];
    }
    else {
        ret = (PyTupleObject *)PyTuple_New(self->nout);
        for (i = 0; i < self->nout; i++) {
            PyTuple_SET_ITEM(ret, i, retobj[i]);
        }
        return (PyObject *)ret;
    }

fail:
    for (i = self->nin; i < self->nargs; i++) {
        Py_XDECREF(mps[i]);
    }
    return NULL;
}

NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *thedict;
    PyObject *res;

    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    if (PyUFunc_PYVALS_NAME == NULL) {
        PyUFunc_PYVALS_NAME = PyUString_InternFromString(UFUNC_PYVALS_NAME);
    }
    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    res = PyDict_GetItem(thedict, PyUFunc_PYVALS_NAME);
    if (res != NULL) {
        Py_INCREF(res);
        return res;
    }
    /* Construct list of defaults */
    res = PyList_New(3);
    if (res == NULL) {
        return NULL;
    }
    PyList_SET_ITEM(res, 0, PyInt_FromLong(PyArray_BUFSIZE));
    PyList_SET_ITEM(res, 1, PyInt_FromLong(UFUNC_ERR_DEFAULT));
    PyList_SET_ITEM(res, 2, Py_None); Py_INCREF(Py_None);
    return res;
}

#if USE_USE_DEFAULTS==1
/*
 * This is a strategy to buy a little speed up and avoid the dictionary
 * look-up in the default case.  It should work in the presence of
 * threads.  If it is deemed too complicated or it doesn't actually work
 * it could be taken out.
 */
static int
ufunc_update_use_defaults(void)
{
    PyObject *errobj = NULL;
    int errmask, bufsize;
    int res;

    PyUFunc_NUM_NODEFAULTS += 1;
    res = PyUFunc_GetPyValues("test", &bufsize, &errmask, &errobj);
    PyUFunc_NUM_NODEFAULTS -= 1;
    if (res < 0) {
        Py_XDECREF(errobj);
        return -1;
    }
    if ((errmask != UFUNC_ERR_DEFAULT) || (bufsize != PyArray_BUFSIZE)
            || (PyTuple_GET_ITEM(errobj, 1) != Py_None)) {
        PyUFunc_NUM_NODEFAULTS += 1;
    }
    else if (PyUFunc_NUM_NODEFAULTS > 0) {
        PyUFunc_NUM_NODEFAULTS -= 1;
    }
    Py_XDECREF(errobj);
    return 0;
}
#endif

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *thedict;
    int res;
    PyObject *val;
    static char *msg = "Error object must be a list of length 3";

    if (!PyArg_ParseTuple(args, "O", &val)) {
        return NULL;
    }
    if (!PyList_CheckExact(val) || PyList_GET_SIZE(val) != 3) {
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }
    if (PyUFunc_PYVALS_NAME == NULL) {
        PyUFunc_PYVALS_NAME = PyUString_InternFromString(UFUNC_PYVALS_NAME);
    }
    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    res = PyDict_SetItem(thedict, PyUFunc_PYVALS_NAME, val);
    if (res < 0) {
        return NULL;
    }
#if USE_USE_DEFAULTS==1
    if (ufunc_update_use_defaults() < 0) {
        return NULL;
    }
#endif
    Py_INCREF(Py_None);
    return Py_None;
}



/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_ReplaceLoopBySignature(PyUFuncObject *func,
                               PyUFuncGenericFunction newfunc,
                               int *signature,
                               PyUFuncGenericFunction *oldfunc)
{
    int i, j;
    int res = -1;
    /* Find the location of the matching signature */
    for (i = 0; i < func->ntypes; i++) {
        for (j = 0; j < func->nargs; j++) {
            if (signature[j] != func->types[i*func->nargs+j]) {
                break;
            }
        }
        if (j < func->nargs) {
            continue;
        }
        if (oldfunc != NULL) {
            *oldfunc = func->functions[i];
        }
        func->functions[i] = newfunc;
        res = 0;
        break;
    }
    return res;
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndData(PyUFuncGenericFunction *func, void **data,
                        char *types, int ntypes,
                        int nin, int nout, int identity,
                        char *name, char *doc, int check_return)
{
    return PyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes,
        nin, nout, identity, name, doc, check_return, NULL);
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     char *name, char *doc,
                                     int check_return, const char *signature)
{
    PyUFuncObject *self;

    self = _pya_malloc(sizeof(PyUFuncObject));
    if (self == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)self, &PyUFunc_Type);

    self->nin = nin;
    self->nout = nout;
    self->nargs = nin+nout;
    self->identity = identity;

    self->functions = func;
    self->data = data;
    self->types = types;
    self->ntypes = ntypes;
    self->check_return = check_return;
    self->ptr = NULL;
    self->obj = NULL;
    self->userloops=NULL;

    if (name == NULL) {
        self->name = "?";
    }
    else {
        self->name = name;
    }
    if (doc == NULL) {
        self->doc = "NULL";
    }
    else {
        self->doc = doc;
    }

    /* generalized ufunc */
    self->core_enabled = 0;
    self->core_num_dim_ix = 0;
    self->core_num_dims = NULL;
    self->core_dim_ixs = NULL;
    self->core_offsets = NULL;
    self->core_signature = NULL;
    if (signature != NULL) {
        if (_parse_signature(self, signature) != 0) {
            return NULL;
        }
    }
    return (PyObject *)self;
}

/* Specify that the loop specified by the given index should use the array of
 * input and arrays as the data pointer to the loop.
 */
/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_SetUsesArraysAsData(void **data, size_t i)
{
    data[i] = (void*)PyUFunc_SetUsesArraysAsData;
    return 0;
}

/* Return 1 if the given data pointer for the loop specifies that it needs the
 * arrays as the data pointer.
 */
static int
_does_loop_use_arrays(void *data)
{
    return (data == PyUFunc_SetUsesArraysAsData);
}


/*
 * This is the first-part of the CObject structure.
 *
 * I don't think this will change, but if it should, then
 * this needs to be fixed.  The exposed C-API was insufficient
 * because I needed to replace the pointer and it wouldn't
 * let me with a destructor set (even though it works fine
 * with the destructor).
 */
typedef struct {
    PyObject_HEAD
    void *c_obj;
} _simple_cobj;

#define _SETCPTR(cobj, val) ((_simple_cobj *)(cobj))->c_obj = (val)

/* return 1 if arg1 > arg2, 0 if arg1 == arg2, and -1 if arg1 < arg2 */
static int
cmp_arg_types(int *arg1, int *arg2, int n)
{
    for (; n > 0; n--, arg1++, arg2++) {
        if (PyArray_EquivTypenums(*arg1, *arg2)) {
            continue;
        }
        if (PyArray_CanCastSafely(*arg1, *arg2)) {
            return -1;
        }
        return 1;
    }
    return 0;
}

/*
 * This frees the linked-list structure when the CObject
 * is destroyed (removed from the internal dictionary)
*/
static NPY_INLINE void
_free_loop1d_list(PyUFunc_Loop1d *data)
{
    while (data != NULL) {
        PyUFunc_Loop1d *next = data->next;
        _pya_free(data->arg_types);
        _pya_free(data);
        data = next;
    }
}

#if PY_VERSION_HEX >= 0x03000000
static void
_loop1d_list_free(PyObject *ptr)
{
    PyUFunc_Loop1d *data = (PyUFunc_Loop1d *)PyCapsule_GetPointer(ptr, NULL);
    _free_loop1d_list(data);
}
#else
static void
_loop1d_list_free(void *ptr)
{
    PyUFunc_Loop1d *data = (PyUFunc_Loop1d *)ptr;
    _free_loop1d_list(data);
}
#endif


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_RegisterLoopForType(PyUFuncObject *ufunc,
                            int usertype,
                            PyUFuncGenericFunction function,
                            int *arg_types,
                            void *data)
{
    PyArray_Descr *descr;
    PyUFunc_Loop1d *funcdata;
    PyObject *key, *cobj;
    int i;
    int *newtypes=NULL;

    descr=PyArray_DescrFromType(usertype);
    if ((usertype < PyArray_USERDEF) || (descr==NULL)) {
        PyErr_SetString(PyExc_TypeError, "unknown user-defined type");
        return -1;
    }
    Py_DECREF(descr);

    if (ufunc->userloops == NULL) {
        ufunc->userloops = PyDict_New();
    }
    key = PyInt_FromLong((long) usertype);
    if (key == NULL) {
        return -1;
    }
    funcdata = _pya_malloc(sizeof(PyUFunc_Loop1d));
    if (funcdata == NULL) {
        goto fail;
    }
    newtypes = _pya_malloc(sizeof(int)*ufunc->nargs);
    if (newtypes == NULL) {
        goto fail;
    }
    if (arg_types != NULL) {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = arg_types[i];
        }
    }
    else {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = usertype;
        }
    }

    funcdata->func = function;
    funcdata->arg_types = newtypes;
    funcdata->data = data;
    funcdata->next = NULL;

    /* Get entry for this user-defined type*/
    cobj = PyDict_GetItem(ufunc->userloops, key);
    /* If it's not there, then make one and return. */
    if (cobj == NULL) {
        cobj = NpyCapsule_FromVoidPtr((void *)funcdata, _loop1d_list_free);
        if (cobj == NULL) {
            goto fail;
        }
        PyDict_SetItem(ufunc->userloops, key, cobj);
        Py_DECREF(cobj);
        Py_DECREF(key);
        return 0;
    }
    else {
        PyUFunc_Loop1d *current, *prev = NULL;
        int cmp = 1;
        /*
         * There is already at least 1 loop. Place this one in
         * lexicographic order.  If the next one signature
         * is exactly like this one, then just replace.
         * Otherwise insert.
         */
        current = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(cobj);
        while (current != NULL) {
            cmp = cmp_arg_types(current->arg_types, newtypes, ufunc->nargs);
            if (cmp >= 0) {
                break;
            }
            prev = current;
            current = current->next;
        }
        if (cmp == 0) {
            /* just replace it with new function */
            current->func = function;
            current->data = data;
            _pya_free(newtypes);
            _pya_free(funcdata);
        }
        else {
            /*
             * insert it before the current one by hacking the internals
             * of cobject to replace the function pointer --- can't use
             * CObject API because destructor is set.
             */
            funcdata->next = current;
            if (prev == NULL) {
                /* place this at front */
                _SETCPTR(cobj, funcdata);
            }
            else {
                prev->next = funcdata;
            }
        }
    }
    Py_DECREF(key);
    return 0;

 fail:
    Py_DECREF(key);
    _pya_free(funcdata);
    _pya_free(newtypes);
    if (!PyErr_Occurred()) PyErr_NoMemory();
    return -1;
}

#undef _SETCPTR


static void
ufunc_dealloc(PyUFuncObject *self)
{
    if (self->core_num_dims) {
        _pya_free(self->core_num_dims);
    }
    if (self->core_dim_ixs) {
        _pya_free(self->core_dim_ixs);
    }
    if (self->core_offsets) {
        _pya_free(self->core_offsets);
    }
    if (self->core_signature) {
        _pya_free(self->core_signature);
    }
    if (self->ptr) {
        _pya_free(self->ptr);
    }
    Py_XDECREF(self->userloops);
    Py_XDECREF(self->obj);
    _pya_free(self);
}

static PyObject *
ufunc_repr(PyUFuncObject *self)
{
    char buf[100];

    sprintf(buf, "<ufunc '%.50s'>", self->name);
    return PyUString_FromString(buf);
}


/******************************************************************************
 ***                          UFUNC METHODS                                 ***
 *****************************************************************************/


/*
 * op.outer(a,b) is equivalent to op(a[:,NewAxis,NewAxis,etc.],b)
 * where a has b.ndim NewAxis terms appended.
 *
 * The result has dimensions a.ndim + b.ndim
 */
static PyObject *
ufunc_outer(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    int i;
    PyObject *ret;
    PyArrayObject *ap1 = NULL, *ap2 = NULL, *ap_new = NULL;
    PyObject *new_args, *tmp;
    PyObject *shape1, *shape2, *newshape;

    if (self->core_enabled) {
        PyErr_Format(PyExc_TypeError,
                     "method outer is not allowed in ufunc with non-trivial"\
                     " signature");
        return NULL;
    }

    if(self->nin != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "outer product only supported "\
                        "for binary functions");
        return NULL;
    }

    if (PySequence_Length(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "exactly two arguments expected");
        return NULL;
    }

    tmp = PySequence_GetItem(args, 0);
    if (tmp == NULL) {
        return NULL;
    }
    ap1 = (PyArrayObject *) PyArray_FromObject(tmp, PyArray_NOTYPE, 0, 0);
    Py_DECREF(tmp);
    if (ap1 == NULL) {
        return NULL;
    }
    tmp = PySequence_GetItem(args, 1);
    if (tmp == NULL) {
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromObject(tmp, PyArray_NOTYPE, 0, 0);
    Py_DECREF(tmp);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }
    /* Construct new shape tuple */
    shape1 = PyTuple_New(ap1->nd);
    if (shape1 == NULL) {
        goto fail;
    }
    for (i = 0; i < ap1->nd; i++) {
        PyTuple_SET_ITEM(shape1, i,
                PyLong_FromLongLong((longlong)ap1->dimensions[i]));
    }
    shape2 = PyTuple_New(ap2->nd);
    for (i = 0; i < ap2->nd; i++) {
        PyTuple_SET_ITEM(shape2, i, PyInt_FromLong((long) 1));
    }
    if (shape2 == NULL) {
        Py_DECREF(shape1);
        goto fail;
    }
    newshape = PyNumber_Add(shape1, shape2);
    Py_DECREF(shape1);
    Py_DECREF(shape2);
    if (newshape == NULL) {
        goto fail;
    }
    ap_new = (PyArrayObject *)PyArray_Reshape(ap1, newshape);
    Py_DECREF(newshape);
    if (ap_new == NULL) {
        goto fail;
    }
    new_args = Py_BuildValue("(OO)", ap_new, ap2);
    Py_DECREF(ap1);
    Py_DECREF(ap2);
    Py_DECREF(ap_new);
    ret = ufunc_generic_call(self, new_args, kwds);
    Py_DECREF(new_args);
    return ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap_new);
    return NULL;
}


static PyObject *
ufunc_reduce(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, UFUNC_REDUCE);
}

static PyObject *
ufunc_accumulate(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, UFUNC_ACCUMULATE);
}

static PyObject *
ufunc_reduceat(PyUFuncObject *self, PyObject *args, PyObject *kwds)
{
    return PyUFunc_GenericReduction(self, args, kwds, UFUNC_REDUCEAT);
}


static struct PyMethodDef ufunc_methods[] = {
    {"reduce",
        (PyCFunction)ufunc_reduce,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"accumulate",
        (PyCFunction)ufunc_accumulate,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"reduceat",
        (PyCFunction)ufunc_reduceat,
        METH_VARARGS | METH_KEYWORDS, NULL },
    {"outer",
        (PyCFunction)ufunc_outer,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"of", /* old generic call */
        (PyCFunction)ufunc_generic_call,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"nf", /* new generic call */
        (PyCFunction)ufunc_generic_call_iter,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};


/******************************************************************************
 ***                           UFUNC GETSET                                 ***
 *****************************************************************************/


/* construct the string y1,y2,...,yn */
static PyObject *
_makeargs(int num, char *ltr, int null_if_none)
{
    PyObject *str;
    int i;

    switch (num) {
    case 0:
        if (null_if_none) {
            return NULL;
        }
        return PyString_FromString("");
    case 1:
        return PyString_FromString(ltr);
    }
    str = PyString_FromFormat("%s1, %s2", ltr, ltr);
    for (i = 3; i <= num; ++i) {
        PyString_ConcatAndDel(&str, PyString_FromFormat(", %s%d", ltr, i));
    }
    return str;
}

static char
_typecharfromnum(int num) {
    PyArray_Descr *descr;
    char ret;

    descr = PyArray_DescrFromType(num);
    ret = descr->type;
    Py_DECREF(descr);
    return ret;
}

static PyObject *
ufunc_get_doc(PyUFuncObject *self)
{
    /*
     * Put docstring first or FindMethod finds it... could so some
     * introspection on name and nin + nout to automate the first part
     * of it the doc string shouldn't need the calling convention
     * construct name(x1, x2, ...,[ out1, out2, ...]) __doc__
     */
    PyObject *outargs, *inargs, *doc;
    outargs = _makeargs(self->nout, "out", 1);
    inargs = _makeargs(self->nin, "x", 0);
    if (outargs == NULL) {
        doc = PyUString_FromFormat("%s(%s)\n\n%s",
                                   self->name,
                                   PyString_AS_STRING(inargs),
                                   self->doc);
    }
    else {
        doc = PyUString_FromFormat("%s(%s[, %s])\n\n%s",
                                   self->name,
                                   PyString_AS_STRING(inargs),
                                   PyString_AS_STRING(outargs),
                                   self->doc);
        Py_DECREF(outargs);
    }
    Py_DECREF(inargs);
    return doc;
}

static PyObject *
ufunc_get_nin(PyUFuncObject *self)
{
    return PyInt_FromLong(self->nin);
}

static PyObject *
ufunc_get_nout(PyUFuncObject *self)
{
    return PyInt_FromLong(self->nout);
}

static PyObject *
ufunc_get_nargs(PyUFuncObject *self)
{
    return PyInt_FromLong(self->nargs);
}

static PyObject *
ufunc_get_ntypes(PyUFuncObject *self)
{
    return PyInt_FromLong(self->ntypes);
}

static PyObject *
ufunc_get_types(PyUFuncObject *self)
{
    /* return a list with types grouped input->output */
    PyObject *list;
    PyObject *str;
    int k, j, n, nt = self->ntypes;
    int ni = self->nin;
    int no = self->nout;
    char *t;
    list = PyList_New(nt);
    if (list == NULL) {
        return NULL;
    }
    t = _pya_malloc(no+ni+2);
    n = 0;
    for (k = 0; k < nt; k++) {
        for (j = 0; j<ni; j++) {
            t[j] = _typecharfromnum(self->types[n]);
            n++;
        }
        t[ni] = '-';
        t[ni+1] = '>';
        for (j = 0; j < no; j++) {
            t[ni + 2 + j] = _typecharfromnum(self->types[n]);
            n++;
        }
        str = PyUString_FromStringAndSize(t, no + ni + 2);
        PyList_SET_ITEM(list, k, str);
    }
    _pya_free(t);
    return list;
}

static PyObject *
ufunc_get_name(PyUFuncObject *self)
{
    return PyUString_FromString(self->name);
}

static PyObject *
ufunc_get_identity(PyUFuncObject *self)
{
    switch(self->identity) {
    case PyUFunc_One:
        return PyInt_FromLong(1);
    case PyUFunc_Zero:
        return PyInt_FromLong(0);
    }
    return Py_None;
}

static PyObject *
ufunc_get_signature(PyUFuncObject *self)
{
    if (!self->core_enabled) {
        Py_RETURN_NONE;
    }
    return PyUString_FromString(self->core_signature);
}

#undef _typecharfromnum

/*
 * Docstring is now set from python
 * static char *Ufunctype__doc__ = NULL;
 */
static PyGetSetDef ufunc_getset[] = {
    {"__doc__",
        (getter)ufunc_get_doc,
        NULL, NULL, NULL},
    {"nin",
        (getter)ufunc_get_nin,
        NULL, NULL, NULL},
    {"nout",
        (getter)ufunc_get_nout,
        NULL, NULL, NULL},
    {"nargs",
        (getter)ufunc_get_nargs,
        NULL, NULL, NULL},
    {"ntypes",
        (getter)ufunc_get_ntypes,
        NULL, NULL, NULL},
    {"types",
        (getter)ufunc_get_types,
        NULL, NULL, NULL},
    {"__name__",
        (getter)ufunc_get_name,
        NULL, NULL, NULL},
    {"identity",
        (getter)ufunc_get_identity,
        NULL, NULL, NULL},
    {"signature",
        (getter)ufunc_get_signature,
        NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};


/******************************************************************************
 ***                        UFUNC TYPE OBJECT                               ***
 *****************************************************************************/

NPY_NO_EXPORT PyTypeObject PyUFunc_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.ufunc",                              /* tp_name */
    sizeof(PyUFuncObject),                      /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)ufunc_dealloc,                  /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    (reprfunc)ufunc_repr,                       /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
#if USE_NEW_ITERATOR_GENFUNC 
    (ternaryfunc)ufunc_generic_call_iter,       /* tp_call */
#else
    (ternaryfunc)ufunc_generic_call,            /* tp_call */
#endif
    (reprfunc)ufunc_repr,                       /* tp_str */
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
    ufunc_methods,                              /* tp_methods */
    0,                                          /* tp_members */
    ufunc_getset,                               /* tp_getset */
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

/* End of code for ufunc objects */
