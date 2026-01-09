#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_config.h"

#include "npy_argparse.h"

#include "conversion_utils.h"

#include "extobj.h"
#include "numpy/ufuncobject.h"

#include "common.h"


#define UFUNC_ERR_IGNORE 0
#define UFUNC_ERR_WARN   1
#define UFUNC_ERR_RAISE  2
#define UFUNC_ERR_CALL   3
#define UFUNC_ERR_PRINT  4
#define UFUNC_ERR_LOG    5

/* Integer mask */
#define UFUNC_MASK_DIVIDEBYZERO 0x07
#define UFUNC_MASK_OVERFLOW (0x07 << UFUNC_SHIFT_OVERFLOW)
#define UFUNC_MASK_UNDERFLOW (0x07 << UFUNC_SHIFT_UNDERFLOW)
#define UFUNC_MASK_INVALID (0x07 << UFUNC_SHIFT_INVALID)

#define UFUNC_SHIFT_DIVIDEBYZERO 0
#define UFUNC_SHIFT_OVERFLOW     3
#define UFUNC_SHIFT_UNDERFLOW    6
#define UFUNC_SHIFT_INVALID      9

/* Default user error mode (underflows are ignored, others warn) */
#define UFUNC_ERR_DEFAULT                               \
        (UFUNC_ERR_WARN << UFUNC_SHIFT_DIVIDEBYZERO) +  \
        (UFUNC_ERR_WARN << UFUNC_SHIFT_OVERFLOW) +      \
        (UFUNC_ERR_WARN << UFUNC_SHIFT_INVALID)


static int
_error_handler(const char *name, int method, PyObject *pyfunc, char *errtype,
               int retstatus);


#define HANDLEIT(NAME, str) {if (retstatus & NPY_FPE_##NAME) {          \
            handle = errmask & UFUNC_MASK_##NAME;                       \
            if (handle &&                                               \
                _error_handler(name, handle >> UFUNC_SHIFT_##NAME,      \
                               pyfunc, str, retstatus) < 0)      \
                return -1;                                              \
        }}


static int
PyUFunc_handlefperr(
        const char *name, int errmask, PyObject *pyfunc, int retstatus)
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


static void
extobj_capsule_destructor(PyObject *capsule)
{
    npy_extobj *extobj = PyCapsule_GetPointer(capsule, "numpy.ufunc.extobj");
    npy_extobj_clear(extobj);
    PyMem_FREE(extobj);
}


static PyObject *
make_extobj_capsule(npy_intp bufsize, int errmask, PyObject *pyfunc)
{
    npy_extobj *extobj = PyMem_Malloc(sizeof(npy_extobj));
    if (extobj == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    extobj->bufsize = bufsize;
    extobj->errmask = errmask;
    Py_XINCREF(pyfunc);
    extobj->pyfunc = pyfunc;

    PyObject *capsule = PyCapsule_New(
            extobj, "numpy.ufunc.extobj",
            (destructor)&extobj_capsule_destructor);
    if (capsule == NULL) {
        npy_extobj_clear(extobj);
        PyMem_Free(extobj);
        return NULL;
    }
    return capsule;
}


/*
 * Fetch the current error/extobj state and fill it into `npy_extobj *extobj`.
 * On success, the filled `extobj` must be cleared using `npy_extobj_clear`.
 * Returns -1 on failure and 0 on success.
 */
static int
fetch_curr_extobj_state(npy_extobj *extobj)
{
    PyObject *capsule;
    if (PyContextVar_Get(
            npy_static_pydata.npy_extobj_contextvar,
            npy_static_pydata.default_extobj_capsule, &capsule) < 0) {
        return -1;
    }
    npy_extobj *obj = PyCapsule_GetPointer(capsule, "numpy.ufunc.extobj");
    if (obj == NULL) {
        Py_DECREF(capsule);
        return -1;
    }

    extobj->bufsize = obj->bufsize;
    extobj->errmask = obj->errmask;
    extobj->pyfunc = obj->pyfunc;
    Py_INCREF(extobj->pyfunc);

    Py_DECREF(capsule);
    return 0;
}


NPY_NO_EXPORT int
init_extobj(void)
{
    npy_static_pydata.default_extobj_capsule = make_extobj_capsule(
            NPY_BUFSIZE, UFUNC_ERR_DEFAULT, Py_None);
    if (npy_static_pydata.default_extobj_capsule == NULL) {
        return -1;
    }
    npy_static_pydata.npy_extobj_contextvar = PyContextVar_New(
            "numpy.ufunc.extobj", npy_static_pydata.default_extobj_capsule);
    if (npy_static_pydata.npy_extobj_contextvar == NULL) {
        Py_CLEAR(npy_static_pydata.default_extobj_capsule);
        return -1;
    }
    return 0;
}


/*
 * Parsing helper for extobj_seterrobj to extract the modes
 * "ignore", "raise", etc.
 */
static int
errmodeconverter(PyObject *obj, int *mode)
{
    if (obj == Py_None) {
        return 1;
    }
    int i = 0;
    for (; i <= UFUNC_ERR_LOG; i++) {
        int eq = PyObject_RichCompareBool(
                obj, npy_interned_str.errmode_strings[i], Py_EQ);
        if (eq == -1) {
            return 0;
        }
        else if (eq) {
            break;
        }
    }
    if (i > UFUNC_ERR_LOG) {
        PyErr_Format(PyExc_ValueError, "invalid error mode %.100R", obj);
        return 0;
    }

    *mode = i;
    return 1;
 }


/*
 * This function is currently exposed as `umath._seterrobj()`, it is private
 * and returns a capsule representing the errstate.  This capsule is then
 * assigned to the `_extobj_contextvar` in Python.
 */
NPY_NO_EXPORT PyObject *
extobj_make_extobj(PyObject *NPY_UNUSED(mod),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    int all_mode = -1;
    int divide_mode = -1;
    int over_mode = -1;
    int under_mode = -1;
    int invalid_mode = -1;
    npy_intp bufsize = -1;
    PyObject *pyfunc = NULL;

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("_seterrobj", args, len_args, kwnames,
            "$all", &errmodeconverter, &all_mode,
            "$divide", &errmodeconverter, &divide_mode,
            "$over", &errmodeconverter, &over_mode,
            "$under", &errmodeconverter, &under_mode,
            "$invalid", &errmodeconverter, &invalid_mode,
            "$bufsize", &PyArray_IntpFromPyIntConverter, &bufsize,
            "$call", NULL, &pyfunc,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    /* Check that the new buffersize is valid (negative ones mean no change) */
    if (bufsize >= 0) {
        if (bufsize > 10e6) {
            PyErr_Format(PyExc_ValueError,
                    "Buffer size, %" NPY_INTP_FMT ", is too big",
                    bufsize);
            return NULL;
        }
        if (bufsize < 5) {
            PyErr_Format(PyExc_ValueError,
                    "Buffer size, %" NPY_INTP_FMT ", is too small",
                    bufsize);
            return NULL;
        }
        if (bufsize % 16 != 0) {
            PyErr_Format(PyExc_ValueError,
                    "Buffer size, %" NPY_INTP_FMT ", is not a multiple of 16",
                    bufsize);
            return NULL;
        }
    }
    /* Validate func (probably): None, callable, or callable write attribute */
    if (pyfunc != NULL && pyfunc != Py_None && !PyCallable_Check(pyfunc)) {
        PyObject *temp;
        temp = PyObject_GetAttrString(pyfunc, "write");
        if (temp == NULL || !PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError,
                            "python object must be callable or have "
                            "a callable write method");
            Py_XDECREF(temp);
            return NULL;
        }
        Py_DECREF(temp);
    }

    /* Fetch the current extobj, we will mutate it and then store it: */
    npy_extobj extobj;
    if (fetch_curr_extobj_state(&extobj) < 0) {
        return NULL;
    }

    if (all_mode != -1) {
        /* if all is passed use it for any mode not passed explicitly */
        divide_mode = divide_mode == -1 ? all_mode : divide_mode;
        over_mode = over_mode == -1 ? all_mode : over_mode;
        under_mode = under_mode == -1 ? all_mode : under_mode;
        invalid_mode = invalid_mode == -1 ? all_mode : invalid_mode;
    }
    if (divide_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_DIVIDEBYZERO;
        extobj.errmask |= divide_mode << UFUNC_SHIFT_DIVIDEBYZERO;
    }
    if (over_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_OVERFLOW;
        extobj.errmask |= over_mode << UFUNC_SHIFT_OVERFLOW;
    }
    if (under_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_UNDERFLOW;
        extobj.errmask |= under_mode << UFUNC_SHIFT_UNDERFLOW;
    }
    if (invalid_mode != -1) {
        extobj.errmask &= ~UFUNC_MASK_INVALID;
        extobj.errmask |= invalid_mode << UFUNC_SHIFT_INVALID;
    }

    if (bufsize > 0) {
        extobj.bufsize = bufsize;
    }
    if (pyfunc != NULL) {
        Py_INCREF(pyfunc);
        Py_SETREF(extobj.pyfunc, pyfunc);
    }
    PyObject *capsule = make_extobj_capsule(
            extobj.bufsize, extobj.errmask, extobj.pyfunc);
    npy_extobj_clear(&extobj);
    return capsule;
}


/*
 * For inspection purposes, allow fetching a dictionary representing the
 * current extobj/errobj.
 */
NPY_NO_EXPORT PyObject *
extobj_get_extobj_dict(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(noarg))
{
    PyObject *result = NULL, *bufsize_obj = NULL;
    npy_extobj extobj;
    int mode;

    if (fetch_curr_extobj_state(&extobj) < 0) {
        goto fail;
    }
    result = PyDict_New();
    if (result == NULL) {
        goto fail;
    }
    /* Set all error modes: */
    mode = (extobj.errmask & UFUNC_MASK_DIVIDEBYZERO) >> UFUNC_SHIFT_DIVIDEBYZERO;
    if (PyDict_SetItemString(result, "divide",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }
    mode = (extobj.errmask & UFUNC_MASK_OVERFLOW) >> UFUNC_SHIFT_OVERFLOW;
    if (PyDict_SetItemString(result, "over",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }
    mode = (extobj.errmask & UFUNC_MASK_UNDERFLOW) >> UFUNC_SHIFT_UNDERFLOW;
    if (PyDict_SetItemString(result, "under",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }
    mode = (extobj.errmask & UFUNC_MASK_INVALID) >> UFUNC_SHIFT_INVALID;
    if (PyDict_SetItemString(result, "invalid",
                             npy_interned_str.errmode_strings[mode]) < 0) {
        goto fail;
    }

    /* Set the callable: */
    if (PyDict_SetItemString(result, "call", extobj.pyfunc) < 0) {
        goto fail;
    }
    /* And the bufsize: */
    bufsize_obj = PyLong_FromSsize_t(extobj.bufsize);
    if (bufsize_obj == NULL) {
        goto fail;
    }
    if (PyDict_SetItemString(result, "bufsize", bufsize_obj) < 0) {
        goto fail;
    }
    Py_DECREF(bufsize_obj);
    npy_extobj_clear(&extobj);
    return result;

  fail:
    Py_XDECREF(result);
    Py_XDECREF(bufsize_obj);
    npy_extobj_clear(&extobj);
    return NULL;
}


/*
 * fpstatus is the ufunc_formatted hardware status
 * errmask is the handling mask specified by the user.
 * pyfunc is a Python callable or write method (logging).
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
_error_handler(const char *name, int method, PyObject *pyfunc, char *errtype,
               int retstatus)
{
    PyObject *ret, *args;
    char msg[100];

    NPY_ALLOW_C_API_DEF

    /* don't need C API for a simple ignore */
    if (method == UFUNC_ERR_IGNORE) {
        return 0;
    }

    /* don't need C API for a simple print */
    if (method == UFUNC_ERR_PRINT) {
        fprintf(stderr, "Warning: %s encountered in %s\n", errtype, name);
        return 0;
    }

    NPY_ALLOW_C_API;
    switch(method) {
    case UFUNC_ERR_WARN:
        PyOS_snprintf(msg, sizeof(msg), "%s encountered in %s", errtype, name);
        if (PyErr_WarnEx(PyExc_RuntimeWarning, msg, 1) < 0) {
            goto fail;
        }
        break;
    case UFUNC_ERR_RAISE:
        PyErr_Format(PyExc_FloatingPointError, "%s encountered in %s",
                errtype, name);
        goto fail;
    case UFUNC_ERR_CALL:
        if (pyfunc == Py_None) {
            PyErr_Format(PyExc_NameError,
                    "python callback specified for %s (in " \
                    " %s) but no function found.",
                    errtype, name);
            goto fail;
        }
        args = Py_BuildValue("NN", PyUnicode_FromString(errtype),
                PyLong_FromLong((long) retstatus));
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
    case UFUNC_ERR_LOG:
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
        break;
    }
    NPY_DISABLE_C_API;
    return 0;

fail:
    NPY_DISABLE_C_API;
    return -1;
}


/*
 * Extracts some values from the global pyvals tuple.
 * all destinations may be NULL, in which case they are not retrieved
 * ref - should hold the global tuple
 * name - is the name of the ufunc (ufuncobj->name)
 *
 * bufsize - receives the buffer size to use
 * errmask - receives the bitmask for error handling
 * pyfunc - receives the python object to call with the error,
 *          if an error handling method is 'call'
 */
static int
_extract_pyvals(int *bufsize, int *errmask, PyObject **pyfunc)
{
    npy_extobj extobj;
    if (fetch_curr_extobj_state(&extobj) < 0) {
        return -1;
    }

    if (bufsize != NULL) {
        *bufsize = extobj.bufsize;
    }

    if (errmask != NULL) {
        *errmask = extobj.errmask;
    }

    if (pyfunc != NULL) {
        *pyfunc = extobj.pyfunc;
        Py_INCREF(*pyfunc);
    }
    npy_extobj_clear(&extobj);
    return 0;
}

/*UFUNC_API
 * Signal a floating point error respecting the error signaling setting in
 * the NumPy errstate. Takes the name of the operation to use in the error
 * message and an integer flag that is one of NPY_FPE_DIVIDEBYZERO,
 * NPY_FPE_OVERFLOW, NPY_FPE_UNDERFLOW, NPY_FPE_INVALID to indicate
 * which errors to check for.
 *
 * Returns -1 on failure (an error was raised) and 0 on success.
 */
NPY_NO_EXPORT int
PyUFunc_GiveFloatingpointErrors(const char *name, int fpe_errors)
{
    int bufsize, errmask;
    PyObject *pyfunc = NULL;

    if (_extract_pyvals(&bufsize, &errmask, &pyfunc) < 0) {
        Py_XDECREF(pyfunc);
        return -1;
    }
    if (PyUFunc_handlefperr(name, errmask, pyfunc, fpe_errors)) {
        Py_XDECREF(pyfunc);
        return -1;
    }
    Py_XDECREF(pyfunc);
    return 0;
}


/*
 * check the floating point status
 *  - errmask: mask of status to check
 *  - extobj: ufunc pyvals object
 *            may be null, in which case the thread global one is fetched
 *  - ufunc_name: name of ufunc
 */
NPY_NO_EXPORT int
_check_ufunc_fperr(int errmask, const char *ufunc_name) {
    int fperr;
    PyObject *pyfunc = NULL;
    int ret;

    if (!errmask) {
        return 0;
    }
    fperr = npy_get_floatstatus_barrier((char*)ufunc_name);
    if (!fperr) {
        return 0;
    }

    /* Get error state parameters */
    if (_extract_pyvals(NULL, NULL, &pyfunc) < 0) {
        Py_XDECREF(pyfunc);
        return -1;
    }

    ret = PyUFunc_handlefperr(ufunc_name, errmask, pyfunc, fperr);
    Py_XDECREF(pyfunc);

    return ret;
}


NPY_NO_EXPORT int
_get_bufsize_errmask(int *buffersize, int *errormask)
{
    return _extract_pyvals(buffersize, errormask, NULL);
}
