#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "npy_config.h"

#include "npy_pycompat.h"

#include "extobj.h"
#include "numpy/ufuncobject.h"

#include "ufunc_object.h"  /* for npy_um_str_pyvals_name */
#include "common.h"


static int
_error_handler(const char *name, int method, PyObject *pyfunc, char *errtype,
               int retstatus, int *first);


#define HANDLEIT(NAME, str) {if (retstatus & NPY_FPE_##NAME) {          \
            handle = errmask & UFUNC_MASK_##NAME;                       \
            if (handle &&                                               \
                _error_handler(name, handle >> UFUNC_SHIFT_##NAME,      \
                               pyfunc, str, retstatus, first) < 0)      \
                return -1;                                              \
        }}


static int
PyUFunc_handlefperr(
        const char *name, int errmask, PyObject *pyfunc, int retstatus,
        int *first)
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
_error_handler(const char *name, int method, PyObject *pyfunc, char *errtype,
               int retstatus, int *first)
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
        if (*first) {
            fprintf(stderr, "Warning: %s encountered in %s\n", errtype, name);
            *first = 0;
        }
        return 0;
    }

    NPY_ALLOW_C_API;
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
        if (first) {
            *first = 0;
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
    NPY_DISABLE_C_API;
    return 0;

fail:
    NPY_DISABLE_C_API;
    return -1;
}



static PyObject *
get_global_ext_obj(void)
{
    PyObject *thedict;
    PyObject *ref = NULL;

    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    ref = PyDict_GetItemWithError(thedict, npy_um_str_pyvals_name);

    return ref;
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
    PyObject *ref = get_global_ext_obj();
    PyObject *retval;

    if (ref == NULL && PyErr_Occurred()) {
        return -1;
    }

    /* default errobj case, skips dictionary lookup */
    if (ref == NULL) {
        if (errmask) {
            *errmask = UFUNC_ERR_DEFAULT;
        }
        if (pyfunc) {
            Py_INCREF(Py_None);
            *pyfunc = Py_None;
        }
        if (bufsize) {
            *bufsize = NPY_BUFSIZE;
        }
        return 0;
    }

    if (!PyList_Check(ref) || (PyList_GET_SIZE(ref)!=3)) {
        PyErr_Format(PyExc_TypeError,
                "%s must be a length 3 list.", UFUNC_PYVALS_NAME);
        return -1;
    }

    if (bufsize != NULL) {
        *bufsize = PyLong_AsLong(PyList_GET_ITEM(ref, 0));
        if (error_converting(*bufsize)) {
            return -1;
        }
        if ((*bufsize < NPY_MIN_BUFSIZE) ||
                (*bufsize > NPY_MAX_BUFSIZE) ||
                (*bufsize % 16 != 0)) {
            PyErr_Format(PyExc_ValueError,
                    "buffer size (%d) is not in range "
                    "(%"NPY_INTP_FMT" - %"NPY_INTP_FMT") or not a multiple of 16",
                    *bufsize, (npy_intp) NPY_MIN_BUFSIZE,
                    (npy_intp) NPY_MAX_BUFSIZE);
            return -1;
        }
    }

    if (errmask != NULL) {
        *errmask = PyLong_AsLong(PyList_GET_ITEM(ref, 1));
        if (*errmask < 0) {
            if (PyErr_Occurred()) {
                return -1;
            }
            PyErr_Format(PyExc_ValueError,
                         "invalid error mask (%d)",
                         *errmask);
            return -1;
        }
    }

    if (pyfunc != NULL) {
        *pyfunc = NULL;
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

        Py_INCREF(retval);
        *pyfunc = retval;
        if (*pyfunc == NULL) {
            return -1;
        }
    }
    return 0;
}

/*
 * Handler which uses the default `np.errstate` given that `fpe_errors` is
 * already set.  `fpe_errors` is typically the (nonzero) result of
 * `npy_get_floatstatus_barrier`.
 *
 * Returns -1 on failure (an error was raised) and 0 on success.
 */
NPY_NO_EXPORT int
PyUFunc_GiveFloatingpointErrors(const char *name, int fpe_errors)
{
    int bufsize, errmask;
    PyObject *pyfunc;

    if (_extract_pyvals(&bufsize, &errmask, &pyfunc) < 0) {
        Py_XDECREF(pyfunc);
        return -1;
    }
    int first = 1;
    if (PyUFunc_handlefperr(name, errmask, pyfunc, fpe_errors, &first)) {
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
    int first = 1;

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

    ret = PyUFunc_handlefperr(ufunc_name, errmask, pyfunc, fperr, &first);
    Py_XDECREF(pyfunc);

    return ret;
}


NPY_NO_EXPORT int
_get_bufsize_errmask(const char *ufunc_name, int *buffersize, int *errormask)
{
    /* Get the buffersize and errormask */
    PyObject *extobj = get_global_ext_obj();
    if (extobj == NULL && PyErr_Occurred()) {
        return -1;
    }
    if (_extract_pyvals(buffersize, errormask, NULL) < 0) {
        return -1;
    }

    return 0;
}
