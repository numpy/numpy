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
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"

#include "npy_config.h"

#define PY_ARRAY_UNIQUE_SYMBOL _npy_umathmodule_ARRAY_API
#define NO_IMPORT_ARRAY

#include "npy_pycompat.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/arrayscalars.h"
#include "lowlevel_strided_loops.h"
#include "ufunc_type_resolution.h"
#include "reduction.h"
#include "mem_overlap.h"

#include "ufunc_object.h"
#include "override.h"
#include "npy_import.h"
#include "extobj.h"
#include "common.h"

/********** PRINTF DEBUG TRACING **************/
#define NPY_UF_DBG_TRACING 0

#if NPY_UF_DBG_TRACING
#define NPY_UF_DBG_PRINT(s) {printf("%s", s);fflush(stdout);}
#define NPY_UF_DBG_PRINT1(s, p1) {printf((s), (p1));fflush(stdout);}
#define NPY_UF_DBG_PRINT2(s, p1, p2) {printf(s, p1, p2);fflush(stdout);}
#define NPY_UF_DBG_PRINT3(s, p1, p2, p3) {printf(s, p1, p2, p3);fflush(stdout);}
#else
#define NPY_UF_DBG_PRINT(s)
#define NPY_UF_DBG_PRINT1(s, p1)
#define NPY_UF_DBG_PRINT2(s, p1, p2)
#define NPY_UF_DBG_PRINT3(s, p1, p2, p3)
#endif
/**********************************************/

/* ---------------------------------------------------------------- */

static int
_does_loop_use_arrays(void *data);

static int
assign_reduce_identity_zero(PyArrayObject *result, void *data);

static int
assign_reduce_identity_minusone(PyArrayObject *result, void *data);

static int
assign_reduce_identity_one(PyArrayObject *result, void *data);


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_getfperr(void)
{
    /*
     * non-clearing get was only added in 1.9 so this function always cleared
     * keep it so just in case third party code relied on the clearing
     */
    return npy_clear_floatstatus();
}

#define HANDLEIT(NAME, str) {if (retstatus & NPY_FPE_##NAME) {          \
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
    /* clearing is done for backward compatibility */
    int retstatus = npy_clear_floatstatus();

    return PyUFunc_handlefperr(errmask, errobj, retstatus, first);
}


/* Checking the status flag clears it */
/*UFUNC_API*/
NPY_NO_EXPORT void
PyUFunc_clearfperr()
{
    npy_clear_floatstatus();
}

/*
 * This function analyzes the input arguments
 * and determines an appropriate __array_prepare__ function to call
 * for the outputs.
 * Assumes subok is already true if check_subok is false.
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
                    PyObject **output_prep, int nin, int nout,
                    int check_subok)
{
    Py_ssize_t nargs;
    int i;
    int np = 0;
    PyObject *with_prep[NPY_MAXARGS], *preps[NPY_MAXARGS];
    PyObject *obj, *prep = NULL;

    /*
     * If a 'subok' parameter is passed and isn't True, don't wrap
     * if check_subok is false it assumed subok in kwds keyword is True
     */
    if (check_subok && kwds != NULL &&
        (obj = PyDict_GetItem(kwds, npy_um_str_subok)) != NULL) {
        if (obj != Py_True) {
            for (i = 0; i < nout; i++) {
                output_prep[i] = NULL;
            }
            return;
        }
    }

    nargs = PyTuple_GET_SIZE(args);
    for (i = 0; i < nin; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        prep = PyObject_GetAttr(obj, npy_um_str_array_prepare);
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
                        NPY_PRIORITY);
            for (i = 1; i < np; ++i) {
                double priority = PyArray_GetPriority(with_prep[i],
                            NPY_PRIORITY);
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
                obj = PyDict_GetItem(kwds, npy_um_str_out);
            }
        }
        /* Output argument one may also be in a keyword argument */
        else if (i == 0 && kwds != NULL) {
            obj = PyDict_GetItem(kwds, npy_um_str_out);
        }

        if (obj != Py_None && obj != NULL) {
            if (PyArray_CheckExact(obj)) {
                /* None signals to not call any wrapping */
                output_prep[i] = Py_None;
            }
            else {
                PyObject *oprep = PyObject_GetAttr(obj,
                                                   npy_um_str_array_prepare);
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


/*UFUNC_API
 *
 * On return, if errobj is populated with a non-NULL value, the caller
 * owns a new reference to errobj.
 */
NPY_NO_EXPORT int
PyUFunc_GetPyValues(char *name, int *bufsize, int *errmask, PyObject **errobj)
{
    PyObject *ref = get_global_ext_obj();

    return _extract_pyvals(ref, name, bufsize, errmask, errobj);
}

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
 * and core_signature in PyUFuncObject "ufunc".  Returns 0 unless an
 * error occurred.
 */
static int
_parse_signature(PyUFuncObject *ufunc, const char *signature)
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
    ufunc->core_signature = PyArray_malloc(sizeof(char) * (len+1));
    if (ufunc->core_signature) {
        strcpy(ufunc->core_signature, signature);
    }
    /* Allocate sufficient memory to store pointers to all dimension names */
    var_names = PyArray_malloc(sizeof(char const*) * len);
    if (var_names == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    ufunc->core_enabled = 1;
    ufunc->core_num_dim_ix = 0;
    ufunc->core_num_dims = PyArray_malloc(sizeof(int) * ufunc->nargs);
    ufunc->core_dim_ixs = PyArray_malloc(sizeof(int) * len); /* shrink this later */
    ufunc->core_offsets = PyArray_malloc(sizeof(int) * ufunc->nargs);
    if (ufunc->core_num_dims == NULL || ufunc->core_dim_ixs == NULL
        || ufunc->core_offsets == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    i = _next_non_white_space(signature, 0);
    while (signature[i] != '\0') {
        /* loop over input/output arguments */
        if (cur_arg == ufunc->nin) {
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
            while (j < ufunc->core_num_dim_ix) {
                if (_is_same_name(signature+i, var_names[j])) {
                    break;
                }
                j++;
            }
            if (j >= ufunc->core_num_dim_ix) {
                var_names[j] = signature+i;
                ufunc->core_num_dim_ix++;
            }
            ufunc->core_dim_ixs[cur_core_dim] = j;
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
        ufunc->core_num_dims[cur_arg] = nd;
        ufunc->core_offsets[cur_arg] = cur_core_dim-nd;
        cur_arg++;
        nd = 0;

        i = _next_non_white_space(signature, i + 1);
        if (cur_arg != ufunc->nin && cur_arg != ufunc->nargs) {
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
    if (cur_arg != ufunc->nargs) {
        parse_error = "incomplete signature: not all arguments found";
        goto fail;
    }
    ufunc->core_dim_ixs = PyArray_realloc(ufunc->core_dim_ixs,
            sizeof(int)*cur_core_dim);
    /* check for trivial core-signature, e.g. "(),()->()" */
    if (cur_core_dim == 0) {
        ufunc->core_enabled = 0;
    }
    PyArray_free((void*)var_names);
    return 0;

fail:
    PyArray_free((void*)var_names);
    if (parse_error) {
        PyErr_Format(PyExc_ValueError,
                     "%s at position %d in \"%s\"",
                     parse_error, i, signature);
    }
    return -1;
}

/*
 * Checks if 'obj' is a valid output array for a ufunc, i.e. it is
 * either None or a writeable array, increments its reference count
 * and stores a pointer to it in 'store'. Returns 0 on success, sets
 * an exception and returns -1 on failure.
 */
static int
_set_out_array(PyObject *obj, PyArrayObject **store)
{
    if (obj == Py_None) {
        /* Translate None to NULL */
        return 0;
    }
    if PyArray_Check(obj) {
        /* If it's an array, store it */
        if (PyArray_FailUnlessWriteable((PyArrayObject *)obj,
                                        "output array") < 0) {
            return -1;
        }
        Py_INCREF(obj);
        *store = (PyArrayObject *)obj;

        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "return arrays must be of ArrayType");

    return -1;
}

/********* GENERIC UFUNC USING ITERATOR *********/

/*
 * Produce a name for the ufunc, if one is not already set
 * This is used in the PyUFunc_handlefperr machinery, and in error messages
 */
NPY_NO_EXPORT const char*
ufunc_get_name_cstr(PyUFuncObject *ufunc) {
    return ufunc->name ? ufunc->name : "<unnamed ufunc>";
}

/*
 * Parses the positional and keyword arguments for a generic ufunc call.
 *
 * Note that if an error is returned, the caller must free the
 * non-zero references in out_op.  This
 * function does not do its own clean-up.
 */
static int
get_ufunc_arguments(PyUFuncObject *ufunc,
                    PyObject *args, PyObject *kwds,
                    PyArrayObject **out_op,
                    NPY_ORDER *out_order,
                    NPY_CASTING *out_casting,
                    PyObject **out_extobj,
                    PyObject **out_typetup,
                    int *out_subok,
                    PyArrayObject **out_wheremask)
{
    int i, nargs;
    int nin = ufunc->nin;
    int nout = ufunc->nout;
    PyObject *obj, *context;
    PyObject *str_key_obj = NULL;
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);
    int type_num;

    int any_flexible = 0, any_object = 0, any_flexible_userloops = 0;
    int has_sig = 0;

    *out_extobj = NULL;
    *out_typetup = NULL;
    if (out_wheremask != NULL) {
        *out_wheremask = NULL;
    }

    /* Check number of arguments */
    nargs = PyTuple_Size(args);
    if ((nargs < nin) || (nargs > ufunc->nargs)) {
        PyErr_SetString(PyExc_ValueError, "invalid number of arguments");
        return -1;
    }

    /* Get input arguments */
    for (i = 0; i < nin; ++i) {
        obj = PyTuple_GET_ITEM(args, i);

        if (PyArray_Check(obj)) {
            PyArrayObject *obj_a = (PyArrayObject *)obj;
            out_op[i] = (PyArrayObject *)PyArray_FromArray(obj_a, NULL, 0);
        }
        else {
            if (!PyArray_IsScalar(obj, Generic)) {
                /*
                 * TODO: There should be a comment here explaining what
                 *       context does.
                 */
                context = Py_BuildValue("OOi", ufunc, args, i);
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
        }

        if (out_op[i] == NULL) {
            return -1;
        }

        type_num = PyArray_DESCR(out_op[i])->type_num;
        if (!any_flexible &&
                PyTypeNum_ISFLEXIBLE(type_num)) {
            any_flexible = 1;
        }
        if (!any_object &&
                PyTypeNum_ISOBJECT(type_num)) {
            any_object = 1;
        }

        /*
         * If any operand is a flexible dtype, check to see if any
         * struct dtype ufuncs are registered. A ufunc has been registered
         * for a struct dtype if ufunc's arg_dtypes array is not NULL.
         */
        if (PyTypeNum_ISFLEXIBLE(type_num) &&
                    !any_flexible_userloops &&
                    ufunc->userloops != NULL) {
                PyUFunc_Loop1d *funcdata;
                PyObject *key, *obj;
                key = PyInt_FromLong(type_num);
            if (key == NULL) {
                continue;
            }
            obj = PyDict_GetItem(ufunc->userloops, key);
            Py_DECREF(key);
            if (obj == NULL) {
                continue;
            }
            funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
            while (funcdata != NULL) {
                if (funcdata->arg_dtypes != NULL) {
                    any_flexible_userloops = 1;
                    break;
                }
                funcdata = funcdata->next;
            }
        }
    }

    if (any_flexible && !any_flexible_userloops && !any_object) {
        /* Traditionally, we return -2 here (meaning "NotImplemented") anytime
         * we hit the above condition.
         *
         * This condition basically means "we are doomed", b/c the "flexible"
         * dtypes -- strings and void -- cannot have their own ufunc loops
         * registered (except via the special "flexible userloops" mechanism),
         * and they can't be cast to anything except object (and we only cast
         * to object if any_object is true). So really we should do nothing
         * here and continue and let the proper error be raised. But, we can't
         * quite yet, b/c of backcompat.
         *
         * Most of the time, this NotImplemented either got returned directly
         * to the user (who can't do anything useful with it), or got passed
         * back out of a special function like __mul__. And fortunately, for
         * almost all special functions, the end result of this was a
         * TypeError. Which is also what we get if we just continue without
         * this special case, so this special case is unnecessary.
         *
         * The only thing that actually depended on the NotImplemented is
         * array_richcompare, which did two things with it. First, it needed
         * to see this NotImplemented in order to implement the special-case
         * comparisons for
         *
         *    string < <= == != >= > string
         *    void == != void
         *
         * Now it checks for those cases first, before trying to call the
         * ufunc, so that's no problem. What it doesn't handle, though, is
         * cases like
         *
         *    float < string
         *
         * or
         *
         *    float == void
         *
         * For those, it just let the NotImplemented bubble out, and accepted
         * Python's default handling. And unfortunately, for comparisons,
         * Python's default is *not* to raise an error. Instead, it returns
         * something that depends on the operator:
         *
         *    ==         return False
         *    !=         return True
         *    < <= >= >  Python 2: use "fallback" (= weird and broken) ordering
         *               Python 3: raise TypeError (hallelujah)
         *
         * In most cases this is straightforwardly broken, because comparison
         * of two arrays should always return an array, and here we end up
         * returning a scalar. However, there is an exception: if we are
         * comparing two scalars for equality, then it actually is correct to
         * return a scalar bool instead of raising an error. If we just
         * removed this special check entirely, then "np.float64(1) == 'foo'"
         * would raise an error instead of returning False, which is genuinely
         * wrong.
         *
         * The proper end goal here is:
         *   1) == and != should be implemented in a proper vectorized way for
         *      all types. The short-term hack for this is just to add a
         *      special case to PyUFunc_DefaultLegacyInnerLoopSelector where
         *      if it can't find a comparison loop for the given types, and
         *      the ufunc is np.equal or np.not_equal, then it returns a loop
         *      that just fills the output array with False (resp. True). Then
         *      array_richcompare could trust that whenever its special cases
         *      don't apply, simply calling the ufunc will do the right thing,
         *      even without this special check.
         *   2) < <= >= > should raise an error if no comparison function can
         *      be found. array_richcompare already handles all string <>
         *      string cases, and void dtypes don't have ordering, so again
         *      this would mean that array_richcompare could simply call the
         *      ufunc and it would do the right thing (i.e., raise an error),
         *      again without needing this special check.
         *
         * So this means that for the transition period, our goal is:
         *   == and != on scalars should simply return NotImplemented like
         *     they always did, since everything ends up working out correctly
         *     in this case only
         *   == and != on arrays should issue a FutureWarning and then return
         *     NotImplemented
         *   < <= >= > on all flexible dtypes on py2 should raise a
         *     DeprecationWarning, and then return NotImplemented. On py3 we
         *     skip the warning, though, b/c it would just be immediately be
         *     followed by an exception anyway.
         *
         * And for all other operations, we let things continue as normal.
         */
        /* strcmp() is a hack but I think we can get away with it for this
         * temporary measure.
         */
        if (!strcmp(ufunc_name, "equal") ||
                !strcmp(ufunc_name, "not_equal")) {
            /* Warn on non-scalar, return NotImplemented regardless */
            assert(nin == 2);
            if (PyArray_NDIM(out_op[0]) != 0 ||
                    PyArray_NDIM(out_op[1]) != 0) {
                if (DEPRECATE_FUTUREWARNING(
                        "elementwise comparison failed; returning scalar "
                        "instead, but in the future will perform elementwise "
                        "comparison") < 0) {
                    return -1;
                }
            }
            return -2;
        }
        else if (!strcmp(ufunc_name, "less") ||
                 !strcmp(ufunc_name, "less_equal") ||
                 !strcmp(ufunc_name, "greater") ||
                 !strcmp(ufunc_name, "greater_equal")) {
#if !defined(NPY_PY3K)
            if (DEPRECATE("unorderable dtypes; returning scalar but in "
                          "the future this will be an error") < 0) {
                return -1;
            }
#endif
            return -2;
        }
    }

    /* Get positional output arguments */
    for (i = nin; i < nargs; ++i) {
        obj = PyTuple_GET_ITEM(args, i);
        if (_set_out_array(obj, out_op + i) < 0) {
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

#if defined(NPY_PY3K)
            Py_XDECREF(str_key_obj);
            str_key_obj = PyUnicode_AsASCIIString(key);
            if (str_key_obj != NULL) {
                key = str_key_obj;
            }
#endif

            if (PyBytes_AsStringAndSize(key, &str, &length) < 0) {
                PyErr_Clear();
                PyErr_SetString(PyExc_TypeError, "invalid keyword argument");
                goto fail;
            }

            switch (str[0]) {
                case 'c':
                    /* Provides a policy for allowed casting */
                    if (strcmp(str, "casting") == 0) {
                        if (!PyArray_CastingConverter(value, out_casting)) {
                            goto fail;
                        }
                        bad_arg = 0;
                    }
                    break;
                case 'd':
                    /* Another way to specify 'sig' */
                    if (strcmp(str, "dtype") == 0) {
                        /* Allow this parameter to be None */
                        PyArray_Descr *dtype;
                        if (!PyArray_DescrConverter2(value, &dtype)) {
                            goto fail;
                        }
                        if (dtype != NULL) {
                            if (*out_typetup != NULL) {
                                PyErr_SetString(PyExc_RuntimeError,
                                    "cannot specify both 'sig' and 'dtype'");
                                goto fail;
                            }
                            *out_typetup = Py_BuildValue("(N)", dtype);
                        }
                        bad_arg = 0;
                    }
                    break;
                case 'e':
                    /*
                     * Overrides the global parameters buffer size,
                     * error mask, and error object
                     */
                    if (strcmp(str, "extobj") == 0) {
                        *out_extobj = value;
                        bad_arg = 0;
                    }
                    break;
                case 'o':
                    /*
                     * Output arrays may be specified as a keyword argument,
                     * either as a single array or None for single output
                     * ufuncs, or as a tuple of arrays and Nones.
                     */
                    if (strcmp(str, "out") == 0) {
                        if (nargs > nin) {
                            PyErr_SetString(PyExc_ValueError,
                                    "cannot specify 'out' as both a "
                                    "positional and keyword argument");
                            goto fail;
                        }
                        if (PyTuple_CheckExact(value)) {
                            if (PyTuple_GET_SIZE(value) != nout) {
                                PyErr_SetString(PyExc_ValueError,
                                        "The 'out' tuple must have exactly "
                                        "one entry per ufunc output");
                                goto fail;
                            }
                            /* 'out' must be a tuple of arrays and Nones */
                            for(i = 0; i < nout; ++i) {
                                PyObject *val = PyTuple_GET_ITEM(value, i);
                                if (_set_out_array(val, out_op+nin+i) < 0) {
                                    goto fail;
                                }
                            }
                        }
                        else if (nout == 1) {
                            /* Can be an array if it only has one output */
                            if (_set_out_array(value, out_op + nin) < 0) {
                                goto fail;
                            }
                        }
                        else {
                            /*
                             * If the deprecated behavior is ever removed,
                             * keep only the else branch of this if-else
                             */
                            if (PyArray_Check(value) || value == Py_None) {
                                if (DEPRECATE("passing a single array to the "
                                              "'out' keyword argument of a "
                                              "ufunc with\n"
                                              "more than one output will "
                                              "result in an error in the "
                                              "future") < 0) {
                                    /* The future error message */
                                    PyErr_SetString(PyExc_TypeError,
                                        "'out' must be a tuple of arrays");
                                    goto fail;
                                }
                                if (_set_out_array(value, out_op+nin) < 0) {
                                    goto fail;
                                }
                            }
                            else {
                                PyErr_SetString(PyExc_TypeError,
                                    nout > 1 ? "'out' must be a tuple "
                                               "of arrays" :
                                               "'out' must be an array or a "
                                               "tuple of a single array");
                                goto fail;
                            }
                        }
                        bad_arg = 0;
                    }
                    /* Allows the default output layout to be overridden */
                    else if (strcmp(str, "order") == 0) {
                        if (!PyArray_OrderConverter(value, out_order)) {
                            goto fail;
                        }
                        bad_arg = 0;
                    }
                    break;
                case 's':
                    /* Allows a specific function inner loop to be selected */
                    if (strcmp(str, "sig") == 0 ||
                            strcmp(str, "signature") == 0) {
                        if (has_sig == 1) {
                            PyErr_SetString(PyExc_ValueError,
                                "cannot specify both 'sig' and 'signature'");
                            goto fail;
                        }
                        if (*out_typetup != NULL) {
                            PyErr_SetString(PyExc_RuntimeError,
                                    "cannot specify both 'sig' and 'dtype'");
                            goto fail;
                        }
                        *out_typetup = value;
                        Py_INCREF(value);
                        bad_arg = 0;
                        has_sig = 1;
                    }
                    else if (strcmp(str, "subok") == 0) {
                        if (!PyBool_Check(value)) {
                            PyErr_SetString(PyExc_TypeError,
                                        "'subok' must be a boolean");
                            goto fail;
                        }
                        *out_subok = (value == Py_True);
                        bad_arg = 0;
                    }
                    break;
                case 'w':
                    /*
                     * Provides a boolean array 'where=' mask if
                     * out_wheremask is supplied.
                     */
                    if (out_wheremask != NULL && strcmp(str, "where") == 0) {
                        PyArray_Descr *dtype;
                        dtype = PyArray_DescrFromType(NPY_BOOL);
                        if (dtype == NULL) {
                            goto fail;
                        }
                        if (value == Py_True) {
                            /*
                             * Optimization: where=True is the same as no
                             * where argument. This lets us document it as a
                             * default argument
                             */
                            bad_arg = 0;
                            break;
                        }
                        *out_wheremask = (PyArrayObject *)PyArray_FromAny(
                                                            value, dtype,
                                                            0, 0, 0, NULL);
                        if (*out_wheremask == NULL) {
                            goto fail;
                        }
                        bad_arg = 0;
                    }
                    break;
            }

            if (bad_arg) {
                char *format = "'%s' is an invalid keyword to ufunc '%s'";
                PyErr_Format(PyExc_TypeError, format, str, ufunc_name);
                goto fail;
            }
        }
    }
    Py_XDECREF(str_key_obj);

    return 0;

fail:
    Py_XDECREF(str_key_obj);
    Py_XDECREF(*out_extobj);
    *out_extobj = NULL;
    Py_XDECREF(*out_typetup);
    *out_typetup = NULL;
    if (out_wheremask != NULL) {
        Py_XDECREF(*out_wheremask);
        *out_wheremask = NULL;
    }
    return -1;
}

/*
 * This checks whether a trivial loop is ok,
 * making copies of scalar and one dimensional operands if that will
 * help.
 *
 * Returns 1 if a trivial loop is ok, 0 if it is not, and
 * -1 if there is an error.
 */
static int
check_for_trivial_loop(PyUFuncObject *ufunc,
                        PyArrayObject **op,
                        PyArray_Descr **dtype,
                        npy_intp buffersize)
{
    npy_intp i, nin = ufunc->nin, nop = nin + ufunc->nout;

    for (i = 0; i < nop; ++i) {
        /*
         * If the dtype doesn't match, or the array isn't aligned,
         * indicate that the trivial loop can't be done.
         */
        if (op[i] != NULL &&
                (!PyArray_ISALIGNED(op[i]) ||
                !PyArray_EquivTypes(dtype[i], PyArray_DESCR(op[i]))
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
                Py_INCREF(dtype[i]);
                tmp = (PyArrayObject *)
                            PyArray_CastToType(op[i], dtype[i], 0);
                if (tmp == NULL) {
                    return -1;
                }
                Py_DECREF(op[i]);
                op[i] = tmp;
            }
            else {
                return 0;
            }
        }
    }

    return 1;
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
    NPY_UF_DBG_PRINT1("two operand loop count %d\n", (int)count[0]);

    if (!needs_api) {
        NPY_BEGIN_THREADS_THRESHOLDED(count[0]);
    }

    innerloop(data, count, stride, innerloopdata);

    NPY_END_THREADS;
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
    NPY_UF_DBG_PRINT1("three operand loop count %d\n", (int)count[0]);

    if (!needs_api) {
        NPY_BEGIN_THREADS_THRESHOLDED(count[0]);
    }

    innerloop(data, count, stride, innerloopdata);

    NPY_END_THREADS;
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
prepare_ufunc_output(PyUFuncObject *ufunc,
                    PyArrayObject **op,
                    PyObject *arr_prep,
                    PyObject *arr_prep_args,
                    int i)
{
    if (arr_prep != NULL && arr_prep != Py_None) {
        PyObject *res;
        PyArrayObject *arr;

        res = PyObject_CallFunction(arr_prep, "O(OOi)",
                    *op, ufunc, arr_prep_args, i);
        if ((res == NULL) || (res == Py_None) || !PyArray_Check(res)) {
            if (!PyErr_Occurred()){
                PyErr_SetString(PyExc_TypeError,
                        "__array_prepare__ must return an "
                        "ndarray or subclass thereof");
            }
            Py_XDECREF(res);
            return -1;
        }
        arr = (PyArrayObject *)res;

        /* If the same object was returned, nothing to do */
        if (arr == *op) {
            Py_DECREF(arr);
        }
        /* If the result doesn't match, throw an error */
        else if (PyArray_NDIM(arr) != PyArray_NDIM(*op) ||
                !PyArray_CompareLists(PyArray_DIMS(arr),
                                      PyArray_DIMS(*op),
                                      PyArray_NDIM(arr)) ||
                !PyArray_CompareLists(PyArray_STRIDES(arr),
                                      PyArray_STRIDES(*op),
                                      PyArray_NDIM(arr)) ||
                !PyArray_EquivTypes(PyArray_DESCR(arr),
                                    PyArray_DESCR(*op))) {
            PyErr_SetString(PyExc_TypeError,
                    "__array_prepare__ must return an "
                    "ndarray or subclass thereof which is "
                    "otherwise identical to its input");
            Py_DECREF(arr);
            return -1;
        }
        /* Replace the op value */
        else {
            Py_DECREF(*op);
            *op = arr;
        }
    }

    return 0;
}

static int
iterator_loop(PyUFuncObject *ufunc,
                    PyArrayObject **op,
                    PyArray_Descr **dtype,
                    NPY_ORDER order,
                    npy_intp buffersize,
                    PyObject **arr_prep,
                    PyObject *arr_prep_args,
                    PyUFuncGenericFunction innerloop,
                    void *innerloopdata)
{
    npy_intp i, iop, nin = ufunc->nin, nout = ufunc->nout;
    npy_intp nop = nin + nout;
    npy_uint32 op_flags[NPY_MAXARGS];
    NpyIter *iter;
    char *baseptrs[NPY_MAXARGS];

    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *stride;
    npy_intp *count_ptr;

    PyArrayObject **op_it;
    npy_uint32 iter_flags;

    NPY_BEGIN_THREADS_DEF;

    /* Set up the flags */
    for (i = 0; i < nin; ++i) {
        op_flags[i] = NPY_ITER_READONLY |
                      NPY_ITER_ALIGNED |
                      NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
        /*
         * If READWRITE flag has been set for this operand,
         * then clear default READONLY flag
         */
        op_flags[i] |= ufunc->op_flags[i];
        if (op_flags[i] & (NPY_ITER_READWRITE | NPY_ITER_WRITEONLY)) {
            op_flags[i] &= ~NPY_ITER_READONLY;
        }
    }
    for (i = nin; i < nop; ++i) {
        op_flags[i] = NPY_ITER_WRITEONLY |
                      NPY_ITER_ALIGNED |
                      NPY_ITER_ALLOCATE |
                      NPY_ITER_NO_BROADCAST |
                      NPY_ITER_NO_SUBTYPE |
                      NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
    }

    iter_flags = ufunc->iter_flags |
                 NPY_ITER_EXTERNAL_LOOP |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_ZEROSIZE_OK |
                 NPY_ITER_BUFFERED |
                 NPY_ITER_GROWINNER |
                 NPY_ITER_DELAY_BUFALLOC |
                 NPY_ITER_COPY_IF_OVERLAP;

    /* Call the __array_prepare__ functions for already existing output arrays.
     * Do this before creating the iterator, as the iterator may UPDATEIFCOPY
     * some of them.
     */
    for (i = 0; i < nout; ++i) {
        if (op[nin+i] == NULL) {
            continue;
        }
        if (prepare_ufunc_output(ufunc, &op[nin+i],
                            arr_prep[i], arr_prep_args, i) < 0) {
            return -1;
        }
    }

    /*
     * Allocate the iterator.  Because the types of the inputs
     * were already checked, we use the casting rule 'unsafe' which
     * is faster to calculate.
     */
    iter = NpyIter_AdvancedNew(nop, op,
                        iter_flags,
                        order, NPY_UNSAFE_CASTING,
                        op_flags, dtype,
                        -1, NULL, NULL, buffersize);
    if (iter == NULL) {
        return -1;
    }

    /* Copy any allocated outputs */
    op_it = NpyIter_GetOperandArray(iter);
    for (i = 0; i < nout; ++i) {
        if (op[nin+i] == NULL) {
            op[nin+i] = op_it[nin+i];
            Py_INCREF(op[nin+i]);

            /* Call the __array_prepare__ functions for the new array */
            if (prepare_ufunc_output(ufunc, &op[nin+i],
                                     arr_prep[i], arr_prep_args, i) < 0) {
                for(iop = 0; iop < nin+i; ++iop) {
                    if (op_it[iop] != op[iop]) {
                        /* ignore errrors */
                        PyArray_ResolveWritebackIfCopy(op_it[iop]);
                    }
                }
                NpyIter_Deallocate(iter);
                return -1;
            }

            /*
             * In case __array_prepare__ returned a different array, put the
             * results directly there, ignoring the array allocated by the
             * iterator.
             *
             * Here, we assume the user-provided __array_prepare__ behaves
             * sensibly and doesn't return an array overlapping in memory
             * with other operands --- the op[nin+i] array passed to it is newly
             * allocated and doesn't have any overlap.
             */
            baseptrs[nin+i] = PyArray_BYTES(op[nin+i]);
        }
        else {
            baseptrs[nin+i] = PyArray_BYTES(op_it[nin+i]);
        }
    }

    /* Only do the loop if the iteration size is non-zero */
    if (NpyIter_GetIterSize(iter) != 0) {
        /* Reset the iterator with the base pointers from possible __array_prepare__ */
        for (i = 0; i < nin; ++i) {
            baseptrs[i] = PyArray_BYTES(op_it[i]);
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

        NPY_BEGIN_THREADS_NDITER(iter);

        /* Execute the loop */
        do {
            NPY_UF_DBG_PRINT1("iterator loop count %d\n", (int)*count_ptr);
            innerloop(dataptr, count_ptr, stride, innerloopdata);
        } while (iternext(iter));

        NPY_END_THREADS;
    }
    for(iop = 0; iop < nop; ++iop) {
        if (op_it[iop] != op[iop]) {
            PyArray_ResolveWritebackIfCopy(op_it[iop]);
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
execute_legacy_ufunc_loop(PyUFuncObject *ufunc,
                    int trivial_loop_ok,
                    PyArrayObject **op,
                    PyArray_Descr **dtypes,
                    NPY_ORDER order,
                    npy_intp buffersize,
                    PyObject **arr_prep,
                    PyObject *arr_prep_args)
{
    npy_intp nin = ufunc->nin, nout = ufunc->nout;
    PyUFuncGenericFunction innerloop;
    void *innerloopdata;
    int needs_api = 0;

    if (ufunc->legacy_inner_loop_selector(ufunc, dtypes,
                    &innerloop, &innerloopdata, &needs_api) < 0) {
        return -1;
    }
    /* If the loop wants the arrays, provide them. */
    if (_does_loop_use_arrays(innerloopdata)) {
        innerloopdata = (void*)op;
    }

    /* First check for the trivial cases that don't need an iterator */
    if (trivial_loop_ok) {
        if (nin == 1 && nout == 1) {
            if (op[1] == NULL &&
                        (order == NPY_ANYORDER || order == NPY_KEEPORDER) &&
                        PyArray_TRIVIALLY_ITERABLE(op[0])) {
                Py_INCREF(dtypes[1]);
                op[1] = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                             dtypes[1],
                             PyArray_NDIM(op[0]),
                             PyArray_DIMS(op[0]),
                             NULL, NULL,
                             PyArray_ISFORTRAN(op[0]) ?
                                            NPY_ARRAY_F_CONTIGUOUS : 0,
                             NULL);
                if (op[1] == NULL) {
                    return -1;
                }

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(ufunc, &op[1],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINT("trivial 1 input with allocated output\n");
                trivial_two_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
            else if (op[1] != NULL &&
                        PyArray_NDIM(op[1]) >= PyArray_NDIM(op[0]) &&
                        PyArray_TRIVIALLY_ITERABLE_PAIR(op[0], op[1],
                                                        PyArray_TRIVIALLY_ITERABLE_OP_READ,
                                                        PyArray_TRIVIALLY_ITERABLE_OP_NOREAD)) {

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(ufunc, &op[1],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINT("trivial 1 input\n");
                trivial_two_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
        }
        else if (nin == 2 && nout == 1) {
            if (op[2] == NULL &&
                        (order == NPY_ANYORDER || order == NPY_KEEPORDER) &&
                        PyArray_TRIVIALLY_ITERABLE_PAIR(op[0], op[1],
                                                        PyArray_TRIVIALLY_ITERABLE_OP_READ,
                                                        PyArray_TRIVIALLY_ITERABLE_OP_READ)) {
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
                Py_INCREF(dtypes[2]);
                op[2] = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type,
                                 dtypes[2],
                                 PyArray_NDIM(tmp),
                                 PyArray_DIMS(tmp),
                                 NULL, NULL,
                                 PyArray_ISFORTRAN(tmp) ?
                                                NPY_ARRAY_F_CONTIGUOUS : 0,
                                 NULL);
                if (op[2] == NULL) {
                    return -1;
                }

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(ufunc, &op[2],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINT("trivial 2 input with allocated output\n");
                trivial_three_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
            else if (op[2] != NULL &&
                    PyArray_NDIM(op[2]) >= PyArray_NDIM(op[0]) &&
                    PyArray_NDIM(op[2]) >= PyArray_NDIM(op[1]) &&
                    PyArray_TRIVIALLY_ITERABLE_TRIPLE(op[0], op[1], op[2],
                                                      PyArray_TRIVIALLY_ITERABLE_OP_READ,
                                                      PyArray_TRIVIALLY_ITERABLE_OP_READ,
                                                      PyArray_TRIVIALLY_ITERABLE_OP_NOREAD)) {

                /* Call the __prepare_array__ if necessary */
                if (prepare_ufunc_output(ufunc, &op[2],
                                    arr_prep[0], arr_prep_args, 0) < 0) {
                    return -1;
                }

                NPY_UF_DBG_PRINT("trivial 2 input\n");
                trivial_three_operand_loop(op, innerloop, innerloopdata);

                return 0;
            }
        }
    }

    /*
     * If no trivial loop matched, an iterator is required to
     * resolve broadcasting, etc
     */

    NPY_UF_DBG_PRINT("iterator loop\n");
    if (iterator_loop(ufunc, op, dtypes, order,
                    buffersize, arr_prep, arr_prep_args,
                    innerloop, innerloopdata) < 0) {
        return -1;
    }

    return 0;
}

/*
 * nin             - number of inputs
 * nout            - number of outputs
 * wheremask       - if not NULL, the 'where=' parameter to the ufunc.
 * op              - the operands (nin + nout of them)
 * order           - the loop execution order/output memory order
 * buffersize      - how big of a buffer to use
 * arr_prep        - the __array_prepare__ functions for the outputs
 * innerloop       - the inner loop function
 * innerloopdata   - data to pass to the inner loop
 */
static int
execute_fancy_ufunc_loop(PyUFuncObject *ufunc,
                    PyArrayObject *wheremask,
                    PyArrayObject **op,
                    PyArray_Descr **dtypes,
                    NPY_ORDER order,
                    npy_intp buffersize,
                    PyObject **arr_prep,
                    PyObject *arr_prep_args)
{
    int retval, i, nin = ufunc->nin, nout = ufunc->nout;
    int nop = nin + nout;
    npy_uint32 op_flags[NPY_MAXARGS];
    NpyIter *iter;
    int needs_api;
    npy_intp default_op_in_flags = 0, default_op_out_flags = 0;

    NpyIter_IterNextFunc *iternext;
    char **dataptr;
    npy_intp *strides;
    npy_intp *countptr;

    PyArrayObject **op_it;
    npy_uint32 iter_flags;

    if (wheremask != NULL) {
        if (nop + 1 > NPY_MAXARGS) {
            PyErr_SetString(PyExc_ValueError,
                    "Too many operands when including where= parameter");
            return -1;
        }
        op[nop] = wheremask;
        dtypes[nop] = NULL;
        default_op_out_flags |= NPY_ITER_WRITEMASKED;
    }

    /* Set up the flags */
    for (i = 0; i < nin; ++i) {
        op_flags[i] = default_op_in_flags |
                      NPY_ITER_READONLY |
                      NPY_ITER_ALIGNED |
                      NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
        /*
         * If READWRITE flag has been set for this operand,
         * then clear default READONLY flag
         */
        op_flags[i] |= ufunc->op_flags[i];
        if (op_flags[i] & (NPY_ITER_READWRITE | NPY_ITER_WRITEONLY)) {
            op_flags[i] &= ~NPY_ITER_READONLY;
        }
    }
    for (i = nin; i < nop; ++i) {
        /*
         * We don't write to all elements, and the iterator may make
         * UPDATEIFCOPY temporary copies. The output arrays (unless they are
         * allocated by the iterator itself) must be considered READWRITE by the
         * iterator, so that the elements we don't write to are copied to the
         * possible temporary array.
         */
        op_flags[i] = default_op_out_flags |
                      (op[i] != NULL ? NPY_ITER_READWRITE : NPY_ITER_WRITEONLY) |
                      NPY_ITER_ALIGNED |
                      NPY_ITER_ALLOCATE |
                      NPY_ITER_NO_BROADCAST |
                      NPY_ITER_NO_SUBTYPE |
                      NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
    }
    if (wheremask != NULL) {
        op_flags[nop] = NPY_ITER_READONLY | NPY_ITER_ARRAYMASK;
    }

    NPY_UF_DBG_PRINT("Making iterator\n");

    iter_flags = ufunc->iter_flags |
                 NPY_ITER_EXTERNAL_LOOP |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_ZEROSIZE_OK |
                 NPY_ITER_BUFFERED |
                 NPY_ITER_GROWINNER |
                 NPY_ITER_COPY_IF_OVERLAP;

    /*
     * Allocate the iterator.  Because the types of the inputs
     * were already checked, we use the casting rule 'unsafe' which
     * is faster to calculate.
     */
    iter = NpyIter_AdvancedNew(nop + ((wheremask != NULL) ? 1 : 0), op,
                        iter_flags,
                        order, NPY_UNSAFE_CASTING,
                        op_flags, dtypes,
                        -1, NULL, NULL, buffersize);
    if (iter == NULL) {
        return -1;
    }

    NPY_UF_DBG_PRINT("Made iterator\n");

    needs_api = NpyIter_IterationNeedsAPI(iter);

    /* Call the __array_prepare__ functions where necessary */
    op_it = NpyIter_GetOperandArray(iter);
    for (i = nin; i < nop; ++i) {
        PyArrayObject *op_tmp, *orig_op_tmp;

        /*
         * The array can be allocated by the iterator -- it is placed in op[i]
         * and returned to the caller, and this needs an extra incref.
         */
        if (op[i] == NULL) {
            op_tmp = op_it[i];
            Py_INCREF(op_tmp);
        }
        else {
            op_tmp = op[i];
        }

        /* prepare_ufunc_output may decref & replace the pointer */
        orig_op_tmp = op_tmp;
        Py_INCREF(op_tmp);

        if (prepare_ufunc_output(ufunc, &op_tmp,
                                 arr_prep[i], arr_prep_args, i) < 0) {
            NpyIter_Deallocate(iter);
            return -1;
        }

        /* Validate that the prepare_ufunc_output didn't mess with pointers */
        if (PyArray_BYTES(op_tmp) != PyArray_BYTES(orig_op_tmp)) {
            PyErr_SetString(PyExc_ValueError,
                        "The __array_prepare__ functions modified the data "
                        "pointer addresses in an invalid fashion");
            Py_DECREF(op_tmp);
            NpyIter_Deallocate(iter);
            return -1;
        }

        /*
         * Put the updated operand back and undo the DECREF above. If
         * COPY_IF_OVERLAP made a temporary copy, the output will be copied
         * by UPDATEIFCOPY even if op[i] was changed by prepare_ufunc_output.
         */
        op[i] = op_tmp;
        Py_DECREF(op_tmp);
    }

    /* Only do the loop if the iteration size is non-zero */
    if (NpyIter_GetIterSize(iter) != 0) {
        PyUFunc_MaskedStridedInnerLoopFunc *innerloop;
        NpyAuxData *innerloopdata;
        npy_intp fixed_strides[2*NPY_MAXARGS];
        PyArray_Descr **iter_dtypes;
        NPY_BEGIN_THREADS_DEF;

        /*
         * Get the inner loop, with the possibility of specialization
         * based on the fixed strides.
         */
        NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
        iter_dtypes = NpyIter_GetDescrArray(iter);
        if (ufunc->masked_inner_loop_selector(ufunc, dtypes,
                        wheremask != NULL ? iter_dtypes[nop]
                                          : iter_dtypes[nop + nin],
                        fixed_strides,
                        wheremask != NULL ? fixed_strides[nop]
                                          : fixed_strides[nop + nin],
                        &innerloop, &innerloopdata, &needs_api) < 0) {
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
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NPY_BEGIN_THREADS_NDITER(iter);

        NPY_UF_DBG_PRINT("Actual inner loop:\n");
        /* Execute the loop */
        do {
            NPY_UF_DBG_PRINT1("iterator loop count %d\n", (int)*countptr);
            innerloop(dataptr, strides,
                        dataptr[nop], strides[nop],
                        *countptr, innerloopdata);
        } while (iternext(iter));

        NPY_END_THREADS;

        NPY_AUXDATA_FREE(innerloopdata);
    }

    retval = 0;
    nop = NpyIter_GetNOp(iter);
    for(i=0; i< nop; ++i) {
        if (PyArray_ResolveWritebackIfCopy(NpyIter_GetOperandArray(iter)[i]) < 0) {
            retval = -1;
        }
    }

    NpyIter_Deallocate(iter);
    return retval;
}

static PyObject *
make_arr_prep_args(npy_intp nin, PyObject *args, PyObject *kwds)
{
    PyObject *out = kwds ? PyDict_GetItem(kwds, npy_um_str_out) : NULL;
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

/*
 * Validate the core dimensions of all the operands, and collect all of
 * the labelled core dimensions into 'core_dim_sizes'.
 *
 * Returns 0 on success, and -1 on failure
 *
 * The behavior has been changed in NumPy 1.10.0, and the following
 * requirements must be fulfilled or an error will be raised:
 *  * Arguments, both input and output, must have at least as many
 *    dimensions as the corresponding number of core dimensions. In
 *    previous versions, 1's were prepended to the shape as needed.
 *  * Core dimensions with same labels must have exactly matching sizes.
 *    In previous versions, core dimensions of size 1 would broadcast
 *    against other core dimensions with the same label.
 *  * All core dimensions must have their size specified by a passed in
 *    input or output argument. In previous versions, core dimensions in
 *    an output argument that were not specified in an input argument,
 *    and whose size could not be inferred from a passed in output
 *    argument, would have their size set to 1.
 */
static int
_get_coredim_sizes(PyUFuncObject *ufunc, PyArrayObject **op,
                   npy_intp* core_dim_sizes) {
    int i;
    int nin = ufunc->nin;
    int nout = ufunc->nout;
    int nop = nin + nout;

    for (i = 0; i < ufunc->core_num_dim_ix; ++i) {
        core_dim_sizes[i] = -1;
    }
    for (i = 0; i < nop; ++i) {
        if (op[i] != NULL) {
            int idim;
            int dim_offset = ufunc->core_offsets[i];
            int num_dims = ufunc->core_num_dims[i];
            int core_start_dim = PyArray_NDIM(op[i]) - num_dims;

            /* Check if operands have enough dimensions */
            if (core_start_dim < 0) {
                PyErr_Format(PyExc_ValueError,
                        "%s: %s operand %d does not have enough "
                        "dimensions (has %d, gufunc core with "
                        "signature %s requires %d)",
                        ufunc_get_name_cstr(ufunc), i < nin ? "Input" : "Output",
                        i < nin ? i : i - nin, PyArray_NDIM(op[i]),
                        ufunc->core_signature, num_dims);
                return -1;
            }

            /*
             * Make sure every core dimension exactly matches all other core
             * dimensions with the same label.
             */
            for (idim = 0; idim < num_dims; ++idim) {
                int core_dim_index = ufunc->core_dim_ixs[dim_offset+idim];
                npy_intp op_dim_size =
                            PyArray_DIM(op[i], core_start_dim+idim);

                if (core_dim_sizes[core_dim_index] == -1) {
                    core_dim_sizes[core_dim_index] = op_dim_size;
                }
                else if (op_dim_size != core_dim_sizes[core_dim_index]) {
                    PyErr_Format(PyExc_ValueError,
                            "%s: %s operand %d has a mismatch in its "
                            "core dimension %d, with gufunc "
                            "signature %s (size %zd is different "
                            "from %zd)",
                            ufunc_get_name_cstr(ufunc), i < nin ? "Input" : "Output",
                            i < nin ? i : i - nin, idim,
                            ufunc->core_signature, op_dim_size,
                            core_dim_sizes[core_dim_index]);
                    return -1;
                }
            }
        }
    }

    /*
     * Make sure no core dimension is unspecified.
     */
    for (i = 0; i < ufunc->core_num_dim_ix; ++i) {
        if (core_dim_sizes[i] == -1) {
            break;
        }
    }
    if (i != ufunc->core_num_dim_ix) {
        /*
         * There is at least one core dimension missing, find in which
         * operand it comes up first (it has to be an output operand).
         */
        const int missing_core_dim = i;
        int out_op;
        for (out_op = nin; out_op < nop; ++out_op) {
            int first_idx = ufunc->core_offsets[out_op];
            int last_idx = first_idx + ufunc->core_num_dims[out_op];
            for (i = first_idx; i < last_idx; ++i) {
                if (ufunc->core_dim_ixs[i] == missing_core_dim) {
                    break;
                }
            }
            if (i < last_idx) {
                /* Change index offsets for error message */
                out_op -= nin;
                i -= first_idx;
                break;
            }
        }
        PyErr_Format(PyExc_ValueError,
                     "%s: Output operand %d has core dimension %d "
                     "unspecified, with gufunc signature %s",
                     ufunc_get_name_cstr(ufunc), out_op, i, ufunc->core_signature);
        return -1;
    }
    return 0;
}

static int
PyUFunc_GeneralizedFunction(PyUFuncObject *ufunc,
                        PyObject *args, PyObject *kwds,
                        PyArrayObject **op)
{
    int nin, nout;
    int i, j, idim, nop;
    const char *ufunc_name;
    int retval = 0, subok = 1;
    int needs_api = 0;

    PyArray_Descr *dtypes[NPY_MAXARGS];

    /* Use remapped axes for generalized ufunc */
    int broadcast_ndim, iter_ndim;
    int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
    int *op_axes[NPY_MAXARGS];

    npy_uint32 op_flags[NPY_MAXARGS];
    npy_intp iter_shape[NPY_MAXARGS];
    NpyIter *iter = NULL;
    npy_uint32 iter_flags;
    npy_intp total_problem_size;

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    /* The selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;
    /* The dimensions which get passed to the inner loop */
    npy_intp inner_dimensions[NPY_MAXDIMS+1];
    /* The strides which get passed to the inner loop */
    npy_intp *inner_strides = NULL;

    /* The sizes of the core dimensions (# entries is ufunc->core_num_dim_ix) */
    npy_intp *core_dim_sizes = inner_dimensions + 1;
    int core_dim_ixs_size;

    /* The __array_prepare__ function to call for each output */
    PyObject *arr_prep[NPY_MAXARGS];
    /*
     * This is either args, or args with the out= parameter from
     * kwds added appropriately.
     */
    PyObject *arr_prep_args = NULL;

    NPY_ORDER order = NPY_KEEPORDER;
    /* Use the default assignment casting rule */
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    /* When provided, extobj and typetup contain borrowed references */
    PyObject *extobj = NULL, *type_tup = NULL;

    if (ufunc == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return -1;
    }

    nin = ufunc->nin;
    nout = ufunc->nout;
    nop = nin + nout;

    ufunc_name = ufunc_get_name_cstr(ufunc);

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_name);

    /* Initialize all the operands and dtypes to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
        dtypes[i] = NULL;
        arr_prep[i] = NULL;
    }

    NPY_UF_DBG_PRINT("Getting arguments\n");

    /* Get all the arguments */
    retval = get_ufunc_arguments(ufunc, args, kwds,
                op, &order, &casting, &extobj,
                &type_tup, &subok, NULL);
    if (retval < 0) {
        goto fail;
    }

    /*
     * Figure out the number of iteration dimensions, which
     * is the broadcast result of all the input non-core
     * dimensions.
     */
    broadcast_ndim = 0;
    for (i = 0; i < nin; ++i) {
        int n = PyArray_NDIM(op[i]) - ufunc->core_num_dims[i];
        if (n > broadcast_ndim) {
            broadcast_ndim = n;
        }
    }

    /*
     * Figure out the number of iterator creation dimensions,
     * which is the broadcast dimensions + all the core dimensions of
     * the outputs, so that the iterator can allocate those output
     * dimensions following the rules of order='F', for example.
     */
    iter_ndim = broadcast_ndim;
    for (i = nin; i < nop; ++i) {
        iter_ndim += ufunc->core_num_dims[i];
    }
    if (iter_ndim > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                    "too many dimensions for generalized ufunc %s",
                    ufunc_name);
        retval = -1;
        goto fail;
    }

    /* Collect the lengths of the labelled core dimensions */
    retval = _get_coredim_sizes(ufunc, op, core_dim_sizes);
    if(retval < 0) {
        goto fail;
    }

    /* Fill in the initial part of 'iter_shape' */
    for (idim = 0; idim < broadcast_ndim; ++idim) {
        iter_shape[idim] = -1;
    }

    /* Fill in op_axes for all the operands */
    j = broadcast_ndim;
    for (i = 0; i < nop; ++i) {
        int n;
        if (op[i]) {
            /*
             * Note that n may be negative if broadcasting
             * extends into the core dimensions.
             */
            n = PyArray_NDIM(op[i]) - ufunc->core_num_dims[i];
        }
        else {
            n = broadcast_ndim;
        }
        /* Broadcast all the unspecified dimensions normally */
        for (idim = 0; idim < broadcast_ndim; ++idim) {
            if (idim >= broadcast_ndim - n) {
                op_axes_arrays[i][idim] = idim - (broadcast_ndim - n);
            }
            else {
                op_axes_arrays[i][idim] = -1;
            }
        }

        /* Any output core dimensions shape should be ignored */
        for (idim = broadcast_ndim; idim < iter_ndim; ++idim) {
            op_axes_arrays[i][idim] = -1;
        }

        /* Except for when it belongs to this output */
        if (i >= nin) {
            int dim_offset = ufunc->core_offsets[i];
            int num_dims = ufunc->core_num_dims[i];
            /* Fill in 'iter_shape' and 'op_axes' for this output */
            for (idim = 0; idim < num_dims; ++idim) {
                iter_shape[j] = core_dim_sizes[
                                        ufunc->core_dim_ixs[dim_offset + idim]];
                op_axes_arrays[i][j] = n + idim;
                ++j;
            }
        }

        op_axes[i] = op_axes_arrays[i];
    }

    /* Get the buffersize and errormask */
    if (_get_bufsize_errmask(extobj, ufunc_name, &buffersize, &errormask) < 0) {
        retval = -1;
        goto fail;
    }

    NPY_UF_DBG_PRINT("Finding inner loop\n");


    retval = ufunc->type_resolver(ufunc, casting,
                            op, type_tup, dtypes);
    if (retval < 0) {
        goto fail;
    }
    /* For the generalized ufunc, we get the loop right away too */
    retval = ufunc->legacy_inner_loop_selector(ufunc, dtypes,
                                    &innerloop, &innerloopdata, &needs_api);
    if (retval < 0) {
        goto fail;
    }

#if NPY_UF_DBG_TRACING
    printf("input types:\n");
    for (i = 0; i < nin; ++i) {
        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
        printf(" ");
    }
    printf("\noutput types:\n");
    for (i = nin; i < nop; ++i) {
        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
        printf(" ");
    }
    printf("\n");
#endif

    if (subok) {
        /*
         * Get the appropriate __array_prepare__ function to call
         * for each output
         */
        _find_array_prepare(args, kwds, arr_prep, nin, nout, 0);

        /* Set up arr_prep_args if a prep function was needed */
        for (i = 0; i < nout; ++i) {
            if (arr_prep[i] != NULL && arr_prep[i] != Py_None) {
                arr_prep_args = make_arr_prep_args(nin, args, kwds);
                break;
            }
        }
    }

    /* If the loop wants the arrays, provide them */
    if (_does_loop_use_arrays(innerloopdata)) {
        innerloopdata = (void*)op;
    }

    /*
     * Set up the iterator per-op flags.  For generalized ufuncs, we
     * can't do buffering, so must COPY or UPDATEIFCOPY.
     */
    for (i = 0; i < nin; ++i) {
        op_flags[i] = NPY_ITER_READONLY |
                      NPY_ITER_COPY |
                      NPY_ITER_ALIGNED |
                      NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
        /*
         * If READWRITE flag has been set for this operand,
         * then clear default READONLY flag
         */
        op_flags[i] |= ufunc->op_flags[i];
        if (op_flags[i] & (NPY_ITER_READWRITE | NPY_ITER_WRITEONLY)) {
            op_flags[i] &= ~NPY_ITER_READONLY;
        }
    }
    for (i = nin; i < nop; ++i) {
        op_flags[i] = NPY_ITER_READWRITE|
                      NPY_ITER_UPDATEIFCOPY|
                      NPY_ITER_ALIGNED|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
    }

    iter_flags = ufunc->iter_flags |
                 NPY_ITER_MULTI_INDEX |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_REDUCE_OK |
                 NPY_ITER_ZEROSIZE_OK |
                 NPY_ITER_COPY_IF_OVERLAP;

    /* Create the iterator */
    iter = NpyIter_AdvancedNew(nop, op, iter_flags,
                           order, NPY_UNSAFE_CASTING, op_flags,
                           dtypes, iter_ndim,
                           op_axes, iter_shape, 0);
    if (iter == NULL) {
        retval = -1;
        goto fail;
    }

    /* Fill in any allocated outputs */
    for (i = nin; i < nop; ++i) {
        if (op[i] == NULL) {
            op[i] = NpyIter_GetOperandArray(iter)[i];
            Py_INCREF(op[i]);
        }
    }

    /*
     * Set up the inner strides array. Because we're not doing
     * buffering, the strides are fixed throughout the looping.
     */
    core_dim_ixs_size = 0;
    for (i = 0; i < nop; ++i) {
        core_dim_ixs_size += ufunc->core_num_dims[i];
    }
    inner_strides = (npy_intp *)PyArray_malloc(
                        NPY_SIZEOF_INTP * (nop+core_dim_ixs_size));
    if (inner_strides == NULL) {
        PyErr_NoMemory();
        retval = -1;
        goto fail;
    }
    /* Copy the strides after the first nop */
    idim = nop;
    for (i = 0; i < nop; ++i) {
        int num_dims = ufunc->core_num_dims[i];
        int core_start_dim = PyArray_NDIM(op[i]) - num_dims;
        /*
         * Need to use the arrays in the iterator, not op, because
         * a copy with a different-sized type may have been made.
         */
        PyArrayObject *arr = NpyIter_GetOperandArray(iter)[i];
        npy_intp *shape = PyArray_SHAPE(arr);
        npy_intp *strides = PyArray_STRIDES(arr);
        for (j = 0; j < num_dims; ++j) {
            if (core_start_dim + j >= 0) {
                /*
                 * Force the stride to zero when the shape is 1, sot
                 * that the broadcasting works right.
                 */
                if (shape[core_start_dim + j] != 1) {
                    inner_strides[idim++] = strides[core_start_dim + j];
                } else {
                    inner_strides[idim++] = 0;
                }
            } else {
                inner_strides[idim++] = 0;
            }
        }
    }

    total_problem_size = NpyIter_GetIterSize(iter);
    if (total_problem_size < 0) {
        /*
         * Only used for threading, if negative (this means that it is
         * larger then ssize_t before axes removal) assume that the actual
         * problem is large enough to be threaded usefully.
         */
        total_problem_size = 1000;
    }

    /* Remove all the core output dimensions from the iterator */
    for (i = broadcast_ndim; i < iter_ndim; ++i) {
        if (NpyIter_RemoveAxis(iter, broadcast_ndim) != NPY_SUCCEED) {
            retval = -1;
            goto fail;
        }
    }
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        retval = -1;
        goto fail;
    }
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        retval = -1;
        goto fail;
    }

    /*
     * The first nop strides are for the inner loop (but only can
     * copy them after removing the core axes
     */
    memcpy(inner_strides, NpyIter_GetInnerStrideArray(iter),
                                    NPY_SIZEOF_INTP * nop);

#if 0
    printf("strides: ");
    for (i = 0; i < nop+core_dim_ixs_size; ++i) {
        printf("%d ", (int)inner_strides[i]);
    }
    printf("\n");
#endif

    /* Start with the floating-point exception flags cleared */
    PyUFunc_clearfperr();

    NPY_UF_DBG_PRINT("Executing inner loop\n");

    if (NpyIter_GetIterSize(iter) != 0) {
        /* Do the ufunc loop */
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *count_ptr;
        NPY_BEGIN_THREADS_DEF;

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            retval = -1;
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        count_ptr = NpyIter_GetInnerLoopSizePtr(iter);

        if (!needs_api && !NpyIter_IterationNeedsAPI(iter)) {
            NPY_BEGIN_THREADS_THRESHOLDED(total_problem_size);
        }
        do {
            inner_dimensions[0] = *count_ptr;
            innerloop(dataptr, inner_dimensions, inner_strides, innerloopdata);
        } while (iternext(iter));

        if (!needs_api && !NpyIter_IterationNeedsAPI(iter)) {
            NPY_END_THREADS;
        }
    } else {
        /**
         * For each output operand, check if it has non-zero size,
         * and assign the identity if it does. For example, a dot
         * product of two zero-length arrays will be a scalar,
         * which has size one.
         */
        for (i = nin; i < nop; ++i) {
            if (PyArray_SIZE(op[i]) != 0) {
                switch (ufunc->identity) {
                    case PyUFunc_Zero:
                        assign_reduce_identity_zero(op[i], NULL);
                        break;
                    case PyUFunc_One:
                        assign_reduce_identity_one(op[i], NULL);
                        break;
                    case PyUFunc_MinusOne:
                        assign_reduce_identity_minusone(op[i], NULL);
                        break;
                    case PyUFunc_None:
                    case PyUFunc_ReorderableNone:
                        PyErr_Format(PyExc_ValueError,
                                "ufunc %s ",
                                ufunc_name);
                        retval = -1;
                        goto fail;
                    default:
                        PyErr_Format(PyExc_ValueError,
                                "ufunc %s has an invalid identity for reduction",
                                ufunc_name);
                        retval = -1;
                        goto fail;
                }
            }
        }
    }

    /* Check whether any errors occurred during the loop */
    if (PyErr_Occurred() ||
        _check_ufunc_fperr(errormask, extobj, ufunc_name) < 0) {
        retval = -1;
        goto fail;
    }

    /* Write back any temporary data from PyArray_SetWritebackIfCopyBase */
    for(i=nin; i< nop; ++i)
        if (PyArray_ResolveWritebackIfCopy(NpyIter_GetOperandArray(iter)[i]) < 0)
            goto fail;

    PyArray_free(inner_strides);
    NpyIter_Deallocate(iter);
    /* The caller takes ownership of all the references in op */
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(dtypes[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(type_tup);
    Py_XDECREF(arr_prep_args);

    NPY_UF_DBG_PRINT("Returning Success\n");

    return 0;

fail:
    NPY_UF_DBG_PRINT1("Returning failure code %d\n", retval);
    PyArray_free(inner_strides);
    NpyIter_Deallocate(iter);
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
        Py_XDECREF(dtypes[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(type_tup);
    Py_XDECREF(arr_prep_args);

    return retval;
}

/*UFUNC_API
 *
 * This generic function is called with the ufunc object, the arguments to it,
 * and an array of (pointers to) PyArrayObjects which are NULL.
 *
 * 'op' is an array of at least NPY_MAXARGS PyArrayObject *.
 */
NPY_NO_EXPORT int
PyUFunc_GenericFunction(PyUFuncObject *ufunc,
                        PyObject *args, PyObject *kwds,
                        PyArrayObject **op)
{
    int nin, nout;
    int i, nop;
    const char *ufunc_name;
    int retval = -1, subok = 1;
    int need_fancy = 0;

    PyArray_Descr *dtypes[NPY_MAXARGS];

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    /* The mask provided in the 'where=' parameter */
    PyArrayObject *wheremask = NULL;

    /* The __array_prepare__ function to call for each output */
    PyObject *arr_prep[NPY_MAXARGS];
    /*
     * This is either args, or args with the out= parameter from
     * kwds added appropriately.
     */
    PyObject *arr_prep_args = NULL;

    int trivial_loop_ok = 0;

    NPY_ORDER order = NPY_KEEPORDER;
    /* Use the default assignment casting rule */
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    /* When provided, extobj and typetup contain borrowed references */
    PyObject *extobj = NULL, *type_tup = NULL;

    if (ufunc == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return -1;
    }

    if (ufunc->core_enabled) {
        return PyUFunc_GeneralizedFunction(ufunc, args, kwds, op);
    }

    nin = ufunc->nin;
    nout = ufunc->nout;
    nop = nin + nout;

    ufunc_name = ufunc_get_name_cstr(ufunc);

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_name);

    /* Initialize all the operands and dtypes to NULL */
    for (i = 0; i < nop; ++i) {
        op[i] = NULL;
        dtypes[i] = NULL;
        arr_prep[i] = NULL;
    }

    NPY_UF_DBG_PRINT("Getting arguments\n");

    /* Get all the arguments */
    retval = get_ufunc_arguments(ufunc, args, kwds,
                op, &order, &casting, &extobj,
                &type_tup, &subok, &wheremask);
    if (retval < 0) {
        goto fail;
    }

    /*
     * Use the masked loop if a wheremask was specified.
     */
    if (wheremask != NULL) {
        need_fancy = 1;
    }

    /* Get the buffersize and errormask */
    if (_get_bufsize_errmask(extobj, ufunc_name, &buffersize, &errormask) < 0) {
        retval = -1;
        goto fail;
    }

    NPY_UF_DBG_PRINT("Finding inner loop\n");

    retval = ufunc->type_resolver(ufunc, casting,
                            op, type_tup, dtypes);
    if (retval < 0) {
        goto fail;
    }

    /* Only do the trivial loop check for the unmasked version. */
    if (!need_fancy) {
        /*
         * This checks whether a trivial loop is ok, making copies of
         * scalar and one dimensional operands if that will help.
         */
        trivial_loop_ok = check_for_trivial_loop(ufunc, op, dtypes, buffersize);
        if (trivial_loop_ok < 0) {
            goto fail;
        }
    }

#if NPY_UF_DBG_TRACING
    printf("input types:\n");
    for (i = 0; i < nin; ++i) {
        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
        printf(" ");
    }
    printf("\noutput types:\n");
    for (i = nin; i < nop; ++i) {
        PyObject_Print((PyObject *)dtypes[i], stdout, 0);
        printf(" ");
    }
    printf("\n");
#endif

    if (subok) {
        /*
         * Get the appropriate __array_prepare__ function to call
         * for each output
         */
        _find_array_prepare(args, kwds, arr_prep, nin, nout, 0);

        /* Set up arr_prep_args if a prep function was needed */
        for (i = 0; i < nout; ++i) {
            if (arr_prep[i] != NULL && arr_prep[i] != Py_None) {
                arr_prep_args = make_arr_prep_args(nin, args, kwds);
                break;
            }
        }
    }

    /* Start with the floating-point exception flags cleared */
    PyUFunc_clearfperr();

    /* Do the ufunc loop */
    if (need_fancy) {
        NPY_UF_DBG_PRINT("Executing fancy inner loop\n");

        retval = execute_fancy_ufunc_loop(ufunc, wheremask,
                            op, dtypes, order,
                            buffersize, arr_prep, arr_prep_args);
    }
    else {
        NPY_UF_DBG_PRINT("Executing legacy inner loop\n");

        retval = execute_legacy_ufunc_loop(ufunc, trivial_loop_ok,
                            op, dtypes, order,
                            buffersize, arr_prep, arr_prep_args);
    }
    if (retval < 0) {
        goto fail;
    }

    /* Check whether any errors occurred during the loop */
    if (PyErr_Occurred() ||
        _check_ufunc_fperr(errormask, extobj, ufunc_name) < 0) {
        retval = -1;
        goto fail;
    }


    /* The caller takes ownership of all the references in op */
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(dtypes[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(type_tup);
    Py_XDECREF(arr_prep_args);
    Py_XDECREF(wheremask);

    NPY_UF_DBG_PRINT("Returning Success\n");

    return 0;

fail:
    NPY_UF_DBG_PRINT1("Returning failure code %d\n", retval);
    for (i = 0; i < nop; ++i) {
        Py_XDECREF(op[i]);
        op[i] = NULL;
        Py_XDECREF(dtypes[i]);
        Py_XDECREF(arr_prep[i]);
    }
    Py_XDECREF(type_tup);
    Py_XDECREF(arr_prep_args);
    Py_XDECREF(wheremask);

    return retval;
}

/*
 * Given the output type, finds the specified binary op.  The
 * ufunc must have nin==2 and nout==1.  The function may modify
 * otype if the given type isn't found.
 *
 * Returns 0 on success, -1 on failure.
 */
static int
get_binary_op_function(PyUFuncObject *ufunc, int *otype,
                        PyUFuncGenericFunction *out_innerloop,
                        void **out_innerloopdata)
{
    int i;
    PyUFunc_Loop1d *funcdata;

    NPY_UF_DBG_PRINT1("Getting binary op function for type number %d\n",
                                *otype);

    /* If the type is custom and there are userloops, search for it here */
    if (ufunc->userloops != NULL && PyTypeNum_ISUSERDEF(*otype)) {
        PyObject *key, *obj;
        key = PyInt_FromLong(*otype);
        if (key == NULL) {
            return -1;
        }
        obj = PyDict_GetItem(ufunc->userloops, key);
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
    for (i = 0; i < ufunc->ntypes; ++i) {
        char *types = ufunc->types + i*ufunc->nargs;

        NPY_UF_DBG_PRINT3("Trying loop with signature %d %d -> %d\n",
                                types[0], types[1], types[2]);

        if (PyArray_CanCastSafely(*otype, types[0]) &&
                    types[0] == types[1] &&
                    (*otype == NPY_OBJECT || types[0] != NPY_OBJECT)) {
            /* If the signature is "xx->x", we found the loop */
            if (types[2] == types[0]) {
                *out_innerloop = ufunc->functions[i];
                *out_innerloopdata = ufunc->data[i];
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
    for (i = 0; i < ufunc->ntypes; ++i) {
        char *types = ufunc->types + i*ufunc->nargs;

        if (PyArray_CanCastSafely(*otype, types[0]) &&
                    types[0] == types[1] &&
                    types[1] == types[2] &&
                    (*otype == NPY_OBJECT || types[0] != NPY_OBJECT)) {
            /* Since the signature is "xx->x", we found the loop */
            *out_innerloop = ufunc->functions[i];
            *out_innerloopdata = ufunc->data[i];
            *otype = types[0];
            return 0;
        }
    }

    return -1;
}

static int
reduce_type_resolver(PyUFuncObject *ufunc, PyArrayObject *arr,
                        PyArray_Descr *odtype, PyArray_Descr **out_dtype)
{
    int i, retcode;
    PyArrayObject *op[3] = {arr, arr, NULL};
    PyArray_Descr *dtypes[3] = {NULL, NULL, NULL};
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);
    PyObject *type_tup = NULL;

    *out_dtype = NULL;

    /*
     * If odtype is specified, make a type tuple for the type
     * resolution.
     */
    if (odtype != NULL) {
        type_tup = PyTuple_Pack(3, odtype, odtype, Py_None);
        if (type_tup == NULL) {
            return -1;
        }
    }

    /* Use the type resolution function to find our loop */
    retcode = ufunc->type_resolver(
                        ufunc, NPY_UNSAFE_CASTING,
                        op, type_tup, dtypes);
    Py_DECREF(type_tup);
    if (retcode == -1) {
        return -1;
    }
    else if (retcode == -2) {
        PyErr_Format(PyExc_RuntimeError,
                "type resolution returned NotImplemented to "
                "reduce ufunc %s", ufunc_name);
        return -1;
    }

    /*
     * The first two type should be equivalent. Because of how
     * reduce has historically behaved in NumPy, the return type
     * could be different, and it is the return type on which the
     * reduction occurs.
     */
    if (!PyArray_EquivTypes(dtypes[0], dtypes[1])) {
        for (i = 0; i < 3; ++i) {
            Py_DECREF(dtypes[i]);
        }
        PyErr_Format(PyExc_RuntimeError,
                "could not find a type resolution appropriate for "
                "reduce ufunc %s", ufunc_name);
        return -1;
    }

    Py_DECREF(dtypes[0]);
    Py_DECREF(dtypes[1]);
    *out_dtype = dtypes[2];

    return 0;
}

static int
assign_reduce_identity_zero(PyArrayObject *result, void *NPY_UNUSED(data))
{
    return PyArray_FillWithScalar(result, PyArrayScalar_False);
}

static int
assign_reduce_identity_one(PyArrayObject *result, void *NPY_UNUSED(data))
{
    return PyArray_FillWithScalar(result, PyArrayScalar_True);
}

static int
assign_reduce_identity_minusone(PyArrayObject *result, void *NPY_UNUSED(data))
{
    static PyObject *MinusOne = NULL;

    if (MinusOne == NULL) {
        if ((MinusOne = PyInt_FromLong(-1)) == NULL) {
            return -1;
        }
    }
    return PyArray_FillWithScalar(result, MinusOne);
}

static int
reduce_loop(NpyIter *iter, char **dataptrs, npy_intp *strides,
            npy_intp *countptr, NpyIter_IterNextFunc *iternext,
            int needs_api, npy_intp skip_first_count, void *data)
{
    PyArray_Descr *dtypes[3], **iter_dtypes;
    PyUFuncObject *ufunc = (PyUFuncObject *)data;
    char *dataptrs_copy[3];
    npy_intp strides_copy[3];

    /* The normal selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;

    NPY_BEGIN_THREADS_DEF;

    /* Get the inner loop */
    iter_dtypes = NpyIter_GetDescrArray(iter);
    dtypes[0] = iter_dtypes[0];
    dtypes[1] = iter_dtypes[1];
    dtypes[2] = iter_dtypes[0];
    if (ufunc->legacy_inner_loop_selector(ufunc, dtypes,
                            &innerloop, &innerloopdata, &needs_api) < 0) {
        return -1;
    }

    NPY_BEGIN_THREADS_NDITER(iter);

    if (skip_first_count > 0) {
        do {
            npy_intp count = *countptr;

            /* Skip any first-visit elements */
            if (NpyIter_IsFirstVisit(iter, 0)) {
                if (strides[0] == 0) {
                    --count;
                    --skip_first_count;
                    dataptrs[1] += strides[1];
                }
                else {
                    skip_first_count -= count;
                    count = 0;
                }
            }

            /* Turn the two items into three for the inner loop */
            dataptrs_copy[0] = dataptrs[0];
            dataptrs_copy[1] = dataptrs[1];
            dataptrs_copy[2] = dataptrs[0];
            strides_copy[0] = strides[0];
            strides_copy[1] = strides[1];
            strides_copy[2] = strides[0];
            innerloop(dataptrs_copy, &count,
                        strides_copy, innerloopdata);

            /* Jump to the faster loop when skipping is done */
            if (skip_first_count == 0) {
                if (iternext(iter)) {
                    break;
                }
                else {
                    goto finish_loop;
                }
            }
        } while (iternext(iter));
    }
    do {
        /* Turn the two items into three for the inner loop */
        dataptrs_copy[0] = dataptrs[0];
        dataptrs_copy[1] = dataptrs[1];
        dataptrs_copy[2] = dataptrs[0];
        strides_copy[0] = strides[0];
        strides_copy[1] = strides[1];
        strides_copy[2] = strides[0];
        innerloop(dataptrs_copy, countptr,
                    strides_copy, innerloopdata);
    } while (iternext(iter));

finish_loop:
    NPY_END_THREADS;

    return (needs_api && PyErr_Occurred()) ? -1 : 0;
}

/*
 * The implementation of the reduction operators with the new iterator
 * turned into a bit of a long function here, but I think the design
 * of this part needs to be changed to be more like einsum, so it may
 * not be worth refactoring it too much.  Consider this timing:
 *
 * >>> a = arange(10000)
 *
 * >>> timeit sum(a)
 * 10000 loops, best of 3: 17 us per loop
 *
 * >>> timeit einsum("i->",a)
 * 100000 loops, best of 3: 13.5 us per loop
 *
 * The axes must already be bounds-checked by the calling function,
 * this function does not validate them.
 */
static PyArrayObject *
PyUFunc_Reduce(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *out,
        int naxes, int *axes, PyArray_Descr *odtype, int keepdims)
{
    int iaxes, reorderable, ndim;
    npy_bool axis_flags[NPY_MAXDIMS];
    PyArray_Descr *dtype;
    PyArrayObject *result;
    PyArray_AssignReduceIdentityFunc *assign_identity = NULL;
    const char *ufunc_name = ufunc_get_name_cstr(ufunc);
    /* These parameters come from a TLS global */
    int buffersize = 0, errormask = 0;

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s.reduce\n", ufunc_name);

    ndim = PyArray_NDIM(arr);

    /* Create an array of flags for reduction */
    memset(axis_flags, 0, ndim);
    for (iaxes = 0; iaxes < naxes; ++iaxes) {
        int axis = axes[iaxes];
        if (axis_flags[axis]) {
            PyErr_SetString(PyExc_ValueError,
                    "duplicate value in 'axis'");
            return NULL;
        }
        axis_flags[axis] = 1;
    }

    switch (ufunc->identity) {
        case PyUFunc_Zero:
            assign_identity = &assign_reduce_identity_zero;
            reorderable = 1;
            /*
             * The identity for a dynamic dtype like
             * object arrays can't be used in general
             */
            if (PyArray_ISOBJECT(arr) && PyArray_SIZE(arr) != 0) {
                assign_identity = NULL;
            }
            break;
        case PyUFunc_One:
            assign_identity = &assign_reduce_identity_one;
            reorderable = 1;
            /*
             * The identity for a dynamic dtype like
             * object arrays can't be used in general
             */
            if (PyArray_ISOBJECT(arr) && PyArray_SIZE(arr) != 0) {
                assign_identity = NULL;
            }
            break;
        case PyUFunc_MinusOne:
            assign_identity = &assign_reduce_identity_minusone;
            reorderable = 1;
            /*
             * The identity for a dynamic dtype like
             * object arrays can't be used in general
             */
            if (PyArray_ISOBJECT(arr) && PyArray_SIZE(arr) != 0) {
                assign_identity = NULL;
            }
            break;

        case PyUFunc_None:
            reorderable = 0;
            break;
        case PyUFunc_ReorderableNone:
            reorderable = 1;
            break;
        default:
            PyErr_Format(PyExc_ValueError,
                    "ufunc %s has an invalid identity for reduction",
                    ufunc_name);
            return NULL;
    }

    if (_get_bufsize_errmask(NULL, "reduce", &buffersize, &errormask) < 0) {
        return NULL;
    }

    /* Get the reduction dtype */
    if (reduce_type_resolver(ufunc, arr, odtype, &dtype) < 0) {
        return NULL;
    }

    result = PyUFunc_ReduceWrapper(arr, out, NULL, dtype, dtype,
                                   NPY_UNSAFE_CASTING,
                                   axis_flags, reorderable,
                                   keepdims, 0,
                                   assign_identity,
                                   reduce_loop,
                                   ufunc, buffersize, ufunc_name, errormask);

    Py_DECREF(dtype);
    return result;
}


static PyObject *
PyUFunc_Accumulate(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *out,
                   int axis, int otype)
{
    PyArrayObject *op[2];
    PyArray_Descr *op_dtypes[2] = {NULL, NULL};
    int op_axes_arrays[2][NPY_MAXDIMS];
    int *op_axes[2] = {op_axes_arrays[0], op_axes_arrays[1]};
    npy_uint32 op_flags[2];
    int idim, ndim, otype_final;
    int needs_api, need_outer_iterator;

    NpyIter *iter = NULL, *iter_inner = NULL;

    /* The selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;

    const char *ufunc_name = ufunc_get_name_cstr(ufunc);

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    NPY_BEGIN_THREADS_DEF;

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s.accumulate\n", ufunc_name);

#if 0
    printf("Doing %s.accumulate on array with dtype :  ", ufunc_name);
    PyObject_Print((PyObject *)PyArray_DESCR(arr), stdout, 0);
    printf("\n");
#endif

    if (_get_bufsize_errmask(NULL, "accumulate", &buffersize, &errormask) < 0) {
        return NULL;
    }

    /* Take a reference to out for later returning */
    Py_XINCREF(out);

    otype_final = otype;
    if (get_binary_op_function(ufunc, &otype_final,
                                &innerloop, &innerloopdata) < 0) {
        PyArray_Descr *dtype = PyArray_DescrFromType(otype);
        PyErr_Format(PyExc_ValueError,
                     "could not find a matching type for %s.accumulate, "
                     "requested type has type code '%c'",
                            ufunc_name, dtype ? dtype->type : '-');
        Py_XDECREF(dtype);
        goto fail;
    }

    ndim = PyArray_NDIM(arr);

    /*
     * Set up the output data type, using the input's exact
     * data type if the type number didn't change to preserve
     * metadata
     */
    if (PyArray_DESCR(arr)->type_num == otype_final) {
        if (PyArray_ISNBO(PyArray_DESCR(arr)->byteorder)) {
            op_dtypes[0] = PyArray_DESCR(arr);
            Py_INCREF(op_dtypes[0]);
        }
        else {
            op_dtypes[0] = PyArray_DescrNewByteorder(PyArray_DESCR(arr),
                                                    NPY_NATIVE);
        }
    }
    else {
        op_dtypes[0] = PyArray_DescrFromType(otype_final);
    }
    if (op_dtypes[0] == NULL) {
        goto fail;
    }

#if NPY_UF_DBG_TRACING
    printf("Found %s.accumulate inner loop with dtype :  ", ufunc_name);
    PyObject_Print((PyObject *)op_dtypes[0], stdout, 0);
    printf("\n");
#endif

    /* Set up the op_axes for the outer loop */
    for (idim = 0; idim < ndim; ++idim) {
        op_axes_arrays[0][idim] = idim;
        op_axes_arrays[1][idim] = idim;
    }

    /* The per-operand flags for the outer loop */
    op_flags[0] = NPY_ITER_READWRITE |
                  NPY_ITER_NO_BROADCAST |
                  NPY_ITER_ALLOCATE |
                  NPY_ITER_NO_SUBTYPE;
    op_flags[1] = NPY_ITER_READONLY;

    op[0] = out;
    op[1] = arr;

    need_outer_iterator = (ndim > 1);
    /* We can't buffer, so must do UPDATEIFCOPY */
    if (!PyArray_ISALIGNED(arr) || (out && !PyArray_ISALIGNED(out)) ||
            !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(arr)) ||
            (out &&
             !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(out)))) {
        need_outer_iterator = 1;
    }
    /* If input and output overlap in memory, use iterator to figure it out */
    else if (out != NULL && solve_may_share_memory(out, arr, NPY_MAY_SHARE_BOUNDS) != 0) {
        need_outer_iterator = 1;
    }

    if (need_outer_iterator) {
        int ndim_iter = 0;
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK|
                           NPY_ITER_REFS_OK|
                           NPY_ITER_COPY_IF_OVERLAP;
        PyArray_Descr **op_dtypes_param = NULL;

        /*
         * The way accumulate is set up, we can't do buffering,
         * so make a copy instead when necessary.
         */
        ndim_iter = ndim;
        flags |= NPY_ITER_MULTI_INDEX;
        /* Add some more flags */
        op_flags[0] |= NPY_ITER_UPDATEIFCOPY|NPY_ITER_ALIGNED;
        op_flags[1] |= NPY_ITER_COPY|NPY_ITER_ALIGNED;
        op_dtypes_param = op_dtypes;
        op_dtypes[1] = op_dtypes[0];
        NPY_UF_DBG_PRINT("Allocating outer iterator\n");
        iter = NpyIter_AdvancedNew(2, op, flags,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                   op_flags,
                                   op_dtypes_param,
                                   ndim_iter, op_axes, NULL, 0);
        if (iter == NULL) {
            goto fail;
        }

        /* In case COPY or UPDATEIFCOPY occurred */
        op[0] = NpyIter_GetOperandArray(iter)[0];
        op[1] = NpyIter_GetOperandArray(iter)[1];

        if (NpyIter_RemoveAxis(iter, axis) != NPY_SUCCEED) {
            goto fail;
        }
        if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
            goto fail;
        }
    }

    /* Get the output */
    if (out == NULL) {
        if (iter) {
            op[0] = out = NpyIter_GetOperandArray(iter)[0];
            Py_INCREF(out);
        }
        else {
            PyArray_Descr *dtype = op_dtypes[0];
            Py_INCREF(dtype);
            op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                    &PyArray_Type, dtype,
                                    ndim, PyArray_DIMS(op[1]), NULL, NULL,
                                    0, NULL);
            if (out == NULL) {
                goto fail;
            }

        }
    }

    /*
     * If the reduction axis has size zero, either return the reduction
     * unit for UFUNC_REDUCE, or return the zero-sized output array
     * for UFUNC_ACCUMULATE.
     */
    if (PyArray_DIM(op[1], axis) == 0) {
        goto finish;
    }
    else if (PyArray_SIZE(op[0]) == 0) {
        goto finish;
    }

    if (iter && NpyIter_GetIterSize(iter) != 0) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];
        npy_intp count_m1, stride0, stride1;

        NpyIter_IterNextFunc *iternext;
        char **dataptr;

        int itemsize = op_dtypes[0]->elsize;

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);


        /* Execute the loop with just the outer iterator */
        count_m1 = PyArray_DIM(op[1], axis)-1;
        stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with just outer iterator\n");

        stride0 = PyArray_STRIDE(op[0], axis);

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        needs_api = NpyIter_IterationNeedsAPI(iter);

        NPY_BEGIN_THREADS_NDITER(iter);

        do {
            dataptr_copy[0] = dataptr[0];
            dataptr_copy[1] = dataptr[1];
            dataptr_copy[2] = dataptr[0];

            /*
             * Copy the first element to start the reduction.
             *
             * Output (dataptr[0]) and input (dataptr[1]) may point to
             * the same memory, e.g. np.add.accumulate(a, out=a).
             */
            if (otype == NPY_OBJECT) {
                /*
                 * Incref before decref to avoid the possibility of the
                 * reference count being zero temporarily.
                 */
                Py_XINCREF(*(PyObject **)dataptr_copy[1]);
                Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                *(PyObject **)dataptr_copy[0] =
                                    *(PyObject **)dataptr_copy[1];
            }
            else {
                memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
            }

            if (count_m1 > 0) {
                /* Turn the two items into three for the inner loop */
                dataptr_copy[1] += stride1;
                dataptr_copy[2] += stride0;
                NPY_UF_DBG_PRINT1("iterator loop count %d\n",
                                                (int)count_m1);
                innerloop(dataptr_copy, &count_m1,
                            stride_copy, innerloopdata);
            }
        } while (iternext(iter));

        NPY_END_THREADS;
    }
    else if (iter == NULL) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];

        int itemsize = op_dtypes[0]->elsize;

        /* Execute the loop with no iterators */
        npy_intp count = PyArray_DIM(op[1], axis);
        npy_intp stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with no iterators\n");

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

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        /* Turn the two items into three for the inner loop */
        dataptr_copy[0] = PyArray_BYTES(op[0]);
        dataptr_copy[1] = PyArray_BYTES(op[1]);
        dataptr_copy[2] = PyArray_BYTES(op[0]);

        /*
         * Copy the first element to start the reduction.
         *
         * Output (dataptr[0]) and input (dataptr[1]) may point to the
         * same memory, e.g. np.add.accumulate(a, out=a).
         */
        if (otype == NPY_OBJECT) {
            /*
             * Incref before decref to avoid the possibility of the
             * reference count being zero temporarily.
             */
            Py_XINCREF(*(PyObject **)dataptr_copy[1]);
            Py_XDECREF(*(PyObject **)dataptr_copy[0]);
            *(PyObject **)dataptr_copy[0] =
                                *(PyObject **)dataptr_copy[1];
        }
        else {
            memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
        }

        if (count > 1) {
            --count;
            dataptr_copy[1] += stride1;
            dataptr_copy[2] += stride0;

            NPY_UF_DBG_PRINT1("iterator loop count %d\n", (int)count);

            needs_api = PyDataType_REFCHK(op_dtypes[0]);

            if (!needs_api) {
                NPY_BEGIN_THREADS_THRESHOLDED(count);
            }

            innerloop(dataptr_copy, &count,
                        stride_copy, innerloopdata);

            NPY_END_THREADS;
        }
    }

finish:
    /* Write back any temporary data from PyArray_SetWritebackIfCopyBase */
    if (PyArray_ResolveWritebackIfCopy(op[0]) < 0)
        goto fail;
    Py_XDECREF(op_dtypes[0]);
    NpyIter_Deallocate(iter);
    NpyIter_Deallocate(iter_inner);

    return (PyObject *)out;

fail:
    Py_XDECREF(out);
    Py_XDECREF(op_dtypes[0]);

    NpyIter_Deallocate(iter);
    NpyIter_Deallocate(iter_inner);

    return NULL;
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
PyUFunc_Reduceat(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *ind,
                 PyArrayObject *out, int axis, int otype)
{
    PyArrayObject *op[3];
    PyArray_Descr *op_dtypes[3] = {NULL, NULL, NULL};
    int op_axes_arrays[3][NPY_MAXDIMS];
    int *op_axes[3] = {op_axes_arrays[0], op_axes_arrays[1],
                            op_axes_arrays[2]};
    npy_uint32 op_flags[3];
    int i, idim, ndim, otype_final;
    int need_outer_iterator;

    NpyIter *iter = NULL;

    /* The reduceat indices - ind must be validated outside this call */
    npy_intp *reduceat_ind;
    npy_intp ind_size, red_axis_size;
    /* The selected inner loop */
    PyUFuncGenericFunction innerloop = NULL;
    void *innerloopdata = NULL;

    const char *ufunc_name = ufunc_get_name_cstr(ufunc);
    char *opname = "reduceat";

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    NPY_BEGIN_THREADS_DEF;

    reduceat_ind = (npy_intp *)PyArray_DATA(ind);
    ind_size = PyArray_DIM(ind, 0);
    red_axis_size = PyArray_DIM(arr, axis);

    /* Check for out-of-bounds values in indices array */
    for (i = 0; i < ind_size; ++i) {
        if (reduceat_ind[i] < 0 || reduceat_ind[i] >= red_axis_size) {
            PyErr_Format(PyExc_IndexError,
                "index %d out-of-bounds in %s.%s [0, %d)",
                (int)reduceat_ind[i], ufunc_name, opname, (int)red_axis_size);
            return NULL;
        }
    }

    NPY_UF_DBG_PRINT2("\nEvaluating ufunc %s.%s\n", ufunc_name, opname);

#if 0
    printf("Doing %s.%s on array with dtype :  ", ufunc_name, opname);
    PyObject_Print((PyObject *)PyArray_DESCR(arr), stdout, 0);
    printf("\n");
    printf("Index size is %d\n", (int)ind_size);
#endif

    if (_get_bufsize_errmask(NULL, opname, &buffersize, &errormask) < 0) {
        return NULL;
    }

    /* Take a reference to out for later returning */
    Py_XINCREF(out);

    otype_final = otype;
    if (get_binary_op_function(ufunc, &otype_final,
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

    /*
     * Set up the output data type, using the input's exact
     * data type if the type number didn't change to preserve
     * metadata
     */
    if (PyArray_DESCR(arr)->type_num == otype_final) {
        if (PyArray_ISNBO(PyArray_DESCR(arr)->byteorder)) {
            op_dtypes[0] = PyArray_DESCR(arr);
            Py_INCREF(op_dtypes[0]);
        }
        else {
            op_dtypes[0] = PyArray_DescrNewByteorder(PyArray_DESCR(arr),
                                                    NPY_NATIVE);
        }
    }
    else {
        op_dtypes[0] = PyArray_DescrFromType(otype_final);
    }
    if (op_dtypes[0] == NULL) {
        goto fail;
    }

#if NPY_UF_DBG_TRACING
    printf("Found %s.%s inner loop with dtype :  ", ufunc_name, opname);
    PyObject_Print((PyObject *)op_dtypes[0], stdout, 0);
    printf("\n");
#endif

    /* Set up the op_axes for the outer loop */
    for (i = 0, idim = 0; idim < ndim; ++idim) {
        /* Use the i-th iteration dimension to match up ind */
        if (idim == axis) {
            op_axes_arrays[0][idim] = axis;
            op_axes_arrays[1][idim] = -1;
            op_axes_arrays[2][idim] = 0;
        }
        else {
            op_axes_arrays[0][idim] = idim;
            op_axes_arrays[1][idim] = idim;
            op_axes_arrays[2][idim] = -1;
        }
    }

    op[0] = out;
    op[1] = arr;
    op[2] = ind;

    if (out != NULL || ndim > 1 || !PyArray_ISALIGNED(arr) ||
            !PyArray_EquivTypes(op_dtypes[0], PyArray_DESCR(arr))) {
        need_outer_iterator = 1;
    }

    if (need_outer_iterator) {
        npy_uint32 flags = NPY_ITER_ZEROSIZE_OK|
                           NPY_ITER_REFS_OK|
                           NPY_ITER_MULTI_INDEX|
                           NPY_ITER_COPY_IF_OVERLAP;

        /*
         * The way reduceat is set up, we can't do buffering,
         * so make a copy instead when necessary using
         * the UPDATEIFCOPY flag
         */

        /* The per-operand flags for the outer loop */
        op_flags[0] = NPY_ITER_READWRITE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_SUBTYPE|
                      NPY_ITER_UPDATEIFCOPY|
                      NPY_ITER_ALIGNED;
        op_flags[1] = NPY_ITER_READONLY|
                      NPY_ITER_COPY|
                      NPY_ITER_ALIGNED;
        op_flags[2] = NPY_ITER_READONLY;

        op_dtypes[1] = op_dtypes[0];

        NPY_UF_DBG_PRINT("Allocating outer iterator\n");
        iter = NpyIter_AdvancedNew(3, op, flags,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                   op_flags,
                                   op_dtypes,
                                   ndim, op_axes, NULL, 0);
        if (iter == NULL) {
            goto fail;
        }

        /* Remove the inner loop axis from the outer iterator */
        if (NpyIter_RemoveAxis(iter, axis) != NPY_SUCCEED) {
            goto fail;
        }
        if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
            goto fail;
        }

        /* In case COPY or UPDATEIFCOPY occurred */
        op[0] = NpyIter_GetOperandArray(iter)[0];
        op[1] = NpyIter_GetOperandArray(iter)[1];
        op[2] = NpyIter_GetOperandArray(iter)[2];

        if (out == NULL) {
            out = op[0];
            Py_INCREF(out);
        }
    }
    /* Allocate the output for when there's no outer iterator */
    else if (out == NULL) {
        Py_INCREF(op_dtypes[0]);
        op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                    &PyArray_Type, op_dtypes[0],
                                    1, &ind_size, NULL, NULL,
                                    0, NULL);
        if (out == NULL) {
            goto fail;
        }
    }

    /*
     * If the output has zero elements, return now.
     */
    if (PyArray_SIZE(op[0]) == 0) {
        goto finish;
    }

    if (iter && NpyIter_GetIterSize(iter) != 0) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];

        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp count_m1;
        npy_intp stride0, stride1;
        npy_intp stride0_ind = PyArray_STRIDE(op[0], axis);

        int itemsize = op_dtypes[0]->elsize;

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);

        /* Execute the loop with just the outer iterator */
        count_m1 = PyArray_DIM(op[1], axis)-1;
        stride0 = 0;
        stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with just outer iterator\n");

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        NPY_BEGIN_THREADS_NDITER(iter);

        do {

            for (i = 0; i < ind_size; ++i) {
                npy_intp start = reduceat_ind[i],
                        end = (i == ind_size-1) ? count_m1+1 :
                                                  reduceat_ind[i+1];
                npy_intp count = end - start;

                dataptr_copy[0] = dataptr[0] + stride0_ind*i;
                dataptr_copy[1] = dataptr[1] + stride1*start;
                dataptr_copy[2] = dataptr[0] + stride0_ind*i;

                /*
                 * Copy the first element to start the reduction.
                 *
                 * Output (dataptr[0]) and input (dataptr[1]) may point
                 * to the same memory, e.g.
                 * np.add.reduceat(a, np.arange(len(a)), out=a).
                 */
                if (otype == NPY_OBJECT) {
                    /*
                     * Incref before decref to avoid the possibility of
                     * the reference count being zero temporarily.
                     */
                    Py_XINCREF(*(PyObject **)dataptr_copy[1]);
                    Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                    *(PyObject **)dataptr_copy[0] =
                                        *(PyObject **)dataptr_copy[1];
                }
                else {
                    memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
                }

                if (count > 1) {
                    /* Inner loop like REDUCE */
                    --count;
                    dataptr_copy[1] += stride1;
                    NPY_UF_DBG_PRINT1("iterator loop count %d\n",
                                                    (int)count);
                    innerloop(dataptr_copy, &count,
                                stride_copy, innerloopdata);
                }
            }
        } while (iternext(iter));

        NPY_END_THREADS;
    }
    else if (iter == NULL) {
        char *dataptr_copy[3];
        npy_intp stride_copy[3];

        int itemsize = op_dtypes[0]->elsize;

        npy_intp stride0_ind = PyArray_STRIDE(op[0], axis);

        /* Execute the loop with no iterators */
        npy_intp stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        int needs_api = PyDataType_REFCHK(op_dtypes[0]);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with no iterators\n");

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        if (!needs_api) {
            NPY_BEGIN_THREADS;
        }

        for (i = 0; i < ind_size; ++i) {
            npy_intp start = reduceat_ind[i],
                    end = (i == ind_size-1) ? PyArray_DIM(arr,axis) :
                                              reduceat_ind[i+1];
            npy_intp count = end - start;

            dataptr_copy[0] = PyArray_BYTES(op[0]) + stride0_ind*i;
            dataptr_copy[1] = PyArray_BYTES(op[1]) + stride1*start;
            dataptr_copy[2] = PyArray_BYTES(op[0]) + stride0_ind*i;

            /*
             * Copy the first element to start the reduction.
             *
             * Output (dataptr[0]) and input (dataptr[1]) may point to
             * the same memory, e.g.
             * np.add.reduceat(a, np.arange(len(a)), out=a).
             */
            if (otype == NPY_OBJECT) {
                /*
                 * Incref before decref to avoid the possibility of the
                 * reference count being zero temporarily.
                 */
                Py_XINCREF(*(PyObject **)dataptr_copy[1]);
                Py_XDECREF(*(PyObject **)dataptr_copy[0]);
                *(PyObject **)dataptr_copy[0] =
                                    *(PyObject **)dataptr_copy[1];
            }
            else {
                memmove(dataptr_copy[0], dataptr_copy[1], itemsize);
            }

            if (count > 1) {
                /* Inner loop like REDUCE */
                --count;
                dataptr_copy[1] += stride1;
                NPY_UF_DBG_PRINT1("iterator loop count %d\n",
                                                (int)count);
                innerloop(dataptr_copy, &count,
                            stride_copy, innerloopdata);
            }
        }

        NPY_END_THREADS;
    }

finish:
    if (op[0] && PyArray_ResolveWritebackIfCopy(op[0]) < 0) {
        goto fail;
    }
    Py_XDECREF(op_dtypes[0]);
    NpyIter_Deallocate(iter);

    return (PyObject *)out;

fail:
    Py_XDECREF(out);
    Py_XDECREF(op_dtypes[0]);

    NpyIter_Deallocate(iter);

    return NULL;
}


/*
 * This code handles reduce, reduceat, and accumulate
 * (accumulate and reduce are special cases of the more general reduceat
 * but they are handled separately for speed)
 */
static PyObject *
PyUFunc_GenericReduction(PyUFuncObject *ufunc, PyObject *args,
                         PyObject *kwds, int operation)
{
    int i, naxes=0, ndim;
    int axes[NPY_MAXDIMS];
    PyObject *axes_in = NULL;
    PyArrayObject *mp = NULL, *ret = NULL;
    PyObject *op, *res = NULL;
    PyObject *obj_ind, *context;
    PyArrayObject *indices = NULL;
    PyArray_Descr *otype = NULL;
    PyArrayObject *out = NULL;
    int keepdims = 0;
    static char *reduce_kwlist[] = {
            "array", "axis", "dtype", "out", "keepdims", NULL};
    static char *accumulate_kwlist[] = {
            "array", "axis", "dtype", "out", NULL};
    static char *reduceat_kwlist[] = {
            "array", "indices", "axis", "dtype", "out", NULL};

    static char *_reduce_type[] = {"reduce", "accumulate", "reduceat", NULL};

    if (ufunc == NULL) {
        PyErr_SetString(PyExc_ValueError, "function not supported");
        return NULL;
    }
    if (ufunc->core_enabled) {
        PyErr_Format(PyExc_RuntimeError,
                     "Reduction not defined on ufunc with signature");
        return NULL;
    }
    if (ufunc->nin != 2) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for binary functions",
                     _reduce_type[operation]);
        return NULL;
    }
    if (ufunc->nout != 1) {
        PyErr_Format(PyExc_ValueError,
                     "%s only supported for functions "
                     "returning a single value",
                     _reduce_type[operation]);
        return NULL;
    }
    /* if there is a tuple of 1 for `out` in kwds, unpack it */
    if (kwds != NULL) {
        PyObject *out_obj = PyDict_GetItem(kwds, npy_um_str_out);
        if (out_obj != NULL && PyTuple_CheckExact(out_obj)) {
            if (PyTuple_GET_SIZE(out_obj) != 1) {
                PyErr_SetString(PyExc_ValueError,
                                "The 'out' tuple must have exactly one entry");
                return NULL;
            }
            out_obj = PyTuple_GET_ITEM(out_obj, 0);
            PyDict_SetItem(kwds, npy_um_str_out, out_obj);
        }
    }

    if (operation == UFUNC_REDUCEAT) {
        PyArray_Descr *indtype;
        indtype = PyArray_DescrFromType(NPY_INTP);
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO&O&:reduceat", reduceat_kwlist,
                                         &op,
                                         &obj_ind,
                                         &axes_in,
                                         PyArray_DescrConverter2, &otype,
                                         PyArray_OutputConverter, &out)) {
            goto fail;
        }
        indices = (PyArrayObject *)PyArray_FromAny(obj_ind, indtype,
                                           1, 1, NPY_ARRAY_CARRAY, NULL);
        if (indices == NULL) {
            goto fail;
        }
    }
    else if (operation == UFUNC_ACCUMULATE) {
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO&O&:accumulate",
                                        accumulate_kwlist,
                                        &op,
                                        &axes_in,
                                        PyArray_DescrConverter2, &otype,
                                        PyArray_OutputConverter, &out)) {
            goto fail;
        }
    }
    else {
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO&O&i:reduce",
                                        reduce_kwlist,
                                        &op,
                                        &axes_in,
                                        PyArray_DescrConverter2, &otype,
                                        PyArray_OutputConverter, &out,
                                        &keepdims)) {
            goto fail;
        }
    }
    /* Ensure input is an array */
    if (!PyArray_Check(op) && !PyArray_IsScalar(op, Generic)) {
        context = Py_BuildValue("O(O)i", ufunc, op, 0);
    }
    else {
        context = NULL;
    }
    mp = (PyArrayObject *)PyArray_FromAny(op, NULL, 0, 0, 0, context);
    Py_XDECREF(context);
    if (mp == NULL) {
        goto fail;
    }

    ndim = PyArray_NDIM(mp);

    /* Check to see that type (and otype) is not FLEXIBLE */
    if (PyArray_ISFLEXIBLE(mp) ||
        (otype && PyTypeNum_ISFLEXIBLE(otype->type_num))) {
        PyErr_Format(PyExc_TypeError,
                     "cannot perform %s with flexible type",
                     _reduce_type[operation]);
        goto fail;
    }

    /* Convert the 'axis' parameter into a list of axes */
    if (axes_in == NULL) {
        naxes = 1;
        axes[0] = 0;
    }
    /* Convert 'None' into all the axes */
    else if (axes_in == Py_None) {
        naxes = ndim;
        for (i = 0; i < naxes; ++i) {
            axes[i] = i;
        }
    }
    else if (PyTuple_Check(axes_in)) {
        naxes = PyTuple_Size(axes_in);
        if (naxes < 0 || naxes > NPY_MAXDIMS) {
            PyErr_SetString(PyExc_ValueError,
                    "too many values for 'axis'");
            goto fail;
        }
        for (i = 0; i < naxes; ++i) {
            PyObject *tmp = PyTuple_GET_ITEM(axes_in, i);
            int axis = PyArray_PyIntAsInt(tmp);
            if (error_converting(axis)) {
                goto fail;
            }
            if (check_and_adjust_axis(&axis, ndim) < 0) {
                goto fail;
            }
            axes[i] = (int)axis;
        }
    }
    /* Try to interpret axis as an integer */
    else {
        int axis = PyArray_PyIntAsInt(axes_in);
        /* TODO: PyNumber_Index would be good to use here */
        if (error_converting(axis)) {
            goto fail;
        }
        /* Special case letting axis={0 or -1} slip through for scalars */
        if (ndim == 0 && (axis == 0 || axis == -1)) {
            axis = 0;
        }
        else if (check_and_adjust_axis(&axis, ndim) < 0) {
            goto fail;
        }
        axes[0] = (int)axis;
        naxes = 1;
    }

    /* Check to see if input is zero-dimensional. */
    if (ndim == 0) {
        /*
         * A reduction with no axes is still valid but trivial.
         * As a special case for backwards compatibility in 'sum',
         * 'prod', et al, also allow a reduction where axis=0, even
         * though this is technically incorrect.
         */
        naxes = 0;

        if (!(operation == UFUNC_REDUCE &&
                    (naxes == 0 || (naxes == 1 && axes[0] == 0)))) {
            PyErr_Format(PyExc_TypeError, "cannot %s on a scalar",
                         _reduce_type[operation]);
            goto fail;
        }
    }

     /*
      * If out is specified it determines otype
      * unless otype already specified.
      */
    if (otype == NULL && out != NULL) {
        otype = PyArray_DESCR(out);
        Py_INCREF(otype);
    }
    if (otype == NULL) {
        /*
         * For integer types --- make sure at least a long
         * is used for add and multiply reduction to avoid overflow
         */
        int typenum = PyArray_TYPE(mp);
        if ((PyTypeNum_ISBOOL(typenum) || PyTypeNum_ISINTEGER(typenum))
            && ((strcmp(ufunc->name,"add") == 0)
                || (strcmp(ufunc->name,"multiply") == 0))) {
            if (PyTypeNum_ISBOOL(typenum)) {
                typenum = NPY_LONG;
            }
            else if ((size_t)PyArray_DESCR(mp)->elsize < sizeof(long)) {
                if (PyTypeNum_ISUNSIGNED(typenum)) {
                    typenum = NPY_ULONG;
                }
                else {
                    typenum = NPY_LONG;
                }
            }
        }
        otype = PyArray_DescrFromType(typenum);
    }


    switch(operation) {
    case UFUNC_REDUCE:
        ret = PyUFunc_Reduce(ufunc, mp, out, naxes, axes,
                                          otype, keepdims);
        break;
    case UFUNC_ACCUMULATE:
        if (naxes != 1) {
            PyErr_SetString(PyExc_ValueError,
                        "accumulate does not allow multiple axes");
            goto fail;
        }
        ret = (PyArrayObject *)PyUFunc_Accumulate(ufunc, mp, out, axes[0],
                                                  otype->type_num);
        break;
    case UFUNC_REDUCEAT:
        if (naxes != 1) {
            PyErr_SetString(PyExc_ValueError,
                        "reduceat does not allow multiple axes");
            goto fail;
        }
        ret = (PyArrayObject *)PyUFunc_Reduceat(ufunc, mp, indices, out,
                                            axes[0], otype->type_num);
        Py_DECREF(indices);
        break;
    }
    Py_DECREF(mp);
    Py_DECREF(otype);

    if (ret == NULL) {
        return NULL;
    }

    /* If an output parameter was provided, don't wrap it */
    if (out != NULL) {
        return (PyObject *)ret;
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

fail:
    Py_XDECREF(otype);
    Py_XDECREF(mp);
    return NULL;
}

/*
 * Returns an incref'ed pointer to the proper wrapping object for a
 * ufunc output argument, given the output argument 'out', and the
 * input's wrapping function, 'wrap'.
 */
static PyObject*
_get_out_wrap(PyObject *out, PyObject *wrap) {
    PyObject *owrap;

    if (out == Py_None) {
        /* Iterator allocated outputs get the input's wrapping */
        Py_XINCREF(wrap);
        return wrap;
    }
    if (PyArray_CheckExact(out)) {
        /* None signals to not call any wrapping */
        Py_RETURN_NONE;
    }
    /*
     * For array subclasses use their __array_wrap__ method, or the
     * input's wrapping if not available
     */
    owrap = PyObject_GetAttr(out, npy_um_str_array_wrap);
    if (owrap == NULL || !PyCallable_Check(owrap)) {
        Py_XDECREF(owrap);
        owrap = wrap;
        Py_XINCREF(wrap);
        PyErr_Clear();
    }
    return owrap;
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
    int i, idx_offset, start_idx;
    int np = 0;
    PyObject *with_wrap[NPY_MAXARGS], *wraps[NPY_MAXARGS];
    PyObject *obj, *wrap = NULL;

    /*
     * If a 'subok' parameter is passed and isn't True, don't wrap but put None
     * into slots with out arguments which means return the out argument
     */
    if (kwds != NULL && (obj = PyDict_GetItem(kwds,
                                              npy_um_str_subok)) != NULL) {
        if (obj != Py_True) {
            /* skip search for wrap members */
            goto handle_out;
        }
    }


    for (i = 0; i < nin; i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        wrap = PyObject_GetAttr(obj, npy_um_str_array_wrap);
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
                        NPY_PRIORITY);
            for (i = 1; i < np; ++i) {
                double priority = PyArray_GetPriority(with_wrap[i],
                            NPY_PRIORITY);
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
handle_out:
    nargs = PyTuple_GET_SIZE(args);
    /* Default is using positional arguments */
    obj = args;
    idx_offset = nin;
    start_idx = 0;
    if (nin == nargs && kwds != NULL) {
        /* There may be a keyword argument we can use instead */
        obj = PyDict_GetItem(kwds, npy_um_str_out);
        if (obj == NULL) {
            /* No, go back to positional (even though there aren't any) */
            obj = args;
        }
        else {
            idx_offset = 0;
            if (PyTuple_Check(obj)) {
                /* If a tuple, must have all nout items */
                nargs = nout;
            }
            else {
                /* If the kwarg is not a tuple then it is an array (or None) */
                output_wrap[0] = _get_out_wrap(obj, wrap);
                start_idx = 1;
                nargs = 1;
            }
        }
    }

    for (i = start_idx; i < nout; ++i) {
        int j = idx_offset + i;

        if (j < nargs) {
            output_wrap[i] = _get_out_wrap(PyTuple_GET_ITEM(obj, j),
                                           wrap);
        }
        else {
            output_wrap[i] = wrap;
            Py_XINCREF(wrap);
        }
    }

    Py_XDECREF(wrap);
    return;
}


static PyObject *
ufunc_generic_call(PyUFuncObject *ufunc, PyObject *args, PyObject *kwds)
{
    int i;
    PyTupleObject *ret;
    PyArrayObject *mps[NPY_MAXARGS];
    PyObject *retobj[NPY_MAXARGS];
    PyObject *wraparr[NPY_MAXARGS];
    PyObject *res;
    PyObject *override = NULL;
    int errval;

    /*
     * Initialize all array objects to NULL to make cleanup easier
     * if something goes wrong.
     */
    for (i = 0; i < ufunc->nargs; i++) {
        mps[i] = NULL;
    }

    errval = PyUFunc_CheckOverride(ufunc, "__call__", args, kwds, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        for (i = 0; i < ufunc->nargs; i++) {
            PyArray_DiscardWritebackIfCopy(mps[i]);
            Py_XDECREF(mps[i]);
        }
        return override;
    }

    errval = PyUFunc_GenericFunction(ufunc, args, kwds, mps);
    if (errval < 0) {
        for (i = 0; i < ufunc->nargs; i++) {
            PyArray_DiscardWritebackIfCopy(mps[i]);
            Py_XDECREF(mps[i]);
        }
        if (errval == -1) {
            return NULL;
        }
        else if (ufunc->nin == 2 && ufunc->nout == 1) {
            /*
             * For array_richcompare's benefit -- see the long comment in
             * get_ufunc_arguments.
             */
            Py_INCREF(Py_NotImplemented);
            return Py_NotImplemented;
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "XX can't happen, please report a bug XX");
            return NULL;
        }
    }

    /* Free the input references */
    for (i = 0; i < ufunc->nin; i++) {
        Py_XDECREF(mps[i]);
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
    _find_array_wrap(args, kwds, wraparr, ufunc->nin, ufunc->nout);

    /* wrap outputs */
    for (i = 0; i < ufunc->nout; i++) {
        int j = ufunc->nin+i;
        PyObject *wrap = wraparr[i];

        if (wrap != NULL) {
            if (wrap == Py_None) {
                Py_DECREF(wrap);
                retobj[i] = (PyObject *)mps[j];
                continue;
            }
            res = PyObject_CallFunction(wrap, "O(OOi)", mps[j], ufunc, args, i);
            /* Handle __array_wrap__ that does not accept a context argument */
            if (res == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                res = PyObject_CallFunctionObjArgs(wrap, mps[j], NULL);
            }
            Py_DECREF(wrap);
            if (res == NULL) {
                goto fail;
            }
            else {
                Py_DECREF(mps[j]);
                retobj[i] = res;
                continue;
            }
        }
        else {
            /* default behavior */
            retobj[i] = PyArray_Return(mps[j]);
        }

    }

    if (ufunc->nout == 1) {
        return retobj[0];
    }
    else {
        ret = (PyTupleObject *)PyTuple_New(ufunc->nout);
        for (i = 0; i < ufunc->nout; i++) {
            PyTuple_SET_ITEM(ret, i, retobj[i]);
        }
        return (PyObject *)ret;
    }

fail:
    for (i = ufunc->nin; i < ufunc->nargs; i++) {
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
    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    res = PyDict_GetItem(thedict, npy_um_str_pyvals_name);
    if (res != NULL) {
        Py_INCREF(res);
        return res;
    }
    /* Construct list of defaults */
    res = PyList_New(3);
    if (res == NULL) {
        return NULL;
    }
    PyList_SET_ITEM(res, 0, PyInt_FromLong(NPY_BUFSIZE));
    PyList_SET_ITEM(res, 1, PyInt_FromLong(UFUNC_ERR_DEFAULT));
    PyList_SET_ITEM(res, 2, Py_None); Py_INCREF(Py_None);
    return res;
}

NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *thedict;
    int res;
    PyObject *val;
    static char *msg = "Error object must be a list of length 3";

    if (!PyArg_ParseTuple(args, "O:seterrobj", &val)) {
        return NULL;
    }
    if (!PyList_CheckExact(val) || PyList_GET_SIZE(val) != 3) {
        PyErr_SetString(PyExc_ValueError, msg);
        return NULL;
    }
    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    res = PyDict_SetItem(thedict, npy_um_str_pyvals_name, val);
    if (res < 0) {
        return NULL;
    }
#if USE_USE_DEFAULTS==1
    if (ufunc_update_use_defaults() < 0) {
        return NULL;
    }
#endif
    Py_RETURN_NONE;
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
                        const char *name, const char *doc, int unused)
{
    return PyUFunc_FromFuncAndDataAndSignature(func, data, types, ntypes,
        nin, nout, identity, name, doc, 0, NULL);
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     int unused, const char *signature)
{
    PyUFuncObject *ufunc;

    if (nin + nout > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Cannot construct a ufunc with more than %d operands "
                     "(requested number were: inputs = %d and outputs = %d)",
                     NPY_MAXARGS, nin, nout);
        return NULL;
    }

    ufunc = PyArray_malloc(sizeof(PyUFuncObject));
    if (ufunc == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)ufunc, &PyUFunc_Type);

    ufunc->reserved1 = 0;
    ufunc->reserved2 = NULL;

    ufunc->nin = nin;
    ufunc->nout = nout;
    ufunc->nargs = nin+nout;
    ufunc->identity = identity;

    ufunc->functions = func;
    ufunc->data = data;
    ufunc->types = types;
    ufunc->ntypes = ntypes;
    ufunc->ptr = NULL;
    ufunc->obj = NULL;
    ufunc->userloops=NULL;

    /* Type resolution and inner loop selection functions */
    ufunc->type_resolver = &PyUFunc_DefaultTypeResolver;
    ufunc->legacy_inner_loop_selector = &PyUFunc_DefaultLegacyInnerLoopSelector;
    ufunc->masked_inner_loop_selector = &PyUFunc_DefaultMaskedInnerLoopSelector;

    if (name == NULL) {
        ufunc->name = "?";
    }
    else {
        ufunc->name = name;
    }
    ufunc->doc = doc;

    ufunc->op_flags = PyArray_malloc(sizeof(npy_uint32)*ufunc->nargs);
    if (ufunc->op_flags == NULL) {
        return PyErr_NoMemory();
    }
    memset(ufunc->op_flags, 0, sizeof(npy_uint32)*ufunc->nargs);

    ufunc->iter_flags = 0;

    /* generalized ufunc */
    ufunc->core_enabled = 0;
    ufunc->core_num_dim_ix = 0;
    ufunc->core_num_dims = NULL;
    ufunc->core_dim_ixs = NULL;
    ufunc->core_offsets = NULL;
    ufunc->core_signature = NULL;
    if (signature != NULL) {
        if (_parse_signature(ufunc, signature) != 0) {
            Py_DECREF(ufunc);
            return NULL;
        }
    }
    return (PyObject *)ufunc;
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

/*
 * Return 1 if the given data pointer for the loop specifies that it needs the
 * arrays as the data pointer.
 *
 * NOTE: This is easier to specify with the type_resolver
 *       in the ufunc object.
 *
 * TODO: Remove this, since this is already basically broken
 *       with the addition of the masked inner loops and
 *       not worth fixing since the new loop selection functions
 *       have access to the full dtypes and can dynamically allocate
 *       arbitrary auxiliary data.
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
    int i;

    while (data != NULL) {
        PyUFunc_Loop1d *next = data->next;
        PyArray_free(data->arg_types);

        if (data->arg_dtypes != NULL) {
            for (i = 0; i < data->nargs; i++) {
                Py_DECREF(data->arg_dtypes[i]);
            }
            PyArray_free(data->arg_dtypes);
        }

        PyArray_free(data);
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


/*
 * This function allows the user to register a 1-d loop with an already
 * created ufunc. This function is similar to RegisterLoopForType except
 * that it allows a 1-d loop to be registered with PyArray_Descr objects
 * instead of dtype type num values. This allows a 1-d loop to be registered
 * for a structured array dtype or a custom dtype. The ufunc is called
 * whenever any of it's input arguments match the user_dtype argument.
 * ufunc - ufunc object created from call to PyUFunc_FromFuncAndData
 * user_dtype - dtype that ufunc will be registered with
 * function - 1-d loop function pointer
 * arg_dtypes - array of dtype objects describing the ufunc operands
 * data - arbitrary data pointer passed in to loop function
 */
/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_RegisterLoopForDescr(PyUFuncObject *ufunc,
                            PyArray_Descr *user_dtype,
                            PyUFuncGenericFunction function,
                            PyArray_Descr **arg_dtypes,
                            void *data)
{
    int i;
    int result = 0;
    int *arg_typenums;
    PyObject *key, *cobj;

    if (user_dtype == NULL) {
        PyErr_SetString(PyExc_TypeError,
            "unknown user defined struct dtype");
        return -1;
    }

    key = PyInt_FromLong((long) user_dtype->type_num);
    if (key == NULL) {
        return -1;
    }

    arg_typenums = PyArray_malloc(ufunc->nargs * sizeof(int));
    if (arg_typenums == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    if (arg_dtypes != NULL) {
        for (i = 0; i < ufunc->nargs; i++) {
            arg_typenums[i] = arg_dtypes[i]->type_num;
        }
    }
    else {
        for (i = 0; i < ufunc->nargs; i++) {
            arg_typenums[i] = user_dtype->type_num;
        }
    }

    result = PyUFunc_RegisterLoopForType(ufunc, user_dtype->type_num,
        function, arg_typenums, data);

    if (result == 0) {
        cobj = PyDict_GetItem(ufunc->userloops, key);
        if (cobj == NULL) {
            PyErr_SetString(PyExc_KeyError,
                "userloop for user dtype not found");
            result = -1;
        }
        else {
            PyUFunc_Loop1d *current;
            int cmp = 1;
            current = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(cobj);
            while (current != NULL) {
                cmp = cmp_arg_types(current->arg_types,
                    arg_typenums, ufunc->nargs);
                if (cmp >= 0 && current->arg_dtypes == NULL) {
                    break;
                }
                current = current->next;
            }
            if (cmp == 0 && current->arg_dtypes == NULL) {
                current->arg_dtypes = PyArray_malloc(ufunc->nargs *
                    sizeof(PyArray_Descr*));
                if (arg_dtypes != NULL) {
                    for (i = 0; i < ufunc->nargs; i++) {
                        current->arg_dtypes[i] = arg_dtypes[i];
                        Py_INCREF(current->arg_dtypes[i]);
                    }
                }
                else {
                    for (i = 0; i < ufunc->nargs; i++) {
                        current->arg_dtypes[i] = user_dtype;
                        Py_INCREF(current->arg_dtypes[i]);
                    }
                }
                current->nargs = ufunc->nargs;
            }
            else {
                result = -1;
            }
        }
    }

    PyArray_free(arg_typenums);

    Py_DECREF(key);

    return result;
}

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
    if ((usertype < NPY_USERDEF && usertype != NPY_VOID) || (descr==NULL)) {
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
    funcdata = PyArray_malloc(sizeof(PyUFunc_Loop1d));
    if (funcdata == NULL) {
        goto fail;
    }
    newtypes = PyArray_malloc(sizeof(int)*ufunc->nargs);
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
    funcdata->arg_dtypes = NULL;
    funcdata->nargs = 0;

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
            PyArray_free(newtypes);
            PyArray_free(funcdata);
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
    PyArray_free(funcdata);
    PyArray_free(newtypes);
    if (!PyErr_Occurred()) PyErr_NoMemory();
    return -1;
}

#undef _SETCPTR


static void
ufunc_dealloc(PyUFuncObject *ufunc)
{
    PyArray_free(ufunc->core_num_dims);
    PyArray_free(ufunc->core_dim_ixs);
    PyArray_free(ufunc->core_offsets);
    PyArray_free(ufunc->core_signature);
    PyArray_free(ufunc->ptr);
    PyArray_free(ufunc->op_flags);
    Py_XDECREF(ufunc->userloops);
    Py_XDECREF(ufunc->obj);
    PyArray_free(ufunc);
}

static PyObject *
ufunc_repr(PyUFuncObject *ufunc)
{
    return PyUString_FromFormat("<ufunc '%s'>", ufunc->name);
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
ufunc_outer(PyUFuncObject *ufunc, PyObject *args, PyObject *kwds)
{
    int i;
    int errval;
    PyObject *override = NULL;
    PyObject *ret;
    PyArrayObject *ap1 = NULL, *ap2 = NULL, *ap_new = NULL;
    PyObject *new_args, *tmp;
    PyObject *shape1, *shape2, *newshape;

    errval = PyUFunc_CheckOverride(ufunc, "outer", args, kwds, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }

    if (ufunc->core_enabled) {
        PyErr_Format(PyExc_TypeError,
                     "method outer is not allowed in ufunc with non-trivial"\
                     " signature");
        return NULL;
    }

    if (ufunc->nin != 2) {
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
    ap1 = (PyArrayObject *) PyArray_FromObject(tmp, NPY_NOTYPE, 0, 0);
    Py_DECREF(tmp);
    if (ap1 == NULL) {
        return NULL;
    }
    tmp = PySequence_GetItem(args, 1);
    if (tmp == NULL) {
        return NULL;
    }
    ap2 = (PyArrayObject *)PyArray_FromObject(tmp, NPY_NOTYPE, 0, 0);
    Py_DECREF(tmp);
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }
    /* Construct new shape tuple */
    shape1 = PyTuple_New(PyArray_NDIM(ap1));
    if (shape1 == NULL) {
        goto fail;
    }
    for (i = 0; i < PyArray_NDIM(ap1); i++) {
        PyTuple_SET_ITEM(shape1, i,
                PyLong_FromLongLong((npy_longlong)PyArray_DIMS(ap1)[i]));
    }
    shape2 = PyTuple_New(PyArray_NDIM(ap2));
    for (i = 0; i < PyArray_NDIM(ap2); i++) {
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
    ret = ufunc_generic_call(ufunc, new_args, kwds);
    Py_DECREF(new_args);
    return ret;

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(ap_new);
    return NULL;
}


static PyObject *
ufunc_reduce(PyUFuncObject *ufunc, PyObject *args, PyObject *kwds)
{
    int errval;
    PyObject *override = NULL;

    errval = PyUFunc_CheckOverride(ufunc, "reduce", args, kwds, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }
    return PyUFunc_GenericReduction(ufunc, args, kwds, UFUNC_REDUCE);
}

static PyObject *
ufunc_accumulate(PyUFuncObject *ufunc, PyObject *args, PyObject *kwds)
{
    int errval;
    PyObject *override = NULL;

    errval = PyUFunc_CheckOverride(ufunc, "accumulate", args, kwds, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }
    return PyUFunc_GenericReduction(ufunc, args, kwds, UFUNC_ACCUMULATE);
}

static PyObject *
ufunc_reduceat(PyUFuncObject *ufunc, PyObject *args, PyObject *kwds)
{
    int errval;
    PyObject *override = NULL;

    errval = PyUFunc_CheckOverride(ufunc, "reduceat", args, kwds, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }
    return PyUFunc_GenericReduction(ufunc, args, kwds, UFUNC_REDUCEAT);
}

/* Helper for ufunc_at, below */
static NPY_INLINE PyArrayObject *
new_array_op(PyArrayObject *op_array, char *data)
{
    npy_intp dims[1] = {1};
    PyObject *r = PyArray_NewFromDescr(&PyArray_Type, PyArray_DESCR(op_array),
                                       1, dims, NULL, data,
                                       NPY_ARRAY_WRITEABLE, NULL);
    return (PyArrayObject *)r;
}

/*
 * Call ufunc only on selected array items and store result in first operand.
 * For add ufunc, method call is equivalent to op1[idx] += op2 with no
 * buffering of the first operand.
 * Arguments:
 * op1 - First operand to ufunc
 * idx - Indices that are applied to first operand. Equivalent to op1[idx].
 * op2 - Second operand to ufunc (if needed). Must be able to broadcast
 *       over first operand.
 */
static PyObject *
ufunc_at(PyUFuncObject *ufunc, PyObject *args)
{
    PyObject *op1 = NULL;
    PyObject *idx = NULL;
    PyObject *op2 = NULL;
    PyArrayObject *op1_array = NULL;
    PyArrayObject *op2_array = NULL;
    PyArrayMapIterObject *iter = NULL;
    PyArrayIterObject *iter2 = NULL;
    PyArray_Descr *dtypes[3] = {NULL, NULL, NULL};
    PyArrayObject *operands[3] = {NULL, NULL, NULL};
    PyArrayObject *array_operands[3] = {NULL, NULL, NULL};

    int needs_api = 0;

    PyUFuncGenericFunction innerloop;
    void *innerloopdata;
    int i;
    int nop;

    /* override vars */
    int errval;
    PyObject *override = NULL;

    NpyIter *iter_buffer;
    NpyIter_IterNextFunc *iternext;
    npy_uint32 op_flags[NPY_MAXARGS];
    int buffersize;
    int errormask = 0;
    char * err_msg = NULL;
    NPY_BEGIN_THREADS_DEF;

    errval = PyUFunc_CheckOverride(ufunc, "at", args, NULL, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }

    if (ufunc->nin > 2) {
        PyErr_SetString(PyExc_ValueError,
            "Only unary and binary ufuncs supported at this time");
        return NULL;
    }

    if (ufunc->nout != 1) {
        PyErr_SetString(PyExc_ValueError,
            "Only single output ufuncs supported at this time");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "OO|O:at", &op1, &idx, &op2)) {
        return NULL;
    }

    if (ufunc->nin == 2 && op2 == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "second operand needed for ufunc");
        return NULL;
    }

    if (!PyArray_Check(op1)) {
        PyErr_SetString(PyExc_TypeError,
                        "first operand must be array");
        return NULL;
    }

    op1_array = (PyArrayObject *)op1;

    /* Create second operand from number array if needed. */
    if (op2 != NULL) {
        op2_array = (PyArrayObject *)PyArray_FromAny(op2, NULL,
                                0, 0, 0, NULL);
        if (op2_array == NULL) {
            goto fail;
        }
    }

    /* Create map iterator */
    iter = (PyArrayMapIterObject *)PyArray_MapIterArrayCopyIfOverlap(
        op1_array, idx, 1, op2_array);
    if (iter == NULL) {
        goto fail;
    }
    op1_array = iter->array;  /* May be updateifcopied on overlap */

    if (op2 != NULL) {
        /*
         * May need to swap axes so that second operand is
         * iterated over correctly
         */
        if ((iter->subspace != NULL) && (iter->consec)) {
            PyArray_MapIterSwapAxes(iter, &op2_array, 0);
            if (op2_array == NULL) {
                goto fail;
            }
        }

        /*
         * Create array iter object for second operand that
         * "matches" the map iter object for the first operand.
         * Then we can just iterate over the first and second
         * operands at the same time and not have to worry about
         * picking the correct elements from each operand to apply
         * the ufunc to.
         */
        if ((iter2 = (PyArrayIterObject *)\
             PyArray_BroadcastToShape((PyObject *)op2_array,
                                        iter->dimensions, iter->nd))==NULL) {
            goto fail;
        }
    }

    /*
     * Create dtypes array for either one or two input operands.
     * The output operand is set to the first input operand
     */
    dtypes[0] = PyArray_DESCR(op1_array);
    operands[0] = op1_array;
    if (op2_array != NULL) {
        dtypes[1] = PyArray_DESCR(op2_array);
        dtypes[2] = dtypes[0];
        operands[1] = op2_array;
        operands[2] = op1_array;
        nop = 3;
    }
    else {
        dtypes[1] = dtypes[0];
        dtypes[2] = NULL;
        operands[1] = op1_array;
        operands[2] = NULL;
        nop = 2;
    }

    if (ufunc->type_resolver(ufunc, NPY_UNSAFE_CASTING,
                            operands, NULL, dtypes) < 0) {
        goto fail;
    }
    if (ufunc->legacy_inner_loop_selector(ufunc, dtypes,
        &innerloop, &innerloopdata, &needs_api) < 0) {
        goto fail;
    }

    Py_INCREF(PyArray_DESCR(op1_array));
    array_operands[0] = new_array_op(op1_array, iter->dataptr);
    if (iter2 != NULL) {
        Py_INCREF(PyArray_DESCR(op2_array));
        array_operands[1] = new_array_op(op2_array, PyArray_ITER_DATA(iter2));
        Py_INCREF(PyArray_DESCR(op1_array));
        array_operands[2] = new_array_op(op1_array, iter->dataptr);
    }
    else {
        Py_INCREF(PyArray_DESCR(op1_array));
        array_operands[1] = new_array_op(op1_array, iter->dataptr);
        array_operands[2] = NULL;
    }

    /* Set up the flags */
    op_flags[0] = NPY_ITER_READONLY|
                  NPY_ITER_ALIGNED;

    if (iter2 != NULL) {
        op_flags[1] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
        op_flags[2] = NPY_ITER_WRITEONLY|
                      NPY_ITER_ALIGNED|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_NO_SUBTYPE;
    }
    else {
        op_flags[1] = NPY_ITER_WRITEONLY|
                      NPY_ITER_ALIGNED|
                      NPY_ITER_ALLOCATE|
                      NPY_ITER_NO_BROADCAST|
                      NPY_ITER_NO_SUBTYPE;
    }

    if (_get_bufsize_errmask(NULL, ufunc->name, &buffersize, &errormask) < 0) {
        goto fail;
    }

    /*
     * Create NpyIter object to "iterate" over single element of each input
     * operand. This is an easy way to reuse the NpyIter logic for dealing
     * with certain cases like casting operands to correct dtype. On each
     * iteration over the MapIterArray object created above, we'll take the
     * current data pointers from that and reset this NpyIter object using
     * those data pointers, and then trigger a buffer copy. The buffer data
     * pointers from the NpyIter object will then be passed to the inner loop
     * function.
     */
    iter_buffer = NpyIter_AdvancedNew(nop, array_operands,
                        NPY_ITER_EXTERNAL_LOOP|
                        NPY_ITER_REFS_OK|
                        NPY_ITER_ZEROSIZE_OK|
                        NPY_ITER_BUFFERED|
                        NPY_ITER_GROWINNER|
                        NPY_ITER_DELAY_BUFALLOC,
                        NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                        op_flags, dtypes,
                        -1, NULL, NULL, buffersize);

    if (iter_buffer == NULL) {
        goto fail;
    }

    needs_api = needs_api | NpyIter_IterationNeedsAPI(iter_buffer);

    iternext = NpyIter_GetIterNext(iter_buffer, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter_buffer);
        goto fail;
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    /*
     * Iterate over first and second operands and call ufunc
     * for each pair of inputs
     */
    i = iter->size;
    while (i > 0)
    {
        char *dataptr[3];
        char **buffer_dataptr;
        /* one element at a time, no stride required but read by innerloop */
        npy_intp count[3] = {1, 0xDEADBEEF, 0xDEADBEEF};
        npy_intp stride[3] = {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF};

        /*
         * Set up data pointers for either one or two input operands.
         * The output data pointer points to the first operand data.
         */
        dataptr[0] = iter->dataptr;
        if (iter2 != NULL) {
            dataptr[1] = PyArray_ITER_DATA(iter2);
            dataptr[2] = iter->dataptr;
        }
        else {
            dataptr[1] = iter->dataptr;
            dataptr[2] = NULL;
        }

        /* Reset NpyIter data pointers which will trigger a buffer copy */
        NpyIter_ResetBasePointers(iter_buffer, dataptr, &err_msg);
        if (err_msg) {
            break;
        }

        buffer_dataptr = NpyIter_GetDataPtrArray(iter_buffer);

        innerloop(buffer_dataptr, count, stride, innerloopdata);

        if (needs_api && PyErr_Occurred()) {
            break;
        }

        /*
         * Call to iternext triggers copy from buffer back to output array
         * after innerloop puts result in buffer.
         */
        iternext(iter_buffer);

        PyArray_MapIterNext(iter);
        if (iter2 != NULL) {
            PyArray_ITER_NEXT(iter2);
        }

        i--;
    }

    NPY_END_THREADS;

    if (err_msg) {
        PyErr_SetString(PyExc_ValueError, err_msg);
    }

    NpyIter_Deallocate(iter_buffer);

    if (op1_array != (PyArrayObject*)op1) {
        PyArray_ResolveWritebackIfCopy(op1_array);
    }
    Py_XDECREF(op2_array);
    Py_XDECREF(iter);
    Py_XDECREF(iter2);
    Py_XDECREF(array_operands[0]);
    Py_XDECREF(array_operands[1]);
    Py_XDECREF(array_operands[2]);

    if (needs_api && PyErr_Occurred()) {
        return NULL;
    }
    else {
        Py_RETURN_NONE;
    }

fail:

    if (op1_array != (PyArrayObject*)op1) {
        PyArray_ResolveWritebackIfCopy(op1_array);
    }
    Py_XDECREF(op2_array);
    Py_XDECREF(iter);
    Py_XDECREF(iter2);
    Py_XDECREF(array_operands[0]);
    Py_XDECREF(array_operands[1]);
    Py_XDECREF(array_operands[2]);

    return NULL;
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
    {"at",
        (PyCFunction)ufunc_at,
        METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}           /* sentinel */
};


/******************************************************************************
 ***                           UFUNC GETSET                                 ***
 *****************************************************************************/


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
ufunc_get_doc(PyUFuncObject *ufunc)
{
    static PyObject *_sig_formatter;
    PyObject *doc;

    npy_cache_import(
        "numpy.core._internal",
        "_ufunc_doc_signature_formatter",
        &_sig_formatter);

    if (_sig_formatter == NULL) {
        return NULL;
    }

    /*
     * Put docstring first or FindMethod finds it... could so some
     * introspection on name and nin + nout to automate the first part
     * of it the doc string shouldn't need the calling convention
     */
    doc = PyObject_CallFunctionObjArgs(
        _sig_formatter, (PyObject *)ufunc, NULL);
    if (doc == NULL) {
        return NULL;
    }
    if (ufunc->doc != NULL) {
        PyUString_ConcatAndDel(&doc,
            PyUString_FromFormat("\n\n%s", ufunc->doc));
    }
    return doc;
}

static PyObject *
ufunc_get_nin(PyUFuncObject *ufunc)
{
    return PyInt_FromLong(ufunc->nin);
}

static PyObject *
ufunc_get_nout(PyUFuncObject *ufunc)
{
    return PyInt_FromLong(ufunc->nout);
}

static PyObject *
ufunc_get_nargs(PyUFuncObject *ufunc)
{
    return PyInt_FromLong(ufunc->nargs);
}

static PyObject *
ufunc_get_ntypes(PyUFuncObject *ufunc)
{
    return PyInt_FromLong(ufunc->ntypes);
}

static PyObject *
ufunc_get_types(PyUFuncObject *ufunc)
{
    /* return a list with types grouped input->output */
    PyObject *list;
    PyObject *str;
    int k, j, n, nt = ufunc->ntypes;
    int ni = ufunc->nin;
    int no = ufunc->nout;
    char *t;
    list = PyList_New(nt);
    if (list == NULL) {
        return NULL;
    }
    t = PyArray_malloc(no+ni+2);
    n = 0;
    for (k = 0; k < nt; k++) {
        for (j = 0; j<ni; j++) {
            t[j] = _typecharfromnum(ufunc->types[n]);
            n++;
        }
        t[ni] = '-';
        t[ni+1] = '>';
        for (j = 0; j < no; j++) {
            t[ni + 2 + j] = _typecharfromnum(ufunc->types[n]);
            n++;
        }
        str = PyUString_FromStringAndSize(t, no + ni + 2);
        PyList_SET_ITEM(list, k, str);
    }
    PyArray_free(t);
    return list;
}

static PyObject *
ufunc_get_name(PyUFuncObject *ufunc)
{
    return PyUString_FromString(ufunc->name);
}

static PyObject *
ufunc_get_identity(PyUFuncObject *ufunc)
{
    switch(ufunc->identity) {
    case PyUFunc_One:
        return PyInt_FromLong(1);
    case PyUFunc_Zero:
        return PyInt_FromLong(0);
    case PyUFunc_MinusOne:
        return PyInt_FromLong(-1);
    }
    Py_RETURN_NONE;
}

static PyObject *
ufunc_get_signature(PyUFuncObject *ufunc)
{
    if (!ufunc->core_enabled) {
        Py_RETURN_NONE;
    }
    return PyUString_FromString(ufunc->core_signature);
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
    (ternaryfunc)ufunc_generic_call,            /* tp_call */
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
    0,                                          /* tp_version_tag */
};

/* End of code for ufunc objects */
