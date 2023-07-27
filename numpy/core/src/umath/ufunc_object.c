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
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>

#include "npy_config.h"
#include "npy_pycompat.h"
#include "npy_argparse.h"

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/arrayscalars.h"
#include "lowlevel_strided_loops.h"
#include "ufunc_type_resolution.h"
#include "reduction.h"
#include "mem_overlap.h"
#include "npy_hashtable.h"

#include "ufunc_object.h"
#include "override.h"
#include "npy_import.h"
#include "extobj.h"

#include "arrayobject.h"
#include "common.h"
#include "ctors.h"
#include "dtypemeta.h"
#include "numpyos.h"
#include "dispatching.h"
#include "convert_datatype.h"
#include "legacy_array_method.h"
#include "abstractdtypes.h"

/* TODO: Only for `NpyIter_GetTransferFlags` until it is public */
#define NPY_ITERATOR_IMPLEMENTATION_CODE
#include "nditer_impl.h"

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

typedef struct {
    PyObject *in;   /* The input arguments to the ufunc, a tuple */
    PyObject *out;  /* The output arguments, a tuple. If no non-None outputs are
                       provided, then this is NULL. */
} ufunc_full_args;

/* C representation of the context argument to __array_wrap__ */
typedef struct {
    PyUFuncObject *ufunc;
    ufunc_full_args args;
    int out_i;
} _ufunc_context;

/* Get the arg tuple to pass in the context argument to __array_wrap__ and
 * __array_prepare__.
 *
 * Output arguments are only passed if at least one is non-None.
 */
static PyObject *
_get_wrap_prepare_args(ufunc_full_args full_args) {
    if (full_args.out == NULL) {
        Py_INCREF(full_args.in);
        return full_args.in;
    }
    else {
        return PySequence_Concat(full_args.in, full_args.out);
    }
}

/* ---------------------------------------------------------------- */

static PyObject *
prepare_input_arguments_for_outer(PyObject *args, PyUFuncObject *ufunc);

static int
resolve_descriptors(int nop,
        PyUFuncObject *ufunc, PyArrayMethodObject *ufuncimpl,
        PyArrayObject *operands[], PyArray_Descr *dtypes[],
        PyArray_DTypeMeta *signature[], NPY_CASTING casting);


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_getfperr(void)
{
    /*
     * non-clearing get was only added in 1.9 so this function always cleared
     * keep it so just in case third party code relied on the clearing
     */
    char param = 0;
    return npy_clear_floatstatus_barrier(&param);
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
    int retstatus;
    retstatus = npy_clear_floatstatus_barrier((char*)&retstatus);

    return PyUFunc_handlefperr(errmask, errobj, retstatus, first);
}


/* Checking the status flag clears it */
/*UFUNC_API*/
NPY_NO_EXPORT void
PyUFunc_clearfperr()
{
    char param = 0;
    npy_clear_floatstatus_barrier(&param);
}

/*
 * This function analyzes the input arguments and determines an appropriate
 * method (__array_prepare__ or __array_wrap__) function to call, taking it
 * from the input with the highest priority. Return NULL if no argument
 * defines the method.
 */
static PyObject*
_find_array_method(PyObject *args, PyObject *method_name)
{
    int i, n_methods;
    PyObject *obj;
    PyObject *with_method[NPY_MAXARGS], *methods[NPY_MAXARGS];
    PyObject *method = NULL;

    n_methods = 0;
    for (i = 0; i < PyTuple_GET_SIZE(args); i++) {
        obj = PyTuple_GET_ITEM(args, i);
        if (PyArray_CheckExact(obj) || PyArray_IsAnyScalar(obj)) {
            continue;
        }
        method = PyObject_GetAttr(obj, method_name);
        if (method) {
            if (PyCallable_Check(method)) {
                with_method[n_methods] = obj;
                methods[n_methods] = method;
                ++n_methods;
            }
            else {
                Py_DECREF(method);
                method = NULL;
            }
        }
        else {
            PyErr_Clear();
        }
    }
    if (n_methods > 0) {
        /* If we have some methods defined, find the one of highest priority */
        method = methods[0];
        if (n_methods > 1) {
            double maxpriority = PyArray_GetPriority(with_method[0],
                                                     NPY_PRIORITY);
            for (i = 1; i < n_methods; ++i) {
                double priority = PyArray_GetPriority(with_method[i],
                                                      NPY_PRIORITY);
                if (priority > maxpriority) {
                    maxpriority = priority;
                    Py_DECREF(method);
                    method = methods[i];
                }
                else {
                    Py_DECREF(methods[i]);
                }
            }
        }
    }
    return method;
}

/*
 * Returns an incref'ed pointer to the proper __array_prepare__/__array_wrap__
 * method for a ufunc output argument, given the output argument `obj`, and the
 * method chosen from the inputs `input_method`.
 */
static PyObject *
_get_output_array_method(PyObject *obj, PyObject *method,
                         PyObject *input_method) {
    if (obj != Py_None) {
        PyObject *ometh;

        if (PyArray_CheckExact(obj)) {
            /*
             * No need to wrap regular arrays - None signals to not call
             * wrap/prepare at all
             */
            Py_RETURN_NONE;
        }

        ometh = PyObject_GetAttr(obj, method);
        if (ometh == NULL) {
            PyErr_Clear();
        }
        else if (!PyCallable_Check(ometh)) {
            Py_DECREF(ometh);
        }
        else {
            /* Use the wrap/prepare method of the output if it's callable */
            return ometh;
        }
    }

    /* Fall back on the input's wrap/prepare */
    Py_XINCREF(input_method);
    return input_method;
}

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
_find_array_prepare(ufunc_full_args args,
                    PyObject **output_prep, int nout)
{
    int i;
    PyObject *prep;

    /*
     * Determine the prepping function given by the input arrays
     * (could be NULL).
     */
    prep = _find_array_method(args.in, npy_um_str_array_prepare);
    /*
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
    if (args.out == NULL) {
        for (i = 0; i < nout; i++) {
            Py_XINCREF(prep);
            output_prep[i] = prep;
        }
    }
    else {
        for (i = 0; i < nout; i++) {
            output_prep[i] = _get_output_array_method(
                PyTuple_GET_ITEM(args.out, i), npy_um_str_array_prepare, prep);
        }
    }
    Py_XDECREF(prep);
    return;
}

#define NPY_UFUNC_DEFAULT_INPUT_FLAGS \
    NPY_ITER_READONLY | \
    NPY_ITER_ALIGNED | \
    NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE

#define NPY_UFUNC_DEFAULT_OUTPUT_FLAGS \
    NPY_ITER_ALIGNED | \
    NPY_ITER_ALLOCATE | \
    NPY_ITER_NO_BROADCAST | \
    NPY_ITER_NO_SUBTYPE | \
    NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE

/* Called at module initialization to set the matmul ufunc output flags */
NPY_NO_EXPORT int
set_matmul_flags(PyObject *d)
{
    PyObject *matmul = _PyDict_GetItemStringWithError(d, "matmul");
    if (matmul == NULL) {
        return -1;
    }
    /*
     * The default output flag NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE allows
     * perfectly overlapping input and output (in-place operations). While
     * correct for the common mathematical operations, this assumption is
     * incorrect in the general case and specifically in the case of matmul.
     *
     * NPY_ITER_UPDATEIFCOPY is added by default in
     * PyUFunc_GeneralizedFunction, which is the variant called for gufuncs
     * with a signature
     *
     * Enabling NPY_ITER_WRITEONLY can prevent a copy in some cases.
     */
    ((PyUFuncObject *)matmul)->op_flags[2] = (NPY_ITER_WRITEONLY |
                                         NPY_ITER_UPDATEIFCOPY |
                                         NPY_UFUNC_DEFAULT_OUTPUT_FLAGS) &
                                         ~NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
    return 0;
}


/*
 * Set per-operand flags according to desired input or output flags.
 * op_flags[i] for i in input (as determined by ufunc->nin) will be
 * merged with op_in_flags, perhaps overriding per-operand flags set
 * in previous stages.
 * op_flags[i] for i in output will be set to op_out_flags only if previously
 * unset.
 * The input flag behavior preserves backward compatibility, while the
 * output flag behaviour is the "correct" one for maximum flexibility.
 */
NPY_NO_EXPORT void
_ufunc_setup_flags(PyUFuncObject *ufunc, npy_uint32 op_in_flags,
                   npy_uint32 op_out_flags, npy_uint32 *op_flags)
{
    int nin = ufunc->nin;
    int nout = ufunc->nout;
    int nop = nin + nout, i;
    /* Set up the flags */
    for (i = 0; i < nin; ++i) {
        op_flags[i] = ufunc->op_flags[i] | op_in_flags;
        /*
         * If READWRITE flag has been set for this operand,
         * then clear default READONLY flag
         */
        if (op_flags[i] & (NPY_ITER_READWRITE | NPY_ITER_WRITEONLY)) {
            op_flags[i] &= ~NPY_ITER_READONLY;
        }
    }
    for (i = nin; i < nop; ++i) {
        op_flags[i] = ufunc->op_flags[i] ? ufunc->op_flags[i] : op_out_flags;
    }
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
_find_array_wrap(ufunc_full_args args, npy_bool subok,
                 PyObject **output_wrap, int nin, int nout)
{
    int i;
    PyObject *wrap = NULL;

    /*
     * If a 'subok' parameter is passed and isn't True, don't wrap but put None
     * into slots with out arguments which means return the out argument
     */
    if (!subok) {
        goto handle_out;
    }

    /*
     * Determine the wrapping function given by the input arrays
     * (could be NULL).
     */
    wrap = _find_array_method(args.in, npy_um_str_array_wrap);

    /*
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
    if (args.out == NULL) {
        for (i = 0; i < nout; i++) {
            Py_XINCREF(wrap);
            output_wrap[i] = wrap;
        }
    }
    else {
        for (i = 0; i < nout; i++) {
            output_wrap[i] = _get_output_array_method(
                PyTuple_GET_ITEM(args.out, i), npy_um_str_array_wrap, wrap);
        }
    }

    Py_XDECREF(wrap);
}


/*
 * Apply the __array_wrap__ function with the given array and content.
 *
 * Interprets wrap=None and wrap=NULL as intended by _find_array_wrap
 *
 * Steals a reference to obj and wrap.
 * Pass context=NULL to indicate there is no context.
 */
static PyObject *
_apply_array_wrap(
            PyObject *wrap, PyArrayObject *obj, _ufunc_context const *context) {
    if (wrap == NULL) {
        /* default behavior */
        return PyArray_Return(obj);
    }
    else if (wrap == Py_None) {
        Py_DECREF(wrap);
        return (PyObject *)obj;
    }
    else {
        PyObject *res;
        PyObject *py_context = NULL;

        /* Convert the context object to a tuple, if present */
        if (context == NULL) {
            py_context = Py_None;
            Py_INCREF(py_context);
        }
        else {
            PyObject *args_tup;
            /* Call the method with appropriate context */
            args_tup = _get_wrap_prepare_args(context->args);
            if (args_tup == NULL) {
                goto fail;
            }
            py_context = Py_BuildValue("OOi",
                context->ufunc, args_tup, context->out_i);
            Py_DECREF(args_tup);
            if (py_context == NULL) {
                goto fail;
            }
        }
        /* try __array_wrap__(obj, context) */
        res = PyObject_CallFunctionObjArgs(wrap, obj, py_context, NULL);
        Py_DECREF(py_context);

        /* try __array_wrap__(obj) if the context argument is not accepted  */
        if (res == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            res = PyObject_CallFunctionObjArgs(wrap, obj, NULL);
        }
        Py_DECREF(wrap);
        Py_DECREF(obj);
        return res;
    fail:
        Py_DECREF(wrap);
        Py_DECREF(obj);
        return NULL;
    }
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
 * Convert a string into a number
 */
static npy_intp
_get_size(const char* str)
{
    char *stop;
    npy_longlong size = NumPyOS_strtoll(str, &stop, 10);

    if (stop == str || _is_alpha_underscore(*stop)) {
        /* not a well formed number */
        return -1;
    }
    if (size >= NPY_MAX_INTP || size <= NPY_MIN_INTP) {
        /* len(str) too long */
        return -1;
    }
    return size;
}

/*
 * Return the ending position of a variable name including optional modifier
 */
static int
_get_end_of_name(const char* str, int offset)
{
    int ret = offset;
    while (_is_alnum_underscore(str[ret])) {
        ret++;
    }
    if (str[ret] == '?') {
        ret ++;
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
 * Sets the following fields in the PyUFuncObject 'ufunc':
 *
 * Field             Type                     Array Length
 * core_enabled      int (effectively bool)   N/A
 * core_num_dim_ix   int                      N/A
 * core_dim_flags    npy_uint32 *             core_num_dim_ix
 * core_dim_sizes    npy_intp *               core_num_dim_ix
 * core_num_dims     int *                    nargs (i.e. nin+nout)
 * core_offsets      int *                    nargs
 * core_dim_ixs      int *                    sum(core_num_dims)
 * core_signature    char *                   strlen(signature) + 1
 *
 * The function assumes that the values that are arrays have not
 * been set already, and sets these pointers to memory allocated
 * with PyArray_malloc.  These are freed when the ufunc dealloc
 * method is called.
 *
 * Returns 0 unless an error occurred.
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
    ufunc->core_offsets = PyArray_malloc(sizeof(int) * ufunc->nargs);
    /* The next three items will be shrunk later */
    ufunc->core_dim_ixs = PyArray_malloc(sizeof(int) * len);
    ufunc->core_dim_sizes = PyArray_malloc(sizeof(npy_intp) * len);
    ufunc->core_dim_flags = PyArray_malloc(sizeof(npy_uint32) * len);

    if (ufunc->core_num_dims == NULL || ufunc->core_dim_ixs == NULL ||
        ufunc->core_offsets == NULL ||
        ufunc->core_dim_sizes == NULL ||
        ufunc->core_dim_flags == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    for (size_t j = 0; j < len; j++) {
        ufunc->core_dim_flags[j] = 0;
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
            int ix, i_end;
            npy_intp frozen_size;
            npy_bool can_ignore;

            if (signature[i] == '\0') {
                parse_error = "unexpected end of signature string";
                goto fail;
            }
            /*
             * Is this a variable or a fixed size dimension?
             */
            if (_is_alpha_underscore(signature[i])) {
                frozen_size = -1;
            }
            else {
                frozen_size = (npy_intp)_get_size(signature + i);
                if (frozen_size <= 0) {
                    parse_error = "expect dimension name or non-zero frozen size";
                    goto fail;
                }
            }
            /* Is this dimension flexible? */
            i_end = _get_end_of_name(signature, i);
            can_ignore = (i_end > 0 && signature[i_end - 1] == '?');
            /*
             * Determine whether we already saw this dimension name,
             * get its index, and set its properties
             */
            for(ix = 0; ix < ufunc->core_num_dim_ix; ix++) {
                if (frozen_size > 0 ?
                    frozen_size == ufunc->core_dim_sizes[ix] :
                    _is_same_name(signature + i, var_names[ix])) {
                    break;
                }
            }
            /*
             * If a new dimension, store its properties; if old, check consistency.
             */
            if (ix == ufunc->core_num_dim_ix) {
                ufunc->core_num_dim_ix++;
                var_names[ix] = signature + i;
                ufunc->core_dim_sizes[ix] = frozen_size;
                if (frozen_size < 0) {
                    ufunc->core_dim_flags[ix] |= UFUNC_CORE_DIM_SIZE_INFERRED;
                }
                if (can_ignore) {
                    ufunc->core_dim_flags[ix] |= UFUNC_CORE_DIM_CAN_IGNORE;
                }
            } else {
                if (can_ignore && !(ufunc->core_dim_flags[ix] &
                                    UFUNC_CORE_DIM_CAN_IGNORE)) {
                    parse_error = "? cannot be used, name already seen without ?";
                    goto fail;
                }
                if (!can_ignore && (ufunc->core_dim_flags[ix] &
                                    UFUNC_CORE_DIM_CAN_IGNORE)) {
                    parse_error = "? must be used, name already seen with ?";
                    goto fail;
                }
            }
            ufunc->core_dim_ixs[cur_core_dim] = ix;
            cur_core_dim++;
            nd++;
            i = _next_non_white_space(signature, i_end);
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
            sizeof(int) * cur_core_dim);
    ufunc->core_dim_sizes = PyArray_realloc(
            ufunc->core_dim_sizes,
            sizeof(npy_intp) * ufunc->core_num_dim_ix);
    ufunc->core_dim_flags = PyArray_realloc(
            ufunc->core_dim_flags,
            sizeof(npy_uint32) * ufunc->core_num_dim_ix);

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
    if (PyArray_Check(obj)) {
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
 * Converters for use in parsing of keywords arguments.
 */
static int
_subok_converter(PyObject *obj, npy_bool *subok)
{
    if (PyBool_Check(obj)) {
        *subok = (obj == Py_True);
        return NPY_SUCCEED;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "'subok' must be a boolean");
        return NPY_FAIL;
    }
}

static int
_keepdims_converter(PyObject *obj, int *keepdims)
{
    if (PyBool_Check(obj)) {
        *keepdims = (obj == Py_True);
        return NPY_SUCCEED;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "'keepdims' must be a boolean");
        return NPY_FAIL;
    }
}

static int
_wheremask_converter(PyObject *obj, PyArrayObject **wheremask)
{
    /*
     * Optimization: where=True is the same as no where argument.
     * This lets us document True as the default.
     */
    if (obj == Py_True) {
        return NPY_SUCCEED;
    }
    else {
        PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
        if (dtype == NULL) {
            return NPY_FAIL;
        }
        /* PyArray_FromAny steals reference to dtype, even on failure */
        *wheremask = (PyArrayObject *)PyArray_FromAny(obj, dtype, 0, 0, 0, NULL);
        if ((*wheremask) == NULL) {
            return NPY_FAIL;
        }
        return NPY_SUCCEED;
    }
}


/*
 * Due to the array override, do the actual parameter conversion
 * only in this step. This function takes the reference objects and
 * parses them into the desired values.
 * This function cleans up after itself and NULLs references on error,
 * however, the caller has to ensure that `out_op[0:nargs]` and `out_whermeask`
 * are NULL initialized.
 */
static int
convert_ufunc_arguments(PyUFuncObject *ufunc,
        ufunc_full_args full_args, PyArrayObject *out_op[],
        PyArray_DTypeMeta *out_op_DTypes[],
        npy_bool *force_legacy_promotion, npy_bool *allow_legacy_promotion,
        npy_bool *promoting_pyscalars,
        PyObject *order_obj, NPY_ORDER *out_order,
        PyObject *casting_obj, NPY_CASTING *out_casting,
        PyObject *subok_obj, npy_bool *out_subok,
        PyObject *where_obj, PyArrayObject **out_wheremask, /* PyArray of bool */
        PyObject *keepdims_obj, int *out_keepdims)
{
    int nin = ufunc->nin;
    int nout = ufunc->nout;
    int nop = ufunc->nargs;
    PyObject *obj;

    /* Convert and fill in input arguments */
    npy_bool all_scalar = NPY_TRUE;
    npy_bool any_scalar = NPY_FALSE;
    *allow_legacy_promotion = NPY_TRUE;
    *force_legacy_promotion = NPY_FALSE;
    *promoting_pyscalars = NPY_FALSE;
    for (int i = 0; i < nin; i++) {
        obj = PyTuple_GET_ITEM(full_args.in, i);

        if (PyArray_Check(obj)) {
            out_op[i] = (PyArrayObject *)obj;
            Py_INCREF(out_op[i]);
        }
        else {
            /* Convert the input to an array and check for special cases */
            out_op[i] = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
            if (out_op[i] == NULL) {
                goto fail;
            }
        }
        out_op_DTypes[i] = NPY_DTYPE(PyArray_DESCR(out_op[i]));
        Py_INCREF(out_op_DTypes[i]);

        if (!NPY_DT_is_legacy(out_op_DTypes[i])) {
            *allow_legacy_promotion = NPY_FALSE;
            // TODO: A subclass of int, float, complex could reach here and
            //       it should not be flagged as "weak" if it does.
        }
        if (PyArray_NDIM(out_op[i]) == 0) {
            any_scalar = NPY_TRUE;
        }
        else {
            all_scalar = NPY_FALSE;
            continue;
        }

        // TODO: Is this equivalent/better by removing the logic which enforces
        //       that we always use weak promotion in the core?
        if (npy_promotion_state == NPY_USE_LEGACY_PROMOTION) {
            continue;  /* Skip use of special dtypes */
        }

        /*
         * Handle the "weak" Python scalars/literals.  We use a special DType
         * for these.
         * Further, we mark the operation array with a special flag to indicate
         * this.  This is because the legacy dtype resolution makes use of
         * `np.can_cast(operand, dtype)`.  The flag is local to this use, but
         * necessary to propagate the information to the legacy type resolution.
         */
        if (npy_mark_tmp_array_if_pyscalar(obj, out_op[i], &out_op_DTypes[i])) {
            if (PyArray_FLAGS(out_op[i]) & NPY_ARRAY_WAS_PYTHON_INT
                    && PyArray_TYPE(out_op[i]) != NPY_LONG) {
                /*
                 * When `np.array(integer)` is not the default integer (mainly
                 * object dtype), this confuses many type resolvers.  Simply
                 * forcing a default integer array is unfortunately easiest.
                 * In this disables the optional NEP 50 warnings, but in
                 * practice when this happens we should _usually_ pick the
                 * default integer loop and that raises an error.
                 * (An exception is `float64(1.) + 10**100` which silently
                 * will give a float64 result rather than a Python float.)
                 *
                 * TODO: Just like the general dual NEP 50/legacy promotion
                 * support this is meant as a temporary hack for NumPy 1.25.
                 */
                static PyArrayObject *zero_arr = NULL;
                if (NPY_UNLIKELY(zero_arr == NULL)) {
                    zero_arr = (PyArrayObject *)PyArray_ZEROS(
                            0, NULL, NPY_LONG, NPY_FALSE);
                    if (zero_arr == NULL) {
                        goto fail;
                    }
                    ((PyArrayObject_fields *)zero_arr)->flags |= (
                        NPY_ARRAY_WAS_PYTHON_INT|NPY_ARRAY_WAS_INT_AND_REPLACED);
                }
                Py_INCREF(zero_arr);
                Py_SETREF(out_op[i], zero_arr);
            }
            *promoting_pyscalars = NPY_TRUE;
        }
    }
    if (*allow_legacy_promotion && (!all_scalar && any_scalar)) {
        *force_legacy_promotion = should_use_min_scalar(nin, out_op, 0, NULL);
    }

    /* Convert and fill in output arguments */
    memset(out_op_DTypes + nin, 0, nout * sizeof(*out_op_DTypes));
    if (full_args.out != NULL) {
        for (int i = 0; i < nout; i++) {
            obj = PyTuple_GET_ITEM(full_args.out, i);
            if (_set_out_array(obj, out_op + i + nin) < 0) {
                goto fail;
            }
            if (out_op[i] != NULL) {
                out_op_DTypes[i + nin] = NPY_DTYPE(PyArray_DESCR(out_op[i]));
                Py_INCREF(out_op_DTypes[i + nin]);
            }
        }
    }

    /*
     * Convert most arguments manually here, since it is easier to handle
     * the ufunc override if we first parse only to objects.
     */
    if (where_obj && !_wheremask_converter(where_obj, out_wheremask)) {
        goto fail;
    }
    if (keepdims_obj && !_keepdims_converter(keepdims_obj, out_keepdims)) {
        goto fail;
    }
    if (casting_obj && !PyArray_CastingConverter(casting_obj, out_casting)) {
        goto fail;
    }
    if (order_obj && !PyArray_OrderConverter(order_obj, out_order)) {
        goto fail;
    }
    if (subok_obj && !_subok_converter(subok_obj, out_subok)) {
        goto fail;
    }
    return 0;

fail:
    if (out_wheremask != NULL) {
        Py_XSETREF(*out_wheremask, NULL);
    }
    for (int i = 0; i < nop; i++) {
        Py_XSETREF(out_op[i], NULL);
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
check_for_trivial_loop(PyArrayMethodObject *ufuncimpl,
        PyArrayObject **op, PyArray_Descr **dtypes,
        NPY_CASTING casting, npy_intp buffersize)
{
    int force_cast_input = ufuncimpl->flags & _NPY_METH_FORCE_CAST_INPUTS;
    int i, nin = ufuncimpl->nin, nop = nin + ufuncimpl->nout;

    for (i = 0; i < nop; ++i) {
        /*
         * If the dtype doesn't match, or the array isn't aligned,
         * indicate that the trivial loop can't be done.
         */
        if (op[i] == NULL) {
            continue;
        }
        int must_copy = !PyArray_ISALIGNED(op[i]);

        if (dtypes[i] != PyArray_DESCR(op[i])) {
            npy_intp view_offset;
            NPY_CASTING safety = PyArray_GetCastInfo(
                    PyArray_DESCR(op[i]), dtypes[i], NULL, &view_offset);
            if (safety < 0 && PyErr_Occurred()) {
                /* A proper error during a cast check, should be rare */
                return -1;
            }
            if (view_offset != 0) {
                /* NOTE: Could possibly implement non-zero view offsets */
                must_copy = 1;
            }

            if (force_cast_input && i < nin) {
                /*
                 * ArrayMethod flagged to ignore casting (logical funcs
                 * can  force cast to bool)
                 */
            }
            else if (PyArray_MinCastSafety(safety, casting) != casting) {
                return 0;  /* the cast is not safe enough */
            }
        }
        if (must_copy) {
            /*
             * If op[j] is a scalar or small one dimensional
             * array input, make a copy to keep the opportunity
             * for a trivial loop.  Outputs are not copied here.
             */
            if (i < nin && (PyArray_NDIM(op[i]) == 0
                            || (PyArray_NDIM(op[i]) == 1
                                && PyArray_DIM(op[i], 0) <= buffersize))) {
                PyArrayObject *tmp;
                Py_INCREF(dtypes[i]);
                tmp = (PyArrayObject *)PyArray_CastToType(op[i], dtypes[i], 0);
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
                    ufunc_full_args full_args,
                    int i)
{
    if (arr_prep != NULL && arr_prep != Py_None) {
        PyObject *res;
        PyArrayObject *arr;
        PyObject *args_tup;

        /* Call with the context argument */
        args_tup = _get_wrap_prepare_args(full_args);
        if (args_tup == NULL) {
            return -1;
        }
        res = PyObject_CallFunction(
            arr_prep, "O(OOi)", *op, ufunc, args_tup, i);
        Py_DECREF(args_tup);

        if (res == NULL) {
            return -1;
        }
        else if (!PyArray_Check(res)) {
            PyErr_SetString(PyExc_TypeError,
                    "__array_prepare__ must return an "
                    "ndarray or subclass thereof");
            Py_DECREF(res);
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


/*
 * Check whether a trivial loop is possible and call the innerloop if it is.
 * A trivial loop is defined as one where a single strided inner-loop call
 * is possible.
 *
 * This function only supports a single output (due to the overlap check).
 * It always accepts 0-D arrays and will broadcast them.  The function
 * cannot broadcast any other array (as it requires a single stride).
 * The function accepts all 1-D arrays, and N-D arrays that are either all
 * C- or all F-contiguous.
 * NOTE: Broadcast outputs are implicitly rejected in the overlap detection.
 *
 * Returns -2 if a trivial loop is not possible, 0 on success and -1 on error.
 */
static int
try_trivial_single_output_loop(PyArrayMethod_Context *context,
        PyArrayObject *op[], NPY_ORDER order,
        PyObject *arr_prep[], ufunc_full_args full_args,
        int errormask, PyObject *extobj)
{
    int nin = context->method->nin;
    int nop = nin + 1;
    assert(context->method->nout == 1);

    /* The order of all N-D contiguous operands, can be fixed by `order` */
    int operation_order = 0;
    if (order == NPY_CORDER) {
        operation_order = NPY_ARRAY_C_CONTIGUOUS;
    }
    else if (order == NPY_FORTRANORDER) {
        operation_order = NPY_ARRAY_F_CONTIGUOUS;
    }

    int operation_ndim = 0;
    npy_intp *operation_shape = NULL;
    npy_intp fixed_strides[NPY_MAXARGS];

    for (int iop = 0; iop < nop; iop++) {
        if (op[iop] == NULL) {
            /* The out argument may be NULL (and only that one); fill later */
            assert(iop == nin);
            continue;
        }

        int op_ndim = PyArray_NDIM(op[iop]);

        /* Special case 0-D since we can handle broadcasting using a 0-stride */
        if (op_ndim == 0 && iop < nin) {
            fixed_strides[iop] = 0;
            continue;
        }

        /* First non 0-D op: fix dimensions, shape (order is fixed later) */
        if (operation_ndim == 0) {
            operation_ndim = op_ndim;
            operation_shape = PyArray_SHAPE(op[iop]);
        }
        else if (op_ndim != operation_ndim) {
            return -2;  /* dimension mismatch (except 0-d input ops) */
        }
        else if (!PyArray_CompareLists(
                operation_shape, PyArray_DIMS(op[iop]), op_ndim)) {
            return -2;  /* shape mismatch */
        }

        if (op_ndim == 1) {
            fixed_strides[iop] = PyArray_STRIDES(op[iop])[0];
        }
        else {
            fixed_strides[iop] = PyArray_ITEMSIZE(op[iop]);  /* contiguous */

            /* This op must match the operation order (and be contiguous) */
            int op_order = (PyArray_FLAGS(op[iop]) &
                            (NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_F_CONTIGUOUS));
            if (op_order == 0) {
                return -2;  /* N-dimensional op must be contiguous */
            }
            else if (operation_order == 0) {
                operation_order = op_order;  /* op fixes order */
            }
            else if (operation_order != op_order) {
                return -2;
            }
        }
    }

    if (op[nin] == NULL) {
        Py_INCREF(context->descriptors[nin]);
        op[nin] = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
                context->descriptors[nin], operation_ndim, operation_shape,
                NULL, NULL, operation_order==NPY_ARRAY_F_CONTIGUOUS, NULL);
        if (op[nin] == NULL) {
            return -1;
        }
        fixed_strides[nin] = context->descriptors[nin]->elsize;
    }
    else {
        /* If any input overlaps with the output, we use the full path. */
        for (int iop = 0; iop < nin; iop++) {
            if (!PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(
                    op[iop], op[nin],
                    PyArray_TRIVIALLY_ITERABLE_OP_READ,
                    PyArray_TRIVIALLY_ITERABLE_OP_NOREAD)) {
                return -2;
            }
        }
        /* Check self-overlap (non 1-D are contiguous, perfect overlap is OK) */
        if (operation_ndim == 1 &&
                PyArray_STRIDES(op[nin])[0] < PyArray_ITEMSIZE(op[nin]) &&
                PyArray_STRIDES(op[nin])[0] != 0) {
            return -2;
        }
    }

    /* Call the __prepare_array__ if necessary */
    if (prepare_ufunc_output((PyUFuncObject *)context->caller, &op[nin],
            arr_prep[0], full_args, 0) < 0) {
        return -1;
    }

    /*
     * We can use the trivial (single inner-loop call) optimization
     * and `fixed_strides` holds the strides for that call.
     */
    char *data[NPY_MAXARGS];
    npy_intp count = PyArray_MultiplyList(operation_shape, operation_ndim);
    if (count == 0) {
        /* Nothing to do */
        return 0;
    }
    NPY_BEGIN_THREADS_DEF;

    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata = NULL;
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (context->method->get_strided_loop(context,
            1, 0, fixed_strides,
            &strided_loop, &auxdata, &flags) < 0) {
        return -1;
    }
    for (int iop=0; iop < nop; iop++) {
        data[iop] = PyArray_BYTES(op[iop]);
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)context);
    }
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        NPY_BEGIN_THREADS_THRESHOLDED(count);
    }

    int res = strided_loop(context, data, &count, fixed_strides, auxdata);

    NPY_END_THREADS;
    NPY_AUXDATA_FREE(auxdata);
    /*
     * An error should only be possible if `res != 0` is already set.
     * But this is not strictly correct for old-style ufuncs (e.g. `power`
     * released the GIL but manually set an Exception).
     */
    if (PyErr_Occurred()) {
        res = -1;
    }

    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        const char *name = ufunc_get_name_cstr((PyUFuncObject *)context->caller);
        res = _check_ufunc_fperr(errormask, extobj, name);
    }
    return res;
}


/*
 * Check casting: It would be nice to just move this into the iterator
 * or pass in the full cast information.  But this can special case
 * the logical functions and prints a better error message.
 */
static inline int
validate_casting(PyArrayMethodObject *method, PyUFuncObject *ufunc,
        PyArrayObject *ops[], PyArray_Descr *descriptors[],
        NPY_CASTING casting)
{
    if (method->resolve_descriptors == &wrapped_legacy_resolve_descriptors) {
        /*
         * In this case the legacy type resolution was definitely called
         * and we do not need to check (astropy/pyerfa relied on this).
         */
        return 0;
    }
    if (method->flags & _NPY_METH_FORCE_CAST_INPUTS) {
        if (PyUFunc_ValidateOutCasting(ufunc, casting, ops, descriptors) < 0) {
            return -1;
        }
    }
    else {
        if (PyUFunc_ValidateCasting(ufunc, casting, ops, descriptors) < 0) {
            return -1;
        }
    }
    return 0;
}


/*
 * The ufunc loop implementation for both normal ufunc calls and masked calls
 * when the iterator has to be used.
 *
 * See `PyUFunc_GenericFunctionInternal` for more information (where this is
 * called from).
 */
static int
execute_ufunc_loop(PyArrayMethod_Context *context, int masked,
        PyArrayObject **op, NPY_ORDER order, npy_intp buffersize,
        NPY_CASTING casting,
        PyObject **arr_prep, ufunc_full_args full_args,
        npy_uint32 *op_flags, int errormask, PyObject *extobj)
{
    PyUFuncObject *ufunc = (PyUFuncObject *)context->caller;
    int nin = context->method->nin, nout = context->method->nout;
    int nop = nin + nout;

    if (validate_casting(context->method,
            ufunc, op, context->descriptors, casting) < 0) {
        return -1;
    }

    if (masked) {
        assert(PyArray_TYPE(op[nop]) == NPY_BOOL);
        if (ufunc->_always_null_previously_masked_innerloop_selector != NULL) {
            if (PyErr_WarnFormat(PyExc_UserWarning, 1,
                    "The ufunc %s has a custom masked-inner-loop-selector."
                    "NumPy assumes that this is NEVER used. If you do make "
                    "use of this please notify the NumPy developers to discuss "
                    "future solutions. (See NEP 41 and 43)\n"
                    "NumPy will continue, but ignore the custom loop selector. "
                    "This should only affect performance.",
                    ufunc_get_name_cstr(ufunc)) < 0) {
                return -1;
            }
        }

        /*
         * NOTE: In the masked version, we consider the output read-write,
         *       this gives a best-effort of preserving the input, but does
         *       not always work.  It could allow the operand to be copied
         *       due to copy-if-overlap, but only if it was passed in.
         *       In that case `__array_prepare__` is called before it happens.
         */
        for (int i = nin; i < nop; ++i) {
            op_flags[i] |= (op[i] != NULL ? NPY_ITER_READWRITE : NPY_ITER_WRITEONLY);
        }
        op_flags[nop] = NPY_ITER_READONLY | NPY_ITER_ARRAYMASK;  /* mask */
    }

    NPY_UF_DBG_PRINT("Making iterator\n");

    npy_uint32 iter_flags = ufunc->iter_flags |
                 NPY_ITER_EXTERNAL_LOOP |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_ZEROSIZE_OK |
                 NPY_ITER_BUFFERED |
                 NPY_ITER_GROWINNER |
                 NPY_ITER_DELAY_BUFALLOC |
                 NPY_ITER_COPY_IF_OVERLAP;

    /*
     * Call the __array_prepare__ functions for already existing output arrays.
     * Do this before creating the iterator, as the iterator may UPDATEIFCOPY
     * some of them.
     */
    for (int i = 0; i < nout; i++) {
        if (op[nin+i] == NULL) {
            continue;
        }
        if (prepare_ufunc_output(ufunc, &op[nin+i],
                arr_prep[i], full_args, i) < 0) {
            return -1;
        }
    }

    /*
     * Allocate the iterator.  Because the types of the inputs
     * were already checked, we use the casting rule 'unsafe' which
     * is faster to calculate.
     */
    NpyIter *iter = NpyIter_AdvancedNew(nop + masked, op,
                        iter_flags,
                        order, NPY_UNSAFE_CASTING,
                        op_flags, context->descriptors,
                        -1, NULL, NULL, buffersize);
    if (iter == NULL) {
        return -1;
    }

    NPY_UF_DBG_PRINT("Made iterator\n");

    /* Call the __array_prepare__ functions for newly allocated arrays */
    PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
    char *baseptrs[NPY_MAXARGS];

    for (int i = 0; i < nout; ++i) {
        if (op[nin + i] == NULL) {
            op[nin + i] = op_it[nin + i];
            Py_INCREF(op[nin + i]);

            /* Call the __array_prepare__ functions for the new array */
            if (prepare_ufunc_output(ufunc,
                    &op[nin + i], arr_prep[i], full_args, i) < 0) {
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
            baseptrs[nin + i] = PyArray_BYTES(op[nin + i]);
        }
        else {
            baseptrs[nin + i] = PyArray_BYTES(op_it[nin + i]);
        }
    }
    /* Only do the loop if the iteration size is non-zero */
    npy_intp full_size = NpyIter_GetIterSize(iter);
    if (full_size == 0) {
        if (!NpyIter_Deallocate(iter)) {
            return -1;
        }
        return 0;
    }

    /*
     * Reset the iterator with the base pointers possibly modified by
     * `__array_prepare__`.
     */
    for (int i = 0; i < nin; i++) {
        baseptrs[i] = PyArray_BYTES(op_it[i]);
    }
    if (masked) {
        baseptrs[nop] = PyArray_BYTES(op_it[nop]);
    }

    /*
     * Get the inner loop, with the possibility of specialization
     * based on the fixed strides.
     */
    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata;
    npy_intp fixed_strides[NPY_MAXARGS];

    NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (masked) {
        if (PyArrayMethod_GetMaskedStridedLoop(context,
                1, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
            NpyIter_Deallocate(iter);
            return -1;
        }
    }
    else {
        if (context->method->get_strided_loop(context,
                1, 0, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
            NpyIter_Deallocate(iter);
            return -1;
        }
    }

    /* Get the variables needed for the loop */
    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NPY_AUXDATA_FREE(auxdata);
        NpyIter_Deallocate(iter);
        return -1;
    }
    char **dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
    npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

    NPY_BEGIN_THREADS_DEF;

    flags = PyArrayMethod_COMBINED_FLAGS(flags, NpyIter_GetTransferFlags(iter));

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)context);
    }
    if (!(flags & NPY_METH_REQUIRES_PYAPI)) {
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);
    }

    /* The reset may copy the first buffer chunk, which could cause FPEs */
    if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
        NPY_AUXDATA_FREE(auxdata);
        NpyIter_Deallocate(iter);
        return -1;
    }

    NPY_UF_DBG_PRINT("Actual inner loop:\n");
    /* Execute the loop */
    int res;
    do {
        NPY_UF_DBG_PRINT1("iterator loop count %d\n", (int)*countptr);
        res = strided_loop(context, dataptr, countptr, strides, auxdata);
    } while (res == 0 && iternext(iter));

    NPY_END_THREADS;
    NPY_AUXDATA_FREE(auxdata);

    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        const char *name = ufunc_get_name_cstr((PyUFuncObject *)context->caller);
        res = _check_ufunc_fperr(errormask, extobj, name);
    }

    if (!NpyIter_Deallocate(iter)) {
        return -1;
    }
    return res;
}


/*
 * Validate that operands have enough dimensions, accounting for
 * possible flexible dimensions that may be absent.
 */
static int
_validate_num_dims(PyUFuncObject *ufunc, PyArrayObject **op,
                   npy_uint32 *core_dim_flags,
                   int *op_core_num_dims) {
    int i, j;
    int nin = ufunc->nin;
    int nop = ufunc->nargs;

    for (i = 0; i < nop; i++) {
        if (op[i] != NULL) {
            int op_ndim = PyArray_NDIM(op[i]);

            if (op_ndim < op_core_num_dims[i]) {
                int core_offset = ufunc->core_offsets[i];
                /* We've too few, but some dimensions might be flexible */
                for (j = core_offset;
                     j < core_offset + ufunc->core_num_dims[i]; j++) {
                    int core_dim_index = ufunc->core_dim_ixs[j];
                    if ((core_dim_flags[core_dim_index] &
                         UFUNC_CORE_DIM_CAN_IGNORE)) {
                        int i1, j1, k;
                        /*
                         * Found a dimension that can be ignored. Flag that
                         * it is missing, and unflag that it can be ignored,
                         * since we are doing so already.
                         */
                        core_dim_flags[core_dim_index] |= UFUNC_CORE_DIM_MISSING;
                        core_dim_flags[core_dim_index] ^= UFUNC_CORE_DIM_CAN_IGNORE;
                        /*
                         * Reduce the number of core dimensions for all
                         * operands that use this one (including ours),
                         * and check whether we're now OK.
                         */
                        for (i1 = 0, k=0; i1 < nop; i1++) {
                            for (j1 = 0; j1 < ufunc->core_num_dims[i1]; j1++) {
                                if (ufunc->core_dim_ixs[k++] == core_dim_index) {
                                    op_core_num_dims[i1]--;
                                }
                            }
                        }
                        if (op_ndim == op_core_num_dims[i]) {
                            break;
                        }
                    }
                }
                if (op_ndim < op_core_num_dims[i]) {
                    PyErr_Format(PyExc_ValueError,
                         "%s: %s operand %d does not have enough "
                         "dimensions (has %d, gufunc core with "
                         "signature %s requires %d)",
                         ufunc_get_name_cstr(ufunc),
                         i < nin ? "Input" : "Output",
                         i < nin ? i : i - nin, PyArray_NDIM(op[i]),
                         ufunc->core_signature, op_core_num_dims[i]);
                    return -1;
                }
            }
        }
    }
    return 0;
}

/*
 * Check whether any of the outputs of a gufunc has core dimensions.
 */
static int
_has_output_coredims(PyUFuncObject *ufunc) {
    int i;
    for (i = ufunc->nin; i < ufunc->nin + ufunc->nout; ++i) {
        if (ufunc->core_num_dims[i] > 0) {
            return 1;
        }
    }
    return 0;
}

/*
 * Check whether the gufunc can be used with axis, i.e., that there is only
 * a single, shared core dimension (which means that operands either have
 * that dimension, or have no core dimensions).  Returns 0 if all is fine,
 * and sets an error and returns -1 if not.
 */
static int
_check_axis_support(PyUFuncObject *ufunc) {
    if (ufunc->core_num_dim_ix != 1) {
        PyErr_Format(PyExc_TypeError,
                     "%s: axis can only be used with a single shared core "
                     "dimension, not with the %d distinct ones implied by "
                     "signature %s.",
                     ufunc_get_name_cstr(ufunc),
                     ufunc->core_num_dim_ix,
                     ufunc->core_signature);
        return -1;
    }
    return 0;
}

/*
 * Check whether the gufunc can be used with keepdims, i.e., that all its
 * input arguments have the same number of core dimension, and all output
 * arguments have no core dimensions. Returns 0 if all is fine, and sets
 * an error and returns -1 if not.
 */
static int
_check_keepdims_support(PyUFuncObject *ufunc) {
    int i;
    int nin = ufunc->nin, nout = ufunc->nout;
    int input_core_dims = ufunc->core_num_dims[0];
    for (i = 1; i < nin + nout; i++) {
        if (ufunc->core_num_dims[i] != (i < nin ? input_core_dims : 0)) {
            PyErr_Format(PyExc_TypeError,
                "%s does not support keepdims: its signature %s requires "
                "%s %d to have %d core dimensions, but keepdims can only "
                "be used when all inputs have the same number of core "
                "dimensions and all outputs have no core dimensions.",
                ufunc_get_name_cstr(ufunc),
                ufunc->core_signature,
                i < nin ? "input" : "output",
                i < nin ? i : i - nin,
                ufunc->core_num_dims[i]);
            return -1;
        }
    }
    return 0;
}

/*
 * Interpret a possible axes keyword argument, using it to fill the remap_axis
 * array which maps default to actual axes for each operand, indexed as
 * as remap_axis[iop][iaxis]. The default axis order has first all broadcast
 * axes and then the core axes the gufunc operates on.
 *
 * Returns 0 on success, and -1 on failure
 */
static int
_parse_axes_arg(PyUFuncObject *ufunc, int op_core_num_dims[], PyObject *axes,
                PyArrayObject **op, int broadcast_ndim, int **remap_axis) {
    static PyObject *AxisError_cls = NULL;

    int nin = ufunc->nin;
    int nop = ufunc->nargs;
    int iop, list_size;

    if (!PyList_Check(axes)) {
        PyErr_SetString(PyExc_TypeError, "axes should be a list.");
        return -1;
    }
    list_size = PyList_Size(axes);
    if (list_size != nop) {
        if (list_size != nin || _has_output_coredims(ufunc)) {
            PyErr_Format(PyExc_ValueError,
                         "axes should be a list with an entry for all "
                         "%d inputs and outputs; entries for outputs can only "
                         "be omitted if none of them has core axes.",
                         nop);
            return -1;
        }
        for (iop = nin; iop < nop; iop++) {
            remap_axis[iop] = NULL;
        }
    }
    for (iop = 0; iop < list_size; ++iop) {
        int op_ndim, op_ncore, op_nbroadcast;
        int have_seen_axis[NPY_MAXDIMS] = {0};
        PyObject *op_axes_tuple, *axis_item;
        int axis, op_axis;

        op_ncore = op_core_num_dims[iop];
        if (op[iop] != NULL) {
            op_ndim = PyArray_NDIM(op[iop]);
            op_nbroadcast = op_ndim - op_ncore;
        }
        else {
            op_nbroadcast = broadcast_ndim;
            op_ndim = broadcast_ndim + op_ncore;
        }
        /*
         * Get axes tuple for operand. If not a tuple already, make it one if
         * there is only one axis (its content is checked later).
         */
        op_axes_tuple = PyList_GET_ITEM(axes, iop);
        if (PyTuple_Check(op_axes_tuple)) {
            if (PyTuple_Size(op_axes_tuple) != op_ncore) {
                /* must have been a tuple with too many entries. */
                npy_cache_import(
                        "numpy.exceptions", "AxisError", &AxisError_cls);
                if (AxisError_cls == NULL) {
                    return -1;
                }
                PyErr_Format(AxisError_cls,
                        "%s: operand %d has %d core dimensions, "
                        "but %zd dimensions are specified by axes tuple.",
                        ufunc_get_name_cstr(ufunc), iop, op_ncore,
                        PyTuple_Size(op_axes_tuple));
                return -1;
            }
            Py_INCREF(op_axes_tuple);
        }
        else if (op_ncore == 1) {
            op_axes_tuple = PyTuple_Pack(1, op_axes_tuple);
            if (op_axes_tuple == NULL) {
                return -1;
            }
        }
        else {
            /* If input is not an integer tell user that a tuple is needed */
            if (error_converting(PyArray_PyIntAsInt(op_axes_tuple))) {
                PyErr_Format(PyExc_TypeError,
                        "%s: axes item %d should be a tuple.",
                        ufunc_get_name_cstr(ufunc), iop);
                return -1;
            }
            /* If it is a single integer, inform user that more are needed */
            npy_cache_import("numpy.exceptions", "AxisError", &AxisError_cls);
            if (AxisError_cls == NULL) {
                return -1;
            }
            PyErr_Format(AxisError_cls,
                    "%s: operand %d has %d core dimensions, "
                    "but the axes item is a single integer.",
                    ufunc_get_name_cstr(ufunc), iop, op_ncore);
            return -1;
        }
        /*
         * Now create the remap, starting with the core dimensions, and then
         * adding the remaining broadcast axes that are to be iterated over.
         */
        for (axis = op_nbroadcast; axis < op_ndim; axis++) {
            axis_item = PyTuple_GET_ITEM(op_axes_tuple, axis - op_nbroadcast);
            op_axis = PyArray_PyIntAsInt(axis_item);
            if (error_converting(op_axis) ||
                    (check_and_adjust_axis(&op_axis, op_ndim) < 0)) {
                Py_DECREF(op_axes_tuple);
                return -1;
            }
            if (have_seen_axis[op_axis]) {
                PyErr_Format(PyExc_ValueError,
                             "axes item %d has value %d repeated",
                             iop, op_axis);
                Py_DECREF(op_axes_tuple);
                return -1;
            }
            have_seen_axis[op_axis] = 1;
            remap_axis[iop][axis] = op_axis;
        }
        Py_DECREF(op_axes_tuple);
        /*
         * Fill the op_nbroadcast=op_ndim-op_ncore axes not yet set,
         * using have_seen_axis to skip over entries set above.
         */
        for (axis = 0, op_axis = 0; axis < op_nbroadcast; axis++) {
            while (have_seen_axis[op_axis]) {
                op_axis++;
            }
            remap_axis[iop][axis] = op_axis++;
        }
        /*
         * Check whether we are actually remapping anything. Here,
         * op_axis can only equal axis if all broadcast axes were the same
         * (i.e., the while loop above was never entered).
         */
        if (axis == op_axis) {
            while (axis < op_ndim && remap_axis[iop][axis] == axis) {
                axis++;
            }
        }
        if (axis == op_ndim) {
            remap_axis[iop] = NULL;
        }
    } /* end of for(iop) loop over operands */
    return 0;
}

/*
 * Simplified version of the above, using axis to fill the remap_axis
 * array, which maps default to actual axes for each operand, indexed as
 * as remap_axis[iop][iaxis]. The default axis order has first all broadcast
 * axes and then the core axes the gufunc operates on.
 *
 * Returns 0 on success, and -1 on failure
 */
static int
_parse_axis_arg(PyUFuncObject *ufunc, const int core_num_dims[], PyObject *axis,
                PyArrayObject **op, int broadcast_ndim, int **remap_axis) {
    int nop = ufunc->nargs;
    int iop, axis_int;

    axis_int = PyArray_PyIntAsInt(axis);
    if (error_converting(axis_int)) {
        return -1;
    }

    for (iop = 0; iop < nop; ++iop) {
        int axis, op_ndim, op_axis;

        /* _check_axis_support ensures core_num_dims is 0 or 1 */
        if (core_num_dims[iop] == 0) {
            remap_axis[iop] = NULL;
            continue;
        }
        if (op[iop]) {
            op_ndim = PyArray_NDIM(op[iop]);
        }
        else {
            op_ndim = broadcast_ndim + 1;
        }
        op_axis = axis_int;  /* ensure we don't modify axis_int */
        if (check_and_adjust_axis(&op_axis, op_ndim) < 0) {
            return -1;
        }
        /* Are we actually remapping away from last axis? */
        if (op_axis == op_ndim - 1) {
            remap_axis[iop] = NULL;
            continue;
        }
        remap_axis[iop][op_ndim - 1] = op_axis;
        for (axis = 0; axis < op_axis; axis++) {
            remap_axis[iop][axis] = axis;
        }
        for (axis = op_axis; axis < op_ndim - 1; axis++) {
            remap_axis[iop][axis] = axis + 1;
        }
    } /* end of for(iop) loop over operands */
    return 0;
}

#define REMAP_AXIS(iop, axis) ((remap_axis != NULL && \
                                remap_axis[iop] != NULL)? \
                               remap_axis[iop][axis] : axis)

/*
 * Validate the core dimensions of all the operands, and collect all of
 * the labelled core dimensions into 'core_dim_sizes'.
 *
 * Returns 0 on success, and -1 on failure
 *
 * The behavior has been changed in NumPy 1.16.0, and the following
 * requirements must be fulfilled or an error will be raised:
 *  * Arguments, both input and output, must have at least as many
 *    dimensions as the corresponding number of core dimensions. In
 *    versions before 1.10, 1's were prepended to the shape as needed.
 *  * Core dimensions with same labels must have exactly matching sizes.
 *    In versions before 1.10, core dimensions of size 1 would broadcast
 *    against other core dimensions with the same label.
 *  * All core dimensions must have their size specified by a passed in
 *    input or output argument. In versions before 1.10, core dimensions in
 *    an output argument that were not specified in an input argument,
 *    and whose size could not be inferred from a passed in output
 *    argument, would have their size set to 1.
 *  * Core dimensions may be fixed, new in NumPy 1.16
 */
static int
_get_coredim_sizes(PyUFuncObject *ufunc, PyArrayObject **op,
                   const int *op_core_num_dims, npy_uint32 *core_dim_flags,
                   npy_intp *core_dim_sizes, int **remap_axis) {
    int i;
    int nin = ufunc->nin;
    int nout = ufunc->nout;
    int nop = nin + nout;

    for (i = 0; i < nop; ++i) {
        if (op[i] != NULL) {
            int idim;
            int dim_offset = ufunc->core_offsets[i];
            int core_start_dim = PyArray_NDIM(op[i]) - op_core_num_dims[i];
            int dim_delta = 0;

            /* checked before this routine gets called */
            assert(core_start_dim >= 0);

            /*
             * Make sure every core dimension exactly matches all other core
             * dimensions with the same label. Note that flexible dimensions
             * may have been removed at this point, if so, they are marked
             * with UFUNC_CORE_DIM_MISSING.
             */
            for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
                int core_index = dim_offset + idim;
                int core_dim_index = ufunc->core_dim_ixs[core_index];
                npy_intp core_dim_size = core_dim_sizes[core_dim_index];
                npy_intp op_dim_size;

                /* can only happen if flexible; dimension missing altogether */
                if (core_dim_flags[core_dim_index] & UFUNC_CORE_DIM_MISSING) {
                    op_dim_size = 1;
                    dim_delta++; /* for indexing in dimensions */
                }
                else {
                    op_dim_size = PyArray_DIM(op[i],
                             REMAP_AXIS(i, core_start_dim + idim - dim_delta));
                }
                if (core_dim_sizes[core_dim_index] < 0) {
                    core_dim_sizes[core_dim_index] = op_dim_size;
                }
                else if (op_dim_size != core_dim_size) {
                    PyErr_Format(PyExc_ValueError,
                            "%s: %s operand %d has a mismatch in its "
                            "core dimension %d, with gufunc "
                            "signature %s (size %zd is different "
                            "from %zd)",
                            ufunc_get_name_cstr(ufunc), i < nin ? "Input" : "Output",
                            i < nin ? i : i - nin, idim - dim_delta,
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
    for (i = nin; i < nop; ++i) {
        int idim;
        int dim_offset = ufunc->core_offsets[i];

        for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
            int core_dim_index = ufunc->core_dim_ixs[dim_offset + idim];

            /* check all cases where the size has not yet been set */
            if (core_dim_sizes[core_dim_index] < 0) {
                /*
                 * Oops, this dimension was never specified
                 * (can only happen if output op not given)
                 */
                PyErr_Format(PyExc_ValueError,
                        "%s: Output operand %d has core dimension %d "
                        "unspecified, with gufunc signature %s",
                        ufunc_get_name_cstr(ufunc), i - nin, idim,
                        ufunc->core_signature);
                return -1;
            }
        }
    }

    return 0;
}

/*
 * Returns a new reference to the ufunc identity.  Note that this identity
 * is only a default identity value stored on the ufunc, since the invidiual
 * ufunc loop (ArrayMethod) is queried for the actual identity.
 *
 * TODO: store a reference in the ufunc object itself, rather than
 *       constructing one each time
 */
NPY_NO_EXPORT PyObject *
PyUFunc_GetDefaultIdentity(PyUFuncObject *ufunc, npy_bool *reorderable)
{
    switch(ufunc->identity) {
    case PyUFunc_One:
        *reorderable = 1;
        return PyLong_FromLong(1);

    case PyUFunc_Zero:
        *reorderable = 1;
        return PyLong_FromLong(0);

    case PyUFunc_MinusOne:
        *reorderable = 1;
        return PyLong_FromLong(-1);

    case PyUFunc_ReorderableNone:
        *reorderable = 1;
        Py_RETURN_NONE;

    case PyUFunc_None:
        *reorderable = 0;
        Py_RETURN_NONE;

    case PyUFunc_IdentityValue:
        *reorderable = 1;
        Py_INCREF(ufunc->identity_value);
        return ufunc->identity_value;

    default:
        PyErr_Format(PyExc_ValueError,
                "ufunc %s has an invalid identity", ufunc_get_name_cstr(ufunc));
        return NULL;
    }
}

/*
 * Copy over parts of the ufunc structure that may need to be
 * changed during execution.  Returns 0 on success; -1 otherwise.
 */
static int
_initialize_variable_parts(PyUFuncObject *ufunc,
                           int op_core_num_dims[],
                           npy_intp core_dim_sizes[],
                           npy_uint32 core_dim_flags[]) {
    int i;

    for (i = 0; i < ufunc->nargs; i++) {
        op_core_num_dims[i] = ufunc->core_num_dims[i];
    }
    for (i = 0; i < ufunc->core_num_dim_ix; i++) {
        core_dim_sizes[i] = ufunc->core_dim_sizes[i];
        core_dim_flags[i] = ufunc->core_dim_flags[i];
    }
    return 0;
}

static int
PyUFunc_GeneralizedFunctionInternal(PyUFuncObject *ufunc,
        PyArrayMethodObject *ufuncimpl, PyArray_Descr *operation_descrs[],
        PyArrayObject *op[], PyObject *extobj,
        NPY_CASTING casting, NPY_ORDER order,
        PyObject *axis, PyObject *axes, int keepdims)
{
    int nin, nout;
    int i, j, idim, nop;
    const char *ufunc_name;
    int retval;
    int needs_api = 0;

    /* Use remapped axes for generalized ufunc */
    int broadcast_ndim, iter_ndim;
    int op_core_num_dims[NPY_MAXARGS];
    int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
    int *op_axes[NPY_MAXARGS];
    npy_uint32 core_dim_flags[NPY_MAXARGS];

    npy_uint32 op_flags[NPY_MAXARGS];
    npy_intp iter_shape[NPY_MAXARGS];
    NpyIter *iter = NULL;
    npy_uint32 iter_flags;
    npy_intp total_problem_size;

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    /* The dimensions which get passed to the inner loop */
    npy_intp inner_dimensions[NPY_MAXDIMS+1];
    /* The strides which get passed to the inner loop */
    npy_intp *inner_strides = NULL;
    /* Auxiliary data allocated by the ufuncimpl (ArrayMethod) */
    NpyAuxData *auxdata = NULL;

    /* The sizes of the core dimensions (# entries is ufunc->core_num_dim_ix) */
    npy_intp *core_dim_sizes = inner_dimensions + 1;
    int core_dim_ixs_size;
    /* swapping around of axes */
    int *remap_axis_memory = NULL;
    int **remap_axis = NULL;

    nin = ufunc->nin;
    nout = ufunc->nout;
    nop = nin + nout;

    ufunc_name = ufunc_get_name_cstr(ufunc);

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_name);

    if (validate_casting(ufuncimpl,
            ufunc, op, operation_descrs, casting) < 0) {
        return -1;
    }

    /* Initialize possibly variable parts to the values from the ufunc */
    retval = _initialize_variable_parts(ufunc, op_core_num_dims,
                                        core_dim_sizes, core_dim_flags);
    if (retval < 0) {
        goto fail;
    }

    /*
     * If keepdims was passed in (and thus changed from the initial value
     * on top), check the gufunc is suitable, i.e., that its inputs share
     * the same number of core dimensions, and its outputs have none.
     */
    if (keepdims != -1) {
        retval = _check_keepdims_support(ufunc);
        if (retval < 0) {
            goto fail;
        }
    }
    if (axis != NULL) {
        retval = _check_axis_support(ufunc);
        if (retval < 0) {
            goto fail;
        }
    }
    /*
     * If keepdims is set and true, which means all input dimensions are
     * the same, signal that all output dimensions will be the same too.
     */
    if (keepdims == 1) {
        int num_dims = op_core_num_dims[0];
        for (i = nin; i < nop; ++i) {
            op_core_num_dims[i] = num_dims;
        }
    }
    else {
        /* keepdims was not set or was false; no adjustment necessary */
        keepdims = 0;
    }
    /*
     * Check that operands have the minimum dimensions required.
     * (Just checks core; broadcast dimensions are tested by the iterator.)
     */
    retval = _validate_num_dims(ufunc, op, core_dim_flags,
                                op_core_num_dims);
    if (retval < 0) {
        goto fail;
    }
    /*
     * Figure out the number of iteration dimensions, which
     * is the broadcast result of all the non-core dimensions.
     * (We do allow outputs to broadcast inputs currently, if they are given.
     * This is in line with what normal ufuncs do.)
     */
    broadcast_ndim = 0;
    for (i = 0; i < nop; ++i) {
        if (op[i] == NULL) {
            continue;
        }
        int n = PyArray_NDIM(op[i]) - op_core_num_dims[i];
        if (n > broadcast_ndim) {
            broadcast_ndim = n;
        }
    }

    /* Possibly remap axes. */
    if (axes != NULL || axis != NULL) {
        assert(!(axes != NULL && axis != NULL));

        remap_axis = PyArray_malloc(sizeof(remap_axis[0]) * nop);
        remap_axis_memory = PyArray_malloc(sizeof(remap_axis_memory[0]) *
                                                  nop * NPY_MAXDIMS);
        if (remap_axis == NULL || remap_axis_memory == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
        for (i=0; i < nop; i++) {
            remap_axis[i] = remap_axis_memory + i * NPY_MAXDIMS;
        }
        if (axis) {
            retval = _parse_axis_arg(ufunc, op_core_num_dims, axis, op,
                                     broadcast_ndim, remap_axis);
        }
        else {
            retval = _parse_axes_arg(ufunc, op_core_num_dims, axes, op,
                                     broadcast_ndim, remap_axis);
        }
        if(retval < 0) {
            goto fail;
        }
    }

    /* Collect the lengths of the labelled core dimensions */
    retval = _get_coredim_sizes(ufunc, op, op_core_num_dims, core_dim_flags,
                                core_dim_sizes, remap_axis);
    if(retval < 0) {
        goto fail;
    }
    /*
     * Figure out the number of iterator creation dimensions,
     * which is the broadcast dimensions + all the core dimensions of
     * the outputs, so that the iterator can allocate those output
     * dimensions following the rules of order='F', for example.
     */
    iter_ndim = broadcast_ndim;
    for (i = nin; i < nop; ++i) {
        iter_ndim += op_core_num_dims[i];
    }
    if (iter_ndim > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                    "too many dimensions for generalized ufunc %s",
                    ufunc_name);
        retval = -1;
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
            n = PyArray_NDIM(op[i]) - op_core_num_dims[i];
        }
        else {
            n = broadcast_ndim;
        }
        /* Broadcast all the unspecified dimensions normally */
        for (idim = 0; idim < broadcast_ndim; ++idim) {
            if (idim >= broadcast_ndim - n) {
                op_axes_arrays[i][idim] =
                    REMAP_AXIS(i, idim - (broadcast_ndim - n));
            }
            else {
                op_axes_arrays[i][idim] = -1;
            }
        }

        /*
         * Any output core dimensions shape should be ignored, so we add
         * it as a Reduce dimension (which can be broadcast with the rest).
         * These will be removed before the actual iteration for gufuncs.
         */
        for (idim = broadcast_ndim; idim < iter_ndim; ++idim) {
            op_axes_arrays[i][idim] = NPY_ITER_REDUCTION_AXIS(-1);
        }

        /* Except for when it belongs to this output */
        if (i >= nin) {
            int dim_offset = ufunc->core_offsets[i];
            int num_removed = 0;
            /*
             * Fill in 'iter_shape' and 'op_axes' for the core dimensions
             * of this output. Here, we have to be careful: if keepdims
             * was used, then the axes are not real core dimensions, but
             * are being added back for broadcasting, so their size is 1.
             * If the axis was removed, we should skip altogether.
             */
            if (keepdims) {
                for (idim = 0; idim < op_core_num_dims[i]; ++idim) {
                    iter_shape[j] = 1;
                    op_axes_arrays[i][j] = REMAP_AXIS(i, n + idim);
                    ++j;
                }
            }
            else {
                for (idim = 0; idim < ufunc->core_num_dims[i]; ++idim) {
                    int core_index = dim_offset + idim;
                    int core_dim_index = ufunc->core_dim_ixs[core_index];
                    if ((core_dim_flags[core_dim_index] &
                         UFUNC_CORE_DIM_MISSING)) {
                        /* skip it */
                        num_removed++;
                        continue;
                    }
                    iter_shape[j] = core_dim_sizes[ufunc->core_dim_ixs[core_index]];
                    op_axes_arrays[i][j] = REMAP_AXIS(i, n + idim - num_removed);
                    ++j;
                }
            }
        }

        op_axes[i] = op_axes_arrays[i];
    }

#if NPY_UF_DBG_TRACING
    printf("iter shapes:");
    for (j=0; j < iter_ndim; j++) {
        printf(" %ld", iter_shape[j]);
    }
    printf("\n");
#endif

    /* Get the buffersize and errormask */
    if (_get_bufsize_errmask(extobj, ufunc_name, &buffersize, &errormask) < 0) {
        retval = -1;
        goto fail;
    }

    NPY_UF_DBG_PRINT("Finding inner loop\n");

    /*
     * We don't write to all elements, and the iterator may make
     * UPDATEIFCOPY temporary copies. The output arrays (unless they are
     * allocated by the iterator itself) must be considered READWRITE by the
     * iterator, so that the elements we don't write to are copied to the
     * possible temporary array.
     */
    _ufunc_setup_flags(ufunc, NPY_ITER_COPY | NPY_UFUNC_DEFAULT_INPUT_FLAGS,
                       NPY_ITER_UPDATEIFCOPY |
                       NPY_ITER_WRITEONLY |
                       NPY_UFUNC_DEFAULT_OUTPUT_FLAGS,
                       op_flags);

    /*
     * Set up the iterator per-op flags.  For generalized ufuncs, we
     * can't do buffering, so must COPY or UPDATEIFCOPY.
     */
    iter_flags = ufunc->iter_flags |
                 NPY_ITER_MULTI_INDEX |
                 NPY_ITER_REFS_OK |
                 NPY_ITER_ZEROSIZE_OK |
                 NPY_ITER_COPY_IF_OVERLAP |
                 NPY_ITER_DELAY_BUFALLOC;

    /* Create the iterator */
    iter = NpyIter_AdvancedNew(nop, op, iter_flags,
                           order, NPY_UNSAFE_CASTING, op_flags,
                           operation_descrs, iter_ndim,
                           op_axes, iter_shape, 0);
    if (iter == NULL) {
        retval = -1;
        goto fail;
    }

    /* Fill in any allocated outputs */
    {
        PyArrayObject **operands = NpyIter_GetOperandArray(iter);
        for (i = nin; i < nop; ++i) {
            if (op[i] == NULL) {
                op[i] = operands[i];
                Py_INCREF(op[i]);
            }
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
        /*
         * Need to use the arrays in the iterator, not op, because
         * a copy with a different-sized type may have been made.
         */
        PyArrayObject *arr = NpyIter_GetOperandArray(iter)[i];
        npy_intp *shape = PyArray_SHAPE(arr);
        npy_intp *strides = PyArray_STRIDES(arr);
        /*
         * Could be negative if flexible dims are used, but not for
         * keepdims, since those dimensions are allocated in arr.
         */
        int core_start_dim = PyArray_NDIM(arr) - op_core_num_dims[i];
        int num_removed = 0;
        int dim_offset = ufunc->core_offsets[i];

        for (j = 0; j < ufunc->core_num_dims[i]; ++j) {
            int core_dim_index = ufunc->core_dim_ixs[dim_offset + j];
            /*
             * Force zero stride when the shape is 1 (always the case for
             * for missing dimensions), so that broadcasting works right.
             */
            if (core_dim_flags[core_dim_index] & UFUNC_CORE_DIM_MISSING) {
                num_removed++;
                inner_strides[idim++] = 0;
            }
            else {
                int remapped_axis = REMAP_AXIS(i, core_start_dim + j - num_removed);
                if (shape[remapped_axis] != 1) {
                    inner_strides[idim++] = strides[remapped_axis];
                } else {
                    inner_strides[idim++] = 0;
                }
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
     * copy them after removing the core axes).  The strides will not change
     * if the iterator is not buffered (they are effectively fixed).
     * Supporting buffering would make sense, but probably would have to be
     * done in the inner-loop itself (not the iterator).
     */
    assert(!NpyIter_IsBuffered(iter));
    memcpy(inner_strides, NpyIter_GetInnerStrideArray(iter),
                                    NPY_SIZEOF_INTP * nop);

    /* Final preparation of the arraymethod call */
    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = operation_descrs,
    };
    PyArrayMethod_StridedLoop *strided_loop;
    NPY_ARRAYMETHOD_FLAGS flags = 0;

    if (ufuncimpl->get_strided_loop(&context, 1, 0, inner_strides,
            &strided_loop, &auxdata, &flags) < 0) {
        goto fail;
    }
    needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    needs_api |= NpyIter_IterationNeedsAPI(iter);
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        npy_clear_floatstatus_barrier((char*)&iter);
    }

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

        if (!needs_api) {
            NPY_BEGIN_THREADS_THRESHOLDED(total_problem_size);
        }
        do {
            inner_dimensions[0] = *count_ptr;
            retval = strided_loop(&context,
                    dataptr, inner_dimensions, inner_strides, auxdata);
        } while (retval == 0 && iternext(iter));

        if (!needs_api && !NpyIter_IterationNeedsAPI(iter)) {
            NPY_END_THREADS;
        }
    }

    if (retval == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        retval = _check_ufunc_fperr(errormask, extobj, ufunc_name);
    }

    PyArray_free(inner_strides);
    NPY_AUXDATA_FREE(auxdata);
    if (!NpyIter_Deallocate(iter)) {
        retval = -1;
    }

    PyArray_free(remap_axis_memory);
    PyArray_free(remap_axis);

    NPY_UF_DBG_PRINT1("Returning code %d\n", retval);

    return retval;

fail:
    NPY_UF_DBG_PRINT1("Returning failure code %d\n", retval);
    PyArray_free(inner_strides);
    NPY_AUXDATA_FREE(auxdata);
    NpyIter_Deallocate(iter);
    PyArray_free(remap_axis_memory);
    PyArray_free(remap_axis);
    return retval;
}


static int
PyUFunc_GenericFunctionInternal(PyUFuncObject *ufunc,
        PyArrayMethodObject *ufuncimpl, PyArray_Descr *operation_descrs[],
        PyArrayObject *op[], PyObject *extobj,
        NPY_CASTING casting, NPY_ORDER order,
        PyObject *output_array_prepare[], ufunc_full_args full_args,
        PyArrayObject *wheremask)
{
    int nin = ufunc->nin, nout = ufunc->nout, nop = nin + nout;

    const char *ufunc_name = ufunc_get_name_cstr(ufunc);

    npy_intp default_op_out_flags;
    npy_uint32 op_flags[NPY_MAXARGS];

    /* These parameters come from extobj= or from a TLS global */
    int buffersize = 0, errormask = 0;

    NPY_UF_DBG_PRINT1("\nEvaluating ufunc %s\n", ufunc_name);

    /* Get the buffersize and errormask */
    if (_get_bufsize_errmask(extobj, ufunc_name, &buffersize, &errormask) < 0) {
        return -1;
    }

    if (wheremask != NULL) {
        /* Set up the flags. */
        default_op_out_flags = NPY_ITER_NO_SUBTYPE |
                               NPY_ITER_WRITEMASKED |
                               NPY_UFUNC_DEFAULT_OUTPUT_FLAGS;
        _ufunc_setup_flags(ufunc, NPY_UFUNC_DEFAULT_INPUT_FLAGS,
                           default_op_out_flags, op_flags);
    }
    else {
        /* Set up the flags. */
        default_op_out_flags = NPY_ITER_WRITEONLY |
                               NPY_UFUNC_DEFAULT_OUTPUT_FLAGS;
        _ufunc_setup_flags(ufunc, NPY_UFUNC_DEFAULT_INPUT_FLAGS,
                           default_op_out_flags, op_flags);
    }

    /* Final preparation of the arraymethod call */
    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = operation_descrs,
    };

    /* Do the ufunc loop */
    if (wheremask != NULL) {
        NPY_UF_DBG_PRINT("Executing masked inner loop\n");

        if (nop + 1 > NPY_MAXARGS) {
            PyErr_SetString(PyExc_ValueError,
                    "Too many operands when including where= parameter");
            return -1;
        }
        op[nop] = wheremask;
        operation_descrs[nop] = NULL;

        return execute_ufunc_loop(&context, 1,
                op, order, buffersize, casting,
                output_array_prepare, full_args, op_flags,
                errormask, extobj);
    }
    else {
        NPY_UF_DBG_PRINT("Executing normal inner loop\n");

        /*
         * This checks whether a trivial loop is ok, making copies of
         * scalar and one dimensional operands if that should help.
         */
        int trivial_ok = check_for_trivial_loop(ufuncimpl,
                op, operation_descrs, casting, buffersize);
        if (trivial_ok < 0) {
            return -1;
        }
        if (trivial_ok && context.method->nout == 1) {
            /* Try to handle everything without using the (heavy) iterator */
            int retval = try_trivial_single_output_loop(&context,
                    op, order, output_array_prepare, full_args,
                    errormask, extobj);
            if (retval != -2) {
                return retval;
            }
        }

        return execute_ufunc_loop(&context, 0,
                op, order, buffersize, casting,
                output_array_prepare, full_args, op_flags,
                errormask, extobj);
    }
}


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_GenericFunction(PyUFuncObject *NPY_UNUSED(ufunc),
        PyObject *NPY_UNUSED(args), PyObject *NPY_UNUSED(kwds),
        PyArrayObject **NPY_UNUSED(op))
{
    /* NumPy 1.21, 2020-03-29 */
    PyErr_SetString(PyExc_RuntimeError,
            "The `PyUFunc_GenericFunction()` C-API function has been disabled. "
            "Please use `PyObject_Call(ufunc, args, kwargs)`, which has "
            "identical behaviour but allows subclass and `__array_ufunc__` "
            "override handling and only returns the normal ufunc result.");
    return -1;
}


/*
 * Promote and resolve a reduction like operation.
 *
 * @param ufunc
 * @param arr The operation array
 * @param out The output array or NULL if not provided.  Note that NumPy always
 *        used out to mean the same as `dtype=out.dtype` and never passed
 *        the array itself to the type-resolution.
 * @param signature The DType signature, which may already be set due to the
 *        dtype passed in by the user, or the special cases (add, multiply).
 *        (Contains strong references and may be modified.)
 * @param enforce_uniform_args If `NPY_TRUE` fully uniform dtypes/descriptors
 *        are enforced as required for accumulate and (currently) reduceat.
 * @param out_descrs New references to the resolved descriptors (on success).
 * @param method The ufunc method, "reduce", "reduceat", or "accumulate".

 * @returns ufuncimpl The `ArrayMethod` implementation to use. Or NULL if an
 *          error occurred.
 */
static PyArrayMethodObject *
reducelike_promote_and_resolve(PyUFuncObject *ufunc,
        PyArrayObject *arr, PyArrayObject *out,
        PyArray_DTypeMeta *signature[3],
        npy_bool enforce_uniform_args, PyArray_Descr *out_descrs[3],
        NPY_CASTING casting, char *method)
{
     /*
      * If no dtype is specified and out is not specified, we override the
      * integer and bool dtype used for add and multiply.
      *
      * TODO: The following should be handled by a promoter!
      */
    if (signature[0] == NULL && out == NULL) {
        /*
         * For integer types --- make sure at least a long
         * is used for add and multiply reduction to avoid overflow
         */
        int typenum = PyArray_TYPE(arr);
        if ((PyTypeNum_ISBOOL(typenum) || PyTypeNum_ISINTEGER(typenum))
                && ((strcmp(ufunc->name, "add") == 0)
                    || (strcmp(ufunc->name, "multiply") == 0))) {
            if (PyTypeNum_ISBOOL(typenum)) {
                typenum = NPY_LONG;
            }
            else if ((size_t)PyArray_DESCR(arr)->elsize < sizeof(long)) {
                if (PyTypeNum_ISUNSIGNED(typenum)) {
                    typenum = NPY_ULONG;
                }
                else {
                    typenum = NPY_LONG;
                }
            }
            signature[0] = PyArray_DTypeFromTypeNum(typenum);
        }
    }
    assert(signature[2] == NULL);  /* we always fill it here */
    Py_XINCREF(signature[0]);
    signature[2] = signature[0];

    /*
     * Note that the `ops` is not really correct.  But legacy resolution
     * cannot quite handle the correct ops (e.g. a NULL first item if `out`
     * is NULL) so we pass `arr` instead in that case.
     */
    PyArrayObject *ops[3] = {out ? out : arr, arr, out};

    /*
     * TODO: This is a dangerous hack, that works by relying on the GIL, it is
     *       terrible, terrifying, and trusts that nobody does crazy stuff
     *       in their type-resolvers.
     *       By mutating the `out` dimension, we ensure that reduce-likes
     *       live in a future without value-based promotion even when legacy
     *       promotion has to be used.
     */
    npy_bool evil_ndim_mutating_hack = NPY_FALSE;
    if (out != NULL && PyArray_NDIM(out) == 0 && PyArray_NDIM(arr) != 0) {
        evil_ndim_mutating_hack = NPY_TRUE;
        ((PyArrayObject_fields *)out)->nd = 1;
    }

    /*
     * TODO: If `out` is not provided, arguably `initial` could define
     *       the first DType (and maybe also the out one), that way
     *       `np.add.reduce([1, 2, 3], initial=3.4)` would return a float
     *       value.  As of 1.20, it returned an integer, so that should
     *       probably go to an error/warning first.
     */
    PyArray_DTypeMeta *operation_DTypes[3] = {
            NULL, NPY_DTYPE(PyArray_DESCR(arr)), NULL};
    Py_INCREF(operation_DTypes[1]);

    if (out != NULL) {
        operation_DTypes[0] = NPY_DTYPE(PyArray_DESCR(out));
        Py_INCREF(operation_DTypes[0]);
        operation_DTypes[2] = operation_DTypes[0];
        Py_INCREF(operation_DTypes[2]);
    }

    PyArrayMethodObject *ufuncimpl = promote_and_get_ufuncimpl(ufunc,
            ops, signature, operation_DTypes, NPY_FALSE, NPY_TRUE,
            NPY_FALSE, NPY_TRUE);
    if (evil_ndim_mutating_hack) {
        ((PyArrayObject_fields *)out)->nd = 0;
    }
    /* DTypes may currently get filled in fallbacks and XDECREF for error: */
    Py_XDECREF(operation_DTypes[0]);
    Py_XDECREF(operation_DTypes[1]);
    Py_XDECREF(operation_DTypes[2]);
    if (ufuncimpl == NULL) {
        return NULL;
    }

    /*
     * Find the correct descriptors for the operation.  We use unsafe casting
     * for historic reasons: The logic ufuncs required it to cast everything to
     * boolean.  However, we now special case the logical ufuncs, so that the
     * casting safety could in principle be set to the default same-kind.
     * (although this should possibly happen through a deprecation)
     */
    if (resolve_descriptors(3, ufunc, ufuncimpl,
            ops, out_descrs, signature, casting) < 0) {
        return NULL;
    }

    /*
     * The first operand and output should be the same array, so they should
     * be identical.  The second argument can be different for reductions,
     * but is checked to be identical for accumulate and reduceat.
     * Ideally, the type-resolver ensures that all are identical, but we do
     * not enforce this here strictly.  Otherwise correct handling of
     * byte-order changes (or metadata) requires a lot of care; see gh-20699.
     */
    if (!PyArray_EquivTypes(out_descrs[0], out_descrs[2]) || (
            enforce_uniform_args && !PyArray_EquivTypes(
                    out_descrs[0], out_descrs[1]))) {
        PyErr_Format(PyExc_TypeError,
                "the resolved dtypes are not compatible with %s.%s. "
                "Resolved (%R, %R, %R)",
                ufunc_get_name_cstr(ufunc), method,
                out_descrs[0], out_descrs[1], out_descrs[2]);
        goto fail;
    }
    /* TODO: This really should _not_ be unsafe casting (same above)! */
    if (validate_casting(ufuncimpl, ufunc, ops, out_descrs, casting) < 0) {
        goto fail;
    }

    return ufuncimpl;

  fail:
    for (int i = 0; i < 3; ++i) {
        Py_CLEAR(out_descrs[i]);
    }
    return NULL;
}


static int
reduce_loop(PyArrayMethod_Context *context,
        PyArrayMethod_StridedLoop *strided_loop, NpyAuxData *auxdata,
        NpyIter *iter, char **dataptrs, npy_intp const *strides,
        npy_intp const *countptr, NpyIter_IterNextFunc *iternext,
        int needs_api, npy_intp skip_first_count)
{
    int retval = 0;
    char *dataptrs_copy[4];
    npy_intp strides_copy[4];
    npy_bool masked;

    NPY_BEGIN_THREADS_DEF;
    /* Get the number of operands, to determine whether "where" is used */
    masked = (NpyIter_GetNOp(iter) == 3);

    if (!needs_api) {
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
    }

    if (skip_first_count > 0) {
        assert(!masked);  /* Path currently not available for masked */
        while (1) {
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
            if (count > 0) {
                /* Turn the two items into three for the inner loop */
                dataptrs_copy[0] = dataptrs[0];
                dataptrs_copy[1] = dataptrs[1];
                dataptrs_copy[2] = dataptrs[0];
                strides_copy[0] = strides[0];
                strides_copy[1] = strides[1];
                strides_copy[2] = strides[0];

                retval = strided_loop(context,
                        dataptrs_copy, &count, strides_copy, auxdata);
                if (retval < 0) {
                    goto finish_loop;
                }
            }

            /* Advance loop, and abort on error (or finish) */
            if (!iternext(iter)) {
                goto finish_loop;
            }

            /* When skipping is done break and continue with faster loop */
            if (skip_first_count == 0) {
                break;
            }
        }
    }

    do {
        /* Turn the two items into three for the inner loop */
        dataptrs_copy[0] = dataptrs[0];
        dataptrs_copy[1] = dataptrs[1];
        dataptrs_copy[2] = dataptrs[0];
        strides_copy[0] = strides[0];
        strides_copy[1] = strides[1];
        strides_copy[2] = strides[0];
        if (masked) {
            dataptrs_copy[3] = dataptrs[2];
            strides_copy[3] = strides[2];
        }

        retval = strided_loop(context,
                dataptrs_copy, countptr, strides_copy, auxdata);
        if (retval < 0) {
            goto finish_loop;
        }

    } while (iternext(iter));

finish_loop:
    NPY_END_THREADS;

    return retval;
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
PyUFunc_Reduce(PyUFuncObject *ufunc,
        PyArrayObject *arr, PyArrayObject *out,
        int naxes, int *axes, PyArray_DTypeMeta *signature[3], int keepdims,
        PyObject *initial, PyArrayObject *wheremask)
{
    int iaxes, ndim;
    npy_bool axis_flags[NPY_MAXDIMS];

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

    if (_get_bufsize_errmask(NULL, "reduce", &buffersize, &errormask) < 0) {
        return NULL;
    }

    PyArray_Descr *descrs[3];
    PyArrayMethodObject *ufuncimpl = reducelike_promote_and_resolve(ufunc,
            arr, out, signature, NPY_FALSE, descrs, NPY_UNSAFE_CASTING, "reduce");
    if (ufuncimpl == NULL) {
        return NULL;
    }

    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = descrs,
    };

    PyArrayObject *result = PyUFunc_ReduceWrapper(&context,
            arr, out, wheremask, axis_flags, keepdims,
            initial, reduce_loop, buffersize, ufunc_name, errormask);

    for (int i = 0; i < 3; i++) {
        Py_DECREF(descrs[i]);
    }
    return result;
}


static PyObject *
PyUFunc_Accumulate(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *out,
                   int axis, PyArray_DTypeMeta *signature[3])
{
    PyArrayObject *op[2];
    int op_axes_arrays[2][NPY_MAXDIMS];
    int *op_axes[2] = {op_axes_arrays[0], op_axes_arrays[1]};
    npy_uint32 op_flags[2];
    int idim, ndim;
    int needs_api, need_outer_iterator;

    int res = 0;

    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata = NULL;

    NpyIter *iter = NULL;

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

    PyArray_Descr *descrs[3];
    PyArrayMethodObject *ufuncimpl = reducelike_promote_and_resolve(ufunc,
            arr, out, signature, NPY_TRUE, descrs, NPY_UNSAFE_CASTING,
            "accumulate");
    if (ufuncimpl == NULL) {
        return NULL;
    }

    /*
     * The below code assumes that all descriptors are interchangeable, we
     * allow them to not be strictly identical (but they typically should be)
     */
    assert(PyArray_EquivTypes(descrs[0], descrs[1])
           && PyArray_EquivTypes(descrs[0], descrs[2]));

    if (PyDataType_REFCHK(descrs[2]) && descrs[2]->type_num != NPY_OBJECT) {
        /* This can be removed, but the initial element copy needs fixing */
        PyErr_SetString(PyExc_TypeError,
                "accumulation currently only supports `object` dtype with "
                "references");
        goto fail;
    }

    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = descrs,
    };

    ndim = PyArray_NDIM(arr);

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
            !PyArray_EquivTypes(descrs[1], PyArray_DESCR(arr)) ||
            (out &&
             !PyArray_EquivTypes(descrs[0], PyArray_DESCR(out)))) {
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

        /*
         * The way accumulate is set up, we can't do buffering,
         * so make a copy instead when necessary.
         */
        ndim_iter = ndim;
        flags |= NPY_ITER_MULTI_INDEX;
        /*
         * Add some more flags.
         *
         * The accumulation outer loop is 'elementwise' over the array, so turn
         * on NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE. That is, in-place
         * accumulate(x, out=x) is safe to do without temporary copies.
         */
        op_flags[0] |= NPY_ITER_UPDATEIFCOPY|NPY_ITER_ALIGNED|NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;
        op_flags[1] |= NPY_ITER_COPY|NPY_ITER_ALIGNED|NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;

        NPY_UF_DBG_PRINT("Allocating outer iterator\n");
        iter = NpyIter_AdvancedNew(2, op, flags,
                                   NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                   op_flags, descrs,
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

    /* Get the output from the iterator if it was allocated */
    if (out == NULL) {
        if (iter) {
            op[0] = out = NpyIter_GetOperandArray(iter)[0];
            Py_INCREF(out);
        }
        else {
            PyArray_Descr *dtype = descrs[0];
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

    npy_intp fixed_strides[3];
    if (need_outer_iterator) {
        NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    }
    else {
        fixed_strides[0] = PyArray_STRIDES(op[0])[axis];
        fixed_strides[1] = PyArray_STRIDES(op[1])[axis];
        fixed_strides[2] = fixed_strides[0];
    }


    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (ufuncimpl->get_strided_loop(&context,
            1, 0, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
        goto fail;
    }
    needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        npy_clear_floatstatus_barrier((char*)&iter);
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

        int itemsize = descrs[0]->elsize;

        /* Get the variables needed for the loop */
        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        needs_api |= NpyIter_IterationNeedsAPI(iter);

        /* Execute the loop with just the outer iterator */
        count_m1 = PyArray_DIM(op[1], axis)-1;
        stride0 = 0, stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with just outer iterator\n");

        stride0 = PyArray_STRIDE(op[0], axis);

        stride_copy[0] = stride0;
        stride_copy[1] = stride1;
        stride_copy[2] = stride0;

        if (!needs_api) {
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
        }

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
            if (descrs[2]->type_num == NPY_OBJECT) {
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
                res = strided_loop(&context,
                        dataptr_copy, &count_m1, stride_copy, auxdata);
            }
        } while (res == 0 && iternext(iter));

        NPY_END_THREADS;
    }
    else if (iter == NULL) {
        char *dataptr_copy[3];

        int itemsize = descrs[0]->elsize;

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
                    "for the accumulation.");
            goto fail;
        }
        stride0 = PyArray_STRIDE(op[0], axis);

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
        if (descrs[2]->type_num == NPY_OBJECT) {
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

            needs_api = PyDataType_REFCHK(descrs[0]);

            if (!needs_api) {
                NPY_BEGIN_THREADS_THRESHOLDED(count);
            }

            res = strided_loop(&context,
                    dataptr_copy, &count, fixed_strides, auxdata);

            NPY_END_THREADS;
        }
    }

finish:
    NPY_AUXDATA_FREE(auxdata);
    Py_DECREF(descrs[0]);
    Py_DECREF(descrs[1]);
    Py_DECREF(descrs[2]);

    if (!NpyIter_Deallocate(iter)) {
        res = -1;
    }

    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        res = _check_ufunc_fperr(errormask, NULL, "accumulate");
    }

    if (res < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return (PyObject *)out;

fail:
    Py_XDECREF(out);

    NPY_AUXDATA_FREE(auxdata);
    Py_XDECREF(descrs[0]);
    Py_XDECREF(descrs[1]);
    Py_XDECREF(descrs[2]);

    NpyIter_Deallocate(iter);

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
 *
 * TODO: Reduceat duplicates too much code from accumulate!
 */
static PyObject *
PyUFunc_Reduceat(PyUFuncObject *ufunc, PyArrayObject *arr, PyArrayObject *ind,
                 PyArrayObject *out, int axis, PyArray_DTypeMeta *signature[3])
{
    PyArrayObject *op[3];
    int op_axes_arrays[3][NPY_MAXDIMS];
    int *op_axes[3] = {op_axes_arrays[0], op_axes_arrays[1],
                            op_axes_arrays[2]};
    npy_uint32 op_flags[3];
    int idim, ndim;
    int needs_api, need_outer_iterator = 0;

    int res = 0;

    NpyIter *iter = NULL;

    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata = NULL;

    /* The reduceat indices - ind must be validated outside this call */
    npy_intp *reduceat_ind;
    npy_intp i, ind_size, red_axis_size;

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
                "index %" NPY_INTP_FMT " out-of-bounds in %s.%s [0, %" NPY_INTP_FMT ")",
                reduceat_ind[i], ufunc_name, opname, red_axis_size);
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

    PyArray_Descr *descrs[3];
    PyArrayMethodObject *ufuncimpl = reducelike_promote_and_resolve(ufunc,
            arr, out, signature, NPY_TRUE, descrs, NPY_UNSAFE_CASTING,
            "reduceat");
    if (ufuncimpl == NULL) {
        return NULL;
    }

    /*
     * The below code assumes that all descriptors are interchangeable, we
     * allow them to not be strictly identical (but they typically should be)
     */
    assert(PyArray_EquivTypes(descrs[0], descrs[1])
           && PyArray_EquivTypes(descrs[0], descrs[2]));

    if (PyDataType_REFCHK(descrs[2]) && descrs[2]->type_num != NPY_OBJECT) {
        /* This can be removed, but the initial element copy needs fixing */
        PyErr_SetString(PyExc_TypeError,
                "reduceat currently only supports `object` dtype with "
                "references");
        goto fail;
    }

    PyArrayMethod_Context context = {
        .caller = (PyObject *)ufunc,
        .method = ufuncimpl,
        .descriptors = descrs,
    };

    ndim = PyArray_NDIM(arr);

#if NPY_UF_DBG_TRACING
    printf("Found %s.%s inner loop with dtype :  ", ufunc_name, opname);
    PyObject_Print((PyObject *)op_dtypes[0], stdout, 0);
    printf("\n");
#endif

    /* Set up the op_axes for the outer loop */
    for (idim = 0; idim < ndim; ++idim) {
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
            !PyArray_EquivTypes(descrs[0], PyArray_DESCR(arr))) {
        need_outer_iterator = 1;
    }

    if (need_outer_iterator) {
        PyArray_Descr *op_dtypes[3] = {descrs[0], descrs[1], NULL};

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
                                   op_flags, op_dtypes,
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
    else {
        /*
         * Allocate the output for when there's no outer iterator, we always
         * use the outer_iteration path when `out` is passed.
         */
        assert(out == NULL);
        Py_INCREF(descrs[0]);
        op[0] = out = (PyArrayObject *)PyArray_NewFromDescr(
                                    &PyArray_Type, descrs[0],
                                    1, &ind_size, NULL, NULL,
                                    0, NULL);
        if (out == NULL) {
            goto fail;
        }
    }

    npy_intp fixed_strides[3];
    if (need_outer_iterator) {
        NpyIter_GetInnerFixedStrideArray(iter, fixed_strides);
    }
    else {
        fixed_strides[1] = PyArray_STRIDES(op[1])[axis];
    }
    /* The reduce axis does not advance here in the strided-loop */
    fixed_strides[0] = 0;
    fixed_strides[2] = 0;

    NPY_ARRAYMETHOD_FLAGS flags = 0;
    if (ufuncimpl->get_strided_loop(&context,
            1, 0, fixed_strides, &strided_loop, &auxdata, &flags) < 0) {
        goto fail;
    }
    needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        npy_clear_floatstatus_barrier((char*)&iter);
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

        int itemsize = descrs[0]->elsize;
        needs_api |= NpyIter_IterationNeedsAPI(iter);

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

        if (!needs_api) {
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));
        }

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
                if (descrs[2]->type_num == NPY_OBJECT) {
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
                    res = strided_loop(&context,
                            dataptr_copy, &count, stride_copy, auxdata);
                }
            }
        } while (res == 0 && iternext(iter));

        NPY_END_THREADS;
    }
    else if (iter == NULL) {
        char *dataptr_copy[3];

        int itemsize = descrs[0]->elsize;

        npy_intp stride0_ind = PyArray_STRIDE(op[0], axis);
        npy_intp stride1 = PyArray_STRIDE(op[1], axis);

        NPY_UF_DBG_PRINT("UFunc: Reduce loop with no iterators\n");

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
            if (descrs[2]->type_num == NPY_OBJECT) {
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
                res = strided_loop(&context,
                        dataptr_copy, &count, fixed_strides, auxdata);
                if (res != 0) {
                    break;
                }
            }
        }

        NPY_END_THREADS;
    }

finish:
    NPY_AUXDATA_FREE(auxdata);
    Py_DECREF(descrs[0]);
    Py_DECREF(descrs[1]);
    Py_DECREF(descrs[2]);

    if (!NpyIter_Deallocate(iter)) {
        res = -1;
    }

    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        res = _check_ufunc_fperr(errormask, NULL, "reduceat");
    }

    if (res < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return (PyObject *)out;

fail:
    Py_XDECREF(out);

    NPY_AUXDATA_FREE(auxdata);
    Py_XDECREF(descrs[0]);
    Py_XDECREF(descrs[1]);
    Py_XDECREF(descrs[2]);

    NpyIter_Deallocate(iter);

    return NULL;
}


static npy_bool
tuple_all_none(PyObject *tup) {
    npy_intp i;
    for (i = 0; i < PyTuple_GET_SIZE(tup); ++i) {
        if (PyTuple_GET_ITEM(tup, i) != Py_None) {
            return NPY_FALSE;
        }
    }
    return NPY_TRUE;
}


static int
_set_full_args_out(int nout, PyObject *out_obj, ufunc_full_args *full_args)
{
    if (PyTuple_CheckExact(out_obj)) {
        if (PyTuple_GET_SIZE(out_obj) != nout) {
            PyErr_SetString(PyExc_ValueError,
                            "The 'out' tuple must have exactly "
                            "one entry per ufunc output");
            return -1;
        }
        if (tuple_all_none(out_obj)) {
            return 0;
        }
        else {
            Py_INCREF(out_obj);
            full_args->out = out_obj;
        }
    }
    else if (nout == 1) {
        if (out_obj == Py_None) {
            return 0;
        }
        /* Can be an array if it only has one output */
        full_args->out = PyTuple_Pack(1, out_obj);
        if (full_args->out == NULL) {
            return -1;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        nout > 1 ? "'out' must be a tuple of arrays" :
                        "'out' must be an array or a tuple with "
                        "a single array");
        return -1;
    }
    return 0;
}


/*
 * Convert function which replaces np._NoValue with NULL.
 * As a converter returns 0 on error and 1 on success.
 */
static int
_not_NoValue(PyObject *obj, PyObject **out)
{
    static PyObject *NoValue = NULL;
    npy_cache_import("numpy", "_NoValue", &NoValue);
    if (NoValue == NULL) {
        return 0;
    }
    if (obj == NoValue) {
        *out = NULL;
    }
    else {
        *out = obj;
    }
    return 1;
}


/* forward declaration */
static PyArray_DTypeMeta * _get_dtype(PyObject *dtype_obj);

/*
 * This code handles reduce, reduceat, and accumulate
 * (accumulate and reduce are special cases of the more general reduceat
 * but they are handled separately for speed)
 */
static PyObject *
PyUFunc_GenericReduction(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames, int operation)
{
    int i, naxes=0, ndim;
    int axes[NPY_MAXDIMS];

    ufunc_full_args full_args = {NULL, NULL};
    PyObject *axes_obj = NULL;
    PyArrayObject *mp = NULL, *wheremask = NULL, *ret = NULL;
    PyObject *op = NULL;
    PyArrayObject *indices = NULL;
    PyArray_DTypeMeta *signature[3] = {NULL, NULL, NULL};
    PyArrayObject *out = NULL;
    int keepdims = 0;
    PyObject *initial = NULL;
    npy_bool out_is_passed_by_position;


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

    /*
     * Perform argument parsing, but start by only extracting. This is
     * just to preserve the behaviour that __array_ufunc__ did not perform
     * any checks on arguments, and we could change this or change it for
     * certain parameters.
     */
    PyObject *otype_obj = NULL, *out_obj = NULL, *indices_obj = NULL;
    PyObject *keepdims_obj = NULL, *wheremask_obj = NULL;
    if (operation == UFUNC_REDUCEAT) {
        NPY_PREPARE_ARGPARSER;

        if (npy_parse_arguments("reduceat", args, len_args, kwnames,
                "array", NULL, &op,
                "indices", NULL, &indices_obj,
                "|axis", NULL, &axes_obj,
                "|dtype", NULL, &otype_obj,
                "|out", NULL, &out_obj,
                NULL, NULL, NULL) < 0) {
            goto fail;
        }
        /* Prepare inputs for PyUfunc_CheckOverride */
        full_args.in = PyTuple_Pack(2, op, indices_obj);
        if (full_args.in == NULL) {
            goto fail;
        }
        out_is_passed_by_position = len_args >= 5;
    }
    else if (operation == UFUNC_ACCUMULATE) {
        NPY_PREPARE_ARGPARSER;

        if (npy_parse_arguments("accumulate", args, len_args, kwnames,
                "array", NULL, &op,
                "|axis", NULL, &axes_obj,
                "|dtype", NULL, &otype_obj,
                "|out", NULL, &out_obj,
                NULL, NULL, NULL) < 0) {
            goto fail;
        }
        /* Prepare input for PyUfunc_CheckOverride */
        full_args.in = PyTuple_Pack(1, op);
        if (full_args.in == NULL) {
            goto fail;
        }
        out_is_passed_by_position = len_args >= 4;
    }
    else {
        NPY_PREPARE_ARGPARSER;

        if (npy_parse_arguments("reduce", args, len_args, kwnames,
                "array", NULL, &op,
                "|axis", NULL, &axes_obj,
                "|dtype", NULL, &otype_obj,
                "|out", NULL, &out_obj,
                "|keepdims", NULL, &keepdims_obj,
                "|initial", &_not_NoValue, &initial,
                "|where", NULL, &wheremask_obj,
                NULL, NULL, NULL) < 0) {
            goto fail;
        }
        /* Prepare input for PyUfunc_CheckOverride */
        full_args.in = PyTuple_Pack(1, op);
        if (full_args.in == NULL) {
            goto fail;
        }
        out_is_passed_by_position = len_args >= 4;
    }

    /* Normalize output for PyUFunc_CheckOverride and conversion. */
    if (out_is_passed_by_position) {
        /* in this branch, out is always wrapped in a tuple. */
        if (out_obj != Py_None) {
            full_args.out = PyTuple_Pack(1, out_obj);
            if (full_args.out == NULL) {
                goto fail;
            }
        }
    }
    else if (out_obj) {
        if (_set_full_args_out(1, out_obj, &full_args) < 0) {
            goto fail;
        }
        /* Ensure that out_obj is the array, not the tuple: */
        if (full_args.out != NULL) {
            out_obj = PyTuple_GET_ITEM(full_args.out, 0);
        }
    }

    /* We now have all the information required to check for Overrides */
    PyObject *override = NULL;
    int errval = PyUFunc_CheckOverride(ufunc, _reduce_type[operation],
            full_args.in, full_args.out, wheremask_obj, args, len_args, kwnames, &override);
    if (errval) {
        return NULL;
    }
    else if (override) {
        Py_XDECREF(full_args.in);
        Py_XDECREF(full_args.out);
        return override;
    }

    /* Finish parsing of all parameters (no matter which reduce-like) */
    if (indices_obj) {
        PyArray_Descr *indtype = PyArray_DescrFromType(NPY_INTP);

        indices = (PyArrayObject *)PyArray_FromAny(indices_obj,
                indtype, 1, 1, NPY_ARRAY_CARRAY, NULL);
        if (indices == NULL) {
            goto fail;
        }
    }
    if (otype_obj && otype_obj != Py_None) {
        /* Use `_get_dtype` because `dtype` is a DType and not the instance */
        signature[0] = _get_dtype(otype_obj);
        if (signature[0] == NULL) {
            goto fail;
        }
    }
    if (out_obj && !PyArray_OutputConverter(out_obj, &out)) {
        goto fail;
    }
    if (keepdims_obj && !PyArray_PythonPyIntFromInt(keepdims_obj, &keepdims)) {
        goto fail;
    }
    if (wheremask_obj && !_wheremask_converter(wheremask_obj, &wheremask)) {
        goto fail;
    }

    /* Ensure input is an array */
    mp = (PyArrayObject *)PyArray_FromAny(op, NULL, 0, 0, 0, NULL);
    if (mp == NULL) {
        goto fail;
    }

    ndim = PyArray_NDIM(mp);

    /* Convert the 'axis' parameter into a list of axes */
    if (axes_obj == NULL) {
        /* apply defaults */
        if (ndim == 0) {
            naxes = 0;
        }
        else {
            naxes = 1;
            axes[0] = 0;
        }
    }
    else if (axes_obj == Py_None) {
        /* Convert 'None' into all the axes */
        naxes = ndim;
        for (i = 0; i < naxes; ++i) {
            axes[i] = i;
        }
    }
    else if (PyTuple_Check(axes_obj)) {
        naxes = PyTuple_Size(axes_obj);
        if (naxes < 0 || naxes > NPY_MAXDIMS) {
            PyErr_SetString(PyExc_ValueError,
                    "too many values for 'axis'");
            goto fail;
        }
        for (i = 0; i < naxes; ++i) {
            PyObject *tmp = PyTuple_GET_ITEM(axes_obj, i);
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
    else {
        /* Try to interpret axis as an integer */
        int axis = PyArray_PyIntAsInt(axes_obj);
        /* TODO: PyNumber_Index would be good to use here */
        if (error_converting(axis)) {
            goto fail;
        }
        /*
         * As a special case for backwards compatibility in 'sum',
         * 'prod', et al, also allow a reduction for scalars even
         * though this is technically incorrect.
         */
        if (ndim == 0 && (axis == 0 || axis == -1)) {
            naxes = 0;
        }
        else if (check_and_adjust_axis(&axis, ndim) < 0) {
            goto fail;
        }
        else {
            axes[0] = (int)axis;
            naxes = 1;
        }
    }

    switch(operation) {
    case UFUNC_REDUCE:
        ret = PyUFunc_Reduce(ufunc,
                mp, out, naxes, axes, signature, keepdims, initial, wheremask);
        Py_XSETREF(wheremask, NULL);
        break;
    case UFUNC_ACCUMULATE:
        if (ndim == 0) {
            PyErr_SetString(PyExc_TypeError, "cannot accumulate on a scalar");
            goto fail;
        }
        if (naxes != 1) {
            PyErr_SetString(PyExc_ValueError,
                        "accumulate does not allow multiple axes");
            goto fail;
        }
        ret = (PyArrayObject *)PyUFunc_Accumulate(ufunc,
                mp, out, axes[0], signature);
        break;
    case UFUNC_REDUCEAT:
        if (ndim == 0) {
            PyErr_SetString(PyExc_TypeError, "cannot reduceat on a scalar");
            goto fail;
        }
        if (naxes != 1) {
            PyErr_SetString(PyExc_ValueError,
                        "reduceat does not allow multiple axes");
            goto fail;
        }
        ret = (PyArrayObject *)PyUFunc_Reduceat(ufunc,
                mp, indices, out, axes[0], signature);
        Py_SETREF(indices, NULL);
        break;
    }
    if (ret == NULL) {
        goto fail;
    }

    Py_DECREF(signature[0]);
    Py_DECREF(signature[1]);
    Py_DECREF(signature[2]);

    Py_DECREF(mp);
    Py_XDECREF(full_args.in);
    Py_XDECREF(full_args.out);

    /* Wrap and return the output */
    {
        /* Find __array_wrap__ - note that these rules are different to the
         * normal ufunc path
         */
        PyObject *wrap;
        if (out != NULL) {
            wrap = Py_None;
            Py_INCREF(wrap);
        }
        else if (Py_TYPE(op) != Py_TYPE(ret)) {
            wrap = PyObject_GetAttr(op, npy_um_str_array_wrap);
            if (wrap == NULL) {
                PyErr_Clear();
            }
            else if (!PyCallable_Check(wrap)) {
                Py_DECREF(wrap);
                wrap = NULL;
            }
        }
        else {
            wrap = NULL;
        }
        return _apply_array_wrap(wrap, ret, NULL);
    }

fail:
    Py_XDECREF(signature[0]);
    Py_XDECREF(signature[1]);
    Py_XDECREF(signature[2]);

    Py_XDECREF(mp);
    Py_XDECREF(wheremask);
    Py_XDECREF(indices);
    Py_XDECREF(full_args.in);
    Py_XDECREF(full_args.out);
    return NULL;
}


/*
 * Perform a basic check on `dtype`, `sig`, and `signature` since only one
 * may be set.  If `sig` is used, writes it into `out_signature` (which should
 * be set to `signature_obj` so that following code only requires to handle
 * `signature_obj`).
 *
 * Does NOT incref the output!  This only copies the borrowed references
 * gotten during the argument parsing.
 *
 * This function does not do any normalization of the input dtype tuples,
 * this happens after the array-ufunc override check currently.
 */
static int
_check_and_copy_sig_to_signature(
        PyObject *sig_obj, PyObject *signature_obj, PyObject *dtype,
        PyObject **out_signature)
{
    *out_signature = NULL;
    if (signature_obj != NULL) {
        *out_signature = signature_obj;
    }

    if (sig_obj != NULL) {
        if (*out_signature != NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "cannot specify both 'sig' and 'signature'");
            *out_signature = NULL;
            return -1;
        }
        *out_signature = sig_obj;
    }

    if (dtype != NULL) {
        if (*out_signature != NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "cannot specify both 'signature' and 'dtype'");
            return -1;
        }
        /* dtype needs to be converted, delay after the override check */
    }
    return 0;
}


/*
 * Note: This function currently lets DType classes pass, but in general
 * the class (not the descriptor instance) is the preferred input, so the
 * parsing should eventually be adapted to prefer classes and possible
 * deprecated instances. (Users should not notice that much, since `np.float64`
 * or "float64" usually denotes the DType class rather than the instance.)
 */
static PyArray_DTypeMeta *
_get_dtype(PyObject *dtype_obj) {
    if (PyObject_TypeCheck(dtype_obj, &PyArrayDTypeMeta_Type)) {
        Py_INCREF(dtype_obj);
        return (PyArray_DTypeMeta *)dtype_obj;
    }
    else {
        PyArray_Descr *descr = NULL;
        if (!PyArray_DescrConverter(dtype_obj, &descr)) {
            return NULL;
        }
        PyArray_DTypeMeta *out = NPY_DTYPE(descr);
        if (NPY_UNLIKELY(!NPY_DT_is_legacy(out))) {
            /* TODO: this path was unreachable when added. */
            PyErr_SetString(PyExc_TypeError,
                    "Cannot pass a new user DType instance to the `dtype` or "
                    "`signature` arguments of ufuncs. Pass the DType class "
                    "instead.");
            Py_DECREF(descr);
            return NULL;
        }
        else if (NPY_UNLIKELY(out->singleton != descr)) {
            /* This does not warn about `metadata`, but units is important. */
            if (out->singleton == NULL
                    || !PyArray_EquivTypes(out->singleton, descr)) {
                PyErr_SetString(PyExc_TypeError,
                        "The `dtype` and `signature` arguments to "
                        "ufuncs only select the general DType and not details "
                        "such as the byte order or time unit. "
                        "You can avoid this error by using the scalar types "
                        "`np.float64` or the dtype string notation.");
                Py_DECREF(descr);
                return NULL;
            }
        }
        Py_INCREF(out);
        Py_DECREF(descr);
        return out;
    }
}


/*
 * Finish conversion parsing of the DType signature.  NumPy always only
 * honored the type number for passed in descriptors/dtypes.
 * The `dtype` argument is interpreted as the first output DType (not
 * descriptor).
 * Unlike the dtype of an `out` array, it influences loop selection!
 *
 * It is the callers responsibility to clean `signature` and NULL it before
 * calling.
 */
static int
_get_fixed_signature(PyUFuncObject *ufunc,
        PyObject *dtype_obj, PyObject *signature_obj,
        PyArray_DTypeMeta **signature)
{
    if (dtype_obj == NULL && signature_obj == NULL) {
        return 0;
    }

    int nin = ufunc->nin, nout = ufunc->nout, nop = nin + nout;

    if (dtype_obj != NULL) {
        if (dtype_obj == Py_None) {
            /* If `dtype=None` is passed, no need to do anything */
            return 0;
        }
        if (nout == 0) {
            /* This may be allowed (NumPy does not do this)? */
            PyErr_SetString(PyExc_TypeError,
                    "Cannot provide `dtype` when a ufunc has no outputs");
            return -1;
        }
        PyArray_DTypeMeta *dtype = _get_dtype(dtype_obj);
        if (dtype == NULL) {
            return -1;
        }
        for (int i = nin; i < nop; i++) {
            Py_INCREF(dtype);
            signature[i] = dtype;
        }
        Py_DECREF(dtype);
        return 0;
    }

    assert(signature_obj != NULL);
    /* Fill in specified_types from the tuple or string (signature_obj) */
    if (PyTuple_Check(signature_obj)) {
        Py_ssize_t n = PyTuple_GET_SIZE(signature_obj);
        if (n == 1 && nop != 1) {
            /*
             * Special handling, because we deprecate this path.  The path
             * probably mainly existed since the `dtype=obj` was passed through
             * as `(obj,)` and parsed later.
             */
            if (PyTuple_GET_ITEM(signature_obj, 0) == Py_None) {
                PyErr_SetString(PyExc_TypeError,
                        "a single item type tuple cannot contain None.");
                return -1;
            }
            if (DEPRECATE("The use of a length 1 tuple for the ufunc "
                          "`signature` is deprecated. Use `dtype` or  fill the"
                          "tuple with `None`s.") < 0) {
                return -1;
            }
            /* Use the same logic as for `dtype=` */
            return _get_fixed_signature(ufunc,
                    PyTuple_GET_ITEM(signature_obj, 0), NULL, signature);
        }
        if (n != nop) {
            PyErr_Format(PyExc_ValueError,
                    "a type-tuple must be specified of length %d for ufunc '%s'",
                    nop, ufunc_get_name_cstr(ufunc));
            return -1;
        }
        for (int i = 0; i < nop; ++i) {
            PyObject *item = PyTuple_GET_ITEM(signature_obj, i);
            if (item == Py_None) {
                continue;
            }
            else {
                signature[i] = _get_dtype(item);
                if (signature[i] == NULL) {
                    return -1;
                }
                else if (i < nin && NPY_DT_is_abstract(signature[i])) {
                    /*
                     * We reject abstract input signatures for now.  These
                     * can probably be defined by finding the common DType with
                     * the actual input and using the result of this for the
                     * promotion.
                     */
                    PyErr_SetString(PyExc_TypeError,
                            "Input DTypes to the signature must not be "
                            "abstract.  The behaviour may be defined in the "
                            "future.");
                    return -1;
                }
            }
        }
    }
    else if (PyBytes_Check(signature_obj) || PyUnicode_Check(signature_obj)) {
        PyObject *str_object = NULL;

        if (PyBytes_Check(signature_obj)) {
            str_object = PyUnicode_FromEncodedObject(signature_obj, NULL, NULL);
            if (str_object == NULL) {
                return -1;
            }
        }
        else {
            Py_INCREF(signature_obj);
            str_object = signature_obj;
        }

        Py_ssize_t length;
        const char *str = PyUnicode_AsUTF8AndSize(str_object, &length);
        if (str == NULL) {
            Py_DECREF(str_object);
            return -1;
        }

        if (length != 1 && (length != nin+nout + 2 ||
                            str[nin] != '-' || str[nin+1] != '>')) {
            PyErr_Format(PyExc_ValueError,
                    "a type-string for %s, %d typecode(s) before and %d after "
                    "the -> sign", ufunc_get_name_cstr(ufunc), nin, nout);
            Py_DECREF(str_object);
            return -1;
        }
        if (length == 1 && nin+nout != 1) {
            Py_DECREF(str_object);
            if (DEPRECATE("The use of a length 1 string for the ufunc "
                          "`signature` is deprecated. Use `dtype` attribute or "
                          "pass a tuple with `None`s.") < 0) {
                return -1;
            }
            /* `signature="l"` is the same as `dtype="l"` */
            return _get_fixed_signature(ufunc, str_object, NULL, signature);
        }
        else {
            for (int i = 0; i < nin+nout; ++i) {
                npy_intp istr = i < nin ? i : i+2;
                PyArray_Descr *descr = PyArray_DescrFromType(str[istr]);
                if (descr == NULL) {
                    Py_DECREF(str_object);
                    return -1;
                }
                signature[i] = NPY_DTYPE(descr);
                Py_INCREF(signature[i]);
                Py_DECREF(descr);
            }
            Py_DECREF(str_object);
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "the signature object to ufunc must be a string or a tuple.");
        return -1;
    }
    return 0;
}


/*
 * Fill in the actual descriptors used for the operation.  This function
 * supports falling back to the legacy `ufunc->type_resolver`.
 *
 * We guarantee the array-method that all passed in descriptors are of the
 * correct DType instance (i.e. a string can just fetch the length, it doesn't
 * need to "cast" to string first).
 */
static int
resolve_descriptors(int nop,
        PyUFuncObject *ufunc, PyArrayMethodObject *ufuncimpl,
        PyArrayObject *operands[], PyArray_Descr *dtypes[],
        PyArray_DTypeMeta *signature[], NPY_CASTING casting)
{
    int retval = -1;
    PyArray_Descr *original_dtypes[NPY_MAXARGS];

    for (int i = 0; i < nop; ++i) {
        if (operands[i] == NULL) {
            original_dtypes[i] = NULL;
        }
        else {
            /*
             * The dtype may mismatch the signature, in which case we need
             * to make it fit before calling the resolution.
             */
            PyArray_Descr *descr = PyArray_DTYPE(operands[i]);
            original_dtypes[i] = PyArray_CastDescrToDType(descr, signature[i]);
            if (original_dtypes[i] == NULL) {
                nop = i;  /* only this much is initialized */
                goto finish;
            }
        }
    }

    NPY_UF_DBG_PRINT("Resolving the descriptors\n");

    if (ufuncimpl->resolve_descriptors != &wrapped_legacy_resolve_descriptors) {
        /* The default: use the `ufuncimpl` as nature intended it */
        npy_intp view_offset = NPY_MIN_INTP;  /* currently ignored */

        NPY_CASTING safety = ufuncimpl->resolve_descriptors(ufuncimpl,
                signature, original_dtypes, dtypes, &view_offset);
        if (safety < 0) {
            goto finish;
        }
        if (NPY_UNLIKELY(PyArray_MinCastSafety(safety, casting) != casting)) {
            /* TODO: Currently impossible to reach (specialized unsafe loop) */
            PyErr_Format(PyExc_TypeError,
                    "The ufunc implementation for %s with the given dtype "
                    "signature is not possible under the casting rule %s",
                    ufunc_get_name_cstr(ufunc), npy_casting_to_string(casting));
            goto finish;
        }
        retval = 0;
    }
    else {
        /*
         * Fall-back to legacy resolver using `operands`, used exclusively
         * for datetime64/timedelta64 and custom ufuncs (in pyerfa/astropy).
         */
        retval = ufunc->type_resolver(ufunc, casting, operands, NULL, dtypes);
    }

  finish:
    for (int i = 0; i < nop; i++) {
        Py_XDECREF(original_dtypes[i]);
    }
    return retval;
}


/**
 * Wraps all outputs and returns the result (which may be NULL on error).
 *
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
 *
 * @param ufunc
 * @param full_args Original inputs and outputs
 * @param subok Whether subclasses are allowed
 * @param result_arrays The ufunc result(s).  REFERENCES ARE STOLEN!
 */
static PyObject *
replace_with_wrapped_result_and_return(PyUFuncObject *ufunc,
        ufunc_full_args full_args, npy_bool subok,
        PyArrayObject *result_arrays[])
{
    PyObject *retobj[NPY_MAXARGS];
    PyObject *wraparr[NPY_MAXARGS];
    _find_array_wrap(full_args, subok, wraparr, ufunc->nin, ufunc->nout);

    /* wrap outputs */
    for (int i = 0; i < ufunc->nout; i++) {
        _ufunc_context context;

        context.ufunc = ufunc;
        context.args = full_args;
        context.out_i = i;

        retobj[i] = _apply_array_wrap(wraparr[i], result_arrays[i], &context);
        result_arrays[i] = NULL;  /* Was DECREF'ed and (probably) wrapped */
        if (retobj[i] == NULL) {
            goto fail;
        }
    }

    if (ufunc->nout == 1) {
        return retobj[0];
    }
    else {
        PyObject *result = PyTuple_New(ufunc->nout);
        if (result == NULL) {
            return NULL;
        }
        for (int i = 0; i < ufunc->nout; i++) {
            PyTuple_SET_ITEM(result, i, retobj[i]);
        }
        return result;
    }

  fail:
    for (int i = 0; i < ufunc->nout; i++) {
        if (result_arrays[i] != NULL) {
            Py_DECREF(result_arrays[i]);
        }
        else {
            Py_XDECREF(retobj[i]);
        }
    }
    return NULL;
}


/*
 * Main ufunc call implementation.
 *
 * This implementation makes use of the "fastcall" way of passing keyword
 * arguments and is called directly from `ufunc_generic_vectorcall` when
 * Python has `tp_vectorcall` (Python 3.8+).
 * If `tp_vectorcall` is not available, the dictionary `kwargs` are unpacked in
 * `ufunc_generic_call` with fairly little overhead.
 */
static PyObject *
ufunc_generic_fastcall(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames,
        npy_bool outer)
{
    int errval;
    int nin = ufunc->nin, nout = ufunc->nout, nop = ufunc->nargs;

    /* All following variables are cleared in the `fail` error path */
    ufunc_full_args full_args;
    PyArrayObject *wheremask = NULL;

    PyArray_DTypeMeta *signature[NPY_MAXARGS];
    PyArrayObject *operands[NPY_MAXARGS];
    PyArray_DTypeMeta *operand_DTypes[NPY_MAXARGS];
    PyArray_Descr *operation_descrs[NPY_MAXARGS];
    PyObject *output_array_prepare[NPY_MAXARGS];
    /* Initialize all arrays (we usually only need a small part) */
    memset(signature, 0, nop * sizeof(*signature));
    memset(operands, 0, nop * sizeof(*operands));
    memset(operand_DTypes, 0, nop * sizeof(*operation_descrs));
    memset(operation_descrs, 0, nop * sizeof(*operation_descrs));
    memset(output_array_prepare, 0, nout * sizeof(*output_array_prepare));

    /*
     * Note that the input (and possibly output) arguments are passed in as
     * positional arguments. We extract these first and check for `out`
     * passed by keyword later.
     * Outputs and inputs are stored in `full_args.in` and `full_args.out`
     * as tuples (or NULL when no outputs are passed).
     */

    /* Check number of arguments */
    if (NPY_UNLIKELY((len_args < nin) || (len_args > nop))) {
        PyErr_Format(PyExc_TypeError,
                "%s() takes from %d to %d positional arguments but "
                "%zd were given",
                ufunc_get_name_cstr(ufunc) , nin, nop, len_args);
        return NULL;
    }

    /* Fetch input arguments. */
    full_args.in = PyArray_TupleFromItems(ufunc->nin, args, 0);
    if (full_args.in == NULL) {
        return NULL;
    }

    /*
     * If there are more arguments, they define the out args. Otherwise
     * full_args.out is NULL for now, and the `out` kwarg may still be passed.
     */
    npy_bool out_is_passed_by_position = len_args > nin;
    if (out_is_passed_by_position) {
        npy_bool all_none = NPY_TRUE;

        full_args.out = PyTuple_New(nout);
        if (full_args.out == NULL) {
            goto fail;
        }
        for (int i = nin; i < nop; i++) {
            PyObject *tmp;
            if (i < (int)len_args) {
                tmp = args[i];
                if (tmp != Py_None) {
                    all_none = NPY_FALSE;
                }
            }
            else {
                tmp = Py_None;
            }
            Py_INCREF(tmp);
            PyTuple_SET_ITEM(full_args.out, i-nin, tmp);
        }
        if (all_none) {
            Py_SETREF(full_args.out, NULL);
        }
    }
    else {
        full_args.out = NULL;
    }

    /*
     * We have now extracted (but not converted) the input arguments.
     * To simplify overrides, extract all other arguments (as objects only)
     */
    PyObject *out_obj = NULL, *where_obj = NULL;
    PyObject *axes_obj = NULL, *axis_obj = NULL;
    PyObject *keepdims_obj = NULL, *casting_obj = NULL, *order_obj = NULL;
    PyObject *subok_obj = NULL, *signature_obj = NULL, *sig_obj = NULL;
    PyObject *dtype_obj = NULL, *extobj = NULL;

    /* Skip parsing if there are no keyword arguments, nothing left to do */
    if (kwnames != NULL) {
        if (!ufunc->core_enabled) {
            NPY_PREPARE_ARGPARSER;

            if (npy_parse_arguments(ufunc->name, args + len_args, 0, kwnames,
                    "$out", NULL, &out_obj,
                    "$where", NULL, &where_obj,
                    "$casting", NULL, &casting_obj,
                    "$order", NULL, &order_obj,
                    "$subok", NULL, &subok_obj,
                    "$dtype", NULL, &dtype_obj,
                    "$signature", NULL, &signature_obj,
                    "$sig", NULL, &sig_obj,
                    "$extobj", NULL, &extobj,
                    NULL, NULL, NULL) < 0) {
                goto fail;
            }
        }
        else {
            NPY_PREPARE_ARGPARSER;

            if (npy_parse_arguments(ufunc->name, args + len_args, 0, kwnames,
                    "$out", NULL, &out_obj,
                    "$axes", NULL, &axes_obj,
                    "$axis", NULL, &axis_obj,
                    "$keepdims", NULL, &keepdims_obj,
                    "$casting", NULL, &casting_obj,
                    "$order", NULL, &order_obj,
                    "$subok", NULL, &subok_obj,
                    "$dtype", NULL, &dtype_obj,
                    "$signature", NULL, &signature_obj,
                    "$sig", NULL, &sig_obj,
                    "$extobj", NULL, &extobj,
                    NULL, NULL, NULL) < 0) {
                goto fail;
            }
            if (NPY_UNLIKELY((axes_obj != NULL) && (axis_obj != NULL))) {
                PyErr_SetString(PyExc_TypeError,
                        "cannot specify both 'axis' and 'axes'");
                goto fail;
            }
        }

        /* Handle `out` arguments passed by keyword */
        if (out_obj != NULL) {
            if (out_is_passed_by_position) {
                PyErr_SetString(PyExc_TypeError,
                        "cannot specify 'out' as both a "
                        "positional and keyword argument");
                goto fail;
            }
            if (_set_full_args_out(nout, out_obj, &full_args) < 0) {
                goto fail;
            }
        }
        /*
         * Only one of signature, sig, and dtype should be passed. If `sig`
         * was passed, this puts it into `signature_obj` instead (these
         * are borrowed references).
         */
        if (_check_and_copy_sig_to_signature(
                sig_obj, signature_obj, dtype_obj, &signature_obj) < 0) {
            goto fail;
        }
    }

    char *method;
    if (!outer) {
        method = "__call__";
    }
    else {
        method = "outer";
    }
    /* We now have all the information required to check for Overrides */
    PyObject *override = NULL;
    errval = PyUFunc_CheckOverride(ufunc, method,
            full_args.in, full_args.out, where_obj,
            args, len_args, kwnames, &override);
    if (errval) {
        goto fail;
    }
    else if (override) {
        Py_DECREF(full_args.in);
        Py_XDECREF(full_args.out);
        return override;
    }

    if (outer) {
        /* Outer uses special preparation of inputs (expand dims) */
        PyObject *new_in = prepare_input_arguments_for_outer(full_args.in, ufunc);
        if (new_in == NULL) {
            goto fail;
        }
        Py_SETREF(full_args.in, new_in);
    }

    /*
     * Parse the passed `dtype` or `signature` into an array containing
     * PyArray_DTypeMeta and/or None.
     */
    if (_get_fixed_signature(ufunc,
            dtype_obj, signature_obj, signature) < 0) {
        goto fail;
    }

    NPY_ORDER order = NPY_KEEPORDER;
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    npy_bool subok = NPY_TRUE;
    int keepdims = -1;  /* We need to know if it was passed */
    npy_bool force_legacy_promotion;
    npy_bool allow_legacy_promotion;
    npy_bool promoting_pyscalars;
    if (convert_ufunc_arguments(ufunc,
            /* extract operand related information: */
            full_args, operands,
            operand_DTypes,
            &force_legacy_promotion, &allow_legacy_promotion,
            &promoting_pyscalars,
            /* extract general information: */
            order_obj, &order,
            casting_obj, &casting,
            subok_obj, &subok,
            where_obj, &wheremask,
            keepdims_obj, &keepdims) < 0) {
        goto fail;
    }

    /*
     * Note that part of the promotion is to the complete the signature
     * (until here it only represents the fixed part and is usually NULLs).
     *
     * After promotion, we could push the following logic into the ArrayMethod
     * in the future.  For now, we do it here.  The type resolution step can
     * be shared between the ufunc and gufunc code.
     */
    PyArrayMethodObject *ufuncimpl = promote_and_get_ufuncimpl(ufunc,
            operands, signature,
            operand_DTypes, force_legacy_promotion, allow_legacy_promotion,
            promoting_pyscalars, NPY_FALSE);
    if (ufuncimpl == NULL) {
        goto fail;
    }

    /* Find the correct descriptors for the operation */
    if (resolve_descriptors(nop, ufunc, ufuncimpl,
            operands, operation_descrs, signature, casting) < 0) {
        goto fail;
    }

    if (promoting_pyscalars) {
        /*
         * Python integers need to be cast specially.  For other python
         * scalars it does not hurt either.  It would be nice to never create
         * the array in this case, but that is difficult until value-based
         * promotion rules are gone.  (After that, we may get away with using
         * dummy arrays rather than real arrays for the legacy resolvers.)
         */
        for (int i = 0; i < nin; i++) {
            int orig_flags = PyArray_FLAGS(operands[i]);
            if (!(orig_flags & NPY_ARRAY_WAS_PYTHON_LITERAL)) {
                continue;
            }
            /*
             * If descriptor matches, no need to convert, but integers may
             * have been too large.
             */
            if (!(orig_flags & NPY_ARRAY_WAS_INT_AND_REPLACED)
                    && PyArray_EquivTypes(
                        PyArray_DESCR(operands[i]), operation_descrs[i])) {
                continue;
            }
            /* Otherwise, replace the operand with a new array */
            PyArray_Descr *descr = operation_descrs[i];
            Py_INCREF(descr);
            PyArrayObject *new = (PyArrayObject *)PyArray_NewFromDescr(
                    &PyArray_Type, descr, 0, NULL, NULL, NULL, 0, NULL);
            Py_SETREF(operands[i], new);
            if (operands[i] == NULL) {
                goto fail;
            }

            PyObject *value = PyTuple_GET_ITEM(full_args.in, i);
            if (PyArray_SETITEM(new, PyArray_BYTES(operands[i]), value) < 0) {
                goto fail;
            }
        }
    }

    if (subok) {
        _find_array_prepare(full_args, output_array_prepare, nout);
    }

    /*
     * Do the final preparations and call the inner-loop.
     */
    if (!ufunc->core_enabled) {
        errval = PyUFunc_GenericFunctionInternal(ufunc, ufuncimpl,
                operation_descrs, operands, extobj, casting, order,
                output_array_prepare, full_args,  /* for __array_prepare__ */
                wheremask);
    }
    else {
        errval = PyUFunc_GeneralizedFunctionInternal(ufunc, ufuncimpl,
                operation_descrs, operands, extobj, casting, order,
                /* GUFuncs never (ever) called __array_prepare__! */
                axis_obj, axes_obj, keepdims);
    }
    if (errval < 0) {
        goto fail;
    }

    /*
     * Clear all variables which are not needed any further.
     * (From here on, we cannot `goto fail` any more.)
     */
    Py_XDECREF(wheremask);
    for (int i = 0; i < nop; i++) {
        Py_DECREF(signature[i]);
        Py_XDECREF(operand_DTypes[i]);
        Py_DECREF(operation_descrs[i]);
        if (i < nin) {
            Py_DECREF(operands[i]);
        }
        else {
            Py_XDECREF(output_array_prepare[i-nin]);
        }
    }
    /* The following steals the references to the outputs: */
    PyObject *result = replace_with_wrapped_result_and_return(ufunc,
            full_args, subok, operands+nin);
    Py_XDECREF(full_args.in);
    Py_XDECREF(full_args.out);

    return result;

fail:
    Py_XDECREF(full_args.in);
    Py_XDECREF(full_args.out);
    Py_XDECREF(wheremask);
    for (int i = 0; i < ufunc->nargs; i++) {
        Py_XDECREF(operands[i]);
        Py_XDECREF(signature[i]);
        Py_XDECREF(operand_DTypes[i]);
        Py_XDECREF(operation_descrs[i]);
        if (i < nout) {
            Py_XDECREF(output_array_prepare[i]);
        }
    }
    return NULL;
}


/*
 * Implement vectorcallfunc which should be defined with Python 3.8+.
 * In principle this could be backported, but the speed gain seems moderate
 * since ufunc calls often do not have keyword arguments and always have
 * a large overhead. The only user would potentially be cython probably.
 */
static PyObject *
ufunc_generic_vectorcall(PyObject *ufunc,
        PyObject *const *args, size_t len_args, PyObject *kwnames)
{
    /*
     * Unlike METH_FASTCALL, `len_args` may have a flag to signal that
     * args[-1] may be (temporarily) used. So normalize it here.
     */
    return ufunc_generic_fastcall((PyUFuncObject *)ufunc,
            args, PyVectorcall_NARGS(len_args), kwnames, NPY_FALSE);
}


NPY_NO_EXPORT PyObject *
ufunc_geterr(PyObject *NPY_UNUSED(dummy), PyObject *NPY_UNUSED(arg))
{
    PyObject *thedict;
    PyObject *res;

    thedict = PyThreadState_GetDict();
    if (thedict == NULL) {
        thedict = PyEval_GetBuiltins();
    }
    res = PyDict_GetItemWithError(thedict, npy_um_str_pyvals_name);
    if (res == NULL && PyErr_Occurred()) {
        return NULL;
    }
    else if (res != NULL) {
        Py_INCREF(res);
        return res;
    }
    /* Construct list of defaults */
    res = PyList_New(3);
    if (res == NULL) {
        return NULL;
    }
    PyList_SET_ITEM(res, 0, PyLong_FromLong(NPY_BUFSIZE));
    PyList_SET_ITEM(res, 1, PyLong_FromLong(UFUNC_ERR_DEFAULT));
    PyList_SET_ITEM(res, 2, Py_None); Py_INCREF(Py_None);
    return res;
}


NPY_NO_EXPORT PyObject *
ufunc_seterr(PyObject *NPY_UNUSED(dummy), PyObject *arg)
{
    PyObject *thedict;
    int res;
    PyObject *val = arg;
    static char *msg = "Error object must be a list of length 3";

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
                               const int *signature,
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
        nin, nout, identity, name, doc, unused, NULL);
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     int unused, const char *signature)
{
    return PyUFunc_FromFuncAndDataAndSignatureAndIdentity(
        func, data, types, ntypes, nin, nout, identity, name, doc,
        unused, signature, NULL);
}

/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyUFunc_FromFuncAndDataAndSignatureAndIdentity(PyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     const int unused, const char *signature,
                                     PyObject *identity_value)
{
    PyUFuncObject *ufunc;
    if (nin + nout > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Cannot construct a ufunc with more than %d operands "
                     "(requested number were: inputs = %d and outputs = %d)",
                     NPY_MAXARGS, nin, nout);
        return NULL;
    }

    ufunc = PyObject_GC_New(PyUFuncObject, &PyUFunc_Type);
    /*
     * We use GC_New here for ufunc->obj, but do not use GC_Track since
     * ufunc->obj is still NULL at the end of this function.
     * See ufunc_frompyfunc where ufunc->obj is set and GC_Track is called.
     */
    if (ufunc == NULL) {
        return NULL;
    }

    ufunc->nin = nin;
    ufunc->nout = nout;
    ufunc->nargs = nin+nout;
    ufunc->identity = identity;
    if (ufunc->identity == PyUFunc_IdentityValue) {
        Py_INCREF(identity_value);
        ufunc->identity_value = identity_value;
    }
    else {
        ufunc->identity_value = NULL;
    }

    ufunc->functions = func;
    ufunc->data = data;
    ufunc->types = types;
    ufunc->ntypes = ntypes;
    ufunc->core_signature = NULL;
    ufunc->core_enabled = 0;
    ufunc->obj = NULL;
    ufunc->core_num_dims = NULL;
    ufunc->core_num_dim_ix = 0;
    ufunc->core_offsets = NULL;
    ufunc->core_dim_ixs = NULL;
    ufunc->core_dim_sizes = NULL;
    ufunc->core_dim_flags = NULL;
    ufunc->userloops = NULL;
    ufunc->ptr = NULL;
    ufunc->vectorcall = &ufunc_generic_vectorcall;
    ufunc->reserved1 = 0;
    ufunc->iter_flags = 0;

    /* Type resolution and inner loop selection functions */
    ufunc->type_resolver = &PyUFunc_DefaultTypeResolver;
    ufunc->legacy_inner_loop_selector = &PyUFunc_DefaultLegacyInnerLoopSelector;
    ufunc->_always_null_previously_masked_innerloop_selector = NULL;

    ufunc->op_flags = NULL;
    ufunc->_loops = NULL;
    if (nin + nout != 0) {
        ufunc->_dispatch_cache = PyArrayIdentityHash_New(nin + nout);
        if (ufunc->_dispatch_cache == NULL) {
            Py_DECREF(ufunc);
            return NULL;
        }
    }
    else {
        /*
         * Work around a test that seems to do this right now, it should not
         * be a valid ufunc at all though, so. TODO: Remove...
         */
        ufunc->_dispatch_cache = NULL;
    }
    ufunc->_loops = PyList_New(0);
    if (ufunc->_loops == NULL) {
        Py_DECREF(ufunc);
        return NULL;
    }

    if (name == NULL) {
        ufunc->name = "?";
    }
    else {
        ufunc->name = name;
    }
    ufunc->doc = doc;

    ufunc->op_flags = PyArray_malloc(sizeof(npy_uint32)*ufunc->nargs);
    if (ufunc->op_flags == NULL) {
        Py_DECREF(ufunc);
        return PyErr_NoMemory();
    }
    memset(ufunc->op_flags, 0, sizeof(npy_uint32)*ufunc->nargs);

    if (signature != NULL) {
        if (_parse_signature(ufunc, signature) != 0) {
            Py_DECREF(ufunc);
            return NULL;
        }
    }

    char *curr_types = ufunc->types;
    for (int i = 0; i < ntypes * (nin + nout); i += nin + nout) {
        /*
         * Add all legacy wrapping loops here. This is normally not necessary,
         * but makes sense.  It could also help/be needed to avoid issues with
         * ambiguous loops such as: `OO->?` and `OO->O` where in theory the
         * wrong loop could be picked if only the second one is added.
         */
        PyObject *info;
        PyArray_DTypeMeta *op_dtypes[NPY_MAXARGS];
        for (int arg = 0; arg < nin + nout; arg++) {
            op_dtypes[arg] = PyArray_DTypeFromTypeNum(curr_types[arg]);
            /* These DTypes are immortal and adding INCREFs: so borrow it */
            Py_DECREF(op_dtypes[arg]);
        }
        curr_types += nin + nout;

        info = add_and_return_legacy_wrapping_ufunc_loop(ufunc, op_dtypes, 1);
        if (info == NULL) {
            Py_DECREF(ufunc);
            return NULL;
        }
    }
    /*
     * TODO: I tried adding a default promoter here (either all object for
     *       some special cases, or all homogeneous).  Those are reasonable
     *       defaults, but short-cut a deprecated SciPy loop, where the
     *       homogeneous loop `ddd->d` was deprecated, but an inhomogeneous
     *       one `dld->d` should be picked.
     *       The default promoter *is* a reasonable default, but switched that
     *       behaviour.
     *       Another problem appeared due to buggy type-resolution for
     *       datetimes, this meant that `timedelta.sum(dtype="f8")` returned
     *       datetimes (and not floats or error), arguably wrong, but...
     */
    return (PyObject *)ufunc;
}


/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_SetUsesArraysAsData(void **NPY_UNUSED(data), size_t NPY_UNUSED(i))
{
    /* NumPy 1.21, 201-03-29 */
    PyErr_SetString(PyExc_RuntimeError,
            "PyUFunc_SetUsesArraysAsData() C-API function has been "
            "disabled.  It was initially deprecated in NumPy 1.19.");
    return -1;
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
static inline void
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

static void
_loop1d_list_free(PyObject *ptr)
{
    PyUFunc_Loop1d *data = (PyUFunc_Loop1d *)PyCapsule_GetPointer(ptr, NULL);
    _free_loop1d_list(data);
}


/*
 * This function allows the user to register a 1-d loop with an already
 * created ufunc. This function is similar to RegisterLoopForType except
 * that it allows a 1-d loop to be registered with PyArray_Descr objects
 * instead of dtype type num values. This allows a 1-d loop to be registered
 * for a structured array dtype or a custom dtype. The ufunc is called
 * whenever any of it's input arguments match the user_dtype argument.
 *
 * ufunc      - ufunc object created from call to PyUFunc_FromFuncAndData
 * user_dtype - dtype that ufunc will be registered with
 * function   - 1-d loop function pointer
 * arg_dtypes - array of dtype objects describing the ufunc operands
 * data       - arbitrary data pointer passed in to loop function
 *
 * returns 0 on success, -1 for failure
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

    key = PyLong_FromLong((long) user_dtype->type_num);
    if (key == NULL) {
        return -1;
    }

    arg_typenums = PyArray_malloc(ufunc->nargs * sizeof(int));
    if (arg_typenums == NULL) {
        Py_DECREF(key);
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
        cobj = PyDict_GetItemWithError(ufunc->userloops, key);
        if (cobj == NULL && PyErr_Occurred()) {
            result = -1;
        }
        else if (cobj == NULL) {
            PyErr_SetString(PyExc_KeyError,
                "userloop for user dtype not found");
            result = -1;
        }
        else {
            int cmp = 1;
            PyUFunc_Loop1d *current = PyCapsule_GetPointer(cobj, NULL);
            if (current == NULL) {
                result = -1;
                goto done;
            }
            while (current != NULL) {
                cmp = cmp_arg_types(current->arg_types,
                    arg_typenums, ufunc->nargs);
                if (cmp >= 0 && current->arg_dtypes == NULL) {
                    break;
                }
                current = current->next;
            }
            if (cmp == 0 && current != NULL && current->arg_dtypes == NULL) {
                current->arg_dtypes = PyArray_malloc(ufunc->nargs *
                    sizeof(PyArray_Descr*));
                if (current->arg_dtypes == NULL) {
                    PyErr_NoMemory();
                    result = -1;
                    goto done;
                }
                else if (arg_dtypes != NULL) {
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
                PyErr_SetString(PyExc_RuntimeError,
                    "loop already registered");
                result = -1;
            }
        }
    }

done:
    PyArray_free(arg_typenums);

    Py_DECREF(key);

    return result;
}

/*UFUNC_API*/
NPY_NO_EXPORT int
PyUFunc_RegisterLoopForType(PyUFuncObject *ufunc,
                            int usertype,
                            PyUFuncGenericFunction function,
                            const int *arg_types,
                            void *data)
{
    PyArray_Descr *descr;
    PyUFunc_Loop1d *funcdata;
    PyObject *key, *cobj;
    PyArray_DTypeMeta *signature[NPY_MAXARGS];
    PyObject *signature_tuple = NULL;
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
    key = PyLong_FromLong((long) usertype);
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
            signature[i] = PyArray_DTypeFromTypeNum(arg_types[i]);
            Py_DECREF(signature[i]);  /* DType can't be deleted... */
        }
    }
    else {
        for (i = 0; i < ufunc->nargs; i++) {
            newtypes[i] = usertype;
            signature[i] = PyArray_DTypeFromTypeNum(usertype);
            Py_DECREF(signature[i]);  /* DType can't be deleted... */
        }
    }

    signature_tuple = PyArray_TupleFromItems(
            ufunc->nargs, (PyObject **)signature, 0);
    if (signature_tuple == NULL) {
        goto fail;
    }
    /*
     * We add the loop to the list of all loops and promoters.  If the
     * equivalent loop was already added, skip this.
     * Note that even then the ufunc is still modified: The legacy ArrayMethod
     * already looks up the inner-loop from the ufunc (and this is replaced
     * below!).
     * If the existing one is not a legacy ArrayMethod, we raise currently:
     * A new-style loop should not be replaced by an old-style one.
     */
    int add_new_loop = 1;
    for (Py_ssize_t j = 0; j < PyList_GET_SIZE(ufunc->_loops); j++) {
        PyObject *item = PyList_GET_ITEM(ufunc->_loops, j);
        PyObject *existing_tuple = PyTuple_GET_ITEM(item, 0);

        int cmp = PyObject_RichCompareBool(existing_tuple, signature_tuple, Py_EQ);
        if (cmp < 0) {
            goto fail;
        }
        if (!cmp) {
            continue;
        }
        PyObject *registered = PyTuple_GET_ITEM(item, 1);
        if (!PyObject_TypeCheck(registered, &PyArrayMethod_Type) || (
                (PyArrayMethodObject *)registered)->get_strided_loop !=
                        &get_wrapped_legacy_ufunc_loop) {
            PyErr_Format(PyExc_TypeError,
                    "A non-compatible loop was already registered for "
                    "ufunc %s and DTypes %S.",
                    ufunc_get_name_cstr(ufunc), signature_tuple);
            goto fail;
        }
        /* The loop was already added */
        add_new_loop = 0;
        break;
    }
    if (add_new_loop) {
        PyObject *info = add_and_return_legacy_wrapping_ufunc_loop(
                ufunc, signature, 0);
        if (info == NULL) {
            goto fail;
        }
    }
    /* Clearing sets it to NULL for the error paths */
    Py_CLEAR(signature_tuple);

    funcdata->func = function;
    funcdata->arg_types = newtypes;
    funcdata->data = data;
    funcdata->next = NULL;
    funcdata->arg_dtypes = NULL;
    funcdata->nargs = 0;

    /* Get entry for this user-defined type*/
    cobj = PyDict_GetItemWithError(ufunc->userloops, key);
    if (cobj == NULL && PyErr_Occurred()) {
        goto fail;
    }
    /* If it's not there, then make one and return. */
    else if (cobj == NULL) {
        cobj = PyCapsule_New((void *)funcdata, NULL, _loop1d_list_free);
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
        current = PyCapsule_GetPointer(cobj, NULL);
        if (current == NULL) {
            goto fail;
        }
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
    Py_XDECREF(signature_tuple);
    PyArray_free(funcdata);
    PyArray_free(newtypes);
    if (!PyErr_Occurred()) PyErr_NoMemory();
    return -1;
}

#undef _SETCPTR


static void
ufunc_dealloc(PyUFuncObject *ufunc)
{
    PyObject_GC_UnTrack((PyObject *)ufunc);
    PyArray_free(ufunc->core_num_dims);
    PyArray_free(ufunc->core_dim_ixs);
    PyArray_free(ufunc->core_dim_sizes);
    PyArray_free(ufunc->core_dim_flags);
    PyArray_free(ufunc->core_offsets);
    PyArray_free(ufunc->core_signature);
    PyArray_free(ufunc->ptr);
    PyArray_free(ufunc->op_flags);
    Py_XDECREF(ufunc->userloops);
    if (ufunc->identity == PyUFunc_IdentityValue) {
        Py_DECREF(ufunc->identity_value);
    }
    Py_XDECREF(ufunc->obj);
    Py_XDECREF(ufunc->_loops);
    if (ufunc->_dispatch_cache != NULL) {
        PyArrayIdentityHash_Dealloc(ufunc->_dispatch_cache);
    }
    PyObject_GC_Del(ufunc);
}

static PyObject *
ufunc_repr(PyUFuncObject *ufunc)
{
    return PyUnicode_FromFormat("<ufunc '%s'>", ufunc->name);
}

static int
ufunc_traverse(PyUFuncObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->obj);
    if (self->identity == PyUFunc_IdentityValue) {
        Py_VISIT(self->identity_value);
    }
    return 0;
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
ufunc_outer(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
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

    if (len_args != 2) {
        PyErr_SetString(PyExc_TypeError, "exactly two arguments expected");
        return NULL;
    }

    return ufunc_generic_fastcall(ufunc, args, len_args, kwnames, NPY_TRUE);
}


static PyObject *
prepare_input_arguments_for_outer(PyObject *args, PyUFuncObject *ufunc)
{
    PyArrayObject *ap1 = NULL;
    PyObject *tmp;
    static PyObject *_numpy_matrix;
    npy_cache_import("numpy", "matrix", &_numpy_matrix);

    const char *matrix_deprecation_msg = (
            "%s.outer() was passed a numpy matrix as %s argument. "
            "Special handling of matrix is deprecated and will result in an "
            "error in most cases. Please convert the matrix to a NumPy "
            "array to retain the old behaviour. You can use `matrix.A` "
            "to achieve this.");

    tmp = PyTuple_GET_ITEM(args, 0);

    if (PyObject_IsInstance(tmp, _numpy_matrix)) {
        /* DEPRECATED 2020-05-13, NumPy 1.20 */
        if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                matrix_deprecation_msg, ufunc->name, "first") < 0) {
            return NULL;
        }
        ap1 = (PyArrayObject *) PyArray_FromObject(tmp, NPY_NOTYPE, 0, 0);
    }
    else {
        ap1 = (PyArrayObject *) PyArray_FROM_O(tmp);
    }
    if (ap1 == NULL) {
        return NULL;
    }

    PyArrayObject *ap2 = NULL;
    tmp = PyTuple_GET_ITEM(args, 1);
    if (PyObject_IsInstance(tmp, _numpy_matrix)) {
        /* DEPRECATED 2020-05-13, NumPy 1.20 */
        if (PyErr_WarnFormat(PyExc_DeprecationWarning, 1,
                matrix_deprecation_msg, ufunc->name, "second") < 0) {
            Py_DECREF(ap1);
            return NULL;
        }
        ap2 = (PyArrayObject *) PyArray_FromObject(tmp, NPY_NOTYPE, 0, 0);
    }
    else {
        ap2 = (PyArrayObject *) PyArray_FROM_O(tmp);
    }
    if (ap2 == NULL) {
        Py_DECREF(ap1);
        return NULL;
    }
    /* Construct new shape from ap1 and ap2 and then reshape */
    PyArray_Dims newdims;
    npy_intp newshape[NPY_MAXDIMS];
    newdims.len = PyArray_NDIM(ap1) + PyArray_NDIM(ap2);
    newdims.ptr = newshape;

    if (newdims.len > NPY_MAXDIMS) {
        PyErr_Format(PyExc_ValueError,
                "maximum supported dimension for an ndarray is %d, but "
                "`%s.outer()` result would have %d.",
                NPY_MAXDIMS, ufunc->name, newdims.len);
        goto fail;
    }
    if (newdims.ptr == NULL) {
        goto fail;
    }
    memcpy(newshape, PyArray_DIMS(ap1), PyArray_NDIM(ap1) * sizeof(npy_intp));
    for (int i = PyArray_NDIM(ap1); i < newdims.len; i++) {
        newshape[i] = 1;
    }

    PyArrayObject *ap_new;
    ap_new = (PyArrayObject *)PyArray_Newshape(ap1, &newdims, NPY_CORDER);
    if (ap_new == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(ap_new) != newdims.len ||
           !PyArray_CompareLists(PyArray_DIMS(ap_new), newshape, newdims.len)) {
        PyErr_Format(PyExc_TypeError,
                "%s.outer() called with ndarray-subclass of type '%s' "
                "which modified its shape after a reshape. `outer()` relies "
                "on reshaping the inputs and is for example not supported for "
                "the 'np.matrix' class (the usage of matrix is generally "
                "discouraged). "
                "To work around this issue, please convert the inputs to "
                "numpy arrays.",
                ufunc->name, Py_TYPE(ap_new)->tp_name);
        Py_DECREF(ap_new);
        goto fail;
    }

    Py_DECREF(ap1);
    return Py_BuildValue("(NN)", ap_new, ap2);

 fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    return NULL;
}


static PyObject *
ufunc_reduce(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return PyUFunc_GenericReduction(
            ufunc, args, len_args, kwnames, UFUNC_REDUCE);
}

static PyObject *
ufunc_accumulate(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return PyUFunc_GenericReduction(
            ufunc, args, len_args, kwnames, UFUNC_ACCUMULATE);
}

static PyObject *
ufunc_reduceat(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return PyUFunc_GenericReduction(
            ufunc, args, len_args, kwnames, UFUNC_REDUCEAT);
}

/* Helper for ufunc_at, below */
static inline PyArrayObject *
new_array_op(PyArrayObject *op_array, char *data)
{
    npy_intp dims[1] = {1};
    Py_INCREF(PyArray_DESCR(op_array));  /* NewFromDescr steals a reference */
    PyObject *r = PyArray_NewFromDescr(&PyArray_Type, PyArray_DESCR(op_array),
                                       1, dims, NULL, data,
                                       NPY_ARRAY_WRITEABLE, NULL);
    return (PyArrayObject *)r;
}

/*
 * Use an indexed loop to do the work
 * Returns 0 if successful
 */
static int
trivial_at_loop(PyArrayMethodObject *ufuncimpl, NPY_ARRAYMETHOD_FLAGS flags,
                    PyArrayMapIterObject *iter,
                    PyArrayObject *op1_array, PyArrayObject *op2_array,
                    PyArrayMethod_Context *context)
{
    int buffersize=0, errormask = 0;
    int res;
    char *args[3];
    npy_intp steps[4];
    args[0] = (char *) iter->baseoffset;
    steps[0] = iter->fancy_strides[0];
    if (ufuncimpl->nin == 1) {
        args[2] = NULL;
        steps[2] = 0;
    } else {
        args[2] = (char *)PyArray_DATA(op2_array);
        if (PyArray_NDIM(op2_array) == 0
            || PyArray_DIM(op2_array, 0) <= 1) {
            steps[2] = 0;
        } else {
            steps[2] = PyArray_STRIDE(op2_array, 0);
        }
    }

    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        npy_clear_floatstatus_barrier((char *)context);
    }

    do {
        npy_intp *inner_size = NpyIter_GetInnerLoopSizePtr(iter->outer);
        npy_intp * indxP = (npy_intp *)iter->outer_ptrs[0];
        args[1] = (char *)indxP;
        steps[1] = iter->outer_strides[0];
        /* 
         * The value of iter->fancy_dims[0] is added to negative indexes
         * inside the inner loop
         */
        steps[3] = iter->fancy_dims[0];

        res = ufuncimpl->contiguous_indexed_loop(
                context, args, inner_size, steps, NULL);

        if (args[2] != NULL) {
            args[2] += (*inner_size) * steps[2];
        }
    } while (res == 0 && iter->outer_next(iter->outer));

    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        const char * ufunc_name =
                        ufunc_get_name_cstr((PyUFuncObject *)context->caller);
        if (_get_bufsize_errmask(NULL, ufunc_name,
                                 &buffersize, &errormask) < 0) {
            return -1;
        }
        res = _check_ufunc_fperr(errormask, NULL, ufunc_name);
    }
    return res;
}

static int
ufunc_at__fast_iter(PyUFuncObject *ufunc, NPY_ARRAYMETHOD_FLAGS flags,
                    PyArrayMapIterObject *iter, PyArrayIterObject *iter2,
                    PyArrayObject *op1_array, PyArrayObject *op2_array,
                    PyArrayMethod_StridedLoop *strided_loop,
                    PyArrayMethod_Context *context,
                    NpyAuxData *auxdata
                    )
{
    int buffersize;
    int errormask = 0;
    int res = 0;
    NPY_BEGIN_THREADS_DEF;

    if (_get_bufsize_errmask(NULL, ufunc->name, &buffersize, &errormask) < 0) {
        return -1;
    }
    int needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    npy_intp strides[3] = {0, 0, 0};
    /*
     * Iterate over first and second operands and call ufunc
     * for each pair of inputs
     */
    for (npy_intp i = iter->size; i > 0; i--)
    {
        char *dataptr[3];
        /* one element at a time, no stride required but read by innerloop */
        npy_intp count = 1;

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

        res = strided_loop(context, dataptr, &count, strides, auxdata);
        if (res != 0) {
            break;
        }

        PyArray_MapIterNext(iter);
        if (iter2 != NULL) {
            PyArray_ITER_NEXT(iter2);
        }
    }

    NPY_END_THREADS;

    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        res = _check_ufunc_fperr(errormask, NULL, "at");
    }
    return res;
}

static int
ufunc_at__slow_iter(PyUFuncObject *ufunc, NPY_ARRAYMETHOD_FLAGS flags,
                    PyArrayMapIterObject *iter, PyArrayIterObject *iter2,
                    PyArrayObject *op1_array, PyArrayObject *op2_array,
                    PyArray_Descr *operation_descrs[3],
                    PyArrayMethod_StridedLoop *strided_loop,
                    PyArrayMethod_Context *context,
                    NpyAuxData *auxdata
                    )
{
    NpyIter *iter_buffer = NULL;
    PyArrayObject *array_operands[3] = {NULL, NULL, NULL};
    int buffersize;
    int errormask = 0;
    int res = 0;
    int nop = 0;
    NpyIter_IterNextFunc *iternext;
    char * err_msg = NULL;
    NPY_BEGIN_THREADS_DEF;

    if (_get_bufsize_errmask(NULL, ufunc->name, &buffersize, &errormask) < 0) {
        return -1;
    }
    array_operands[0] = new_array_op(op1_array, iter->dataptr);
    if (iter2 != NULL) {
        array_operands[1] = new_array_op(op2_array, PyArray_ITER_DATA(iter2));
        array_operands[2] = new_array_op(op1_array, iter->dataptr);
        nop = 3;
    }
    else {
        array_operands[1] = new_array_op(op1_array, iter->dataptr);
        array_operands[2] = NULL;
        nop = 2;
    }
    /* Set up the flags */
    npy_uint32 op_flags[3];
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
                        op_flags, operation_descrs,
                        -1, NULL, NULL, buffersize);

    if (iter_buffer == NULL) {
        /* will fail only on memory allocation errors */
        for (int i = 0; i < 3; i++) {
            Py_XDECREF(array_operands[i]);
        }
        return -1;
    }

    iternext = NpyIter_GetIterNext(iter_buffer, NULL);
    if (iternext == NULL) {
        /* can not really happen, iter_buffer creation is tightly controlled */
        NpyIter_Deallocate(iter_buffer);
        for (int i = 0; i < 3; i++) {
            Py_XDECREF(array_operands[i]);
        }
        return -1;
    }

    int needs_api = (flags & NPY_METH_REQUIRES_PYAPI) != 0;
    needs_api |= NpyIter_IterationNeedsAPI(iter_buffer);
    if (!(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* Start with the floating-point exception flags cleared */
        npy_clear_floatstatus_barrier((char*)&iter);
    }

    npy_intp strides[3] = {0, 0, 0};
    if (!needs_api) {
        NPY_BEGIN_THREADS;
    }

    /*
     * Iterate over first and second operands and call ufunc
     * for each pair of inputs
     */
    for (npy_intp i = iter->size; i > 0; i--)
    {
        char *dataptr[3];
        char **buffer_dataptr;
        /* one element at a time, no stride required but read by innerloop */
        npy_intp count = 1;

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
            res = -1;
            break;
        }

        buffer_dataptr = NpyIter_GetDataPtrArray(iter_buffer);

        res = strided_loop(context, buffer_dataptr, &count, strides, auxdata);
        if (res != 0) {
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
    }

    NPY_END_THREADS;

    if (res != 0 && err_msg) {
        PyErr_SetString(PyExc_ValueError, err_msg);
    }
    if (res == 0 && !(flags & NPY_METH_NO_FLOATINGPOINT_ERRORS)) {
        /* NOTE: We could check float errors even when `res < 0` */
        res = _check_ufunc_fperr(errormask, NULL, "at");
    }
    NpyIter_Deallocate(iter_buffer);
    for (int i = 0; i < 3; i++) {
        Py_XDECREF(array_operands[i]);
    }
    return res;
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
    PyArray_Descr *operation_descrs[3] = {NULL, NULL, NULL};

    int nop;

    /* override vars */
    int errval;
    PyObject *override = NULL;
    int res = -1;   /* start with fail condition so "goto fail" will error */

    PyArrayMethod_StridedLoop *strided_loop;
    NpyAuxData *auxdata = NULL;

    if (ufunc->core_enabled) {
        PyErr_Format(PyExc_TypeError,
            "%s.at does not support ufunc with non-trivial signature: %s has signature %s.",
            ufunc->name, ufunc->name, ufunc->core_signature);
        return NULL;
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

    if (ufunc->nin == 1 && op2 != NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "second operand provided when ufunc is unary");
        return NULL;
    }
    errval = PyUFunc_CheckOverride(ufunc, "at",
            args, NULL, NULL, NULL, 0, NULL, &override);

    if (errval) {
        return NULL;
    }
    else if (override) {
        return override;
    }

    if (!PyArray_Check(op1)) {
        PyErr_SetString(PyExc_TypeError,
                        "first operand must be array");
        return NULL;
    }

    op1_array = (PyArrayObject *)op1;

    /* Create second operand from number array if needed. */
    if (op2 == NULL) {
        nop = 2;
    }
    else {
        nop = 3;
        op2_array = (PyArrayObject *)PyArray_FromAny(op2, NULL,
                                0, 0, 0, NULL);
        if (op2_array == NULL) {
            goto fail;
        }
    }

    PyArrayMethodObject *ufuncimpl = NULL;
    {
        /* Do all the dtype handling and find the correct ufuncimpl */

        PyArrayObject *tmp_operands[3] = {NULL, NULL, NULL};
        PyArray_DTypeMeta *signature[3] = {NULL, NULL, NULL};
        PyArray_DTypeMeta *operand_DTypes[3] = {NULL, NULL, NULL};
        /*
         * Create dtypes array for either one or two input operands.
         * Compare to the logic in `convert_ufunc_arguments`.
         * TODO: It may be good to review some of this behaviour, since the
         *       operand array is special (it is written to) similar to reductions.
         *       Using unsafe-casting as done here, is likely not desirable.
         */
        tmp_operands[0] = op1_array;
        operand_DTypes[0] = NPY_DTYPE(PyArray_DESCR(op1_array));
        Py_INCREF(operand_DTypes[0]);
        int force_legacy_promotion = 0;
        int allow_legacy_promotion = NPY_DT_is_legacy(operand_DTypes[0]);

        if (op2_array != NULL) {
            tmp_operands[1] = op2_array;
            operand_DTypes[1] = NPY_DTYPE(PyArray_DESCR(op2_array));
            Py_INCREF(operand_DTypes[1]);
            allow_legacy_promotion &= NPY_DT_is_legacy(operand_DTypes[1]);
            tmp_operands[2] = tmp_operands[0];
            operand_DTypes[2] = operand_DTypes[0];
            Py_INCREF(operand_DTypes[2]);

            if (allow_legacy_promotion && ((PyArray_NDIM(op1_array) == 0)
                                           != (PyArray_NDIM(op2_array) == 0))) {
                    /* both are legacy and only one is 0-D: force legacy */
                    force_legacy_promotion = should_use_min_scalar(2, tmp_operands, 0, NULL);
                }
        }
        else {
            tmp_operands[1] = tmp_operands[0];
            operand_DTypes[1] = operand_DTypes[0];
            Py_INCREF(operand_DTypes[1]);
            tmp_operands[2] = NULL;
        }

        ufuncimpl = promote_and_get_ufuncimpl(ufunc, tmp_operands, signature,
                        operand_DTypes, force_legacy_promotion,
                        allow_legacy_promotion, NPY_FALSE, NPY_FALSE);
        if (ufuncimpl == NULL) {
            for (int i = 0; i < 3; i++) {
                Py_XDECREF(signature[i]);
                Py_XDECREF(operand_DTypes[i]);
            }
            goto fail;
        }

        /* Find the correct operation_descrs for the operation */
        int resolve_result = resolve_descriptors(nop, ufunc, ufuncimpl,
                tmp_operands, operation_descrs, signature, NPY_UNSAFE_CASTING);
        for (int i = 0; i < 3; i++) {
            Py_XDECREF(signature[i]);
            Py_XDECREF(operand_DTypes[i]);
        }
        if (resolve_result < 0) {
            goto fail;
        }
    }

    iter = (PyArrayMapIterObject *)PyArray_MapIterArrayCopyIfOverlap(
        op1_array, idx, 1, op2_array);
    if (iter == NULL) {
        goto fail;
    }
    op1_array = iter->array;  /* May be updateifcopied on overlap */

    if (op2_array != NULL) {
        /*
         * May need to swap axes so that second operand is
         * iterated over correctly
         */
        if ((iter->subspace != NULL) && (iter->consec)) {
            PyArray_MapIterSwapAxes(iter, &op2_array, 0);
            if (op2_array == NULL) {
                /* only on memory allocation failure */
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

    PyArrayMethod_Context context = {
            .caller = (PyObject *)ufunc,
            .method = ufuncimpl,
            .descriptors = operation_descrs,
    };

    /* Use contiguous strides; if there is such a loop it may be faster */
    npy_intp strides[3] = {
            operation_descrs[0]->elsize, operation_descrs[1]->elsize, 0};
    if (nop == 3) {
        strides[2] = operation_descrs[2]->elsize;
    }

    NPY_ARRAYMETHOD_FLAGS flags;
    if (ufuncimpl->get_strided_loop(&context, 1, 0, strides,
            &strided_loop, &auxdata, &flags) < 0) {
        goto fail;
    }
    int fast_path = 1;
    /* check no casting, alignment */
    if (PyArray_DESCR(op1_array) != operation_descrs[0]) {
        fast_path = 0;
    }
    if (PyArray_DESCR(op1_array) != operation_descrs[nop - 1]) {
        /* output casting */
        fast_path = 0;
    }
    if (!PyArray_ISALIGNED(op1_array)) {
        fast_path = 0;
    }
    if (nop >2) {
        if  (PyArray_DESCR(op2_array) != operation_descrs[1]) {
            fast_path = 0;
        }
        if (!PyArray_ISALIGNED(op2_array)) {
            fast_path = 0;
        }
    }
    if (fast_path == 1) {
        /*
         * Try to use trivial loop (1d, no casting, aligned) if
         * - the matching info has a indexed loop
         * - idx must be exactly one integer index array
         * - all operands are 1d
         * A future enhancement could loosen the restriction on 1d operands
         * by adding an iteration loop inside trivial_at_loop
         */
        if ((ufuncimpl->contiguous_indexed_loop != NULL) &&
                (PyArray_NDIM(op1_array) == 1)  &&
                (op2_array == NULL || PyArray_NDIM(op2_array) <= 1) &&
                (iter->subspace_iter == NULL) && (iter->numiter == 1)) {
            res = trivial_at_loop(ufuncimpl, flags, iter, op1_array,
                        op2_array, &context);

        }
        else {
            /* Couldn't use the fastest path, use the faster path */
            res = ufunc_at__fast_iter(ufunc, flags, iter, iter2, op1_array,
                        op2_array, strided_loop, &context, auxdata);
        }
    } else {
        res = ufunc_at__slow_iter(ufunc, flags, iter, iter2, op1_array,
                        op2_array, operation_descrs, strided_loop, &context,
                        auxdata);
    }

fail:
    NPY_AUXDATA_FREE(auxdata);

    Py_XDECREF(op2_array);
    Py_XDECREF(iter);
    Py_XDECREF(iter2);
    for (int i = 0; i < nop; i++) {
        Py_XDECREF(operation_descrs[i]);
    }

    /*
     * An error should only be possible if needs_api is true or `res != 0`,
     * but this is not strictly correct for old-style ufuncs
     * (e.g. `power` released the GIL but manually set an Exception).
     */
    if (res != 0 || PyErr_Occurred()) {
        /* iter_buffer has already been deallocated, don't use NpyIter_Dealloc */
        if (PyArray_FLAGS(op1_array) & NPY_ARRAY_WRITEBACKIFCOPY) {
            PyArray_DiscardWritebackIfCopy(op1_array);
        }
        return NULL;
    }
    else {
        Py_RETURN_NONE;
    }
}


typedef struct {
    PyArrayMethod_StridedLoop *strided_loop;
    PyArrayMethod_Context *context;
    NpyAuxData *auxdata;
    /* Should move to flags, but lets keep it bools for now: */
    npy_bool requires_pyapi;
    npy_bool no_floatingpoint_errors;
    PyArrayMethod_Context _full_context;
    PyArray_Descr *_descrs[];
} ufunc_call_info;


void
free_ufunc_call_info(PyObject *self)
{
    ufunc_call_info *call_info = PyCapsule_GetPointer(
            self, "numpy_1.24_ufunc_call_info");

    PyArrayMethod_Context *context = call_info->context;

    int nargs = context->method->nin + context->method->nout;
    for (int i = 0; i < nargs; i++) {
        Py_DECREF(context->descriptors[i]);
    }
    Py_DECREF(context->caller);
    Py_DECREF(context->method);
    NPY_AUXDATA_FREE(call_info->auxdata);

    PyObject_Free(call_info);
}


/*
 * Python entry-point to ufunc promotion and dtype/descr resolution.
 *
 * This function does most of the work required to execute ufunc without
 * actually executing it.
 * This can be very useful for downstream libraries that reimplement NumPy
 * functionality, such as Numba or Dask.
 */
static PyObject *
py_resolve_dtypes_generic(PyUFuncObject *ufunc, npy_bool return_context,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    NPY_PREPARE_ARGPARSER;

    PyObject *descrs_tuple;
    PyObject *signature_obj = NULL;
    NPY_CASTING casting = NPY_DEFAULT_ASSIGN_CASTING;
    npy_bool reduction = NPY_FALSE;

    if (npy_parse_arguments("resolve_dtypes", args, len_args, kwnames,
            "", NULL, &descrs_tuple,
            "$signature", NULL, &signature_obj,
            "$casting", &PyArray_CastingConverter, &casting,
            "$reduction", &PyArray_BoolConverter, &reduction,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    if (reduction && (ufunc->nin != 2 || ufunc->nout != 1)) {
        PyErr_SetString(PyExc_ValueError,
                "ufunc is not compatible with reduction operations.");
        return NULL;
    }

    /*
     * Legacy type resolvers expect NumPy arrays as input.  Until NEP 50 is
     * adopted, it is most convenient to ensure that we have an "array" object
     * before calling the type promotion.  Eventually, this hack may be moved
     * into the legacy type resolution code itself (probably after NumPy stops
     * using legacy type resolution itself for the most part).
     *
     * We make the pretty safe assumptions here that:
     * - Nobody will actually do anything with the array objects besides
     *   checking the descriptor or calling CanCast.
     * - No type resolver will cause weird paths that mess with our promotion
     *   state (or mind us messing with it).
     */
    PyObject *result = NULL;
    PyObject *result_dtype_tuple = NULL;

    PyArrayObject *dummy_arrays[NPY_MAXARGS] = {NULL};
    PyArray_DTypeMeta *DTypes[NPY_MAXARGS] = {NULL};
    PyArray_DTypeMeta *signature[NPY_MAXARGS] = {NULL};
    PyArray_Descr *operation_descrs[NPY_MAXARGS] = {NULL};

    /* This entry-point to promotion lives in the NEP 50 future: */
    int original_promotion_state = npy_promotion_state;
    npy_promotion_state = NPY_USE_WEAK_PROMOTION;

    npy_bool promoting_pyscalars = NPY_FALSE;
    npy_bool allow_legacy_promotion = NPY_TRUE;

    if (_get_fixed_signature(ufunc, NULL, signature_obj, signature) < 0) {
        goto finish;
    }

    if (!PyTuple_CheckExact(descrs_tuple)
            || PyTuple_Size(descrs_tuple) != ufunc->nargs)  {
        PyErr_SetString(PyExc_TypeError,
                "resolve_dtypes: The dtypes must be a tuple of "
                "`ufunc.nargs` length.");
        goto finish;
    }
    for (int i=0; i < ufunc->nargs; i++) {
        /*
         * We create dummy arrays for now.  It should be OK to make this
         * truly "dummy" (not even proper objects), but that is a hack better
         * left for the legacy_type_resolution wrapper when NEP 50 is done.
         */
        PyObject *descr_obj = PyTuple_GET_ITEM(descrs_tuple, i);
        PyArray_Descr *descr;

        if (PyArray_DescrCheck(descr_obj)) {
            descr = (PyArray_Descr *)descr_obj;
            Py_INCREF(descr);
            dummy_arrays[i] = (PyArrayObject *)PyArray_NewFromDescr_int(
                    &PyArray_Type, descr, 0, NULL, NULL, NULL,
                    0, NULL, NULL, _NPY_ARRAY_ENSURE_DTYPE_IDENTITY);
            if (dummy_arrays[i] == NULL) {
                goto finish;
            }
            DTypes[i] = NPY_DTYPE(descr);
            Py_INCREF(DTypes[i]);
            if (!NPY_DT_is_legacy(DTypes[i])) {
                allow_legacy_promotion = NPY_FALSE;
            }
        }
         /* Explicitly allow int, float, and complex for the "weak" types. */
        else if (descr_obj == (PyObject *)&PyLong_Type) {
            descr = PyArray_DescrFromType(NPY_LONG);
            dummy_arrays[i] = (PyArrayObject *)PyArray_Empty(0, NULL, descr, 0);
            if (dummy_arrays[i] == NULL) {
                goto finish;
            }
            PyArray_ENABLEFLAGS(dummy_arrays[i], NPY_ARRAY_WAS_PYTHON_INT);
            Py_INCREF(&PyArray_PyIntAbstractDType);
            DTypes[i] = &PyArray_PyIntAbstractDType;
            promoting_pyscalars = NPY_TRUE;
        }
        else if (descr_obj == (PyObject *)&PyFloat_Type) {
            descr = PyArray_DescrFromType(NPY_DOUBLE);
            dummy_arrays[i] = (PyArrayObject *)PyArray_Empty(0, NULL, descr, 0);
            if (dummy_arrays[i] == NULL) {
                goto finish;
            }
            PyArray_ENABLEFLAGS(dummy_arrays[i], NPY_ARRAY_WAS_PYTHON_FLOAT);
            Py_INCREF(&PyArray_PyFloatAbstractDType);
            DTypes[i] = &PyArray_PyFloatAbstractDType;
            promoting_pyscalars = NPY_TRUE;
        }
        else if (descr_obj == (PyObject *)&PyComplex_Type) {
            descr = PyArray_DescrFromType(NPY_CDOUBLE);
            dummy_arrays[i] = (PyArrayObject *)PyArray_Empty(0, NULL, descr, 0);
            if (dummy_arrays[i] == NULL) {
                goto finish;
            }
            PyArray_ENABLEFLAGS(dummy_arrays[i], NPY_ARRAY_WAS_PYTHON_COMPLEX);
            Py_INCREF(&PyArray_PyComplexAbstractDType);
            DTypes[i] = &PyArray_PyComplexAbstractDType;
            promoting_pyscalars = NPY_TRUE;
        }
        else if (descr_obj == Py_None) {
            if (i < ufunc->nin && !(reduction && i == 0)) {
                PyErr_SetString(PyExc_TypeError,
                        "All input dtypes must be provided "
                        "(except the first one in reductions)");
                goto finish;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                    "Provided dtype must be a valid NumPy dtype, "
                    "int, float, complex, or None.");
            goto finish;
        }
    }

    PyArrayMethodObject *ufuncimpl;
    if (!reduction) {
        ufuncimpl = promote_and_get_ufuncimpl(ufunc,
                dummy_arrays, signature, DTypes, NPY_FALSE,
                allow_legacy_promotion, promoting_pyscalars, NPY_FALSE);
        if (ufuncimpl == NULL) {
            goto finish;
        }

        /* Find the correct descriptors for the operation */
        if (resolve_descriptors(ufunc->nargs, ufunc, ufuncimpl,
                dummy_arrays, operation_descrs, signature, casting) < 0) {
            goto finish;
        }

        if (validate_casting(
                ufuncimpl, ufunc, dummy_arrays, operation_descrs, casting) < 0) {
            goto finish;
        }
    }
    else {  /* reduction */
        if (signature[2] != NULL) {
            PyErr_SetString(PyExc_ValueError,
                    "Reduction signature must end with None, instead pass "
                    "the first DType in the signature.");
            goto finish;
        }

        if (dummy_arrays[2] != NULL) {
            PyErr_SetString(PyExc_TypeError,
                    "Output dtype must not be passed for reductions, "
                    "pass the first input instead.");
            goto finish;
        }

        ufuncimpl = reducelike_promote_and_resolve(ufunc,
                dummy_arrays[1], dummy_arrays[0], signature, NPY_FALSE,
                operation_descrs, casting, "resolve_dtypes");

        if (ufuncimpl == NULL) {
            goto finish;
        }
    }

    result = PyArray_TupleFromItems(
            ufunc->nargs, (PyObject **)operation_descrs, 0);

    if (result == NULL || !return_context) {
        goto finish;
    }
    /* Result will be (dtype_tuple, call_info), so move it and clear result */
    result_dtype_tuple = result;
    result = NULL;

    /* We may have to return the context: */
    ufunc_call_info *call_info;
    call_info = PyObject_Malloc(sizeof(ufunc_call_info)
                              + ufunc->nargs * sizeof(PyArray_Descr *));
    if (call_info == NULL) {
        PyErr_NoMemory();
        goto finish;
    }
    call_info->strided_loop = NULL;
    call_info->auxdata = NULL;
    call_info->context = &call_info->_full_context;

    /*
     * We create a capsule with NumPy 1.24 in the name to signal that it is
     * prone to change in version updates (it doesn't have to).
     * This capsule is documented in the `ufunc._resolve_dtypes_and_context`
     * docstring.
     */
    PyObject *capsule = PyCapsule_New(
            call_info, "numpy_1.24_ufunc_call_info", &free_ufunc_call_info);
    if (capsule == NULL) {
        PyObject_Free(call_info);
        goto finish;
    }

    PyArrayMethod_Context *context = call_info->context;

    Py_INCREF(ufunc);
    context->caller = (PyObject *)ufunc;
    Py_INCREF(ufuncimpl);
    context->method = ufuncimpl;
    context->descriptors = call_info->_descrs;
    for (int i=0; i < ufunc->nargs; i++) {
        Py_INCREF(operation_descrs[i]);
        context->descriptors[i] = operation_descrs[i];
    }

    result = PyTuple_Pack(2, result_dtype_tuple, capsule);
    /* cleanup and return */
    Py_DECREF(capsule);

  finish:
    npy_promotion_state = original_promotion_state;

    Py_XDECREF(result_dtype_tuple);
    for (int i = 0; i < ufunc->nargs; i++) {
        Py_XDECREF(signature[i]);
        Py_XDECREF(dummy_arrays[i]);
        Py_XDECREF(operation_descrs[i]);
        Py_XDECREF(DTypes[i]);
    }

    return result;
}


static PyObject *
py_resolve_dtypes(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return py_resolve_dtypes_generic(ufunc, NPY_FALSE, args, len_args, kwnames);
}


static PyObject *
py_resolve_dtypes_and_context(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    return py_resolve_dtypes_generic(ufunc, NPY_TRUE, args, len_args, kwnames);
}


static PyObject *
py_get_strided_loop(PyUFuncObject *ufunc,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    NPY_PREPARE_ARGPARSER;

    PyObject *call_info_obj;
    PyObject *fixed_strides_obj = Py_None;
    npy_intp fixed_strides[NPY_MAXARGS];

    if (npy_parse_arguments("_get_strided_loop", args, len_args, kwnames,
            "", NULL, &call_info_obj,
            "$fixed_strides", NULL, &fixed_strides_obj,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    ufunc_call_info *call_info = PyCapsule_GetPointer(
                call_info_obj, "numpy_1.24_ufunc_call_info");
    if (call_info == NULL) {
        /* Cannot have a context with NULL inside... */
        assert(PyErr_Occurred());
        return NULL;
    }
    if (call_info->strided_loop != NULL) {
        PyErr_SetString(PyExc_TypeError,
                "ufunc call_info has already been filled/used!");
        return NULL;
    }

    if (call_info->context->caller != (PyObject *)ufunc) {
        PyErr_SetString(PyExc_TypeError,
                "calling get_strided_loop with incompatible context");
        return NULL;
    }

    /*
     * Strict conversion of fixed_strides, None, or tuple of int or None.
     */
    if (fixed_strides_obj == Py_None) {
        for (int i = 0; i < ufunc->nargs; i++) {
            fixed_strides[i] = NPY_MAX_INTP;
        }
    }
    else if (PyTuple_CheckExact(fixed_strides_obj)
            && PyTuple_Size(fixed_strides_obj) == ufunc->nargs) {
        for (int i = 0; i < ufunc->nargs; i++) {
            PyObject *stride = PyTuple_GET_ITEM(fixed_strides_obj, i);
            if (PyLong_CheckExact(stride)) {
                fixed_strides[i] = PyLong_AsSsize_t(stride);
                if (error_converting(fixed_strides[i])) {
                    return NULL;
                }
            }
            else if (stride == Py_None) {
                fixed_strides[i] = NPY_MAX_INTP;
            }
            else {
                PyErr_SetString(PyExc_TypeError,
                    "_get_strided_loop(): fixed_strides tuple must contain "
                    "Python ints or None");
                return NULL;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError,
            "_get_strided_loop(): fixed_strides must be a tuple or None");
        return NULL;
    }

    NPY_ARRAYMETHOD_FLAGS flags;
    if (call_info->context->method->get_strided_loop(call_info->context,
            1, 0, fixed_strides, &call_info->strided_loop, &call_info->auxdata,
            &flags) < 0) {
        return NULL;
    }

    call_info->requires_pyapi = flags & NPY_METH_REQUIRES_PYAPI;
    call_info->no_floatingpoint_errors = (
            flags & NPY_METH_NO_FLOATINGPOINT_ERRORS);

    Py_RETURN_NONE;
}


static struct PyMethodDef ufunc_methods[] = {
    {"reduce",
        (PyCFunction)ufunc_reduce,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"accumulate",
        (PyCFunction)ufunc_accumulate,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"reduceat",
        (PyCFunction)ufunc_reduceat,
        METH_FASTCALL | METH_KEYWORDS, NULL },
    {"outer",
        (PyCFunction)ufunc_outer,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"at",
        (PyCFunction)ufunc_at,
        METH_VARARGS, NULL},
    /* Lower level methods: */
    {"resolve_dtypes",
        (PyCFunction)py_resolve_dtypes,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    /*
     * The following two functions are public API, but underscored since they
     * are C-user specific and allow direct access to the core of ufunc loops.
     * (See their documentation for API stability.)
     */
    {"_resolve_dtypes_and_context",
        (PyCFunction)py_resolve_dtypes_and_context,
        METH_FASTCALL | METH_KEYWORDS, NULL},
    {"_get_strided_loop",
        (PyCFunction)py_get_strided_loop,
        METH_FASTCALL | METH_KEYWORDS, NULL},
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
ufunc_get_doc(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
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
    doc = PyObject_CallFunctionObjArgs(_sig_formatter,
                                       (PyObject *)ufunc, NULL);
    if (doc == NULL) {
        return NULL;
    }
    if (ufunc->doc != NULL) {
        Py_SETREF(doc, PyUnicode_FromFormat("%S\n\n%s", doc, ufunc->doc));
    }
    return doc;
}


static PyObject *
ufunc_get_nin(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(ufunc->nin);
}

static PyObject *
ufunc_get_nout(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(ufunc->nout);
}

static PyObject *
ufunc_get_nargs(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(ufunc->nargs);
}

static PyObject *
ufunc_get_ntypes(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyLong_FromLong(ufunc->ntypes);
}

static PyObject *
ufunc_get_types(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
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
        str = PyUnicode_FromStringAndSize(t, no + ni + 2);
        PyList_SET_ITEM(list, k, str);
    }
    PyArray_free(t);
    return list;
}

static PyObject *
ufunc_get_name(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    return PyUnicode_FromString(ufunc->name);
}

static PyObject *
ufunc_get_identity(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    npy_bool reorderable;
    return PyUFunc_GetDefaultIdentity(ufunc, &reorderable);
}

static PyObject *
ufunc_get_signature(PyUFuncObject *ufunc, void *NPY_UNUSED(ignored))
{
    if (!ufunc->core_enabled) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString(ufunc->core_signature);
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
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "numpy.ufunc",
    .tp_basicsize = sizeof(PyUFuncObject),
    .tp_dealloc = (destructor)ufunc_dealloc,
    .tp_vectorcall_offset = offsetof(PyUFuncObject, vectorcall),
    .tp_repr = (reprfunc)ufunc_repr,
    .tp_call = &PyVectorcall_Call,
    .tp_str = (reprfunc)ufunc_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT |
        _Py_TPFLAGS_HAVE_VECTORCALL |
        Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)ufunc_traverse,
    .tp_methods = ufunc_methods,
    .tp_getset = ufunc_getset,
};

/* End of code for ufunc objects */
