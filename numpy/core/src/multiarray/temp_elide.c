#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#include "npy_config.h"
#include "numpy/arrayobject.h"

#define NPY_NUMBER_MAX(a, b) ((a) > (b) ? (a) : (b))

/*
 * Functions used to try to avoid/elide temporaries in python expressions
 * of type a + b + b by translating some operations into in-place operations.
 * This example translates to this bytecode:
 *
 *        0 LOAD_FAST                0 (a)
 *        3 LOAD_FAST                1 (b)
 *        6 BINARY_ADD
 *        7 LOAD_FAST                1 (b)
 *       10 BINARY_ADD
 *
 * The two named variables get their reference count increased by the load
 * instructions so they always have a reference count larger than 1.
 * The temporary of the first BINARY_ADD on the other hand only has a count of
 * 1. Only temporaries can have a count of 1 in python so we can use this to
 * transform the second operation into an in-place operation and not affect the
 * output of the program.
 * CPython does the same thing to resize memory instead of copying when doing
 * string concatenation.
 * The gain can be very significant (4x-6x) when avoiding the temporary allows
 * the operation to remain in the cpu caches and can still be 50% faster for
 * array larger than cpu cache size.
 *
 * A complication is that a DSO (dynamic shared object) module (e.g. cython)
 * could call the PyNumber functions directly on arrays with reference count of
 * 1.
 * This is handled by checking the call stack to verify that we have been
 * called directly from the cpython interpreter.
 * To achieve this we check that all functions in the callstack until the
 * cpython frame evaluation function are located in cpython or numpy.
 * This is an expensive operation so temporaries are only avoided for rather
 * large arrays.
 *
 * A possible future improvement would be to change cpython to give us access
 * to the top of the stack. Then we could just check that the objects involved
 * are on the cpython stack instead of checking the function callstack.
 *
 * Elision can be applied to all operations that do have in-place variants and
 * do not change types (addition, subtraction, multiplication, float division,
 * logical and bitwise operations ...)
 * For commutative operations (addition, multiplication, ...) if eliding into
 * the lefthand side fails it can succeed on the righthand side by swapping the
 * arguments. E.g. b * (a * 2) can be elided by changing it to (2 * a) * b.
 *
 * TODO only supports systems with backtrace(), Windows can probably be
 * supported too by using the appropriate Windows APIs.
 */

#if defined HAVE_BACKTRACE && defined HAVE_DLFCN_H && ! defined PYPY_VERSION
/* 1 prints elided operations, 2 prints stacktraces */
#define NPY_ELIDE_DEBUG 0
#define NPY_MAX_STACKSIZE 10

#if PY_VERSION_HEX >= 0x03060000
/* TODO can pep523 be used to somehow? */
#define PYFRAMEEVAL_FUNC "_PyEval_EvalFrameDefault"
#else
#define PYFRAMEEVAL_FUNC "PyEval_EvalFrameEx"
#endif
/*
 * Heuristic size of the array in bytes at which backtrace overhead generation
 * becomes less than speed gained by in-place operations. Depends on stack depth
 * being checked.  Measurements with 10 stacks show it getting worthwhile
 * around 100KiB but to be conservative put it higher around where the L2 cache
 * spills.
 */
#ifndef Py_DEBUG
#define NPY_MIN_ELIDE_BYTES (256 * 1024)
#else
/*
 * in debug mode always elide but skip scalars as these can convert to 0d array
 * during in-place operations
 */
#define NPY_MIN_ELIDE_BYTES (32)
#endif
#include <dlfcn.h>
#include <execinfo.h>

/*
 * linear search pointer in table
 * number of pointers is usually quite small but if a performance impact can be
 * measured this could be converted to a binary search
 */
static int
find_addr(void * addresses[], npy_intp naddr, void * addr)
{
    npy_intp j;
    for (j = 0; j < naddr; j++) {
        if (addr == addresses[j]) {
            return 1;
        }
    }
    return 0;
}

static int
check_callers(int * cannot)
{
    /*
     * get base addresses of multiarray and python, check if
     * backtrace is in these libraries only calling dladdr if a new max address
     * is found.
     * When after the initial multiarray stack everything is inside python we
     * can elide as no C-API user could have messed up the reference counts.
     * Only check until the python frame evaluation function is found
     * approx 10us overhead for stack size of 10
     *
     * TODO some calls go over scalarmath in umath but we cannot get the base
     * address of it from multiarraymodule as it is not linked against it
     */
    static int init = 0;
    /*
     * measured DSO object memory start and end, if an address is located
     * inside these bounds it is part of that library so we don't need to call
     * dladdr on it (assuming linear memory)
     */
    static void * pos_python_start;
    static void * pos_python_end;
    static void * pos_ma_start;
    static void * pos_ma_end;

    /* known address storage to save dladdr calls */
    static void * py_addr[64];
    static void * pyeval_addr[64];
    static npy_intp n_py_addr = 0;
    static npy_intp n_pyeval = 0;

    void *buffer[NPY_MAX_STACKSIZE];
    int i, nptrs;
    int ok = 0;
    /* cannot determine callers */
    if (init == -1) {
        *cannot = 1;
        return 0;
    }

    nptrs = backtrace(buffer, NPY_MAX_STACKSIZE);
    if (nptrs == 0) {
        /* complete failure, disable elision */
        init = -1;
        *cannot = 1;
        return 0;
    }

    /* setup DSO base addresses, ends updated later */
    if (NPY_UNLIKELY(init == 0)) {
        Dl_info info;
        /* get python base address */
        if (dladdr(&PyNumber_Or, &info)) {
            pos_python_start = info.dli_fbase;
            pos_python_end = info.dli_fbase;
        }
        else {
            init = -1;
            return 0;
        }
        /* get multiarray base address */
        if (dladdr(&PyArray_SetNumericOps, &info)) {
            pos_ma_start = info.dli_fbase;
            pos_ma_end = info.dli_fbase;
        }
        else {
            init = -1;
            return 0;
        }
        init = 1;
    }

    /* loop over callstack addresses to check if they leave numpy or cpython */
    for (i = 0; i < nptrs; i++) {
        Dl_info info;
        int in_python = 0;
        int in_multiarray = 0;
#if NPY_ELIDE_DEBUG >= 2
        dladdr(buffer[i], &info);
        printf("%s(%p) %s(%p)\n", info.dli_fname, info.dli_fbase,
               info.dli_sname, info.dli_saddr);
#endif

        /* check stored DSO boundaries first */
        if (buffer[i] >= pos_python_start && buffer[i] <= pos_python_end) {
            in_python = 1;
        }
        else if (buffer[i] >= pos_ma_start && buffer[i] <= pos_ma_end) {
            in_multiarray = 1;
        }

        /* update DSO boundaries via dladdr if necessary */
        if (!in_python && !in_multiarray) {
            if (dladdr(buffer[i], &info) == 0) {
                init = -1;
                ok = 0;
                break;
            }
            /* update DSO end */
            if (info.dli_fbase == pos_python_start) {
                pos_python_end = NPY_NUMBER_MAX(buffer[i], pos_python_end);
                in_python = 1;
            }
            else if (info.dli_fbase == pos_ma_start) {
                pos_ma_end = NPY_NUMBER_MAX(buffer[i], pos_ma_end);
                in_multiarray = 1;
            }
        }

        /* no longer in ok libraries and not reached PyEval -> no elide */
        if (!in_python && !in_multiarray) {
            ok = 0;
            break;
        }

        /* in python check if the frame eval function was reached */
        if (in_python) {
            /* if reached eval we are done */
            if (find_addr(pyeval_addr, n_pyeval, buffer[i])) {
                ok = 1;
                break;
            }
            /*
             * check if its some other function, use pointer lookup table to
             * save expensive dladdr calls
             */
            if (find_addr(py_addr, n_py_addr, buffer[i])) {
                continue;
            }

            /* new python address, check for PyEvalFrame */
            if (dladdr(buffer[i], &info) == 0) {
                init = -1;
                ok = 0;
                break;
            }
            if (info.dli_sname &&
                    strcmp(info.dli_sname, PYFRAMEEVAL_FUNC) == 0) {
                if (n_pyeval < sizeof(pyeval_addr) / sizeof(pyeval_addr[0])) {
                    /* store address to not have to dladdr it again */
                    pyeval_addr[n_pyeval++] = buffer[i];
                }
                ok = 1;
                break;
            }
            else if (n_py_addr < sizeof(py_addr) / sizeof(py_addr[0])) {
                /* store other py function to not have to dladdr it again */
                py_addr[n_py_addr++] = buffer[i];
            }
        }
    }

    /* all stacks after numpy are from python, we can elide */
    if (ok) {
        *cannot = 0;
        return 1;
    }
    else {
#if NPY_ELIDE_DEBUG != 0
        puts("cannot elide due to c-api usage");
#endif
        *cannot = 1;
        return 0;
    }
}

/*
 * check if in "alhs @op@ orhs" that alhs is a temporary (refcnt == 1) so we
 * can do in-place operations instead of creating a new temporary
 * "cannot" is set to true if it cannot be done even with swapped arguments
 */
static int
can_elide_temp(PyArrayObject * alhs, PyObject * orhs, int * cannot)
{
    /*
     * to be a candidate the array needs to have reference count 1, be an exact
     * array of a basic type, own its data and size larger than threshold
     */
    if (Py_REFCNT(alhs) != 1 || !PyArray_CheckExact(alhs) ||
            !PyArray_ISNUMBER(alhs) ||
            !PyArray_CHKFLAGS(alhs, NPY_ARRAY_OWNDATA) ||
            !PyArray_ISWRITEABLE(alhs) ||
            PyArray_CHKFLAGS(alhs, NPY_ARRAY_UPDATEIFCOPY) ||
            PyArray_CHKFLAGS(alhs, NPY_ARRAY_WRITEBACKIFCOPY) ||
            PyArray_NBYTES(alhs) < NPY_MIN_ELIDE_BYTES) {
        return 0;
    }
    if (PyArray_CheckExact(orhs) ||
        PyArray_CheckAnyScalar(orhs)) {
        PyArrayObject * arhs;

        /* create array from right hand side */
        Py_INCREF(orhs);
        arhs = (PyArrayObject *)PyArray_EnsureArray(orhs);
        if (arhs == NULL) {
            return 0;
        }

        /*
         * if rhs is not a scalar dimensions must match
         * TODO: one could allow broadcasting on equal types
         */
        if (!(PyArray_NDIM(arhs) == 0 ||
              (PyArray_NDIM(arhs) == PyArray_NDIM(alhs) &&
               PyArray_CompareLists(PyArray_DIMS(alhs), PyArray_DIMS(arhs),
                                    PyArray_NDIM(arhs))))) {
                Py_DECREF(arhs);
                return 0;
        }

        /* must be safe to cast (checks values for scalar in rhs) */
        if (PyArray_CanCastArrayTo(arhs, PyArray_DESCR(alhs),
                                   NPY_SAFE_CASTING)) {
            Py_DECREF(arhs);
            return check_callers(cannot);
        }
        Py_DECREF(arhs);
    }

    return 0;
}

/*
 * try eliding a binary op, if commutative is true also try swapped arguments
 */
NPY_NO_EXPORT int
try_binary_elide(PyArrayObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative)
{
    /* set when no elision can be done independent of argument order */
    int cannot = 0;
    if (can_elide_temp(m1, m2, &cannot)) {
        *res = inplace_op(m1, m2);
#if NPY_ELIDE_DEBUG != 0
        puts("elided temporary in binary op");
#endif
        return 1;
    }
    else if (commutative && !cannot) {
        if (can_elide_temp((PyArrayObject *)m2, (PyObject *)m1, &cannot)) {
            *res = inplace_op((PyArrayObject *)m2, (PyObject *)m1);
#if NPY_ELIDE_DEBUG != 0
            puts("elided temporary in commutative binary op");
#endif
            return 1;
        }
    }
    *res = NULL;
    return 0;
}

/* try elide unary temporary */
NPY_NO_EXPORT int
can_elide_temp_unary(PyArrayObject * m1)
{
    int cannot;
    if (Py_REFCNT(m1) != 1 || !PyArray_CheckExact(m1) ||
            !PyArray_ISNUMBER(m1) ||
            !PyArray_CHKFLAGS(m1, NPY_ARRAY_OWNDATA) ||
            !PyArray_ISWRITEABLE(m1) ||
            PyArray_CHKFLAGS(m1, NPY_ARRAY_UPDATEIFCOPY) ||
            PyArray_NBYTES(m1) < NPY_MIN_ELIDE_BYTES) {
        return 0;
    }
    if (check_callers(&cannot)) {
#if NPY_ELIDE_DEBUG != 0
        puts("elided temporary in unary op");
#endif
        return 1;
    }
    else {
        return 0;
    }
}
#else /* unsupported interpreter or missing backtrace */
NPY_NO_EXPORT int
can_elide_temp_unary(PyArrayObject * m1)
{
    return 0;
}

NPY_NO_EXPORT int
try_binary_elide(PyArrayObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative)
{
    *res = NULL;
    return 0;
}
#endif
