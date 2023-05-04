#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"

#include "common_dtype.h"
#include "convert_datatype.h"
#include "dtypemeta.h"
#include "abstractdtypes.h"


/*
 * This file defines all logic necessary for generic "common dtype"
 * operations.  This is unfortunately surprisingly complicated to get right
 * due to the value based logic NumPy uses and the fact that NumPy has
 * no clear (non-transitive) type promotion hierarchy.
 * Unlike most languages `int32 + float32 -> float64` instead of `float32`.
 * The other complicated thing is value-based-promotion, which means that
 * in many cases a Python 1, may end up as an `int8` or `uint8`.
 *
 * This file implements the necessary logic so that `np.result_type(...)`
 * can give the correct result for any order of inputs and can further
 * generalize to user DTypes.
 */


/**
 * This function defines the common DType operator.
 *
 * Note that the common DType will not be "object" (unless one of the dtypes
 * is object), even though object can technically represent all values
 * correctly.
 *
 * TODO: Before exposure, we should review the return value (e.g. no error
 *       when no common DType is found).
 *
 * @param dtype1 DType class to find the common type for.
 * @param dtype2 Second DType class.
 * @return The common DType or NULL with an error set
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_CommonDType(PyArray_DTypeMeta *dtype1, PyArray_DTypeMeta *dtype2)
{
    if (dtype1 == dtype2) {
        Py_INCREF(dtype1);
        return dtype1;
    }

    PyArray_DTypeMeta *common_dtype;

    common_dtype = NPY_DT_CALL_common_dtype(dtype1, dtype2);
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(common_dtype);
        common_dtype = NPY_DT_CALL_common_dtype(dtype2, dtype1);
    }
    if (common_dtype == NULL) {
        return NULL;
    }
    if (common_dtype == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_DECREF(Py_NotImplemented);
        PyErr_Format(npy_DTypePromotionError,
                "The DTypes %S and %S do not have a common DType. "
                "For example they cannot be stored in a single array unless "
                "the dtype is `object`.", dtype1, dtype2);
        return NULL;
    }
    return common_dtype;
}


/**
 * This function takes a list of dtypes and "reduces" them (in a sense,
 * it finds the maximal dtype). Note that "maximum" here is defined by
 * knowledge (or category or domain). A user DType must always "know"
 * about all NumPy dtypes, floats "know" about integers, integers "know"
 * about unsigned integers.
 *
 *           c
 *          / \
 *         a   \    <-- The actual promote(a, b) may be c or unknown.
 *        / \   \
 *       a   b   c
 *
 * The reduction is done "pairwise". In the above `a.__common_dtype__(b)`
 * has a result (so `a` knows more) and `a.__common_dtype__(c)` returns
 * NotImplemented (so `c` knows more).  You may notice that the result
 * `res = a.__common_dtype__(b)` is not important.  We could try to use it
 * to remove the whole branch if `res is c` or by checking if
 * `c.__common_dtype(res) is c`.
 * Right now, we only clear initial elements in the most simple case where
 * `a.__common_dtype(b) is a` (and thus `b` cannot alter the end-result).
 * Clearing means, we do not have to worry about them later.
 *
 * There is one further subtlety. If we have an abstract DType and a
 * non-abstract one, we "prioritize" the non-abstract DType here.
 * In this sense "prioritizing" means that we use:
 *       abstract.__common_dtype__(other)
 * If both return NotImplemented (which is acceptable and even expected in
 * this case, see later) then `other` will be considered to know more.
 *
 * The reason why this may be acceptable for abstract DTypes, is that
 * the value-dependent abstract DTypes may provide default fall-backs.
 * The priority inversion effectively means that abstract DTypes are ordered
 * just below their concrete counterparts.
 * (This fall-back is convenient but not perfect, it can lead to
 * non-minimal promotions: e.g. `np.uint24 + 2**20 -> int32`. And such
 * cases may also be possible in some mixed type scenarios; they can be
 * avoided by defining the promotion explicitly in the user DType.)
 *
 * @param length Number of DTypes
 * @param dtypes
 */
static PyArray_DTypeMeta *
reduce_dtypes_to_most_knowledgeable(
        npy_intp length, PyArray_DTypeMeta **dtypes)
{
    assert(length >= 2);
    npy_intp half = length / 2;

    PyArray_DTypeMeta *res = NULL;

    for (npy_intp low = 0; low < half; low++) {
        npy_intp high = length - 1 - low;
        if (dtypes[high] == dtypes[low]) {
            Py_INCREF(dtypes[low]);
            Py_XSETREF(res, dtypes[low]);
        }
        else {
            if (NPY_DT_is_abstract(dtypes[high])) {
                /*
                 * Priority inversion, start with abstract, because if it
                 * returns `other`, we can let other pass instead.
                 */
                PyArray_DTypeMeta *tmp = dtypes[low];
                dtypes[low] = dtypes[high];
                dtypes[high] = tmp;
            }

            Py_XSETREF(res, NPY_DT_CALL_common_dtype(dtypes[low], dtypes[high]));
            if (res == NULL) {
                return NULL;
            }
        }

        if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            PyArray_DTypeMeta *tmp = dtypes[low];
            dtypes[low] = dtypes[high];
            dtypes[high] = tmp;
        }
        if (res == dtypes[low]) {
            /* `dtypes[high]` cannot influence the final result, so clear: */
            dtypes[high] = NULL;
        }
    }

    if (length == 2) {
        return res;
    }
    Py_DECREF(res);
    return reduce_dtypes_to_most_knowledgeable(length - half, dtypes);
}


/**
 * Promotes a list of DTypes with each other in a way that should guarantee
 * stable results even when changing the order.
 *
 * In general this approach always works as long as the most generic dtype
 * is either strictly larger, or compatible with all other dtypes.
 * For example promoting float16 with any other float, integer, or unsigned
 * integer again gives a floating point number. And any floating point number
 * promotes in the "same way" as `float16`.
 * If a user inserts more than one type into the NumPy type hierarchy, this
 * can break. Given:
 *     uint24 + int32 -> int48  # Promotes to a *new* dtype!
 *
 * The following becomes problematic (order does not matter):
 *         uint24 +      int16  +           uint32  -> int64
 *    <==      (uint24 + int16) + (uint24 + uint32) -> int64
 *    <==                int32  +           uint32  -> int64
 *
 * It is impossible to achieve an `int48` result in the above.
 *
 * This is probably only resolvable by asking `uint24` to take over the
 * whole reduction step; which we currently do not do.
 * (It may be possible to notice the last up-cast and implement use something
 * like: `uint24.nextafter(int32).__common_dtype__(uint32)`, but that seems
 * even harder to grasp.)
 *
 * Note that a case where two dtypes are mixed (and know nothing about each
 * other) will always generate an error:
 *     uint24 + int48 + int64 -> Error
 *
 * Even though `int64` is a safe solution, since `uint24 + int64 -> int64` and
 * `int48 + int64 -> int64` and `int64` and there cannot be a smaller solution.
 *
 * //TODO: Maybe this function should allow not setting an error?
 *
 * @param length Number of dtypes (and values) must be at least 1
 * @param dtypes The concrete or abstract DTypes to promote
 * @return NULL or the promoted DType.
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_PromoteDTypeSequence(
        npy_intp length, PyArray_DTypeMeta **dtypes_in)
{
    if (length == 1) {
        Py_INCREF(dtypes_in[0]);
        return dtypes_in[0];
    }
    PyArray_DTypeMeta *result = NULL;

    /* Copy dtypes so that we can reorder them (only allocate when many) */
    PyObject *_scratch_stack[NPY_MAXARGS];
    PyObject **_scratch_heap = NULL;
    PyArray_DTypeMeta **dtypes = (PyArray_DTypeMeta **)_scratch_stack;

    if (length > NPY_MAXARGS) {
        _scratch_heap = PyMem_Malloc(length * sizeof(PyObject *));
        if (_scratch_heap == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        dtypes = (PyArray_DTypeMeta **)_scratch_heap;
    }

    memcpy(dtypes, dtypes_in, length * sizeof(PyObject *));

    /*
     * `result` is the last promotion result, which can usually be reused if
     * it is not NotImplemneted.
     * The passed in dtypes are partially sorted (and cleared, when clearly
     * not relevant anymore).
     * `dtypes[0]` will be the most knowledgeable (highest category) which
     * we consider the "main_dtype" here.
     */
    result = reduce_dtypes_to_most_knowledgeable(length, dtypes);
    if (result == NULL) {
        goto finish;
    }
    PyArray_DTypeMeta *main_dtype = dtypes[0];

    npy_intp reduce_start = 1;
    if (result == (PyArray_DTypeMeta *)Py_NotImplemented) {
        Py_SETREF(result, NULL);
    }
    else {
        /* (new) first value is already taken care of in `result` */
        reduce_start = 2;
    }
    /*
     * At this point, we have only looked at every DType at most once.
     * The `main_dtype` must know all others (or it will be a failure) and
     * all dtypes returned by its `common_dtype` must be guaranteed to succeed
     * promotion with one another.
     * It is the job of the "main DType" to ensure that at this point order
     * is irrelevant.
     * If this turns out to be a limitation, this "reduction" will have to
     * become a default version and we have to allow DTypes to override it.
     */
    PyArray_DTypeMeta *prev = NULL;
    for (npy_intp i = reduce_start; i < length; i++) {
        if (dtypes[i] == NULL || dtypes[i] == prev) {
            continue;
        }
        /*
         * "Promote" the current dtype with the main one (which should be
         * a higher category). We assume that the result is not in a lower
         * category.
         */
        PyArray_DTypeMeta *promotion = NPY_DT_CALL_common_dtype(
                main_dtype, dtypes[i]);
        if (promotion == NULL) {
            Py_XSETREF(result, NULL);
            goto finish;
        }
        else if ((PyObject *)promotion == Py_NotImplemented) {
            Py_DECREF(Py_NotImplemented);
            Py_XSETREF(result, NULL);
            PyObject *dtypes_in_tuple = PyTuple_New(length);
            if (dtypes_in_tuple == NULL) {
                goto finish;
            }
            for (npy_intp l=0; l < length; l++) {
                Py_INCREF(dtypes_in[l]);
                PyTuple_SET_ITEM(dtypes_in_tuple, l, (PyObject *)dtypes_in[l]);
            }
            PyErr_Format(npy_DTypePromotionError,
                    "The DType %S could not be promoted by %S. This means that "
                    "no common DType exists for the given inputs. "
                    "For example they cannot be stored in a single array unless "
                    "the dtype is `object`. The full list of DTypes is: %S",
                    dtypes[i], main_dtype, dtypes_in_tuple);
            Py_DECREF(dtypes_in_tuple);
            goto finish;
        }
        if (result == NULL) {
            result = promotion;
            continue;
        }

        /*
         * The above promoted, now "reduce" with the current result; note that
         * in the typical cases we expect this step to be a no-op.
         */
        Py_SETREF(result, PyArray_CommonDType(result, promotion));
        Py_DECREF(promotion);
        if (result == NULL) {
            goto finish;
        }
    }

  finish:
    PyMem_Free(_scratch_heap);
    return result;
}
