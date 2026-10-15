#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#include "numpy/npy_common.h"
#include "numpy/arrayobject.h"

#include "alloc.h"
#include "convert_datatype.h"
#include "dtypemeta.h"
#include "abstractdtypes.h"
#include "npy_static_data.h"
#include "module_state.h"


/*
 * This file defines all logic necessary for generic "common dtype"
 * operations on DType classes.  Promotion is not generally associative:
 * for example, combining int8 and uint8 first gives int16, which promotes
 * with float16 to float32. Promoting float16 with either integer separately
 * leaves float16 unchanged.
 *
 * This file implements the necessary logic so that `np.result_type(...)`
 * can resolve such cases consistently across input orders, including user
 * DTypes.
 */


/*NUMPY_API
 * This function defines the common DType operator.
 *
 * Similar to np.result_type, but works on DType classes rather than dtype
 * instances.  Object is not used as an automatic fallback for incompatible
 * inputs, even though it can represent all values.
 *
 * Identical inputs are returned directly.  Otherwise, call dtype1's
 * common_dtype slot first, then dtype2's if the first returns NotImplemented.
 * If both defer, raise DTypePromotionError.  Errors raised by either slot
 * are propagated.
 *
 * @param dtype1 DType class to find the common type for.
 * @param dtype2 Second DType class.
 * @return New reference to the common DType, or NULL with an error set
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
        PyErr_Format(_npy_module_state->static_pydata.DTypePromotionError,
                "The DTypes %S and %S do not have a common DType. "
                "For example they cannot be stored in a single array unless "
                "the dtype is `object`.", dtype1, dtype2);
        return NULL;
    }
    return common_dtype;
}


/**
 * Select and remove the most knowledgeable input DType. "Knowledge" describes
 * which DType implements promotion for a pair.
 * While doing so, `dtypes` are removed if it is clear that they cannot affect
 * the result anymore (because the promotion returned the candidate unchanged).
 *
 * Starting with the first DType as the candidate, call its common_dtype slot
 * with each remaining input. Only NotImplemented replaces the candidate;
 * intermediate promotion results are discarded.
 * At the end `selected` is the "most knowledgeable" which must promote with
 * all others (if there is no promotion error).
 *
 * Abstract DTypes participate in the same selection as concrete DTypes.
 * Their promotion priorities are implemented by the common_dtype slots;
 * see also default_builtin_common_dtype.
 *
 * @param length Number of DTypes, must be at least 1
 * @param dtypes Borrowed DType pointers; survivors are compacted in place
 * @param selected Output: borrowed reference to the selected input on success
 * @return Number of surviving inputs, excluding the selected input, or -1
 *         with an error set.  On success, the surviving inputs occupy that
 *         many entries at the front of dtypes; later entries are ignored.
 */
static npy_intp
select_most_knowledgeable_dtype(
        npy_intp length, PyArray_DTypeMeta **dtypes, PyArray_DTypeMeta **selected)
{
    assert(length >= 1);
    PyArray_DTypeMeta *candidate = dtypes[0];
    npy_intp remaining = 0;
    for (npy_intp i = 1; i < length; i++) {
        PyArray_DTypeMeta *other = dtypes[i];
        if (other == candidate) {
            /* Fast path for identical dtypes: do not call common_dtype */
            continue;
        }
        PyArray_DTypeMeta *res = NPY_DT_CALL_common_dtype(
                candidate, other);
        if (res == NULL) {
            return -1;
        }

        if (res == (PyArray_DTypeMeta *)Py_NotImplemented) {
            /* guess at other being more "knowledgeable" */
            dtypes[remaining++] = candidate;
            candidate = other;
        }
        else if (res != candidate) {
            /* The input may still affect the result: retain it. */
            dtypes[remaining++] = other;
        }
        Py_DECREF(res);
    }

    *selected = candidate;
    return remaining;
}


/*NUMPY_API
 * Promote a nonempty list of concrete or abstract DType classes.  The input
 * array is not modified.  A single input is returned unchanged, and two
 * inputs use PyArray_CommonDType directly.
 *
 * For longer lists, repeatedly select an original input by promotion
 * knowledge while discarding any inputs that clearly cannot affect the result
 * anymore (see `select_most_knowledgeable_dtype`).
 * These repeatedly selected dtypes are then promoted together.
 *
 * This is necessary to ensure promotion is not sensitive to the input order.
 * For example when promoting `int4, uint8, float16` we have:
 *   int4 + float16 -> float16
 *   uint8 + float16 -> float16
 *   ((int4 + uint8) -> int16) + float16 -> float32
 *
 * But the above approach ensures we order them as `int4` (most knowledgeable)
 * first and then `float16` second.
 * This order independence relies on consistent promotion priorities and
 * promotion rules.
 *
 * Each selection is linear. Repeated selection is theoretically quadratic when
 * inputs cannot be eliminated early. In practice, this should be rare and we do
 * not expect many distinct dtypes.
 *
 * Note that this process does have limitations when trying to interleave
 * unsigned and signed custom integers with NumPy ones. If we defined the
 * `uint24` and `int48` with `uint24 + int32 -> int48`, then:
 * (uint24 + uint32) + int16 -> uint32 + int16 -> int64
 * Although int48 could represent all inputs, choosing it for the remaining
 * int16 + uint32 pair would require changing NumPy's builtin promotion rules.
 *
 * @param length Number of DType classes, must be at least 1
 * @param dtypes_in The concrete or abstract DType classes to promote
 * @return New reference to the promoted DType, or NULL with an error set
 */
NPY_NO_EXPORT PyArray_DTypeMeta *
PyArray_PromoteDTypeSequence(
        npy_intp length, PyArray_DTypeMeta **dtypes_in)
{
    if (length == 1) {
        Py_INCREF(dtypes_in[0]);
        return dtypes_in[0];
    }
    if (length == 2) {
        return PyArray_CommonDType(dtypes_in[0], dtypes_in[1]);
    }
    PyArray_DTypeMeta *result = NULL;

    NPY_ALLOC_WORKSPACE(dtypes, PyArray_DTypeMeta *, 16, length);
    if (dtypes == NULL) {
        return NULL;
    }
    memcpy(dtypes, dtypes_in, length * sizeof(PyObject *));

    while (length > 0) {
        /* Select most knowledgeable (may remove some DTypes in the process). */
        PyArray_DTypeMeta *selected;
        length = select_most_knowledgeable_dtype(length, dtypes, &selected);
        if (length < 0) {
            Py_CLEAR(result);
            break;
        }
        if (result == NULL) {
            result = selected;
            Py_INCREF(result);
        }
        else {
            /* Combine the selected DType with the accumulated result. */
            Py_SETREF(result, PyArray_CommonDType(result, selected));
            if (result == NULL) {
                break;
            }
        }
    }

    npy_free_workspace(dtypes);
    return result;
}
