#ifndef _MULTIARRAYMODULE
#error You should not include this
#endif

#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_

#ifdef __cplusplus
extern "C" {
#endif

extern NPY_NO_EXPORT npy_bool numpy_warn_if_no_mem_policy;

NPY_NO_EXPORT PyObject *
_strings_richcompare(PyArrayObject *self, PyArrayObject *other, int cmp_op,
                     int rstrip);

NPY_NO_EXPORT PyObject *
array_richcompare(PyArrayObject *self, PyObject *other, int cmp_op);

NPY_NO_EXPORT int
array_might_be_written(PyArrayObject *obj);

/*
 * This flag is used to mark arrays which we would like to, in the future,
 * turn into views. It causes a warning to be issued on the first attempt to
 * write to the array (but the write is allowed to succeed).
 *
 * This flag is for internal use only, and may be removed in a future release,
 * which is why the #define is not exposed to user code. Currently it is set
 * on arrays returned by ndarray.diagonal.
 */
static const int NPY_ARRAY_WARN_ON_WRITE = (1 << 31);


/*
 * These flags are used internally to indicate an array that was previously
 * a Python scalar (int, float, complex).  The dtype of such an array should
 * be considered as any integer, floating, or complex rather than the explicit
 * dtype attached to the array.
 *
 * These flags must only be used in local context when the array in question
 * is not returned.  Use three flags, to avoid having to double check the
 * actual dtype when the flags are used.
 */
static const int NPY_ARRAY_WAS_PYTHON_INT = (1 << 30);
static const int NPY_ARRAY_WAS_PYTHON_FLOAT = (1 << 29);
static const int NPY_ARRAY_WAS_PYTHON_COMPLEX = (1 << 28);
/*
 * Mark that this was a huge int which was turned into an object array (or
 * unsigned/non-default integer array), but then replaced by a temporary
 * array for further processing. This flag is only used in the ufunc machinery
 * where it is tricky to cover correctly all type resolution paths.
 */
static const int NPY_ARRAY_WAS_INT_AND_REPLACED = (1 << 27);
static const int NPY_ARRAY_WAS_PYTHON_LITERAL = (1 << 30 | 1 << 29 | 1 << 28);

/*
 * This flag allows same kind casting, similar to NPY_ARRAY_FORCECAST.
 *
 * An array never has this flag set; they're only used as parameter
 * flags to the various FromAny functions.
 */
static const int NPY_ARRAY_SAME_KIND_CASTING = (1 << 26);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAYOBJECT_H_ */
