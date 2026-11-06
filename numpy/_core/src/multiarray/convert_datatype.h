#ifndef NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_
#define NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_

#include "array_method.h"

#ifdef __cplusplus
extern "C" {
#endif

extern NPY_NO_EXPORT npy_intp REQUIRED_STR_LEN[];

NPY_NO_EXPORT PyObject *
PyArray_GetCastingImpl(PyArray_DTypeMeta *from, PyArray_DTypeMeta *to);

NPY_NO_EXPORT PyObject *
_get_castingimpl(PyObject *NPY_UNUSED(module), PyObject *args);

NPY_NO_EXPORT PyObject *
_get_all_cast_information(PyObject *NPY_UNUSED(module),
                          PyObject *NPY_UNUSED(args));

NPY_NO_EXPORT PyObject *
_is_view_safe_cast(PyObject *NPY_UNUSED(module), PyObject *const *args,
                   Py_ssize_t len_args);

NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num);

NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type);

NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn);

NPY_NO_EXPORT PyArray_Descr *
PyArray_CastToDTypeAndPromoteDescriptors(
        npy_intp ndescr, PyArray_Descr *descrs[], PyArray_DTypeMeta *DType);

NPY_NO_EXPORT int
PyArray_CheckLegacyResultType(
        PyArray_Descr **new_result,
        npy_intp narrs, PyArrayObject **arr,
        npy_intp ndtypes, PyArray_Descr **dtypes);

NPY_NO_EXPORT int
PyArray_ValidType(int type);

NPY_NO_EXPORT int
dtype_kind_to_ordering(char kind);

NPY_NO_EXPORT npy_bool
can_cast_pyscalar_scalar_to(
        int flags, PyArray_Descr *to, NPY_CASTING casting);

NPY_NO_EXPORT int
should_use_min_scalar(npy_intp narrs, PyArrayObject **arr,
                      npy_intp ndtypes, PyArray_Descr **dtypes);

NPY_NO_EXPORT int
should_use_min_scalar_weak_literals(int narrs, PyArrayObject **arr);

/*
 * Unlike the public versions, these propagate warnings raised as errors.
 * Return 1 if valid, 0 if invalid, and -1 on error.
 */
NPY_NO_EXPORT int
PyArray_CanCastTypeTo_int(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting);

NPY_NO_EXPORT int
PyArray_CanCastArrayTo_int(PyArrayObject *arr, PyArray_Descr *to,
        NPY_CASTING casting);

NPY_NO_EXPORT int
npy_coercion_is_strict(void);

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

NPY_NO_EXPORT void
npy_set_invalid_cast_error(
        PyArray_Descr *src_dtype, PyArray_Descr *dst_dtype,
        NPY_CASTING casting, npy_bool scalar);

NPY_NO_EXPORT PyArray_Descr *
PyArray_CastDescrToDType(PyArray_Descr *descr, PyArray_DTypeMeta *given_DType);

NPY_NO_EXPORT PyArray_Descr *
PyArray_FindConcatenationDescriptor(
        npy_intp n, PyArrayObject **arrays, PyArray_Descr *requested_dtype);

NPY_NO_EXPORT int
PyArray_AddCastingImplementation(PyBoundArrayMethodObject *meth);

NPY_NO_EXPORT int
PyArray_AddCastingImplementation_FromSpec(PyArrayMethod_Spec *spec, int private_);

NPY_NO_EXPORT NPY_CASTING
PyArray_MinCastSafety(NPY_CASTING casting1, NPY_CASTING casting2);

/*
 * Flags accumulated during cast resolution, including casts for structured
 * dtype fields and subarray elements. Add flags here for future casting
 * deprecations.
 *
 * Deprecated NumPy 2.6, 2026-10-13: "safe" and "same_kind" casts between
 * bytes (S) and text (U or T). NPY_CAST_TEXT_BYTES identifies these casts.
 *
 * numpy._core._coercion_policy reads NPY_COERCION_POLICY at import; its
 * _STRICT value is stored in the C module state for npy_coercion_is_strict().
 * can_cast_to checks these flags after resolution: the default "warn" policy
 * preserves the old rules and warns for otherwise allowed casts; "strict"
 * rejects them. Explicit "unsafe" casts remain supported, and public casting
 * queries do not warn.
 */
#define NPY_CAST_TEXT_BYTES 1

NPY_NO_EXPORT NPY_CASTING
PyArray_GetCastInfo(
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype,
        npy_intp *view_offset, unsigned int *cast_flags);

NPY_NO_EXPORT npy_intp
PyArray_SafeCast(PyArray_Descr *type1, PyArray_Descr *type2,
                 npy_intp* view_offset, NPY_CASTING minimum_safety,
                 npy_intp ignore_errors);

NPY_NO_EXPORT int
PyArray_CheckCastSafety(NPY_CASTING casting,
        PyArray_Descr *from, PyArray_Descr *to, PyArray_DTypeMeta *to_dtype,
        unsigned int *cast_flags);

NPY_NO_EXPORT NPY_CASTING
legacy_same_dtype_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset);

NPY_NO_EXPORT int
legacy_cast_get_strided_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references, npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT NPY_CASTING
simple_cast_resolve_descriptors(
        PyArrayMethodObject *self,
        PyArray_DTypeMeta *const dtypes[2],
        PyArray_Descr *const input_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset);

NPY_NO_EXPORT int
PyArray_InitializeCasts(void);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_CONVERT_DATATYPE_H_ */
