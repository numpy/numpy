#ifndef _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_SORTS_H_
#define _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_SORTS_H_

#ifdef __cplusplus
extern "C" {
#endif

#define SMALL_STRING_MERGESORT 20

NPY_NO_EXPORT int
_cmp(char *a, char *b, PyArray_StringDTypeObject *descr, int descending);

NPY_NO_EXPORT int
stringdtype_stablesort(PyArrayMethod_Context *context, char *const *data,
                       const npy_intp *dimensions, const npy_intp *NPY_UNUSED(strides),
                       NpyAuxData *NPY_UNUSED(auxdata));

NPY_NO_EXPORT int
stringdtype_sort_get_loop(PyArrayMethod_Context *context, int aligned,
                          int move_references, const npy_intp *strides,
                          PyArrayMethod_StridedLoop **out_loop,
                          NpyAuxData **out_transferdata, NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT int
stringdtype_stableargsort(PyArrayMethod_Context *context, char *const *data,
                          const npy_intp *dimensions,
                          const npy_intp *NPY_UNUSED(strides),
                          NpyAuxData *NPY_UNUSED(auxdata));

NPY_NO_EXPORT int
stringdtype_argsort_get_loop(PyArrayMethod_Context *context, int aligned,
                             int move_references, const npy_intp *strides,
                             PyArrayMethod_StridedLoop **out_loop,
                             NpyAuxData **out_transferdata,
                             NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT int
init_stringdtype_sorts(void);

#ifdef __cplusplus
}
#endif

#endif /* _NPY_CORE_SRC_MULTIARRAY_STRINGDTYPE_SORTS_H_ */
