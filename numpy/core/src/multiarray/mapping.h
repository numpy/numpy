#ifndef _NPY_ARRAYMAPPING_H_
#define _NPY_ARRAYMAPPING_H_

extern NPY_NO_EXPORT PyMappingMethods array_as_mapping;

/*
 * Plain indexing is python code writing arr[...]. Fancy means explicitly
 * the old behavious, legacy is mostly like fancy, but does not accept
 * array-likes as tuples.
 */
#define PLAIN_INDEXING 1
#define OUTER_INDEXING 2
#define VECTOR_INDEXING 4
#define FANCY_INDEXING 8

/*
 * Struct into which indices are parsed.
 * I.e. integer ones should only be parsed once, slices and arrays
 * need to be validated later and for the ellipsis we need to find how
 * many slices it represents.
 */
typedef struct {
    /*
     * Object of index: slice, array, or NULL. Owns a reference.
     */
    PyObject *object;
    /*
     * Value of an integer index, number of slices an Ellipsis is worth,
     * -1 if input was an integer array and the original size of the
     * boolean array if it is a converted boolean array.
     */
    npy_intp value;
    /* kind of index, see constants in mapping.c */
    int type;
    /* original index number, mostly for boolean. -1 if implicit Ellipsis */
    int orig_index;
} npy_index_info;


NPY_NO_EXPORT Py_ssize_t
array_length(PyArrayObject *self);

NPY_NO_EXPORT PyObject *
array_item_asarray(PyArrayObject *self, npy_intp i);

NPY_NO_EXPORT PyObject *
array_item_asscalar(PyArrayObject *self, npy_intp i);

NPY_NO_EXPORT PyObject *
array_item(PyArrayObject *self, Py_ssize_t i);

NPY_NO_EXPORT PyObject *
array_subscript_asarray(PyArrayObject *self, PyObject *op);

NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op, int indexing_method);

NPY_NO_EXPORT int
array_assign_item(PyArrayObject *self, Py_ssize_t i, PyObject *v);

NPY_NO_EXPORT int
array_assign_subscript(PyArrayObject *self, PyObject *ind, PyObject *op,
                       int indexing_method, int allow_getitem_hack);

/*
 * Prototypes for Mapping calls --- not part of the C-API
 * because only useful as part of a getitem call.
 */
NPY_NO_EXPORT void
PyArray_MapIterReset(PyArrayMapIterObject *mit);

NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit);

NPY_NO_EXPORT int
PyArray_MapIterCheckIndices(PyArrayMapIterObject *mit);

NPY_NO_EXPORT void
PyArray_MapIterSwapAxes(PyArrayMapIterObject *mit, PyArrayObject **ret, int getmap);

NPY_NO_EXPORT PyObject*
PyArray_MapIterNew(npy_index_info *indices , int index_num, int index_type,
                   int ndim, int fancy_ndim,
                   PyArrayObject *arr, PyArrayObject *subspace,
                   npy_uint32 subspace_iter_flags, npy_uint32 subspace_flags,
                   npy_uint32 extra_op_flags, PyArrayObject *extra_op,
                   PyArray_Descr *extra_op_dtype, int outer_indexing);

/*
 * Prototypes for attribute indexing helper (arr.oindex, etc.)
 */
typedef struct {
        PyObject_HEAD
        /*
         * Attribute information portion.
         */
        PyArrayObject *array;
        int indexing_method;  /* See mapping.h */
} PyArrayAttributeIndexer;


NPY_NO_EXPORT PyObject *
PyArray_AttributeIndexerNew(PyArrayObject *array, int indexing_method);


/*
 * Prototypes for multi index objects passed to subclasses by arr.oindex, etc.
 */
typedef struct {
        PyObject_HEAD
        /*
         * Attribute information portion.
         */
        PyObject *index;            /* The indexing object */
        int indexing_method;        /* See mapping.h */
        /* If bound is 1, the following are information about the array */
        int bound;
        npy_intp orig_shape[NPY_MAXDIMS];
        int orig_ndim;
        PyArray_Descr *orig_dtype;
} PyArrayMultiIndex;


NPY_NO_EXPORT PyObject *
PyArray_MultiIndexNew(PyObject *index, PyArrayObject *array,
                      int indexing_method);


NPY_NO_EXPORT PyObject *
arrayattributeindexer_subscript(PyArrayAttributeIndexer *attr_indexer,
                                 PyObject *op);

NPY_NO_EXPORT int
arrayattributeindexer_assign_subscript(PyArrayAttributeIndexer *attr_indexer,
                                        PyObject *op, PyObject *vals);


NPY_NO_EXPORT npy_intp
arrayattributeindexer_length(PyArrayAttributeIndexer *attr_indexer);
#endif
