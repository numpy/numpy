#ifndef NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_

extern NPY_NO_EXPORT PyMappingMethods array_as_mapping;


/*
 * Store the information needed for fancy-indexing over an array. The
 * fields are slightly unordered to keep consec, dataptr and subspace
 * where they were originally.
 */
typedef struct {
        PyObject_HEAD

        int                   numiter;                 /* number of index-array
                                                          iterators */
        npy_intp              size;                    /* size of broadcasted
                                                          result */
        int                   nd;                      /* number of dims */
        npy_intp              dimensions[NPY_MAXDIMS]; /* dimensions */
        NpyIter               *outer;                  /* index objects
                                                          iterator */
        PyArrayObject         *array;

        /* Subspace array. */
        PyArrayObject         *subspace;

        /*
         * if subspace iteration, then this is the array of axes in
         * the underlying array represented by the index objects
         */
        int                   iteraxes[NPY_MAXDIMS];
        npy_intp              fancy_strides[NPY_MAXDIMS];

        /* pointer when all fancy indices are 0 */
        char                  *baseoffset;

        /*
         * after binding consec denotes at which axis the fancy axes
         * are inserted.
         */
        int                   consec;
        char                  *dataptr;

        int                   nd_fancy;
        npy_intp              fancy_dims[NPY_MAXDIMS];

        /*
         * Extra op information.
         */
        PyArrayObject         *extra_op;
        PyArray_Descr         *extra_op_dtype;         /* desired dtype */
        npy_uint32            *extra_op_flags;         /* Iterator flags */

        NpyIter               *extra_op_iter;
        NpyIter_IterNextFunc  *extra_op_next;
        char                  **extra_op_ptrs;

        /*
         * Information about the iteration state.
         */
        NpyIter_IterNextFunc  *outer_next;
        char                  **outer_ptrs;
        npy_intp              *outer_strides;

        /*
         * Information about the subspace iterator.
         */
        NpyIter               *subspace_iter;
        NpyIter_IterNextFunc  *subspace_next;
        char                  **subspace_ptrs;
        npy_intp              *subspace_strides;

        /* Count for the external loop (which ever it is) for API iteration */
        npy_intp              iter_count;

} PyArrayMapIterObject;

extern NPY_NO_EXPORT PyTypeObject PyArrayMapIter_Type;

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
array_subscript(PyArrayObject *self, PyObject *op);

NPY_NO_EXPORT int
array_assign_item(PyArrayObject *self, Py_ssize_t i, PyObject *v);

/*
 * Prototypes for Mapping calls --- not part of the C-API
 * because only useful as part of a getitem call.
 */
NPY_NO_EXPORT int
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
                   PyArray_Descr *extra_op_dtype);

NPY_NO_EXPORT PyObject *
PyArray_MapIterArrayCopyIfOverlap(PyArrayObject * a, PyObject * index,
                                  int copy_if_overlap, PyArrayObject *extra_op);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_ */
