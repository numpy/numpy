#ifndef NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_
#define NUMPY_CORE_SRC_MULTIARRAY_MAPPING_H_

extern NPY_NO_EXPORT PyMappingMethods array_as_mapping;


/*
 * Object to store information needed for advanced (also fancy) indexing.
 * Not public, so does not have to be a Python object in principle.
 */
typedef struct {
        PyObject_HEAD
        /* number of advanced indexing arrays ("fancy" indices) */
        int                   num_fancy;
        /* Total size of the result (see `nd` and `dimensions` below) */
        npy_intp              size;

        /*
         * Arrays used in the iteration:
         * - The original array indexed (subscription or assignment)
         * - extra-op: is used for the subscription result array or the
         *   values being assigned.
         *   (As of writing `ufunc.at` does not use this mechanism.)
         * - "subspace": If not all dimensions are indexed using advanced
         *   indices, the "subspace" is a view into `array` that omits all
         *   dimensions indexed by advanced indices.
         *   (The result of replacing all indexing arrays by a `0`.)
         */
        PyArrayObject         *array;
        PyArrayObject         *extra_op;
        PyArrayObject         *subspace;

        /*
         * Pointer into the array when all index arrays are 0, used as base
         * for all calculations.
         */
        char                  *baseoffset;

        /*
         * Iterator and information for all indexing (fancy) arrays.
         * When no "subspace" is needed it also iterates the `extra_op`.
         */
        NpyIter               *outer;
        NpyIter_IterNextFunc  *outer_next;
        char                  **outer_ptrs;
        npy_intp              *outer_strides;

        /*
         * When a "subspace" is used, `extra_op` needs a dedicated iterator
         * and we need yet another iterator for the original array.
         */
        NpyIter               *extra_op_iter;
        NpyIter_IterNextFunc  *extra_op_next;
        char                  **extra_op_ptrs;

        NpyIter               *subspace_iter;
        NpyIter_IterNextFunc  *subspace_next;
        char                  **subspace_ptrs;
        npy_intp              *subspace_strides;

        /*
         * Total number of total dims of the result and how many of those
         * are created by the advanced indexing result.
         * (This is not the same as `num_fancy` as advanced indices can add
         * or remove dimensions.)
         */
        int                   nd;
        int                   nd_fancy;
        /*
         * After binding "consec" denotes at which axis the fancy axes
         * are inserted.  When all advanced indices are consecutive, NumPy
         * preserves their position in the result (see advanced indexing docs).
         */
        int                   consec;
        /* Result dimensions/shape */
        npy_intp              dimensions[NPY_MAXDIMS];

        /*
         * The axes iterated by the advanced index and the length and strides
         * for each of these axes. (The fast paths copy some of this.)
         */
        int                   iteraxes[NPY_MAXDIMS];
        npy_intp              fancy_dims[NPY_MAXDIMS];
        npy_intp              fancy_strides[NPY_MAXDIMS];

        /* Count and pointer used as state by the slow `PyArray_MapIterNext` */
        npy_intp              iter_count;
        char                  *dataptr;

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
