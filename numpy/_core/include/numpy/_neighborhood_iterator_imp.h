#ifndef NUMPY_CORE_INCLUDE_NUMPY__NEIGHBORHOOD_IMP_H_
#error You should not include this header directly
#endif
/*
 * Private API (here for inline)
 */
static inline int
_PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter);

/*
 * Update to next item of the iterator
 *
 * Note: this simply increment the coordinates vector, last dimension
 * incremented first , i.e, for dimension 3
 * ...
 * -1, -1, -1
 * -1, -1,  0
 * -1, -1,  1
 *  ....
 * -1,  0, -1
 * -1,  0,  0
 *  ....
 * 0,  -1, -1
 * 0,  -1,  0
 *  ....
 */
#define _UPDATE_COORD_ITER(c) \
    wb = PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->coordinates[c] < PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->bounds[c][1]; \
    if (wb) { \
        PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->coordinates[c] += 1; \
        return 0; \
    } \
    else { \
        PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->coordinates[c] = PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->bounds[c][0]; \
    }

static inline int
_PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp i, wb;

    for (i = PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->nd - 1; i >= 0; --i) {
        _UPDATE_COORD_ITER(i)
    }

    return 0;
}

/*
 * Version optimized for 2d arrays, manual loop unrolling
 */
static inline int
_PyArrayNeighborhoodIter_IncrCoord2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp wb;

    _UPDATE_COORD_ITER(1)
    _UPDATE_COORD_ITER(0)

    return 0;
}
#undef _UPDATE_COORD_ITER

/*
 * Advance to the next neighbour
 */
static inline int
PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord (iter);
    PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->dataptr = PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->translate((PyArrayIterObject*)iter, PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->coordinates);

    return 0;
}

/*
 * Reset functions
 */
static inline int
PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp i;

    for (i = 0; i < PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->nd; ++i) {
        PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->coordinates[i] = PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->bounds[i][0];
    }
    PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->dataptr = PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->translate((PyArrayIterObject*)iter, PyArrayNeighborhoodIter_GET_ITEM_DATA(iter)->coordinates);

    return 0;
}
