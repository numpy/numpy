#ifndef _NPY_INCLUDE_NEIGHBORHOOD_IMP
#error You should not include this header directly
#endif
/*
 * Private API (here for inline)
 */
static NPY_INLINE int
_PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrConstant(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrMirror(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrCircular(PyArrayNeighborhoodIterObject* iter);

static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrConstant2D(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrMirror2D(PyArrayNeighborhoodIterObject* iter);
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrCircular2D(PyArrayNeighborhoodIterObject* iter);
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
    wb = iter->coordinates[c] < iter->bounds[c][1]; \
    if (wb) { \
        iter->coordinates[c] += 1; \
        return 0; \
    } \
    else { \
        iter->coordinates[c] = iter->bounds[c][0]; \
    }

static NPY_INLINE int _PyArrayNeighborhoodIter_IncrCoord(PyArrayNeighborhoodIterObject* iter)
{
    int i, wb;

    for (i = iter->nd - 1; i >= 0; --i) {
        _UPDATE_COORD_ITER(i)
    }

    return 0;
}

/*
 * Version optimized for 2d arrays, manual loop unrolling
 */
static NPY_INLINE int _PyArrayNeighborhoodIter_IncrCoord2D(PyArrayNeighborhoodIterObject* iter)
{
    int wb;

    _UPDATE_COORD_ITER(1)
    _UPDATE_COORD_ITER(0)

    return 0;
}
#undef _UPDATE_COORD_ITER

#define _INF_SET_PTR(c) \
    bd = iter->coordinates[c] + iter->_internal_iter->coordinates[c]; \
    if (bd < 0 || bd >= iter->dimensions[c]) { \
        iter->dataptr = iter->constant; \
        return 1; \
    } \
    offset = iter->coordinates[c] * iter->strides[c]; \
    iter->dataptr += offset;

/* set the dataptr from its current coordinates */
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrConstant(PyArrayNeighborhoodIterObject* iter)
{
    int i;
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

    for(i = 0; i < iter->nd; ++i) {
        _INF_SET_PTR(i)
    }

    return 0;
}

static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrConstant2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

    _INF_SET_PTR(0)
    _INF_SET_PTR(1)

    return 0;
}

static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtr2D(PyArrayNeighborhoodIterObject* iter)
{
    switch (iter->mode) {
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
            _PyArrayNeighborhoodIter_SetPtrConstant2D(iter);
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
            _PyArrayNeighborhoodIter_SetPtrMirror2D(iter);
            break;
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            _PyArrayNeighborhoodIter_SetPtrCircular2D(iter);
            break;
    }

    return 0;
}
#undef _INF_SET_PTR

#define _NPY_IS_EVEN(x) ((x) % 2 == 0)

/* For an array x of dimension n, and given index i, returns j, 0 <= j < n
 * such as x[i] = x[j], with x assumed to be mirrored. For example, for x =
 * {1, 2, 3} (n = 3)
 *
 * index -5 -4 -3 -2 -1 0 1 2 3 4 5 6
 * value  2  3  3  2  1 1 2 3 3 2 1 1
 *
 * _npy_pos_index_mirror(4, 3) will return 1, because x[4] = x[1]*/
static NPY_INLINE npy_intp _npy_pos_remainder(npy_intp i, npy_intp n)
{
        npy_intp k, l, j;

        /* Mirror i such as it is guaranteed to be positive */
        if (i < 0) {
                i = - i - 1;
        }

        /* compute k and l such as i = k * n + l, 0 <= l < k */
        k = i / n;
        l = i - k * n;

        if (_NPY_IS_EVEN(k)) {
                j = l;
        } else {
                j = n - 1 - l;
        }
        return j;
}
#undef _NPY_IS_EVEN

#define _INF_SET_PTR_MIRROR(c) \
    bd = iter->coordinates[c] + iter->_internal_iter->coordinates[c]; \
    truepos = _npy_pos_remainder(bd, iter->dimensions[c]); \
    offset = (truepos - iter->_internal_iter->coordinates[c]) * iter->strides[c]; \
    iter->dataptr += offset;

/* set the dataptr from its current coordinates */
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrMirror(PyArrayNeighborhoodIterObject* iter)
{
    int i;
    npy_intp offset, bd, truepos;

    iter->dataptr = iter->_internal_iter->dataptr;

    for(i = 0; i < iter->nd; ++i) {
        _INF_SET_PTR_MIRROR(i)
    }

    return 0;
}

static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrMirror2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp offset, bd, truepos;

    iter->dataptr = iter->_internal_iter->dataptr;

    _INF_SET_PTR_MIRROR(0)
    _INF_SET_PTR_MIRROR(1)

    return 0;
}
#undef _INF_SET_PTR_MIRROR

/* compute l such as i = k * n + l, 0 <= l < |k| */
static NPY_INLINE npy_intp _npy_euclidean_division(npy_intp i, npy_intp n)
{
    npy_intp l;

    l = i % n;
    if (l < 0) {
        l += n;
    }
    return l;
}
#define _INF_SET_PTR_CIRCULAR(c) \
    bd = iter->coordinates[c] + iter->_internal_iter->coordinates[c]; \
    truepos = _npy_euclidean_division(bd, iter->dimensions[c]); \
    offset = (truepos - iter->_internal_iter->coordinates[c]) * iter->strides[c]; \
    iter->dataptr += offset;

/* set the dataptr from its current coordinates */
static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrCircular(PyArrayNeighborhoodIterObject* iter)
{
    int i;
    npy_intp offset, bd, truepos;

    iter->dataptr = iter->_internal_iter->dataptr;

    for(i = 0; i < iter->nd; ++i) {
        _INF_SET_PTR_CIRCULAR(i)
    }

    return 0;
}

static NPY_INLINE int
_PyArrayNeighborhoodIter_SetPtrCircular2D(PyArrayNeighborhoodIterObject* iter)
{
    npy_intp offset, bd, truepos;

    iter->dataptr = iter->_internal_iter->dataptr;

    _INF_SET_PTR_CIRCULAR(0)
    _INF_SET_PTR_CIRCULAR(1)

    return 0;
}
#undef _INF_SET_PTR_CIRCULAR

/*
 * Advance to the next neighbour
 */
static NPY_INLINE int
PyArrayNeighborhoodIter_Next2D(PyArrayNeighborhoodIterObject* iter)
{
    assert(iter->nd == 2);

    _PyArrayNeighborhoodIter_IncrCoord2D(iter);
    _PyArrayNeighborhoodIter_SetPtr2D(iter);

    return 0;
}

static NPY_INLINE int
PyArrayNeighborhoodIter_NextConstant2D(PyArrayNeighborhoodIterObject* iter)
{
    assert((iter->mode == NPY_NEIGHBORHOOD_ITER_ONE_PADDING)
           | (iter->mode == NPY_NEIGHBORHOOD_ITER_ZERO_PADDING)
           | (iter->mode == NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING));
    assert(iter->nd == 2);

    _PyArrayNeighborhoodIter_IncrCoord2D(iter);
    _PyArrayNeighborhoodIter_SetPtrConstant2D(iter);

    return 0;
}

static NPY_INLINE
int PyArrayNeighborhoodIter_NextMirror2D(PyArrayNeighborhoodIterObject* iter)
{
    assert(iter->mode == NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING);
    assert(iter->nd == 2);

    _PyArrayNeighborhoodIter_IncrCoord2D(iter);
    _PyArrayNeighborhoodIter_SetPtrMirror2D(iter);

    return 0;
}

static NPY_INLINE
int PyArrayNeighborhoodIter_NextCircular2D(PyArrayNeighborhoodIterObject* iter)
{
    assert(iter->mode == NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING);
    assert(iter->nd == 2);

    _PyArrayNeighborhoodIter_IncrCoord2D(iter);
    _PyArrayNeighborhoodIter_SetPtrCircular2D(iter);

    return 0;
}

static NPY_INLINE int
PyArrayNeighborhoodIter_NextConstant(PyArrayNeighborhoodIterObject* iter)
{
    assert((iter->mode == NPY_NEIGHBORHOOD_ITER_ONE_PADDING)
           | (iter->mode == NPY_NEIGHBORHOOD_ITER_ZERO_PADDING)
           | (iter->mode == NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING));

    _PyArrayNeighborhoodIter_IncrCoord(iter);
    _PyArrayNeighborhoodIter_SetPtrConstant(iter);

    return 0;
}

static NPY_INLINE
int PyArrayNeighborhoodIter_NextMirror(PyArrayNeighborhoodIterObject* iter)
{
    assert(iter->mode == NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING);

    _PyArrayNeighborhoodIter_IncrCoord(iter);
    _PyArrayNeighborhoodIter_SetPtrMirror(iter);

    return 0;
}

static NPY_INLINE
int PyArrayNeighborhoodIter_NextCircular(PyArrayNeighborhoodIterObject* iter)
{
    assert(iter->mode == NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING);

    _PyArrayNeighborhoodIter_IncrCoord(iter);
    _PyArrayNeighborhoodIter_SetPtrCircular(iter);

    return 0;
}

#define _INF_SET_PTR(c) \
    bd = niter->coordinates[c] + p->coordinates[c]; \
    if (bd < p->bounds[c][0] || bd > p->bounds[c][1]) { \
        return niter->constant; \
    } \
    _coordinates[c] = bd;

static inline char*
_neigh_iter_get_ptr_const(PyArrayNeighborhoodIterObject* niter)
{
    int i;
    npy_intp bd;
    PyArrayIterObject *p = niter->_internal_iter;
    npy_intp _coordinates[NPY_MAXDIMS];

    for(i = 0; i < niter->nd; ++i) {
        _INF_SET_PTR(i)
    }

    return p->translate(p, _coordinates);
}
#undef _INF_SET_PTR

static NPY_INLINE int
PyArrayNeighborhoodIter_Next(PyArrayNeighborhoodIterObject* iter)
{
    _PyArrayNeighborhoodIter_IncrCoord (iter);
    switch (iter->mode) {
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
            //iter->dataptr = _neigh_iter_get_ptr_const(iter);
            iter->dataptr = iter->translate((PyArrayIterObject*)iter, iter->coordinates);
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
            //_PyArrayNeighborhoodIter_SetPtrMirror(iter);
            iter->dataptr = iter->translate((PyArrayIterObject*)iter, iter->coordinates);
            break;
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            _PyArrayNeighborhoodIter_SetPtrCircular(iter);
            break;
    }

    return 0;
}

/*
 * Reset functions
 */
static NPY_INLINE int
PyArrayNeighborhoodIter_ResetConstant(PyArrayNeighborhoodIterObject* iter)
{
    int i;

    assert((iter->mode == NPY_NEIGHBORHOOD_ITER_ONE_PADDING)
           | (iter->mode == NPY_NEIGHBORHOOD_ITER_ZERO_PADDING)
           | (iter->mode == NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING));

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    _PyArrayNeighborhoodIter_SetPtrConstant(iter);

    return 0;
}

static NPY_INLINE int
PyArrayNeighborhoodIter_ResetMirror(PyArrayNeighborhoodIterObject* iter)
{
    int i;

    assert(iter->mode == NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING);

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    _PyArrayNeighborhoodIter_SetPtrMirror(iter);

    return 0;
}

static NPY_INLINE int
PyArrayNeighborhoodIter_ResetCircular(PyArrayNeighborhoodIterObject* iter)
{
    int i;

    assert(iter->mode == NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING);

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    _PyArrayNeighborhoodIter_SetPtrCircular(iter);

    return 0;
}

static NPY_INLINE int
PyArrayNeighborhoodIter_Reset(PyArrayNeighborhoodIterObject* iter)
{
    int i;

    for (i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    iter->dataptr = iter->translate((PyArrayIterObject*)iter, iter->coordinates);

#if 0
    switch (iter->mode) {
        case NPY_NEIGHBORHOOD_ITER_ZERO_PADDING:
        case NPY_NEIGHBORHOOD_ITER_ONE_PADDING:
        case NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING:
	    iter->dataptr = _neigh_iter_get_ptr_const(iter);
            break;
        case NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING:
            _PyArrayNeighborhoodIter_SetPtrMirror(iter);
            break;
        case NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING:
            _PyArrayNeighborhoodIter_SetPtrCircular(iter);
            break;
	default:
	    printf("oups\n");
	    exit(-1);
    }
#endif

    return 0;
}
