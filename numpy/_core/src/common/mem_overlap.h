#ifndef NUMPY_CORE_SRC_COMMON_MEM_OVERLAP_H_
#define NUMPY_CORE_SRC_COMMON_MEM_OVERLAP_H_

#include "npy_config.h"
#include "numpy/ndarraytypes.h"


/* Bounds check only */
#define NPY_MAY_SHARE_BOUNDS 0

/* Exact solution */
#define NPY_MAY_SHARE_EXACT -1


typedef enum {
    MEM_OVERLAP_NO = 0,        /* no solution exists */
    MEM_OVERLAP_YES = 1,       /* solution found */
    MEM_OVERLAP_TOO_HARD = -1, /* max_work exceeded */
    MEM_OVERLAP_OVERFLOW = -2, /* algorithm failed due to integer overflow */
    MEM_OVERLAP_ERROR = -3     /* invalid input */
} mem_overlap_t;


typedef struct {
    npy_int64 a;
    npy_int64 ub;
} diophantine_term_t;

NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_diophantine(unsigned int n, diophantine_term_t *E,
                  npy_int64 b, Py_ssize_t max_work, int require_nontrivial,
                  npy_int64 *x);

NPY_VISIBILITY_HIDDEN int
diophantine_simplify(unsigned int *n, diophantine_term_t *E, npy_int64 b);

NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_may_share_memory(PyArrayObject *a, PyArrayObject *b,
                       Py_ssize_t max_work);

NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_may_have_internal_overlap(PyArrayObject *a, Py_ssize_t max_work);

NPY_VISIBILITY_HIDDEN void
offset_bounds_from_strides(const int itemsize, const int nd,
                           const npy_intp *dims, const npy_intp *strides,
                           npy_intp *lower_offset, npy_intp *upper_offset);

#endif  /* NUMPY_CORE_SRC_COMMON_MEM_OVERLAP_H_ */
