/*
  Solving memory overlap integer programs and bounded Diophantine equations with
  positive coefficients.

  Asking whether two strided arrays `a` and `b` overlap is equivalent to
  asking whether there is a solution to the following problem::

      sum(stride_a[i] * x_a[i] for i in range(ndim_a))
      -
      sum(stride_b[i] * x_b[i] for i in range(ndim_b))
      ==
      base_b - base_a

      0 <= x_a[i] < shape_a[i]
      0 <= x_b[i] < shape_b[i]

  for some integer x_a, x_b.  Itemsize needs to be considered as an additional
  dimension with stride 1 and size itemsize.

  Negative strides can be changed to positive (and vice versa) by changing
  variables x[i] -> shape[i] - 1 - x[i], and zero strides can be dropped, so
  that the problem can be recast into a bounded Diophantine equation with
  positive coefficients::

     sum(a[i] * x[i] for i in range(n)) == b

     a[i] > 0

     0 <= x[i] <= ub[i]

  This problem is NP-hard --- runtime of algorithms grows exponentially with
  increasing ndim.


  *Algorithm description*

  A straightforward algorithm that excludes infeasible solutions using GCD-based
  pruning is outlined in Ref. [1]. It is implemented below. A number of other
  algorithms exist in the literature; however, this one seems to have
  performance satisfactory for the present purpose.

  The idea is that an equation::

      a_1 x_1 + a_2 x_2 + ... + a_n x_n = b
      0 <= x_i <= ub_i, i = 1...n

  implies::

      a_2' x_2' + a_3 x_3 + ... + a_n x_n = b

      0 <= x_i <= ub_i, i = 2...n

      0 <= x_1' <= c_1 ub_1 + c_2 ub_2

  with a_2' = gcd(a_1, a_2) and x_2' = c_1 x_1 + c_2 x_2 with c_1 = (a_1/a_1'),
  and c_2 = (a_2/a_1').  This procedure can be repeated to obtain::

      a_{n-1}' x_{n-1}' + a_n x_n = b

      0 <= x_{n-1}' <= ub_{n-1}'

      0 <= x_n <= ub_n

  Now, one can enumerate all candidate solutions for x_n.  For each, one can use
  the previous-level equation to enumerate potential solutions for x_{n-1}, with
  transformed right-hand side b -> b - a_n x_n.  And so forth, until after n-1
  nested for loops we either arrive at a candidate solution for x_1 (in which
  case we have found one solution to the problem), or find that the equations do
  not allow any solutions either for x_1 or one of the intermediate x_i (in
  which case we have proved there is no solution for the upper-level candidates
  chosen). If no solution is found for any candidate x_n, we have proved the
  problem is infeasible --- which for the memory overlap problem means there is
  no overlap.


  *Performance*

  Some common ndarray cases are easy for the algorithm:

  - Two arrays whose memory ranges do not overlap.

    These will be excluded by the bounds on x_n, with max_work=1. We also add
    this check as a fast path, to avoid computing GCDs needlessly, as this can
    take some time.

  - Arrays produced by continuous slicing of a continuous parent array (no
    internal overlap), e.g., a=x[:,0,:], b=x[:,1,:]. The strides taken together,
    mapped positive, and duplicates then satisfy gcd(stride[0], .., stride[j]) =
    stride[j] for some ordering.

    In this case, for each x[i] at most one candidate exists, given that the
    algorithm runs with strides sorted from largest to smallest. The problem can
    be written as::

       sum a_j x_j ?= b = sum a_j z_j

       a_j = n_{j+1} * n_{j+2} * ... * n_d,  a_d = 1
       0 <= x_j <= u_j <= 2*n_j - 2
       0 <= z_j <= n_j - 1

    b is the offset of the last element of the second array from the start of
    the first.  z_j are uniquely determined because of the gcd property. For
    each x_j, the bounds at first sight allow x_j=z_j and x_j=z_j+n_j. However,
    u_j <= n_j - 1 + z_j, so that at most one candidate is left.

  - Two arrays with stride-incommensurate starting points. For example,
    a=x[:,::2], b=x[:,1::2].

    The base address difference is incommensurate with all strides, so that
    there are no solution candidates to consider. For itemsize != 1, similar
    result is obtained for x_{n-1}.

  The above cases cover arrays produced by typical slicing of well-behaved
  parent arrays. More generally, more difficult cases can result::

      x = np.arange(4*20).reshape(4, 20).astype(np.int8)
      a = x[:,::7]
      b = x[:,3::3]

      <=>

      20*x1 + 7*x2 + 3*x3 = 78    (= 3 + 3*20 + 5*3)
      0 <= x1 <= 6, 0 <= x2 <= 2, 0 <= x3 <= 5

  Non-overlapping in this case relies on x.shape[1] <= lcm(7, 3) = 21.  However,
  elimination of x1 does not restrict candidate values for x3, so the algorithm
  ends up considering all values x3=0...5 separately.

  The upper bound for work done is prod(shape_a)*prod(shape_b), which scales
  faster than than work done by binary ufuncs, after broadcasting,
  prod(shape_a). The bound may be loose, but it is possible to construct hard
  instances where ufunc is faster (adapted from [2,3])::

      from numpy.lib.stride_tricks import as_strided
      # Construct non-overlapping x1 and x2
      x = np.zeros([192163377], dtype=np.int8)
      x1 = as_strided(x, strides=(36674, 61119, 85569), shape=(1049, 1049, 1049))
      x2 = as_strided(x[64023025:], strides=(12223, 12224, 1), shape=(1049, 1049, 1))

  To avoid such worst cases, the amount of work done needs to be capped. If the
  overlap problem is related to ufuncs, one suitable cap choice is to scale
  max_work with the number of elements of the array. (Ref. [3] describes a more
  efficient algorithm for solving problems similar to the above --- however,
  also it must scale exponentially.)


  *Integer overflows*

  The algorithm is written in fixed-width integers, and can terminate with
  failure if integer overflow is detected (the implementation catches all
  cases). Potential failure modes:

  - Array extent sum(stride*(shape-1)) is too large (for int64).

  - Minimal solutions to a_i x_i + a_j x_j == b are too large,
    in some of the intermediate equations.

    We do this part of the computation in 128-bit integers.

  In general, overflows are expected only if array size is close to
  NPY_INT64_MAX, requiring ~exabyte size arrays, which is usually not possible.

  References
  ----------
  .. [1] P. Ramachandran, ''Use of Extended Euclidean Algorithm in Solving
         a System of Linear Diophantine Equations with Bounded Variables''.
         Algorithmic Number Theory, Lecture Notes in Computer Science **4076**,
         182-192 (2006). doi:10.1007/11792086_14

  .. [2] Cornuejols, Urbaniak, Weismantel, and Wolsey,
         ''Decomposition of integer programs and of generating sets.'',
         Lecture Notes in Computer Science 1284, 92-103 (1997).

  .. [3] K. Aardal, A.K. Lenstra,
         ''Hard equality constrained integer knapsacks'',
         Lecture Notes in Computer Science 2337, 350-366 (2002).
*/  

/*
  Copyright (c) 2015 Pauli Virtanen
  All rights reserved.
  Licensed under 3-clause BSD license, see LICENSE.txt.
*/
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/ndarraytypes.h"
#include "mem_overlap.h"
#include "npy_extint128.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>


#define MAX(a, b) (((a) >= (b)) ? (a) : (b))
#define MIN(a, b) (((a) <= (b)) ? (a) : (b))


/**
 * Euclid's algorithm for GCD.
 *
 * Solves for gamma*a1 + epsilon*a2 == gcd(a1, a2)
 * providing |gamma| < |a2|/gcd, |epsilon| < |a1|/gcd.
 */
static void
euclid(npy_int64 a1, npy_int64 a2, npy_int64 *a_gcd, npy_int64 *gamma, npy_int64 *epsilon)
{
    npy_int64 gamma1, gamma2, epsilon1, epsilon2, r;

    assert(a1 > 0);
    assert(a2 > 0);

    gamma1 = 1;
    gamma2 = 0;
    epsilon1 = 0;
    epsilon2 = 1;

    /* The numbers remain bounded by |a1|, |a2| during
       the iteration, so no integer overflows */
    while (1) {
        if (a2 > 0) {
            r = a1/a2;
            a1 -= r*a2;
            gamma1 -= r*gamma2;
            epsilon1 -= r*epsilon2;
        }
        else {
            *a_gcd = a1;
            *gamma = gamma1;
            *epsilon = epsilon1;
            break;
        }

        if (a1 > 0) {
            r = a2/a1;
            a2 -= r*a1;
            gamma2 -= r*gamma1;
            epsilon2 -= r*epsilon1;
        }
        else {
            *a_gcd = a2;
            *gamma = gamma2;
            *epsilon = epsilon2;
            break;
        }
    }
}


/**
 * Precompute GCD and bounds transformations
 */
static int
diophantine_precompute(unsigned int n,
                       diophantine_term_t *E,
                       diophantine_term_t *Ep,
                       npy_int64 *Gamma, npy_int64 *Epsilon)
{
    npy_int64 a_gcd, gamma, epsilon, c1, c2;
    unsigned int j;
    char overflow = 0;

    assert(n >= 2);

    euclid(E[0].a, E[1].a, &a_gcd, &gamma, &epsilon);
    Ep[0].a = a_gcd;
    Gamma[0] = gamma;
    Epsilon[0] = epsilon;

    if (n > 2) {
        c1 = E[0].a / a_gcd;
        c2 = E[1].a / a_gcd;

        /* Ep[0].ub = E[0].ub * c1 + E[1].ub * c2; */
        Ep[0].ub = safe_add(safe_mul(E[0].ub, c1, &overflow),
                            safe_mul(E[1].ub, c2, &overflow), &overflow);
        if (overflow) {
            return 1;
        }
    }

    for (j = 2; j < n; ++j) {
        euclid(Ep[j-2].a, E[j].a, &a_gcd, &gamma, &epsilon);
        Ep[j-1].a = a_gcd;
        Gamma[j-1] = gamma;
        Epsilon[j-1] = epsilon;

        if (j < n - 1) {
            c1 = Ep[j-2].a / a_gcd;
            c2 = E[j].a / a_gcd;

            /* Ep[j-1].ub = c1 * Ep[j-2].ub + c2 * E[j].ub; */
            Ep[j-1].ub = safe_add(safe_mul(c1, Ep[j-2].ub, &overflow),
                                  safe_mul(c2, E[j].ub, &overflow), &overflow);

            if (overflow) {
                return 1;
            }
        }
    }

    return 0;
}


/**
 * Depth-first bounded Euclid search
 */
static mem_overlap_t
diophantine_dfs(unsigned int n,
                unsigned int v,
                diophantine_term_t *E,
                diophantine_term_t *Ep,
                npy_int64 *Gamma, npy_int64 *Epsilon,
                npy_int64 b,
                Py_ssize_t max_work,
                int require_ub_nontrivial,
                npy_int64 *x,
                Py_ssize_t *count)
{
    npy_int64 a_gcd, gamma, epsilon, a1, u1, a2, u2, c, r, c1, c2, t, t_l, t_u, b2, x1, x2;
    npy_extint128_t x10, x20, t_l1, t_l2, t_u1, t_u2;
    mem_overlap_t res;
    char overflow = 0;

    if (max_work >= 0 && *count >= max_work) {
        return MEM_OVERLAP_TOO_HARD;
    }

    /* Fetch precomputed values for the reduced problem */
    if (v == 1) {
        a1 = E[0].a;
        u1 = E[0].ub;
    }
    else {
        a1 = Ep[v-2].a;
        u1 = Ep[v-2].ub;
    }

    a2 = E[v].a;
    u2 = E[v].ub;

    a_gcd = Ep[v-1].a;
    gamma = Gamma[v-1];
    epsilon = Epsilon[v-1];

    /* Generate set of allowed solutions */
    c = b / a_gcd;
    r = b % a_gcd;
    if (r != 0) {
        ++*count;
        return MEM_OVERLAP_NO;
    }

    c1 = a2 / a_gcd;
    c2 = a1 / a_gcd;

    /*
      The set to enumerate is:
      x1 = gamma*c + c1*t
      x2 = epsilon*c - c2*t
      t integer
      0 <= x1 <= u1
      0 <= x2 <= u2
      and we have c, c1, c2 >= 0
     */

    x10 = mul_64_64(gamma, c);
    x20 = mul_64_64(epsilon, c);

    t_l1 = ceildiv_128_64(neg_128(x10), c1);
    t_l2 = ceildiv_128_64(sub_128(x20, to_128(u2), &overflow), c2);

    t_u1 = floordiv_128_64(sub_128(to_128(u1), x10, &overflow), c1);
    t_u2 = floordiv_128_64(x20, c2);

    if (overflow) {
        return MEM_OVERLAP_OVERFLOW;
    }

    if (gt_128(t_l2, t_l1)) {
        t_l1 = t_l2;
    }

    if (gt_128(t_u1, t_u2)) {
        t_u1 = t_u2;
    }

    if (gt_128(t_l1, t_u1)) {
        ++*count;
        return MEM_OVERLAP_NO;
    }

    t_l = to_64(t_l1, &overflow);
    t_u = to_64(t_u1, &overflow);

    x10 = add_128(x10, mul_64_64(c1, t_l), &overflow);
    x20 = sub_128(x20, mul_64_64(c2, t_l), &overflow);

    t_u = safe_sub(t_u, t_l, &overflow);
    t_l = 0;
    x1 = to_64(x10, &overflow);
    x2 = to_64(x20, &overflow);

    if (overflow) {
        return MEM_OVERLAP_OVERFLOW;
    }

    /* The bounds t_l, t_u ensure the x computed below do not overflow */

    if (v == 1) {
        /* Base case */
        if (t_u >= t_l) {
            x[0] = x1 + c1*t_l;
            x[1] = x2 - c2*t_l;
            if (require_ub_nontrivial) {
                unsigned int j;
                int is_ub_trivial;

                is_ub_trivial = 1;
                for (j = 0; j < n; ++j) {
                    if (x[j] != E[j].ub/2) {
                        is_ub_trivial = 0;
                        break;
                    }
                }

                if (is_ub_trivial) {
                    /* Ignore 'trivial' solution */
                    ++*count;
                    return MEM_OVERLAP_NO;
                }
            }
            return MEM_OVERLAP_YES;
        }
        ++*count;
        return MEM_OVERLAP_NO;
    }
    else {
        /* Recurse to all candidates */
        for (t = t_l; t <= t_u; ++t) {
            x[v] = x2 - c2*t;

            /* b2 = b - a2*x[v]; */
            b2 = safe_sub(b, safe_mul(a2, x[v], &overflow), &overflow);
            if (overflow) {
                return MEM_OVERLAP_OVERFLOW;
            }

            res = diophantine_dfs(n, v-1, E, Ep, Gamma, Epsilon,
                                  b2, max_work, require_ub_nontrivial,
                                  x, count);
            if (res != MEM_OVERLAP_NO) {
                return res;
            }
        }
        ++*count;
        return MEM_OVERLAP_NO;
    }
}


/**
 * Solve bounded Diophantine equation
 *
 * The problem considered is::
 *
 *     A[0] x[0] + A[1] x[1] + ... + A[n-1] x[n-1] == b
 *     0 <= x[i] <= U[i]
 *     A[i] > 0
 *
 * Solve via depth-first Euclid's algorithm, as explained in [1].
 *
 * If require_ub_nontrivial!=0, look for solutions to the problem
 * where b = A[0]*(U[0]/2) + ... + A[n]*(U[n-1]/2) but ignoring
 * the trivial solution x[i] = U[i]/2. All U[i] must be divisible by 2.
 * The value given for `b` is ignored in this case.
 */
NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_diophantine(unsigned int n, diophantine_term_t *E, npy_int64 b,
                  Py_ssize_t max_work, int require_ub_nontrivial, npy_int64 *x)
{
    mem_overlap_t res;
    unsigned int j;

    for (j = 0; j < n; ++j) {
        if (E[j].a <= 0) {
            return MEM_OVERLAP_ERROR;
        }
        else if (E[j].ub < 0) {
            return MEM_OVERLAP_NO;
        }
    }

    if (require_ub_nontrivial) {
        npy_int64 ub_sum = 0;
        char overflow = 0;
        for (j = 0; j < n; ++j) {
            if (E[j].ub % 2 != 0) {
                return MEM_OVERLAP_ERROR;
            }
            ub_sum = safe_add(ub_sum,
                              safe_mul(E[j].a, E[j].ub/2, &overflow),
                              &overflow);
        }
        if (overflow) {
            return MEM_OVERLAP_ERROR;
        }
        b = ub_sum;
    }

    if (b < 0) {
        return MEM_OVERLAP_NO;
    }

    if (n == 0) {
        if (require_ub_nontrivial) {
            /* Only trivial solution for 0-variable problem */
            return MEM_OVERLAP_NO;
        }
        if (b == 0) {
            return MEM_OVERLAP_YES;
        }
        return MEM_OVERLAP_NO;
    }
    else if (n == 1) {
        if (require_ub_nontrivial) {
            /* Only trivial solution for 1-variable problem */
            return MEM_OVERLAP_NO;
        }
        if (b % E[0].a == 0) {
            x[0] = b / E[0].a;
            if (x[0] >= 0 && x[0] <= E[0].ub) {
                return MEM_OVERLAP_YES;
            }
        }
        return MEM_OVERLAP_NO;
    }
    else {
        Py_ssize_t count = 0;
        diophantine_term_t *Ep = NULL;
        npy_int64 *Epsilon = NULL, *Gamma = NULL;

        Ep = malloc(n * sizeof(diophantine_term_t));
        Epsilon = malloc(n * sizeof(npy_int64));
        Gamma = malloc(n * sizeof(npy_int64));
        if (Ep == NULL || Epsilon == NULL || Gamma == NULL) {
            res = MEM_OVERLAP_ERROR;
        }
        else if (diophantine_precompute(n, E, Ep, Gamma, Epsilon)) {
            res = MEM_OVERLAP_OVERFLOW;
        }
        else {
            res = diophantine_dfs(n, n-1, E, Ep, Gamma, Epsilon, b, max_work,
                                  require_ub_nontrivial, x, &count);
        }
        free(Ep);
        free(Gamma);
        free(Epsilon);
        return res;
    }
}


static int
diophantine_sort_A(const void *xp, const void *yp)
{
    npy_int64 xa = ((diophantine_term_t*)xp)->a;
    npy_int64 ya = ((diophantine_term_t*)yp)->a;

    if (xa < ya) {
        return 1;
    }
    else if (ya < xa) {
        return -1;
    }
    else {
        return 0;
    }
}


/**
 * Simplify Diophantine decision problem.
 *
 * Combine identical coefficients, remove unnecessary variables, and trim
 * bounds.
 *
 * The feasible/infeasible decision result is retained.
 *
 * Returns: 0 (success), -1 (integer overflow).
 */
NPY_VISIBILITY_HIDDEN int
diophantine_simplify(unsigned int *n, diophantine_term_t *E, npy_int64 b)
{
    unsigned int i, j, m;
    char overflow = 0;

    /* Skip obviously infeasible cases */
    for (j = 0; j < *n; ++j) {
        if (E[j].ub < 0) {
            return 0;
        }
    }

    if (b < 0) {
        return 0;
    }

    /* Sort vs. coefficients */
    qsort(E, *n, sizeof(diophantine_term_t), diophantine_sort_A);

    /* Combine identical coefficients */
    m = *n;
    i = 0;
    for (j = 1; j < m; ++j) {
        if (E[i].a == E[j].a) {
            E[i].ub = safe_add(E[i].ub, E[j].ub, &overflow);
            --*n;
        }
        else {
            ++i;
            if (i != j) {
                E[i] = E[j];
            }
        }
    }

    /* Trim bounds and remove unnecessary variables */
    m = *n;
    i = 0;
    for (j = 0; j < m; ++j) {
        E[j].ub = MIN(E[j].ub, b / E[j].a);
        if (E[j].ub == 0) {
            /* If the problem is feasible at all, x[i]=0 */
            --*n;
        }
        else {
            if (i != j) {
                E[i] = E[j];
            }
            ++i;
        }
    }

    if (overflow) {
        return -1;
    }
    else {
        return 0;
    }
}


/* Gets a half-open range [start, end) of offsets from the data pointer */
NPY_VISIBILITY_HIDDEN void
offset_bounds_from_strides(const int itemsize, const int nd,
                           const npy_intp *dims, const npy_intp *strides,
                           npy_intp *lower_offset, npy_intp *upper_offset)
{
    npy_intp max_axis_offset;
    npy_intp lower = 0;
    npy_intp upper = 0;
    int i;

    for (i = 0; i < nd; i++) {
        if (dims[i] == 0) {
            /* If the array size is zero, return an empty range */
            *lower_offset = 0;
            *upper_offset = 0;
            return;
        }
        /* Expand either upwards or downwards depending on stride */
        max_axis_offset = strides[i] * (dims[i] - 1);
        if (max_axis_offset > 0) {
            upper += max_axis_offset;
        }
        else {
            lower += max_axis_offset;
        }
    }
    /* Return a half-open range */
    upper += itemsize;
    *lower_offset = lower;
    *upper_offset = upper;
}


/* Gets a half-open range [start, end) which contains the array data */
static void
get_array_memory_extents(PyArrayObject *arr,
                         npy_uintp *out_start, npy_uintp *out_end,
                         npy_uintp *num_bytes)
{
    npy_intp low, upper;
    int j;
    offset_bounds_from_strides(PyArray_ITEMSIZE(arr), PyArray_NDIM(arr),
                               PyArray_DIMS(arr), PyArray_STRIDES(arr),
                               &low, &upper);
    *out_start = (npy_uintp)PyArray_DATA(arr) + (npy_uintp)low;
    *out_end = (npy_uintp)PyArray_DATA(arr) + (npy_uintp)upper;

    *num_bytes = PyArray_ITEMSIZE(arr);
    for (j = 0; j < PyArray_NDIM(arr); ++j) {
        *num_bytes *= PyArray_DIM(arr, j);
    }
}


static int
strides_to_terms(PyArrayObject *arr, diophantine_term_t *terms,
                 unsigned int *nterms, int skip_empty)
{
    int i;

    for (i = 0; i < PyArray_NDIM(arr); ++i) {
        if (skip_empty) {
            if (PyArray_DIM(arr, i) <= 1 || PyArray_STRIDE(arr, i) == 0) {
                continue;
            }
        }

        terms[*nterms].a = PyArray_STRIDE(arr, i);

        if (terms[*nterms].a < 0) {
            terms[*nterms].a = -terms[*nterms].a;
        }

        if (terms[*nterms].a < 0) {
            /* integer overflow */
            return 1;
        }

        terms[*nterms].ub = PyArray_DIM(arr, i) - 1;
        ++*nterms;
    }

    return 0;
}


/**
 * Determine whether two arrays share some memory.
 *
 * Returns: 0 (no shared memory), 1 (shared memory), or < 0 (failed to solve).
 *
 * Note that failures to solve can occur due to integer overflows, or effort
 * required solving the problem exceeding max_work.  The general problem is
 * NP-hard and worst case runtime is exponential in the number of dimensions.
 * max_work controls the amount of work done, either exact (max_work == -1), only
 * a simple memory extent check (max_work == 0), or set an upper bound
 * max_work > 0 for the number of solution candidates considered.
 */
NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_may_share_memory(PyArrayObject *a, PyArrayObject *b,
                       Py_ssize_t max_work)
{
    npy_int64 rhs;
    diophantine_term_t terms[2*NPY_MAXDIMS + 2];
    npy_uintp start1 = 0, end1 = 0, size1 = 0;
    npy_uintp start2 = 0, end2 = 0, size2 = 0;
    npy_uintp uintp_rhs;
    npy_int64 x[2*NPY_MAXDIMS + 2];
    unsigned int nterms;

    get_array_memory_extents(a, &start1, &end1, &size1);
    get_array_memory_extents(b, &start2, &end2, &size2);

    if (!(start1 < end2 && start2 < end1 && start1 < end1 && start2 < end2)) {
        /* Memory extents don't overlap */
        return MEM_OVERLAP_NO;
    }

    if (max_work == 0) {
        /* Too much work required, give up */
        return MEM_OVERLAP_TOO_HARD;
    }

    /* Convert problem to Diophantine equation form with positive coefficients.
       The bounds computed by offset_bounds_from_strides correspond to
       all-positive strides.

       start1 + sum(abs(stride1)*x1)
       == start2 + sum(abs(stride2)*x2)
       == end1 - 1 - sum(abs(stride1)*x1')
       == end2 - 1 - sum(abs(stride2)*x2')

       <=>

       sum(abs(stride1)*x1) + sum(abs(stride2)*x2')
       == end2 - 1 - start1

       OR

       sum(abs(stride1)*x1') + sum(abs(stride2)*x2)
       == end1 - 1 - start2

       We pick the problem with the smaller RHS (they are non-negative due to
       the extent check above.)
    */

    uintp_rhs = MIN(end2 - 1 - start1, end1 - 1 - start2);
    if (uintp_rhs > NPY_MAX_INT64) {
        /* Integer overflow */
        return MEM_OVERLAP_OVERFLOW;
    }
    rhs = (npy_int64)uintp_rhs;

    nterms = 0;
    if (strides_to_terms(a, terms, &nterms, 1)) {
        return MEM_OVERLAP_OVERFLOW;
    }
    if (strides_to_terms(b, terms, &nterms, 1)) {
        return MEM_OVERLAP_OVERFLOW;
    }
    if (PyArray_ITEMSIZE(a) > 1) {
        terms[nterms].a = 1;
        terms[nterms].ub = PyArray_ITEMSIZE(a) - 1;
        ++nterms;
    }
    if (PyArray_ITEMSIZE(b) > 1) {
        terms[nterms].a = 1;
        terms[nterms].ub = PyArray_ITEMSIZE(b) - 1;
        ++nterms;
    }

    /* Simplify, if possible */
    if (diophantine_simplify(&nterms, terms, rhs)) {
        /* Integer overflow */
        return MEM_OVERLAP_OVERFLOW;
    }

    /* Solve */
    return solve_diophantine(nterms, terms, rhs, max_work, 0, x);
}


/**
 * Determine whether an array has internal overlap.
 *
 * Returns: 0 (no overlap), 1 (overlap), or < 0 (failed to solve).
 *
 * max_work and reasons for solver failures are as in solve_may_share_memory.
 */
NPY_VISIBILITY_HIDDEN mem_overlap_t
solve_may_have_internal_overlap(PyArrayObject *a, Py_ssize_t max_work)
{
    diophantine_term_t terms[NPY_MAXDIMS+1];
    npy_int64 x[NPY_MAXDIMS+1];
    unsigned int i, j, nterms;

    if (PyArray_ISCONTIGUOUS(a)) {
        /* Quick case */
        return MEM_OVERLAP_NO;
    }

    /* The internal memory overlap problem is looking for two different
       solutions to

           sum(a*x) = b,   0 <= x[i] <= ub[i]

       for any b. Equivalently,

           sum(a*x0) - sum(a*x1) = 0

       Mapping the coefficients on the left by x0'[i] = x0[i] if a[i] > 0
       else ub[i]-x0[i] and opposite for x1, we have

           sum(abs(a)*(x0' + x1')) = sum(abs(a)*ub)

       Now, x0!=x1 if for some i we have x0'[i] + x1'[i] != ub[i].
       We can now change variables to z[i] = x0'[i] + x1'[i] so the problem
       becomes

           sum(abs(a)*z) = sum(abs(a)*ub),   0 <= z[i] <= 2*ub[i],   z != ub

       This can be solved with solve_diophantine.
    */

    nterms = 0;
    if (strides_to_terms(a, terms, &nterms, 0)) {
        return MEM_OVERLAP_OVERFLOW;
    }
    if (PyArray_ITEMSIZE(a) > 1) {
        terms[nterms].a = 1;
        terms[nterms].ub = PyArray_ITEMSIZE(a) - 1;
        ++nterms;
    }

    /* Get rid of zero coefficients and empty terms */
    i = 0;
    for (j = 0; j < nterms; ++j) {
        if (terms[j].ub == 0) {
            continue;
        }
        else if (terms[j].ub < 0) {
            return MEM_OVERLAP_NO;
        }
        else if (terms[j].a == 0) {
            return MEM_OVERLAP_YES;
        }
        if (i != j) {
            terms[i] = terms[j];
        }
        ++i;
    }
    nterms = i;

    /* Double bounds to get the internal overlap problem */
    for (j = 0; j < nterms; ++j) {
        terms[j].ub *= 2;
    }

    /* Sort vs. coefficients; cannot call diophantine_simplify because it may
       change the decision problem inequality part */
    qsort(terms, nterms, sizeof(diophantine_term_t), diophantine_sort_A);

    /* Solve */
    return solve_diophantine(nterms, terms, -1, max_work, 1, x);
}
