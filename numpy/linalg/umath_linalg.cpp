/* -*- c -*- */

/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include "npy_pycompat.h"

#include "npy_config.h"

#include "npy_cblas.h"

#include <cstddef>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <utility>


static const char* umath_linalg_version_string = "0.1.5";

struct scalar_trait {};
struct complex_trait {};
template<typename typ>
using dispatch_scalar = typename std::conditional<std::is_scalar<typ>::value, scalar_trait, complex_trait>::type;

/*
 ****************************************************************************
 *                        Debugging support                                 *
 ****************************************************************************
 */
#define TRACE_TXT(...) do { fprintf (stderr, __VA_ARGS__); } while (0)
#define STACK_TRACE do {} while (0)
#define TRACE\
    do {                                        \
        fprintf (stderr,                        \
                 "%s:%d:%s\n",                  \
                 __FILE__,                      \
                 __LINE__,                      \
                 __FUNCTION__);                 \
        STACK_TRACE;                            \
    } while (0)

#if 0
#if defined HAVE_EXECINFO_H
#include <execinfo.h>
#elif defined HAVE_LIBUNWIND_H
#include <libunwind.h>
#endif
void
dbg_stack_trace()
{
    void *trace[32];
    size_t size;

    size = backtrace(trace, sizeof(trace)/sizeof(trace[0]));
    backtrace_symbols_fd(trace, size, 1);
}

#undef STACK_TRACE
#define STACK_TRACE do { dbg_stack_trace(); } while (0)
#endif

/*
 *****************************************************************************
 *                    BLAS/LAPACK calling macros                             *
 *****************************************************************************
 */

#define FNAME(x) BLAS_FUNC(x)

typedef CBLAS_INT         fortran_int;

typedef struct { float r, i; } f2c_complex;
typedef struct { double r, i; } f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

typedef float             fortran_real;
typedef double            fortran_doublereal;
typedef f2c_complex       fortran_complex;
typedef f2c_doublecomplex fortran_doublecomplex;

extern "C" fortran_int
FNAME(sgeev)(char *jobvl, char *jobvr, fortran_int *n,
             float a[], fortran_int *lda, float wr[], float wi[],
             float vl[], fortran_int *ldvl, float vr[], fortran_int *ldvr,
             float work[], fortran_int lwork[],
             fortran_int *info);
extern "C" fortran_int
FNAME(dgeev)(char *jobvl, char *jobvr, fortran_int *n,
             double a[], fortran_int *lda, double wr[], double wi[],
             double vl[], fortran_int *ldvl, double vr[], fortran_int *ldvr,
             double work[], fortran_int lwork[],
             fortran_int *info);
extern "C" fortran_int
FNAME(cgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_complex a[], fortran_int *lda,
             f2c_complex w[],
             f2c_complex vl[], fortran_int *ldvl,
             f2c_complex vr[], fortran_int *ldvr,
             f2c_complex work[], fortran_int *lwork,
             float rwork[],
             fortran_int *info);
extern "C" fortran_int
FNAME(zgeev)(char *jobvl, char *jobvr, fortran_int *n,
             f2c_doublecomplex a[], fortran_int *lda,
             f2c_doublecomplex w[],
             f2c_doublecomplex vl[], fortran_int *ldvl,
             f2c_doublecomplex vr[], fortran_int *ldvr,
             f2c_doublecomplex work[], fortran_int *lwork,
             double rwork[],
             fortran_int *info);

extern "C" fortran_int
FNAME(ssyevd)(char *jobz, char *uplo, fortran_int *n,
              float a[], fortran_int *lda, float w[], float work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *liwork,
              fortran_int *info);
extern "C" fortran_int
FNAME(dsyevd)(char *jobz, char *uplo, fortran_int *n,
              double a[], fortran_int *lda, double w[], double work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *liwork,
              fortran_int *info);
extern "C" fortran_int
FNAME(cheevd)(char *jobz, char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              float w[], f2c_complex work[],
              fortran_int *lwork, float rwork[], fortran_int *lrwork, fortran_int iwork[],
              fortran_int *liwork,
              fortran_int *info);
extern "C" fortran_int
FNAME(zheevd)(char *jobz, char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              double w[], f2c_doublecomplex work[],
              fortran_int *lwork, double rwork[], fortran_int *lrwork, fortran_int iwork[],
              fortran_int *liwork,
              fortran_int *info);

extern "C" fortran_int
FNAME(sgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              float a[], fortran_int *lda, float b[], fortran_int *ldb,
              float s[], float *rcond, fortran_int *rank,
              float work[], fortran_int *lwork, fortran_int iwork[],
              fortran_int *info);
extern "C" fortran_int
FNAME(dgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              double a[], fortran_int *lda, double b[], fortran_int *ldb,
              double s[], double *rcond, fortran_int *rank,
              double work[], fortran_int *lwork, fortran_int iwork[],
              fortran_int *info);
extern "C" fortran_int
FNAME(cgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              f2c_complex a[], fortran_int *lda,
              f2c_complex b[], fortran_int *ldb,
              float s[], float *rcond, fortran_int *rank,
              f2c_complex work[], fortran_int *lwork,
              float rwork[], fortran_int iwork[],
              fortran_int *info);
extern "C" fortran_int
FNAME(zgelsd)(fortran_int *m, fortran_int *n, fortran_int *nrhs,
              f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex b[], fortran_int *ldb,
              double s[], double *rcond, fortran_int *rank,
              f2c_doublecomplex work[], fortran_int *lwork,
              double rwork[], fortran_int iwork[],
              fortran_int *info);

extern "C" fortran_int
FNAME(dgeqrf)(fortran_int *m, fortran_int *n, double a[], fortran_int *lda,
              double tau[], double work[],
              fortran_int *lwork, fortran_int *info);
extern "C" fortran_int
FNAME(zgeqrf)(fortran_int *m, fortran_int *n, f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex tau[], f2c_doublecomplex work[],
              fortran_int *lwork, fortran_int *info);

extern "C" fortran_int
FNAME(dorgqr)(fortran_int *m, fortran_int *n, fortran_int *k, double a[], fortran_int *lda,
              double tau[], double work[],
              fortran_int *lwork, fortran_int *info);
extern "C" fortran_int
FNAME(zungqr)(fortran_int *m, fortran_int *n, fortran_int *k, f2c_doublecomplex a[],
              fortran_int *lda, f2c_doublecomplex tau[],
              f2c_doublecomplex work[], fortran_int *lwork, fortran_int *info);

extern "C" fortran_int
FNAME(sgesv)(fortran_int *n, fortran_int *nrhs,
             float a[], fortran_int *lda,
             fortran_int ipiv[],
             float b[], fortran_int *ldb,
             fortran_int *info);
extern "C" fortran_int
FNAME(dgesv)(fortran_int *n, fortran_int *nrhs,
             double a[], fortran_int *lda,
             fortran_int ipiv[],
             double b[], fortran_int *ldb,
             fortran_int *info);
extern "C" fortran_int
FNAME(cgesv)(fortran_int *n, fortran_int *nrhs,
             f2c_complex a[], fortran_int *lda,
             fortran_int ipiv[],
             f2c_complex b[], fortran_int *ldb,
             fortran_int *info);
extern "C" fortran_int
FNAME(zgesv)(fortran_int *n, fortran_int *nrhs,
             f2c_doublecomplex a[], fortran_int *lda,
             fortran_int ipiv[],
             f2c_doublecomplex b[], fortran_int *ldb,
             fortran_int *info);

extern "C" fortran_int
FNAME(sgetrf)(fortran_int *m, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
extern "C" fortran_int
FNAME(dgetrf)(fortran_int *m, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
extern "C" fortran_int
FNAME(cgetrf)(fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);
extern "C" fortran_int
FNAME(zgetrf)(fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int ipiv[],
              fortran_int *info);

extern "C" fortran_int
FNAME(spotrf)(char *uplo, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(dpotrf)(char *uplo, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(cpotrf)(char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(zpotrf)(char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int *info);

extern "C" fortran_int
FNAME(sgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              float a[], fortran_int *lda, float s[], float u[],
              fortran_int *ldu, float vt[], fortran_int *ldvt, float work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
extern "C" fortran_int
FNAME(dgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              double a[], fortran_int *lda, double s[], double u[],
              fortran_int *ldu, double vt[], fortran_int *ldvt, double work[],
              fortran_int *lwork, fortran_int iwork[], fortran_int *info);
extern "C" fortran_int
FNAME(cgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              float s[], f2c_complex u[], fortran_int *ldu,
              f2c_complex vt[], fortran_int *ldvt,
              f2c_complex work[], fortran_int *lwork,
              float rwork[], fortran_int iwork[], fortran_int *info);
extern "C" fortran_int
FNAME(zgesdd)(char *jobz, fortran_int *m, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              double s[], f2c_doublecomplex u[], fortran_int *ldu,
              f2c_doublecomplex vt[], fortran_int *ldvt,
              f2c_doublecomplex work[], fortran_int *lwork,
              double rwork[], fortran_int iwork[], fortran_int *info);

extern "C" fortran_int
FNAME(spotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              float a[], fortran_int *lda,
              float b[], fortran_int *ldb,
              fortran_int *info);
extern "C" fortran_int
FNAME(dpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              double a[], fortran_int *lda,
              double b[], fortran_int *ldb,
              fortran_int *info);
extern "C" fortran_int
FNAME(cpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              f2c_complex a[], fortran_int *lda,
              f2c_complex b[], fortran_int *ldb,
              fortran_int *info);
extern "C" fortran_int
FNAME(zpotrs)(char *uplo, fortran_int *n, fortran_int *nrhs,
              f2c_doublecomplex a[], fortran_int *lda,
              f2c_doublecomplex b[], fortran_int *ldb,
              fortran_int *info);

extern "C" fortran_int
FNAME(spotri)(char *uplo, fortran_int *n,
              float a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(dpotri)(char *uplo, fortran_int *n,
              double a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(cpotri)(char *uplo, fortran_int *n,
              f2c_complex a[], fortran_int *lda,
              fortran_int *info);
extern "C" fortran_int
FNAME(zpotri)(char *uplo, fortran_int *n,
              f2c_doublecomplex a[], fortran_int *lda,
              fortran_int *info);

extern "C" fortran_int
FNAME(scopy)(fortran_int *n,
             float *sx, fortran_int *incx,
             float *sy, fortran_int *incy);
extern "C" fortran_int
FNAME(dcopy)(fortran_int *n,
             double *sx, fortran_int *incx,
             double *sy, fortran_int *incy);
extern "C" fortran_int
FNAME(ccopy)(fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
extern "C" fortran_int
FNAME(zcopy)(fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);

extern "C" float
FNAME(sdot)(fortran_int *n,
            float *sx, fortran_int *incx,
            float *sy, fortran_int *incy);
extern "C" double
FNAME(ddot)(fortran_int *n,
            double *sx, fortran_int *incx,
            double *sy, fortran_int *incy);
extern "C" void
FNAME(cdotu)(f2c_complex *ret, fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
extern "C" void
FNAME(zdotu)(f2c_doublecomplex *ret, fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);
extern "C" void
FNAME(cdotc)(f2c_complex *ret, fortran_int *n,
             f2c_complex *sx, fortran_int *incx,
             f2c_complex *sy, fortran_int *incy);
extern "C" void
FNAME(zdotc)(f2c_doublecomplex *ret, fortran_int *n,
             f2c_doublecomplex *sx, fortran_int *incx,
             f2c_doublecomplex *sy, fortran_int *incy);

extern "C" fortran_int
FNAME(sgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             float *alpha,
             float *a, fortran_int *lda,
             float *b, fortran_int *ldb,
             float *beta,
             float *c, fortran_int *ldc);
extern "C" fortran_int
FNAME(dgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             double *alpha,
             double *a, fortran_int *lda,
             double *b, fortran_int *ldb,
             double *beta,
             double *c, fortran_int *ldc);
extern "C" fortran_int
FNAME(cgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             f2c_complex *alpha,
             f2c_complex *a, fortran_int *lda,
             f2c_complex *b, fortran_int *ldb,
             f2c_complex *beta,
             f2c_complex *c, fortran_int *ldc);
extern "C" fortran_int
FNAME(zgemm)(char *transa, char *transb,
             fortran_int *m, fortran_int *n, fortran_int *k,
             f2c_doublecomplex *alpha,
             f2c_doublecomplex *a, fortran_int *lda,
             f2c_doublecomplex *b, fortran_int *ldb,
             f2c_doublecomplex *beta,
             f2c_doublecomplex *c, fortran_int *ldc);


#define LAPACK_T(FUNC)                                          \
    TRACE_TXT("Calling LAPACK ( " # FUNC " )\n");               \
    FNAME(FUNC)

#define BLAS(FUNC)                              \
    FNAME(FUNC)

#define LAPACK(FUNC)                            \
    FNAME(FUNC)


/*
 *****************************************************************************
 **                      Some handy functions                               **
 *****************************************************************************
 */

static inline int
get_fp_invalid_and_clear(void)
{
    int status;
    status = npy_clear_floatstatus_barrier((char*)&status);
    return !!(status & NPY_FPE_INVALID);
}

static inline void
set_fp_invalid_or_clear(int error_occurred)
{
    if (error_occurred) {
        npy_set_floatstatus_invalid();
    }
    else {
        npy_clear_floatstatus_barrier((char*)&error_occurred);
    }
}

/*
 *****************************************************************************
 **                      Some handy constants                               **
 *****************************************************************************
 */

#define UMATH_LINALG_MODULE_NAME "_umath_linalg"

template<typename T>
struct numeric_limits;

template<>
struct numeric_limits<float> {
static constexpr float one = 1.0f;
static constexpr float zero = 0.0f;
static constexpr float minus_one = -1.0f;
static const float ninf;
static const float nan;
};
constexpr float numeric_limits<float>::one;
constexpr float numeric_limits<float>::zero;
constexpr float numeric_limits<float>::minus_one;
const float numeric_limits<float>::ninf = -NPY_INFINITYF;
const float numeric_limits<float>::nan = NPY_NANF;

template<>
struct numeric_limits<double> {
static constexpr double one = 1.0;
static constexpr double zero = 0.0;
static constexpr double minus_one = -1.0;
static const double ninf;
static const double nan;
};
constexpr double numeric_limits<double>::one;
constexpr double numeric_limits<double>::zero;
constexpr double numeric_limits<double>::minus_one;
const double numeric_limits<double>::ninf = -NPY_INFINITY;
const double numeric_limits<double>::nan = NPY_NAN;

template<>
struct numeric_limits<npy_cfloat> {
static constexpr npy_cfloat one = {1.0f, 0.0f};
static constexpr npy_cfloat zero = {0.0f, 0.0f};
static constexpr npy_cfloat minus_one = {-1.0f, 0.0f};
static const npy_cfloat ninf;
static const npy_cfloat nan;
};
constexpr npy_cfloat numeric_limits<npy_cfloat>::one;
constexpr npy_cfloat numeric_limits<npy_cfloat>::zero;
constexpr npy_cfloat numeric_limits<npy_cfloat>::minus_one;
const npy_cfloat numeric_limits<npy_cfloat>::ninf = {-NPY_INFINITYF, 0.0f};
const npy_cfloat numeric_limits<npy_cfloat>::nan = {NPY_NANF, NPY_NANF};

template<>
struct numeric_limits<f2c_complex> {
static constexpr f2c_complex one = {1.0f, 0.0f};
static constexpr f2c_complex zero = {0.0f, 0.0f};
static constexpr f2c_complex minus_one = {-1.0f, 0.0f};
static const f2c_complex ninf;
static const f2c_complex nan;
};
constexpr f2c_complex numeric_limits<f2c_complex>::one;
constexpr f2c_complex numeric_limits<f2c_complex>::zero;
constexpr f2c_complex numeric_limits<f2c_complex>::minus_one;
const f2c_complex numeric_limits<f2c_complex>::ninf = {-NPY_INFINITYF, 0.0f};
const f2c_complex numeric_limits<f2c_complex>::nan = {NPY_NANF, NPY_NANF};

template<>
struct numeric_limits<npy_cdouble> {
static constexpr npy_cdouble one = {1.0, 0.0};
static constexpr npy_cdouble zero = {0.0, 0.0};
static constexpr npy_cdouble minus_one = {-1.0, 0.0};
static const npy_cdouble ninf;
static const npy_cdouble nan;
};
constexpr npy_cdouble numeric_limits<npy_cdouble>::one;
constexpr npy_cdouble numeric_limits<npy_cdouble>::zero;
constexpr npy_cdouble numeric_limits<npy_cdouble>::minus_one;
const npy_cdouble numeric_limits<npy_cdouble>::ninf = {-NPY_INFINITY, 0.0};
const npy_cdouble numeric_limits<npy_cdouble>::nan = {NPY_NAN, NPY_NAN};

template<>
struct numeric_limits<f2c_doublecomplex> {
static constexpr f2c_doublecomplex one = {1.0, 0.0};
static constexpr f2c_doublecomplex zero = {0.0, 0.0};
static constexpr f2c_doublecomplex minus_one = {-1.0, 0.0};
static const f2c_doublecomplex ninf;
static const f2c_doublecomplex nan;
};
constexpr f2c_doublecomplex numeric_limits<f2c_doublecomplex>::one;
constexpr f2c_doublecomplex numeric_limits<f2c_doublecomplex>::zero;
constexpr f2c_doublecomplex numeric_limits<f2c_doublecomplex>::minus_one;
const f2c_doublecomplex numeric_limits<f2c_doublecomplex>::ninf = {-NPY_INFINITY, 0.0};
const f2c_doublecomplex numeric_limits<f2c_doublecomplex>::nan = {NPY_NAN, NPY_NAN};

/*
 *****************************************************************************
 **               Structs used for data rearrangement                       **
 *****************************************************************************
 */


/*
 * this struct contains information about how to linearize a matrix in a local
 * buffer so that it can be used by blas functions.  All strides are specified
 * in bytes and are converted to elements later in type specific functions.
 *
 * rows: number of rows in the matrix
 * columns: number of columns in the matrix
 * row_strides: the number bytes between consecutive rows.
 * column_strides: the number of bytes between consecutive columns.
 * output_lead_dim: BLAS/LAPACK-side leading dimension, in elements
 */
typedef struct linearize_data_struct
{
  npy_intp rows;
  npy_intp columns;
  npy_intp row_strides;
  npy_intp column_strides;
  npy_intp output_lead_dim;
} LINEARIZE_DATA_t;

static inline void
init_linearize_data_ex(LINEARIZE_DATA_t *lin_data,
                       npy_intp rows,
                       npy_intp columns,
                       npy_intp row_strides,
                       npy_intp column_strides,
                       npy_intp output_lead_dim)
{
    lin_data->rows = rows;
    lin_data->columns = columns;
    lin_data->row_strides = row_strides;
    lin_data->column_strides = column_strides;
    lin_data->output_lead_dim = output_lead_dim;
}

static inline void
init_linearize_data(LINEARIZE_DATA_t *lin_data,
                    npy_intp rows,
                    npy_intp columns,
                    npy_intp row_strides,
                    npy_intp column_strides)
{
    init_linearize_data_ex(
        lin_data, rows, columns, row_strides, column_strides, columns);
}

static inline void
dump_ufunc_object(PyUFuncObject* ufunc)
{
    TRACE_TXT("\n\n%s '%s' (%d input(s), %d output(s), %d specialization(s).\n",
              ufunc->core_enabled? "generalized ufunc" : "scalar ufunc",
              ufunc->name, ufunc->nin, ufunc->nout, ufunc->ntypes);
    if (ufunc->core_enabled) {
        int arg;
        int dim;
        TRACE_TXT("\t%s (%d dimension(s) detected).\n",
                  ufunc->core_signature, ufunc->core_num_dim_ix);

        for (arg = 0; arg < ufunc->nargs; arg++){
            int * arg_dim_ix = ufunc->core_dim_ixs + ufunc->core_offsets[arg];
            TRACE_TXT("\t\targ %d (%s) has %d dimension(s): (",
                      arg, arg < ufunc->nin? "INPUT" : "OUTPUT",
                      ufunc->core_num_dims[arg]);
            for (dim = 0; dim < ufunc->core_num_dims[arg]; dim ++) {
                TRACE_TXT(" %d", arg_dim_ix[dim]);
            }
            TRACE_TXT(" )\n");
        }
    }
}

static inline void
dump_linearize_data(const char* name, const LINEARIZE_DATA_t* params)
{
    TRACE_TXT("\n\t%s rows: %zd columns: %zd"\
              "\n\t\trow_strides: %td column_strides: %td"\
              "\n", name, params->rows, params->columns,
              params->row_strides, params->column_strides);
}

static inline void
print(npy_float s)
{
    TRACE_TXT(" %8.4f", s);
}
static inline void
print(npy_double d)
{
    TRACE_TXT(" %10.6f", d);
}
static inline void
print(npy_cfloat c)
{
    float* c_parts = (float*)&c;
    TRACE_TXT("(%8.4f, %8.4fj)", c_parts[0], c_parts[1]);
}
static inline void
print(npy_cdouble z)
{
    double* z_parts = (double*)&z;
    TRACE_TXT("(%8.4f, %8.4fj)", z_parts[0], z_parts[1]);
}

template<typename typ>
static inline void
dump_matrix(const char* name,
                   size_t rows, size_t columns,
                  const typ* ptr)
{
    size_t i, j;

    TRACE_TXT("\n%s %p (%zd, %zd)\n", name, ptr, rows, columns);
    for (i = 0; i < rows; i++)
    {
        TRACE_TXT("| ");
        for (j = 0; j < columns; j++)
        {
            print(ptr[j*rows + i]);
            TRACE_TXT(", ");
        }
        TRACE_TXT(" |\n");
    }
}


/*
 *****************************************************************************
 **                            Basics                                       **
 *****************************************************************************
 */

static inline fortran_int
fortran_int_min(fortran_int x, fortran_int y) {
    return x < y ? x : y;
}

static inline fortran_int
fortran_int_max(fortran_int x, fortran_int y) {
    return x > y ? x : y;
}

#define INIT_OUTER_LOOP_1 \
    npy_intp dN = *dimensions++;\
    npy_intp N_;\
    npy_intp s0 = *steps++;

#define INIT_OUTER_LOOP_2 \
    INIT_OUTER_LOOP_1\
    npy_intp s1 = *steps++;

#define INIT_OUTER_LOOP_3 \
    INIT_OUTER_LOOP_2\
    npy_intp s2 = *steps++;

#define INIT_OUTER_LOOP_4 \
    INIT_OUTER_LOOP_3\
    npy_intp s3 = *steps++;

#define INIT_OUTER_LOOP_5 \
    INIT_OUTER_LOOP_4\
    npy_intp s4 = *steps++;

#define INIT_OUTER_LOOP_6  \
    INIT_OUTER_LOOP_5\
    npy_intp s5 = *steps++;

#define INIT_OUTER_LOOP_7  \
    INIT_OUTER_LOOP_6\
    npy_intp s6 = *steps++;

#define BEGIN_OUTER_LOOP_2 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1) {

#define BEGIN_OUTER_LOOP_3 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2) {

#define BEGIN_OUTER_LOOP_4 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3) {

#define BEGIN_OUTER_LOOP_5 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4) {

#define BEGIN_OUTER_LOOP_6 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4,\
             args[5] += s5) {

#define BEGIN_OUTER_LOOP_7 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4,\
             args[5] += s5,\
             args[6] += s6) {

#define END_OUTER_LOOP  }

static inline void
update_pointers(npy_uint8** bases, ptrdiff_t* offsets, size_t count)
{
    size_t i;
    for (i = 0; i < count; ++i) {
        bases[i] += offsets[i];
    }
}


/* disable -Wmaybe-uninitialized as there is some code that generate false
   positives with this warning
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

/*
 *****************************************************************************
 **                             DISPATCHER FUNCS                            **
 *****************************************************************************
 */
static fortran_int copy(fortran_int *n,
        float *sx, fortran_int *incx,
        float *sy, fortran_int *incy) { return FNAME(scopy)(n, sx, incx,
            sy, incy);
}
static fortran_int copy(fortran_int *n,
        double *sx, fortran_int *incx,
        double *sy, fortran_int *incy) { return FNAME(dcopy)(n, sx, incx,
            sy, incy);
}
static fortran_int copy(fortran_int *n,
        f2c_complex *sx, fortran_int *incx,
        f2c_complex *sy, fortran_int *incy) { return FNAME(ccopy)(n, sx, incx,
            sy, incy);
}
static fortran_int copy(fortran_int *n,
        f2c_doublecomplex *sx, fortran_int *incx,
        f2c_doublecomplex *sy, fortran_int *incy) { return FNAME(zcopy)(n, sx, incx,
            sy, incy);
}

static fortran_int getrf(fortran_int *m, fortran_int *n, float a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(sgetrf)(m, n, a, lda, ipiv, info);
}
static fortran_int getrf(fortran_int *m, fortran_int *n, double a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(dgetrf)(m, n, a, lda, ipiv, info);
}
static fortran_int getrf(fortran_int *m, fortran_int *n, f2c_complex a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(cgetrf)(m, n, a, lda, ipiv, info);
}
static fortran_int getrf(fortran_int *m, fortran_int *n, f2c_doublecomplex a[], fortran_int
*lda, fortran_int ipiv[], fortran_int *info) {
 return LAPACK(zgetrf)(m, n, a, lda, ipiv, info);
}

/*
 *****************************************************************************
 **                             HELPER FUNCS                                **
 *****************************************************************************
 */
template<typename T>
struct fortran_type {
using type = T;
};

template<> struct fortran_type<npy_cfloat> { using type = f2c_complex;};
template<> struct fortran_type<npy_cdouble> { using type = f2c_doublecomplex;};
template<typename T>
using fortran_type_t = typename fortran_type<T>::type;

template<typename T>
struct basetype {
using type = T;
};
template<> struct basetype<npy_cfloat> { using type = npy_float;};
template<> struct basetype<npy_cdouble> { using type = npy_double;};
template<> struct basetype<f2c_complex> { using type = fortran_real;};
template<> struct basetype<f2c_doublecomplex> { using type = fortran_doublereal;};
template<typename T>
using basetype_t = typename basetype<T>::type;

             /* rearranging of 2D matrices using blas */

template<typename typ>
static inline void *
linearize_matrix(typ *dst,
                        typ *src,
                        const LINEARIZE_DATA_t* data)
{
    using ftyp = fortran_type_t<typ>;
    if (dst) {
        int i, j;
        typ* rv = dst;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
            (fortran_int)(data->column_strides/sizeof(typ));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            if (column_strides > 0) {
                copy(&columns,
                              (ftyp*)src, &column_strides,
                              (ftyp*)dst, &one);
            }
            else if (column_strides < 0) {
                copy(&columns,
                              ((ftyp*)src + (columns-1)*column_strides),
                              &column_strides,
                              (ftyp*)dst, &one);
            }
            else {
                /*
                 * Zero stride has undefined behavior in some BLAS
                 * implementations (e.g. OSX Accelerate), so do it
                 * manually
                 */
                for (j = 0; j < columns; ++j) {
                    memcpy(dst + j, src, sizeof(typ));
                }
            }
            src += data->row_strides/sizeof(typ);
            dst += data->output_lead_dim;
        }
        return rv;
    } else {
        return src;
    }
}

template<typename typ>
static inline void *
delinearize_matrix(typ *dst,
                          typ *src,
                          const LINEARIZE_DATA_t* data)
{
using ftyp = fortran_type_t<typ>;

    if (src) {
        int i;
        typ *rv = src;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
            (fortran_int)(data->column_strides/sizeof(typ));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            if (column_strides > 0) {
                copy(&columns,
                              (ftyp*)src, &one,
                              (ftyp*)dst, &column_strides);
            }
            else if (column_strides < 0) {
                copy(&columns,
                              (ftyp*)src, &one,
                              ((ftyp*)dst + (columns-1)*column_strides),
                              &column_strides);
            }
            else {
                /*
                 * Zero stride has undefined behavior in some BLAS
                 * implementations (e.g. OSX Accelerate), so do it
                 * manually
                 */
                if (columns > 0) {
                    memcpy(dst,
                           src + (columns-1),
                           sizeof(typ));
                }
            }
            src += data->output_lead_dim;
            dst += data->row_strides/sizeof(typ);
        }

        return rv;
    } else {
        return src;
    }
}

template<typename typ>
static inline void
nan_matrix(typ *dst, const LINEARIZE_DATA_t* data)
{
    int i, j;
    for (i = 0; i < data->rows; i++) {
        typ *cp = dst;
        ptrdiff_t cs = data->column_strides/sizeof(typ);
        for (j = 0; j < data->columns; ++j) {
            *cp = numeric_limits<typ>::nan;
            cp += cs;
        }
        dst += data->row_strides/sizeof(typ);
    }
}

template<typename typ>
static inline void
zero_matrix(typ *dst, const LINEARIZE_DATA_t* data)
{
    int i, j;
    for (i = 0; i < data->rows; i++) {
        typ *cp = dst;
        ptrdiff_t cs = data->column_strides/sizeof(typ);
        for (j = 0; j < data->columns; ++j) {
            *cp = numeric_limits<typ>::zero;
            cp += cs;
        }
        dst += data->row_strides/sizeof(typ);
    }
}

               /* identity square matrix generation */
template<typename typ>
static inline void
identity_matrix(typ *matrix, size_t n)
{
    size_t i;
    /* in IEEE floating point, zeroes are represented as bitwise 0 */
    memset(matrix, 0, n*n*sizeof(typ));

    for (i = 0; i < n; ++i)
    {
        *matrix = numeric_limits<typ>::one;
        matrix += n+1;
    }
}

         /* lower/upper triangular matrix using blas (in place) */

template<typename typ>
static inline void
triu_matrix(typ *matrix, size_t n)
{
    size_t i, j;
    matrix += n;
    for (i = 1; i < n; ++i) {
        for (j = 0; j < i; ++j) {
            matrix[j] = numeric_limits<typ>::zero;
        }
        matrix += n;
    }
}


/* -------------------------------------------------------------------------- */
                          /* Determinants */

static npy_float npylog(npy_float f) { return npy_logf(f);}
static npy_double npylog(npy_double d) { return npy_log(d);}
static npy_float npyexp(npy_float f) { return npy_expf(f);}
static npy_double npyexp(npy_double d) { return npy_exp(d);}

template<typename typ>
static inline void
slogdet_from_factored_diagonal(typ* src,
                                      fortran_int m,
                                      typ *sign,
                                      typ *logdet)
{
    typ acc_sign = *sign;
    typ acc_logdet = numeric_limits<typ>::zero;
    int i;
    for (i = 0; i < m; i++) {
        typ abs_element = *src;
        if (abs_element < numeric_limits<typ>::zero) {
            acc_sign = -acc_sign;
            abs_element = -abs_element;
        }

        acc_logdet += npylog(abs_element);
        src += m+1;
    }

    *sign = acc_sign;
    *logdet = acc_logdet;
}

template<typename typ>
static inline typ
det_from_slogdet(typ sign, typ logdet)
{
    typ result = sign * npyexp(logdet);
    return result;
}


npy_float npyabs(npy_cfloat z) { return npy_cabsf(z);}
npy_double npyabs(npy_cdouble z) { return npy_cabs(z);}

#define RE(COMPLEX) (COMPLEX).real
#define IM(COMPLEX) (COMPLEX).imag

template<typename typ>
static inline typ
mult(typ op1, typ op2)
{
    typ rv;

    RE(rv) = RE(op1)*RE(op2) - IM(op1)*IM(op2);
    IM(rv) = RE(op1)*IM(op2) + IM(op1)*RE(op2);

    return rv;
}


template<typename typ, typename basetyp>
static inline void
slogdet_from_factored_diagonal(typ* src,
                                      fortran_int m,
                                      typ *sign,
                                      basetyp *logdet)
{
    int i;
    typ sign_acc = *sign;
    basetyp logdet_acc = numeric_limits<basetyp>::zero;

    for (i = 0; i < m; i++)
    {
        basetyp abs_element = npyabs(*src);
        typ sign_element;
        RE(sign_element) = RE(*src) / abs_element;
        IM(sign_element) = IM(*src) / abs_element;

        sign_acc = mult(sign_acc, sign_element);
        logdet_acc += npylog(abs_element);
        src += m + 1;
    }

    *sign = sign_acc;
    *logdet = logdet_acc;
}

template<typename typ, typename basetyp>
static inline typ
det_from_slogdet(typ sign, basetyp logdet)
{
    typ tmp;
    RE(tmp) = npyexp(logdet);
    IM(tmp) = numeric_limits<basetyp>::zero;
    return mult(sign, tmp);
}
#undef RE
#undef IM


/* As in the linalg package, the determinant is computed via LU factorization
 * using LAPACK.
 * slogdet computes sign + log(determinant).
 * det computes sign * exp(slogdet).
 */
template<typename typ, typename basetyp>
static inline void
slogdet_single_element(fortran_int m,
                              typ* src,
                              fortran_int* pivots,
                              typ *sign,
                              basetyp *logdet)
{
using ftyp = fortran_type_t<typ>;
    fortran_int info = 0;
    fortran_int lda = fortran_int_max(m, 1);
    int i;
    /* note: done in place */
    getrf(&m, &m, (ftyp*)src, &lda, pivots, &info);

    if (info == 0) {
        int change_sign = 0;
        /* note: fortran uses 1 based indexing */
        for (i = 0; i < m; i++)
        {
            change_sign += (pivots[i] != (i+1));
        }

        *sign = (change_sign % 2)?numeric_limits<typ>::minus_one:numeric_limits<typ>::one;
        slogdet_from_factored_diagonal(src, m, sign, logdet);
    } else {
        /*
          if getrf fails, use 0 as sign and -inf as logdet
        */
        *sign = numeric_limits<typ>::zero;
        *logdet = numeric_limits<basetyp>::ninf;
    }
}

template<typename typ, typename basetyp>
static void
slogdet(char **args,
               npy_intp const *dimensions,
               npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    fortran_int m;
    char *tmp_buff = NULL;
    size_t matrix_size;
    size_t pivot_size;
    size_t safe_m;
    /* notes:
     *   matrix will need to be copied always, as factorization in lapack is
     *          made inplace
     *   matrix will need to be in column-major order, as expected by lapack
     *          code (fortran)
     *   always a square matrix
     *   need to allocate memory for both, matrix_buffer and pivot buffer
     */
    INIT_OUTER_LOOP_3
    m = (fortran_int) dimensions[0];
    /* avoid empty malloc (buffers likely unused) and ensure m is `size_t` */
    safe_m = m != 0 ? m : 1;
    matrix_size = safe_m * safe_m * sizeof(typ);
    pivot_size = safe_m * sizeof(fortran_int);
    tmp_buff = (char *)malloc(matrix_size + pivot_size);

    if (tmp_buff) {
        LINEARIZE_DATA_t lin_data;
        /* swapped steps to get matrix in FORTRAN order */
        init_linearize_data(&lin_data, m, m, steps[1], steps[0]);
        BEGIN_OUTER_LOOP_3
            linearize_matrix((typ*)tmp_buff, (typ*)args[0], &lin_data);
            slogdet_single_element(m,
                                   (typ*)tmp_buff,
                                          (fortran_int*)(tmp_buff+matrix_size),
                                          (typ*)args[1],
                                          (basetyp*)args[2]);
        END_OUTER_LOOP

        free(tmp_buff);
    }
    else {
        /* TODO: Requires use of new ufunc API to indicate error return */
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
}

template<typename typ, typename basetyp>
static void
det(char **args,
           npy_intp const *dimensions,
           npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    fortran_int m;
    char *tmp_buff;
    size_t matrix_size;
    size_t pivot_size;
    size_t safe_m;
    /* notes:
     *   matrix will need to be copied always, as factorization in lapack is
     *       made inplace
     *   matrix will need to be in column-major order, as expected by lapack
     *       code (fortran)
     *   always a square matrix
     *   need to allocate memory for both, matrix_buffer and pivot buffer
     */
    INIT_OUTER_LOOP_2
    m = (fortran_int) dimensions[0];
    /* avoid empty malloc (buffers likely unused) and ensure m is `size_t` */
    safe_m = m != 0 ? m : 1;
    matrix_size = safe_m * safe_m * sizeof(typ);
    pivot_size = safe_m * sizeof(fortran_int);
    tmp_buff = (char *)malloc(matrix_size + pivot_size);

    if (tmp_buff) {
        LINEARIZE_DATA_t lin_data;
        typ sign;
        basetyp logdet;
        /* swapped steps to get matrix in FORTRAN order */
        init_linearize_data(&lin_data, m, m, steps[1], steps[0]);

        BEGIN_OUTER_LOOP_2
            linearize_matrix((typ*)tmp_buff, (typ*)args[0], &lin_data);
            slogdet_single_element(m,
                                         (typ*)tmp_buff,
                                          (fortran_int*)(tmp_buff + matrix_size),
                                          &sign,
                                          &logdet);
            *(typ *)args[1] = det_from_slogdet(sign, logdet);
        END_OUTER_LOOP

        free(tmp_buff);
    }
    else {
        /* TODO: Requires use of new ufunc API to indicate error return */
        NPY_ALLOW_C_API_DEF
        NPY_ALLOW_C_API;
        PyErr_NoMemory();
        NPY_DISABLE_C_API;
    }
}


/* -------------------------------------------------------------------------- */
                          /* Eigh family */

template<typename typ>
struct EIGH_PARAMS_t {
    typ *A;     /* matrix */
    basetype_t<typ> *W;     /* eigenvalue vector */
    typ *WORK;  /* main work buffer */
    basetype_t<typ> *RWORK; /* secondary work buffer (for complex versions) */
    fortran_int *IWORK;
    fortran_int N;
    fortran_int LWORK;
    fortran_int LRWORK;
    fortran_int LIWORK;
    char JOBZ;
    char UPLO;
    fortran_int LDA;
} ;

static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_float> *params)
{
    fortran_int rv;
    LAPACK(ssyevd)(&params->JOBZ, &params->UPLO, &params->N,
                          params->A, &params->LDA, params->W,
                          params->WORK, &params->LWORK,
                          params->IWORK, &params->LIWORK,
                          &rv);
    return rv;
}
static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_double> *params)
{
    fortran_int rv;
    LAPACK(dsyevd)(&params->JOBZ, &params->UPLO, &params->N,
                          params->A, &params->LDA, params->W,
                          params->WORK, &params->LWORK,
                          params->IWORK, &params->LIWORK,
                          &rv);
    return rv;
}


/*
 * Initialize the parameters to use in for the lapack function _syevd
 * Handles buffer allocation
 */
template<typename typ>
static inline int
init_evd(EIGH_PARAMS_t<typ>* params, char JOBZ, char UPLO,
                   fortran_int N, scalar_trait)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    fortran_int lwork;
    fortran_int liwork;
    npy_uint8 *a, *w, *work, *iwork;
    size_t safe_N = N;
    size_t alloc_size = safe_N * (safe_N + 1) * sizeof(typ);
    fortran_int lda = fortran_int_max(N, 1);

    mem_buff = (npy_uint8 *)malloc(alloc_size);

    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    w = mem_buff + safe_N * safe_N * sizeof(typ);

    params->A = (typ*)a;
    params->W = (typ*)w;
    params->RWORK = NULL; /* unused */
    params->N = N;
    params->LRWORK = 0; /* unused */
    params->JOBZ = JOBZ;
    params->UPLO = UPLO;
    params->LDA = lda;

    /* Work size query */
    {
        typ query_work_size;
        fortran_int query_iwork_size;

        params->LWORK = -1;
        params->LIWORK = -1;
        params->WORK = &query_work_size;
        params->IWORK = &query_iwork_size;

        if (call_evd(params) != 0) {
            goto error;
        }

        lwork = (fortran_int)query_work_size;
        liwork = query_iwork_size;
    }

    mem_buff2 = (npy_uint8 *)malloc(lwork*sizeof(typ) + liwork*sizeof(fortran_int));
    if (!mem_buff2) {
        goto error;
    }

    work = mem_buff2;
    iwork = mem_buff2 + lwork*sizeof(typ);

    params->LWORK = lwork;
    params->WORK = (typ*)work;
    params->LIWORK = liwork;
    params->IWORK = (fortran_int*)iwork;

    return 1;

 error:
    /* something failed */
    memset(params, 0, sizeof(*params));
    free(mem_buff2);
    free(mem_buff);

    return 0;
}


static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_cfloat> *params)
{
    fortran_int rv;
    LAPACK(cheevd)(&params->JOBZ, &params->UPLO, &params->N,
                          (fortran_type_t<npy_cfloat>*)params->A, &params->LDA, params->W,
                          (fortran_type_t<npy_cfloat>*)params->WORK, &params->LWORK,
                          params->RWORK, &params->LRWORK,
                          params->IWORK, &params->LIWORK,
                          &rv);
    return rv;
}

static inline fortran_int
call_evd(EIGH_PARAMS_t<npy_cdouble> *params)
{
    fortran_int rv;
    LAPACK(zheevd)(&params->JOBZ, &params->UPLO, &params->N,
                          (fortran_type_t<npy_cdouble>*)params->A, &params->LDA, params->W,
                          (fortran_type_t<npy_cdouble>*)params->WORK, &params->LWORK,
                          params->RWORK, &params->LRWORK,
                          params->IWORK, &params->LIWORK,
                          &rv);
    return rv;
}

template<typename typ>
static inline int
init_evd(EIGH_PARAMS_t<typ> *params,
                   char JOBZ,
                   char UPLO,
                   fortran_int N, complex_trait)
{
    using basetyp = basetype_t<typ>;
using ftyp = fortran_type_t<typ>;
using fbasetyp = fortran_type_t<basetyp>;
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    fortran_int lwork;
    fortran_int lrwork;
    fortran_int liwork;
    npy_uint8 *a, *w, *work, *rwork, *iwork;
    size_t safe_N = N;
    fortran_int lda = fortran_int_max(N, 1);

    mem_buff = (npy_uint8 *)malloc(safe_N * safe_N * sizeof(typ) +
                      safe_N * sizeof(basetyp));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    w = mem_buff + safe_N * safe_N * sizeof(typ);

    params->A = (typ*)a;
    params->W = (basetyp*)w;
    params->N = N;
    params->JOBZ = JOBZ;
    params->UPLO = UPLO;
    params->LDA = lda;

    /* Work size query */
    {
        ftyp query_work_size;
        fbasetyp query_rwork_size;
        fortran_int query_iwork_size;

        params->LWORK = -1;
        params->LRWORK = -1;
        params->LIWORK = -1;
        params->WORK = (typ*)&query_work_size;
        params->RWORK = (basetyp*)&query_rwork_size;
        params->IWORK = &query_iwork_size;

        if (call_evd(params) != 0) {
            goto error;
        }

        lwork = (fortran_int)*(fbasetyp*)&query_work_size;
        lrwork = (fortran_int)query_rwork_size;
        liwork = query_iwork_size;
    }

    mem_buff2 = (npy_uint8 *)malloc(lwork*sizeof(typ) +
                       lrwork*sizeof(basetyp) +
                       liwork*sizeof(fortran_int));
    if (!mem_buff2) {
        goto error;
    }

    work = mem_buff2;
    rwork = work + lwork*sizeof(typ);
    iwork = rwork + lrwork*sizeof(basetyp);

    params->WORK = (typ*)work;
    params->RWORK = (basetyp*)rwork;
    params->IWORK = (fortran_int*)iwork;
    params->LWORK = lwork;
    params->LRWORK = lrwork;
    params->LIWORK = liwork;

    return 1;

    /* something failed */
error:
    memset(params, 0, sizeof(*params));
    free(mem_buff2);
    free(mem_buff);

    return 0;
}

/*
 * (M, M)->(M,)(M, M)
 * dimensions[1] -> M
 * args[0] -> A[in]
 * args[1] -> W
 * args[2] -> A[out]
 */

template<typename typ>
static inline void
release_evd(EIGH_PARAMS_t<typ> *params)
{
    /* allocated memory in A and WORK */
    free(params->A);
    free(params->WORK);
    memset(params, 0, sizeof(*params));
}


template<typename typ>
static inline void
eigh_wrapper(char JOBZ,
                    char UPLO,
                    char**args,
                    npy_intp const *dimensions,
                    npy_intp const *steps)
{
    using basetyp = basetype_t<typ>;
    ptrdiff_t outer_steps[3];
    size_t iter;
    size_t outer_dim = *dimensions++;
    size_t op_count = (JOBZ=='N')?2:3;
    EIGH_PARAMS_t<typ> eigh_params;
    int error_occurred = get_fp_invalid_and_clear();

    for (iter = 0; iter < op_count; ++iter) {
        outer_steps[iter] = (ptrdiff_t) steps[iter];
    }
    steps += op_count;

    if (init_evd(&eigh_params,
                           JOBZ,
                           UPLO,
                           (fortran_int)dimensions[0], dispatch_scalar<typ>())) {
        LINEARIZE_DATA_t matrix_in_ld;
        LINEARIZE_DATA_t eigenvectors_out_ld;
        LINEARIZE_DATA_t eigenvalues_out_ld;

        init_linearize_data(&matrix_in_ld,
                            eigh_params.N, eigh_params.N,
                            steps[1], steps[0]);
        init_linearize_data(&eigenvalues_out_ld,
                            1, eigh_params.N,
                            0, steps[2]);
        if ('V' == eigh_params.JOBZ) {
            init_linearize_data(&eigenvectors_out_ld,
                                eigh_params.N, eigh_params.N,
                                steps[4], steps[3]);
        }

        for (iter = 0; iter < outer_dim; ++iter) {
            int not_ok;
            /* copy the matrix in */
            linearize_matrix((typ*)eigh_params.A, (typ*)args[0], &matrix_in_ld);
            not_ok = call_evd(&eigh_params);
            if (!not_ok) {
                /* lapack ok, copy result out */
                delinearize_matrix((basetyp*)args[1],
                                              (basetyp*)eigh_params.W,
                                              &eigenvalues_out_ld);

                if ('V' == eigh_params.JOBZ) {
                    delinearize_matrix((typ*)args[2],
                                              (typ*)eigh_params.A,
                                              &eigenvectors_out_ld);
                }
            } else {
                /* lapack fail, set result to nan */
                error_occurred = 1;
                nan_matrix((basetyp*)args[1], &eigenvalues_out_ld);
                if ('V' == eigh_params.JOBZ) {
                    nan_matrix((typ*)args[2], &eigenvectors_out_ld);
                }
            }
            update_pointers((npy_uint8**)args, outer_steps, op_count);
        }

        release_evd(&eigh_params);
    }

    set_fp_invalid_or_clear(error_occurred);
}


template<typename typ>
static void
eighlo(char **args,
              npy_intp const *dimensions,
              npy_intp const *steps,
              void *NPY_UNUSED(func))
{
    eigh_wrapper<typ>('V', 'L', args, dimensions, steps);
}

template<typename typ>
static void
eighup(char **args,
              npy_intp const *dimensions,
              npy_intp const *steps,
              void* NPY_UNUSED(func))
{
    eigh_wrapper<typ>('V', 'U', args, dimensions, steps);
}

template<typename typ>
static void
eigvalshlo(char **args,
                  npy_intp const *dimensions,
                  npy_intp const *steps,
                  void* NPY_UNUSED(func))
{
    eigh_wrapper<typ>('N', 'L', args, dimensions, steps);
}

template<typename typ>
static void
eigvalshup(char **args,
                  npy_intp const *dimensions,
                  npy_intp const *steps,
                  void* NPY_UNUSED(func))
{
    eigh_wrapper<typ>('N', 'U', args, dimensions, steps);
}

/* -------------------------------------------------------------------------- */
                  /* Solve family (includes inv) */

template<typename typ>
struct GESV_PARAMS_t
{
    typ *A; /* A is (N, N) of base type */
    typ *B; /* B is (N, NRHS) of base type */
    fortran_int * IPIV; /* IPIV is (N) */

    fortran_int N;
    fortran_int NRHS;
    fortran_int LDA;
    fortran_int LDB;
};

static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    LAPACK(sgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}

static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    LAPACK(dgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}

static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_complex> *params)
{
    fortran_int rv;
    LAPACK(cgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}

static inline fortran_int
call_gesv(GESV_PARAMS_t<fortran_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zgesv)(&params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->IPIV,
                          params->B, &params->LDB,
                          &rv);
    return rv;
}


/*
 * Initialize the parameters to use in for the lapack function _heev
 * Handles buffer allocation
 */
template<typename ftyp>
static inline int
init_gesv(GESV_PARAMS_t<ftyp> *params, fortran_int N, fortran_int NRHS)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *a, *b, *ipiv;
    size_t safe_N = N;
    size_t safe_NRHS = NRHS;
    fortran_int ld = fortran_int_max(N, 1);
    mem_buff = (npy_uint8 *)malloc(safe_N * safe_N * sizeof(ftyp) +
                      safe_N * safe_NRHS*sizeof(ftyp) +
                      safe_N * sizeof(fortran_int));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_N * safe_N * sizeof(ftyp);
    ipiv = b + safe_N * safe_NRHS * sizeof(ftyp);

    params->A = (ftyp*)a;
    params->B = (ftyp*)b;
    params->IPIV = (fortran_int*)ipiv;
    params->N = N;
    params->NRHS = NRHS;
    params->LDA = ld;
    params->LDB = ld;

    return 1;
 error:
    free(mem_buff);
    memset(params, 0, sizeof(*params));

    return 0;
}

template<typename ftyp>
static inline void
release_gesv(GESV_PARAMS_t<ftyp> *params)
{
    /* memory block base is in A */
    free(params->A);
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static void
solve(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;
    GESV_PARAMS_t<ftyp> params;
    fortran_int n, nrhs;
    int error_occurred = get_fp_invalid_and_clear();
    INIT_OUTER_LOOP_3

    n = (fortran_int)dimensions[0];
    nrhs = (fortran_int)dimensions[1];
    if (init_gesv(&params, n, nrhs)) {
        LINEARIZE_DATA_t a_in, b_in, r_out;

        init_linearize_data(&a_in, n, n, steps[1], steps[0]);
        init_linearize_data(&b_in, nrhs, n, steps[3], steps[2]);
        init_linearize_data(&r_out, nrhs, n, steps[5], steps[4]);

        BEGIN_OUTER_LOOP_3
            int not_ok;
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.B, (typ*)args[1], &b_in);
            not_ok =call_gesv(&params);
            if (!not_ok) {
                delinearize_matrix((typ*)args[2], (typ*)params.B, &r_out);
            } else {
                error_occurred = 1;
                nan_matrix((typ*)args[2], &r_out);
            }
        END_OUTER_LOOP

        release_gesv(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}


template<typename typ>
static void
solve1(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;
    GESV_PARAMS_t<ftyp> params;
    int error_occurred = get_fp_invalid_and_clear();
    fortran_int n;
    INIT_OUTER_LOOP_3

    n = (fortran_int)dimensions[0];
    if (init_gesv(&params, n, 1)) {
        LINEARIZE_DATA_t a_in, b_in, r_out;
        init_linearize_data(&a_in, n, n, steps[1], steps[0]);
        init_linearize_data(&b_in, 1, n, 1, steps[2]);
        init_linearize_data(&r_out, 1, n, 1, steps[3]);

        BEGIN_OUTER_LOOP_3
            int not_ok;
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.B, (typ*)args[1], &b_in);
            not_ok = call_gesv(&params);
            if (!not_ok) {
                delinearize_matrix((typ*)args[2], (typ*)params.B, &r_out);
            } else {
                error_occurred = 1;
                nan_matrix((typ*)args[2], &r_out);
            }
        END_OUTER_LOOP

        release_gesv(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}

template<typename typ>
static void
inv(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;
    GESV_PARAMS_t<ftyp> params;
    fortran_int n;
    int error_occurred = get_fp_invalid_and_clear();
    INIT_OUTER_LOOP_2

    n = (fortran_int)dimensions[0];
    if (init_gesv(&params, n, n)) {
        LINEARIZE_DATA_t a_in, r_out;
        init_linearize_data(&a_in, n, n, steps[1], steps[0]);
        init_linearize_data(&r_out, n, n, steps[3], steps[2]);

        BEGIN_OUTER_LOOP_2
            int not_ok;
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            identity_matrix((typ*)params.B, n);
            not_ok = call_gesv(&params);
            if (!not_ok) {
                delinearize_matrix((typ*)args[1], (typ*)params.B, &r_out);
            } else {
                error_occurred = 1;
                nan_matrix((typ*)args[1], &r_out);
            }
        END_OUTER_LOOP

        release_gesv(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}


/* -------------------------------------------------------------------------- */
                     /* Cholesky decomposition */

template<typename typ>
struct POTR_PARAMS_t
{
    typ *A;
    fortran_int N;
    fortran_int LDA;
    char UPLO;
};


static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    LAPACK(spotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    LAPACK(dpotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_complex> *params)
{
    fortran_int rv;
    LAPACK(cpotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

static inline fortran_int
call_potrf(POTR_PARAMS_t<fortran_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zpotrf)(&params->UPLO,
                          &params->N, params->A, &params->LDA,
                          &rv);
    return rv;
}

template<typename ftyp>
static inline int
init_potrf(POTR_PARAMS_t<ftyp> *params, char UPLO, fortran_int N)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *a;
    size_t safe_N = N;
    fortran_int lda = fortran_int_max(N, 1);

    mem_buff = (npy_uint8 *)malloc(safe_N * safe_N * sizeof(ftyp));
    if (!mem_buff) {
        goto error;
    }

    a = mem_buff;

    params->A = (ftyp*)a;
    params->N = N;
    params->LDA = lda;
    params->UPLO = UPLO;

    return 1;
 error:
    free(mem_buff);
    memset(params, 0, sizeof(*params));

    return 0;
}

template<typename ftyp>
static inline void
release_potrf(POTR_PARAMS_t<ftyp> *params)
{
    /* memory block base in A */
    free(params->A);
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static void
cholesky(char uplo, char **args, npy_intp const *dimensions, npy_intp const *steps)
{
    using ftyp = fortran_type_t<typ>;
    POTR_PARAMS_t<ftyp> params;
    int error_occurred = get_fp_invalid_and_clear();
    fortran_int n;
    INIT_OUTER_LOOP_2

    assert(uplo == 'L');

    n = (fortran_int)dimensions[0];
    if (init_potrf(&params, uplo, n)) {
        LINEARIZE_DATA_t a_in, r_out;
        init_linearize_data(&a_in, n, n, steps[1], steps[0]);
        init_linearize_data(&r_out, n, n, steps[3], steps[2]);
        BEGIN_OUTER_LOOP_2
            int not_ok;
            linearize_matrix(params.A, (ftyp*)args[0], &a_in);
            not_ok = call_potrf(&params);
            if (!not_ok) {
                triu_matrix(params.A, params.N);
                delinearize_matrix((ftyp*)args[1], params.A, &r_out);
            } else {
                error_occurred = 1;
                nan_matrix((ftyp*)args[1], &r_out);
            }
        END_OUTER_LOOP
        release_potrf(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}

template<typename typ>
static void
cholesky_lo(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func))
{
    cholesky<typ>('L', args, dimensions, steps);
}

/* -------------------------------------------------------------------------- */
                          /* eig family  */

template<typename typ>
struct GEEV_PARAMS_t {
    typ *A;
    basetype_t<typ> *WR; /* RWORK in complex versions, REAL W buffer for (sd)geev*/
    typ *WI;
    typ *VLR; /* REAL VL buffers for _geev where _ is s, d */
    typ *VRR; /* REAL VR buffers for _geev where _ is s, d */
    typ *WORK;
    typ *W;  /* final w */
    typ *VL; /* final vl */
    typ *VR; /* final vr */

    fortran_int N;
    fortran_int LDA;
    fortran_int LDVL;
    fortran_int LDVR;
    fortran_int LWORK;

    char JOBVL;
    char JOBVR;
};

template<typename typ>
static inline void
dump_geev_params(const char *name, GEEV_PARAMS_t<typ>* params)
{
    TRACE_TXT("\n%s\n"

              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\
              "\t%10s: %p\n"\

              "\t%10s: %d\n"\
              "\t%10s: %d\n"\
              "\t%10s: %d\n"\
              "\t%10s: %d\n"\
              "\t%10s: %d\n"\

              "\t%10s: %c\n"\
              "\t%10s: %c\n",

              name,

              "A", params->A,
              "WR", params->WR,
              "WI", params->WI,
              "VLR", params->VLR,
              "VRR", params->VRR,
              "WORK", params->WORK,
              "W", params->W,
              "VL", params->VL,
              "VR", params->VR,

              "N", (int)params->N,
              "LDA", (int)params->LDA,
              "LDVL", (int)params->LDVL,
              "LDVR", (int)params->LDVR,
              "LWORK", (int)params->LWORK,

              "JOBVL", params->JOBVL,
              "JOBVR", params->JOBVR);
}

static inline fortran_int
call_geev(GEEV_PARAMS_t<float>* params)
{
    fortran_int rv;
    LAPACK(sgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->WR, params->WI,
                          params->VLR, &params->LDVL,
                          params->VRR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}

static inline fortran_int
call_geev(GEEV_PARAMS_t<double>* params)
{
    fortran_int rv;
    LAPACK(dgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->WR, params->WI,
                          params->VLR, &params->LDVL,
                          params->VRR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}


template<typename typ>
static inline int
init_geev(GEEV_PARAMS_t<typ> *params, char jobvl, char jobvr, fortran_int n,
scalar_trait)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *wr, *wi, *vlr, *vrr, *work, *w, *vl, *vr;
    size_t safe_n = n;
    size_t a_size = safe_n * safe_n * sizeof(typ);
    size_t wr_size = safe_n * sizeof(typ);
    size_t wi_size = safe_n * sizeof(typ);
    size_t vlr_size = jobvl=='V' ? safe_n * safe_n * sizeof(typ) : 0;
    size_t vrr_size = jobvr=='V' ? safe_n * safe_n * sizeof(typ) : 0;
    size_t w_size = wr_size*2;
    size_t vl_size = vlr_size*2;
    size_t vr_size = vrr_size*2;
    size_t work_count = 0;
    fortran_int ld = fortran_int_max(n, 1);

    /* allocate data for known sizes (all but work) */
    mem_buff = (npy_uint8 *)malloc(a_size + wr_size + wi_size +
                      vlr_size + vrr_size +
                      w_size + vl_size + vr_size);
    if (!mem_buff) {
        goto error;
    }

    a = mem_buff;
    wr = a + a_size;
    wi = wr + wr_size;
    vlr = wi + wi_size;
    vrr = vlr + vlr_size;
    w = vrr + vrr_size;
    vl = w + w_size;
    vr = vl + vl_size;

    params->A = (typ*)a;
    params->WR = (typ*)wr;
    params->WI = (typ*)wi;
    params->VLR = (typ*)vlr;
    params->VRR = (typ*)vrr;
    params->W = (typ*)w;
    params->VL = (typ*)vl;
    params->VR = (typ*)vr;
    params->N = n;
    params->LDA = ld;
    params->LDVL = ld;
    params->LDVR = ld;
    params->JOBVL = jobvl;
    params->JOBVR = jobvr;

    /* Work size query */
    {
        typ work_size_query;

        params->LWORK = -1;
        params->WORK = &work_size_query;

        if (call_geev(params) != 0) {
            goto error;
        }

        work_count = (size_t)work_size_query;
    }

    mem_buff2 = (npy_uint8 *)malloc(work_count*sizeof(typ));
    if (!mem_buff2) {
        goto error;
    }
    work = mem_buff2;

    params->LWORK = (fortran_int)work_count;
    params->WORK = (typ*)work;

    return 1;
 error:
    free(mem_buff2);
    free(mem_buff);
    memset(params, 0, sizeof(*params));

    return 0;
}

template<typename complextyp, typename typ>
static inline void
mk_complex_array_from_real(complextyp *c, const typ *re, size_t n)
{
    size_t iter;
    for (iter = 0; iter < n; ++iter) {
        c[iter].r = re[iter];
        c[iter].i = numeric_limits<typ>::zero;
    }
}

template<typename complextyp, typename typ>
static inline void
mk_complex_array(complextyp *c,
                        const typ *re,
                        const typ *im,
                        size_t n)
{
    size_t iter;
    for (iter = 0; iter < n; ++iter) {
        c[iter].r = re[iter];
        c[iter].i = im[iter];
    }
}

template<typename complextyp, typename typ>
static inline void
mk_complex_array_conjugate_pair(complextyp *c,
                                       const typ *r,
                                       size_t n)
{
    size_t iter;
    for (iter = 0; iter < n; ++iter) {
        typ re = r[iter];
        typ im = r[iter+n];
        c[iter].r = re;
        c[iter].i = im;
        c[iter+n].r = re;
        c[iter+n].i = -im;
    }
}

/*
 * make the complex eigenvectors from the real array produced by sgeev/zgeev.
 * c is the array where the results will be left.
 * r is the source array of reals produced by sgeev/zgeev
 * i is the eigenvalue imaginary part produced by sgeev/zgeev
 * n is so that the order of the matrix is n by n
 */
template<typename complextyp, typename typ>
static inline void
mk_geev_complex_eigenvectors(complextyp *c,
                                      const typ *r,
                                      const typ *i,
                                      size_t n)
{
    size_t iter = 0;
    while (iter < n)
    {
        if (i[iter] ==  numeric_limits<typ>::zero) {
            /* eigenvalue was real, eigenvectors as well...  */
            mk_complex_array_from_real(c, r, n);
            c += n;
            r += n;
            iter ++;
        } else {
            /* eigenvalue was complex, generate a pair of eigenvectors */
            mk_complex_array_conjugate_pair(c, r, n);
            c += 2*n;
            r += 2*n;
            iter += 2;
        }
    }
}


template<typename complextyp, typename typ>
static inline void
process_geev_results(GEEV_PARAMS_t<typ> *params, scalar_trait)
{
    /* REAL versions of geev need the results to be translated
     * into complex versions. This is the way to deal with imaginary
     * results. In our gufuncs we will always return complex arrays!
     */
    mk_complex_array((complextyp*)params->W, (typ*)params->WR, (typ*)params->WI, params->N);

    /* handle the eigenvectors */
    if ('V' == params->JOBVL) {
        mk_geev_complex_eigenvectors((complextyp*)params->VL, (typ*)params->VLR,
                                              (typ*)params->WI, params->N);
    }
    if ('V' == params->JOBVR) {
        mk_geev_complex_eigenvectors((complextyp*)params->VR, (typ*)params->VRR,
                                              (typ*)params->WI, params->N);
    }
}


static inline fortran_int
call_geev(GEEV_PARAMS_t<fortran_complex>* params)
{
    fortran_int rv;

    LAPACK(cgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->W,
                          params->VL, &params->LDVL,
                          params->VR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          params->WR, /* actually RWORK */
                          &rv);
    return rv;
}
static inline fortran_int
call_geev(GEEV_PARAMS_t<fortran_doublecomplex>* params)
{
    fortran_int rv;

    LAPACK(zgeev)(&params->JOBVL, &params->JOBVR,
                          &params->N, params->A, &params->LDA,
                          params->W,
                          params->VL, &params->LDVL,
                          params->VR, &params->LDVR,
                          params->WORK, &params->LWORK,
                          params->WR, /* actually RWORK */
                          &rv);
    return rv;
}

template<typename ftyp>
static inline int
init_geev(GEEV_PARAMS_t<ftyp>* params,
                   char jobvl,
                   char jobvr,
                   fortran_int n, complex_trait)
{
using realtyp = basetype_t<ftyp>;
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *w, *vl, *vr, *work, *rwork;
    size_t safe_n = n;
    size_t a_size = safe_n * safe_n * sizeof(ftyp);
    size_t w_size = safe_n * sizeof(ftyp);
    size_t vl_size = jobvl=='V'? safe_n * safe_n * sizeof(ftyp) : 0;
    size_t vr_size = jobvr=='V'? safe_n * safe_n * sizeof(ftyp) : 0;
    size_t rwork_size = 2 * safe_n * sizeof(realtyp);
    size_t work_count = 0;
    size_t total_size = a_size + w_size + vl_size + vr_size + rwork_size;
    fortran_int ld = fortran_int_max(n, 1);

    mem_buff = (npy_uint8 *)malloc(total_size);
    if (!mem_buff) {
        goto error;
    }

    a = mem_buff;
    w = a + a_size;
    vl = w + w_size;
    vr = vl + vl_size;
    rwork = vr + vr_size;

    params->A = (ftyp*)a;
    params->WR = (realtyp*)rwork;
    params->WI = NULL;
    params->VLR = NULL;
    params->VRR = NULL;
    params->VL = (ftyp*)vl;
    params->VR = (ftyp*)vr;
    params->W = (ftyp*)w;
    params->N = n;
    params->LDA = ld;
    params->LDVL = ld;
    params->LDVR = ld;
    params->JOBVL = jobvl;
    params->JOBVR = jobvr;

    /* Work size query */
    {
        ftyp work_size_query;

        params->LWORK = -1;
        params->WORK = &work_size_query;

        if (call_geev(params) != 0) {
            goto error;
        }

        work_count = (size_t) work_size_query.r;
        /* Fix a bug in lapack 3.0.0 */
        if(work_count == 0) work_count = 1;
    }

    mem_buff2 = (npy_uint8 *)malloc(work_count*sizeof(ftyp));
    if (!mem_buff2) {
        goto error;
    }

    work = mem_buff2;

    params->LWORK = (fortran_int)work_count;
    params->WORK = (ftyp*)work;

    return 1;
 error:
    free(mem_buff2);
    free(mem_buff);
    memset(params, 0, sizeof(*params));

    return 0;
}

template<typename complextyp, typename typ>
static inline void
process_geev_results(GEEV_PARAMS_t<typ> *NPY_UNUSED(params), complex_trait)
{
    /* nothing to do here, complex versions are ready to copy out */
}



template<typename typ>
static inline void
release_geev(GEEV_PARAMS_t<typ> *params)
{
    free(params->WORK);
    free(params->A);
    memset(params, 0, sizeof(*params));
}

template<typename fctype, typename ftype>
static inline void
eig_wrapper(char JOBVL,
                   char JOBVR,
                   char**args,
                   npy_intp const *dimensions,
                   npy_intp const *steps)
{
    ptrdiff_t outer_steps[4];
    size_t iter;
    size_t outer_dim = *dimensions++;
    size_t op_count = 2;
    int error_occurred = get_fp_invalid_and_clear();
    GEEV_PARAMS_t<ftype> geev_params;

    assert(JOBVL == 'N');

    STACK_TRACE;
    op_count += 'V'==JOBVL?1:0;
    op_count += 'V'==JOBVR?1:0;

    for (iter = 0; iter < op_count; ++iter) {
        outer_steps[iter] = (ptrdiff_t) steps[iter];
    }
    steps += op_count;

    if (init_geev(&geev_params,
                           JOBVL, JOBVR,
                           (fortran_int)dimensions[0], dispatch_scalar<ftype>())) {
        LINEARIZE_DATA_t a_in;
        LINEARIZE_DATA_t w_out;
        LINEARIZE_DATA_t vl_out;
        LINEARIZE_DATA_t vr_out;

        init_linearize_data(&a_in,
                            geev_params.N, geev_params.N,
                            steps[1], steps[0]);
        steps += 2;
        init_linearize_data(&w_out,
                            1, geev_params.N,
                            0, steps[0]);
        steps += 1;
        if ('V' == geev_params.JOBVL) {
            init_linearize_data(&vl_out,
                                geev_params.N, geev_params.N,
                                steps[1], steps[0]);
            steps += 2;
        }
        if ('V' == geev_params.JOBVR) {
            init_linearize_data(&vr_out,
                                geev_params.N, geev_params.N,
                                steps[1], steps[0]);
        }

        for (iter = 0; iter < outer_dim; ++iter) {
            int not_ok;
            char **arg_iter = args;
            /* copy the matrix in */
            linearize_matrix((ftype*)geev_params.A, (ftype*)*arg_iter++, &a_in);
            not_ok = call_geev(&geev_params);

            if (!not_ok) {
                process_geev_results<fctype>(&geev_params,
dispatch_scalar<ftype>{});
                delinearize_matrix((fctype*)*arg_iter++,
                                                 (fctype*)geev_params.W,
                                                 &w_out);

                if ('V' == geev_params.JOBVL) {
                    delinearize_matrix((fctype*)*arg_iter++,
                                                     (fctype*)geev_params.VL,
                                                     &vl_out);
                }
                if ('V' == geev_params.JOBVR) {
                    delinearize_matrix((fctype*)*arg_iter++,
                                                     (fctype*)geev_params.VR,
                                                     &vr_out);
                }
            } else {
                /* geev failed */
                error_occurred = 1;
                nan_matrix((fctype*)*arg_iter++, &w_out);
                if ('V' == geev_params.JOBVL) {
                    nan_matrix((fctype*)*arg_iter++, &vl_out);
                }
                if ('V' == geev_params.JOBVR) {
                    nan_matrix((fctype*)*arg_iter++, &vr_out);
                }
            }
            update_pointers((npy_uint8**)args, outer_steps, op_count);
        }

        release_geev(&geev_params);
    }

    set_fp_invalid_or_clear(error_occurred);
}

template<typename fctype, typename ftype>
static void
eig(char **args,
           npy_intp const *dimensions,
           npy_intp const *steps,
           void *NPY_UNUSED(func))
{
    eig_wrapper<fctype, ftype>('N', 'V', args, dimensions, steps);
}

template<typename fctype, typename ftype>
static void
eigvals(char **args,
               npy_intp const *dimensions,
               npy_intp const *steps,
               void *NPY_UNUSED(func))
{
    eig_wrapper<fctype, ftype>('N', 'N', args, dimensions, steps);
}



/* -------------------------------------------------------------------------- */
                 /* singular value decomposition  */

template<typename ftyp>
struct GESDD_PARAMS_t
{
    ftyp *A;
    basetype_t<ftyp> *S;
    ftyp *U;
    ftyp *VT;
    ftyp *WORK;
    basetype_t<ftyp> *RWORK;
    fortran_int *IWORK;

    fortran_int M;
    fortran_int N;
    fortran_int LDA;
    fortran_int LDU;
    fortran_int LDVT;
    fortran_int LWORK;
    char JOBZ;
} ;


template<typename ftyp>
static inline void
dump_gesdd_params(const char *name,
                  GESDD_PARAMS_t<ftyp> *params)
{
    TRACE_TXT("\n%s:\n"\

              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\

              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\

              "%14s: %15c'%c'\n",

              name,

              "A", params->A,
              "S", params->S,
              "U", params->U,
              "VT", params->VT,
              "WORK", params->WORK,
              "RWORK", params->RWORK,
              "IWORK", params->IWORK,

              "M", (int)params->M,
              "N", (int)params->N,
              "LDA", (int)params->LDA,
              "LDU", (int)params->LDU,
              "LDVT", (int)params->LDVT,
              "LWORK", (int)params->LWORK,

              "JOBZ", ' ', params->JOBZ);
}

static inline int
compute_urows_vtcolumns(char jobz,
                        fortran_int m, fortran_int n,
                        fortran_int *urows, fortran_int *vtcolumns)
{
    fortran_int min_m_n = fortran_int_min(m, n);
    switch(jobz)
    {
    case 'N':
        *urows = 0;
        *vtcolumns = 0;
        break;
    case 'A':
        *urows = m;
        *vtcolumns = n;
        break;
    case 'S':
        {
            *urows = min_m_n;
            *vtcolumns = min_m_n;
        }
        break;
    default:
        return 0;
    }

    return 1;
}

static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    LAPACK(sgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          (fortran_int*)params->IWORK,
                          &rv);
    return rv;
}
static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    LAPACK(dgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          (fortran_int*)params->IWORK,
                          &rv);
    return rv;
}

template<typename ftyp>
static inline int
init_gesdd(GESDD_PARAMS_t<ftyp> *params,
                   char jobz,
                   fortran_int m,
                   fortran_int n, scalar_trait)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *s, *u, *vt, *work, *iwork;
    size_t safe_m = m;
    size_t safe_n = n;
    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_min_m_n = min_m_n;
    size_t s_size = safe_min_m_n * sizeof(ftyp);
    fortran_int u_row_count, vt_column_count;
    size_t safe_u_row_count, safe_vt_column_count;
    size_t u_size, vt_size;
    fortran_int work_count;
    size_t work_size;
    size_t iwork_size = 8 * safe_min_m_n * sizeof(fortran_int);
    fortran_int ld = fortran_int_max(m, 1);

    if (!compute_urows_vtcolumns(jobz, m, n, &u_row_count, &vt_column_count)) {
        goto error;
    }

    safe_u_row_count = u_row_count;
    safe_vt_column_count = vt_column_count;

    u_size = safe_u_row_count * safe_m * sizeof(ftyp);
    vt_size = safe_n * safe_vt_column_count * sizeof(ftyp);

    mem_buff = (npy_uint8 *)malloc(a_size + s_size + u_size + vt_size + iwork_size);

    if (!mem_buff) {
        goto error;
    }

    a = mem_buff;
    s = a + a_size;
    u = s + s_size;
    vt = u + u_size;
    iwork = vt + vt_size;

    /* fix vt_column_count so that it is a valid lapack parameter (0 is not) */
    vt_column_count = fortran_int_max(1, vt_column_count);

    params->M = m;
    params->N = n;
    params->A = (ftyp*)a;
    params->S = (ftyp*)s;
    params->U = (ftyp*)u;
    params->VT = (ftyp*)vt;
    params->RWORK = NULL;
    params->IWORK = (fortran_int*)iwork;
    params->LDA = ld;
    params->LDU = ld;
    params->LDVT = vt_column_count;
    params->JOBZ = jobz;

    /* Work size query */
    {
        ftyp work_size_query;

        params->LWORK = -1;
        params->WORK = &work_size_query;

        if (call_gesdd(params) != 0) {
            goto error;
        }

        work_count = (fortran_int)work_size_query;
        /* Fix a bug in lapack 3.0.0 */
        if(work_count == 0) work_count = 1;
        work_size = (size_t)work_count * sizeof(ftyp);
    }

    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2) {
        goto error;
    }

    work = mem_buff2;

    params->LWORK = work_count;
    params->WORK = (ftyp*)work;

    return 1;
 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));

    return 0;
}

static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_complex> *params)
{
    fortran_int rv;
    LAPACK(cgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          params->RWORK,
                          params->IWORK,
                          &rv);
    return rv;
}
static inline fortran_int
call_gesdd(GESDD_PARAMS_t<fortran_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zgesdd)(&params->JOBZ, &params->M, &params->N,
                          params->A, &params->LDA,
                          params->S,
                          params->U, &params->LDU,
                          params->VT, &params->LDVT,
                          params->WORK, &params->LWORK,
                          params->RWORK,
                          params->IWORK,
                          &rv);
    return rv;
}

template<typename ftyp>
static inline int
init_gesdd(GESDD_PARAMS_t<ftyp> *params,
                   char jobz,
                   fortran_int m,
                   fortran_int n, complex_trait)
{
using frealtyp = basetype_t<ftyp>;
    npy_uint8 *mem_buff = NULL, *mem_buff2 = NULL;
    npy_uint8 *a,*s, *u, *vt, *work, *rwork, *iwork;
    size_t a_size, s_size, u_size, vt_size, work_size, rwork_size, iwork_size;
    size_t safe_u_row_count, safe_vt_column_count;
    fortran_int u_row_count, vt_column_count, work_count;
    size_t safe_m = m;
    size_t safe_n = n;
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_min_m_n = min_m_n;
    fortran_int ld = fortran_int_max(m, 1);

    if (!compute_urows_vtcolumns(jobz, m, n, &u_row_count, &vt_column_count)) {
        goto error;
    }

    safe_u_row_count = u_row_count;
    safe_vt_column_count = vt_column_count;

    a_size = safe_m * safe_n * sizeof(ftyp);
    s_size = safe_min_m_n * sizeof(frealtyp);
    u_size = safe_u_row_count * safe_m * sizeof(ftyp);
    vt_size = safe_n * safe_vt_column_count * sizeof(ftyp);
    rwork_size = 'N'==jobz?
        (7 * safe_min_m_n) :
        (5*safe_min_m_n * safe_min_m_n + 5*safe_min_m_n);
    rwork_size *= sizeof(ftyp);
    iwork_size = 8 * safe_min_m_n* sizeof(fortran_int);

    mem_buff = (npy_uint8 *)malloc(a_size +
                      s_size +
                      u_size +
                      vt_size +
                      rwork_size +
                      iwork_size);
    if (!mem_buff) {
        goto error;
    }

    a = mem_buff;
    s = a + a_size;
    u = s + s_size;
    vt = u + u_size;
    rwork = vt + vt_size;
    iwork = rwork + rwork_size;

    /* fix vt_column_count so that it is a valid lapack parameter (0 is not) */
    vt_column_count = fortran_int_max(1, vt_column_count);

    params->A = (ftyp*)a;
    params->S = (frealtyp*)s;
    params->U = (ftyp*)u;
    params->VT = (ftyp*)vt;
    params->RWORK = (frealtyp*)rwork;
    params->IWORK = (fortran_int*)iwork;
    params->M = m;
    params->N = n;
    params->LDA = ld;
    params->LDU = ld;
    params->LDVT = vt_column_count;
    params->JOBZ = jobz;

    /* Work size query */
    {
        ftyp work_size_query;

        params->LWORK = -1;
        params->WORK = &work_size_query;

        if (call_gesdd(params) != 0) {
            goto error;
        }

        work_count = (fortran_int)(*(frealtyp*)&work_size_query);
        /* Fix a bug in lapack 3.0.0 */
        if(work_count == 0) work_count = 1;
        work_size = (size_t)work_count * sizeof(ftyp);
    }

    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2) {
        goto error;
    }

    work = mem_buff2;

    params->LWORK = work_count;
    params->WORK = (ftyp*)work;

    return 1;
 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff2);
    free(mem_buff);
    memset(params, 0, sizeof(*params));

    return 0;
}

template<typename typ>
static inline void
release_gesdd(GESDD_PARAMS_t<typ>* params)
{
    /* A and WORK contain allocated blocks */
    free(params->A);
    free(params->WORK);
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static inline void
svd_wrapper(char JOBZ,
                   char **args,
                   npy_intp const *dimensions,
                   npy_intp const *steps)
{
using basetyp = basetype_t<typ>;
    ptrdiff_t outer_steps[4];
    int error_occurred = get_fp_invalid_and_clear();
    size_t iter;
    size_t outer_dim = *dimensions++;
    size_t op_count = (JOBZ=='N')?2:4;
    GESDD_PARAMS_t<typ> params;

    for (iter = 0; iter < op_count; ++iter) {
        outer_steps[iter] = (ptrdiff_t) steps[iter];
    }
    steps += op_count;

    if (init_gesdd(&params,
                   JOBZ,
                   (fortran_int)dimensions[0],
                   (fortran_int)dimensions[1],
dispatch_scalar<typ>())) {
        LINEARIZE_DATA_t a_in, u_out, s_out, v_out;
        fortran_int min_m_n = params.M < params.N ? params.M : params.N;

        init_linearize_data(&a_in, params.N, params.M, steps[1], steps[0]);
        if ('N' == params.JOBZ) {
            /* only the singular values are wanted */
            init_linearize_data(&s_out, 1, min_m_n, 0, steps[2]);
        } else {
            fortran_int u_columns, v_rows;
            if ('S' == params.JOBZ) {
                u_columns = min_m_n;
                v_rows = min_m_n;
            } else { /* JOBZ == 'A' */
                u_columns = params.M;
                v_rows = params.N;
            }
            init_linearize_data(&u_out,
                                u_columns, params.M,
                                steps[3], steps[2]);
            init_linearize_data(&s_out,
                                1, min_m_n,
                                0, steps[4]);
            init_linearize_data(&v_out,
                                params.N, v_rows,
                                steps[6], steps[5]);
        }

        for (iter = 0; iter < outer_dim; ++iter) {
            int not_ok;
            /* copy the matrix in */
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            not_ok = call_gesdd(&params);
            if (!not_ok) {
                if ('N' == params.JOBZ) {
                    delinearize_matrix((basetyp*)args[1], (basetyp*)params.S, &s_out);
                } else {
                    if ('A' == params.JOBZ && min_m_n == 0) {
                        /* Lapack has betrayed us and left these uninitialized,
                         * so produce an identity matrix for whichever of u
                         * and v is not empty.
                         */
                        identity_matrix((typ*)params.U, params.M);
                        identity_matrix((typ*)params.VT, params.N);
                    }

                    delinearize_matrix((typ*)args[1], (typ*)params.U, &u_out);
                    delinearize_matrix((basetyp*)args[2], (basetyp*)params.S, &s_out);
                    delinearize_matrix((typ*)args[3], (typ*)params.VT, &v_out);
                }
            } else {
                error_occurred = 1;
                if ('N' == params.JOBZ) {
                    nan_matrix((basetyp*)args[1], &s_out);
                } else {
                    nan_matrix((typ*)args[1], &u_out);
                    nan_matrix((basetyp*)args[2], &s_out);
                    nan_matrix((typ*)args[3], &v_out);
                }
            }
            update_pointers((npy_uint8**)args, outer_steps, op_count);
        }

        release_gesdd(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}


template<typename typ>
static void
svd_N(char **args,
             npy_intp const *dimensions,
             npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    svd_wrapper<fortran_type_t<typ>>('N', args, dimensions, steps);
}

template<typename typ>
static void
svd_S(char **args,
             npy_intp const *dimensions,
             npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    svd_wrapper<fortran_type_t<typ>>('S', args, dimensions, steps);
}

template<typename typ>
static void
svd_A(char **args,
             npy_intp const *dimensions,
             npy_intp const *steps,
             void *NPY_UNUSED(func))
{
    svd_wrapper<fortran_type_t<typ>>('A', args, dimensions, steps);
}

/* -------------------------------------------------------------------------- */
                 /* qr (modes - r, raw) */

template<typename typ>
struct GEQRF_PARAMS_t
{
    fortran_int M;
    fortran_int N;
    typ *A;
    fortran_int LDA;
    typ* TAU;
    typ *WORK;
    fortran_int LWORK;
};


template<typename typ>
static inline void
dump_geqrf_params(const char *name,
                  GEQRF_PARAMS_t<typ> *params)
{
    TRACE_TXT("\n%s:\n"\

              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n",

              name,

              "A", params->A,
              "TAU", params->TAU,
              "WORK", params->WORK,

              "M", (int)params->M,
              "N", (int)params->N,
              "LDA", (int)params->LDA,
              "LWORK", (int)params->LWORK);
}

static inline fortran_int
call_geqrf(GEQRF_PARAMS_t<double> *params)
{
    fortran_int rv;
    LAPACK(dgeqrf)(&params->M, &params->N,
                          params->A, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}
static inline fortran_int
call_geqrf(GEQRF_PARAMS_t<f2c_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zgeqrf)(&params->M, &params->N,
                          params->A, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}


static inline int
init_geqrf(GEQRF_PARAMS_t<fortran_doublereal> *params,
                   fortran_int m,
                   fortran_int n)
{
using ftyp = fortran_doublereal;
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *tau, *work;
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_min_m_n = min_m_n;
    size_t safe_m = m;
    size_t safe_n = n;

    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t tau_size = safe_min_m_n * sizeof(ftyp);

    fortran_int work_count;
    size_t work_size;
    fortran_int lda = fortran_int_max(1, m);

    mem_buff = (npy_uint8 *)malloc(a_size + tau_size);

    if (!mem_buff)
        goto error;

    a = mem_buff;
    tau = a + a_size;
    memset(tau, 0, tau_size);


    params->M = m;
    params->N = n;
    params->A = (ftyp*)a;
    params->TAU = (ftyp*)tau;
    params->LDA = lda;

    {
        /* compute optimal work size */

        ftyp work_size_query;

        params->WORK = &work_size_query;
        params->LWORK = -1;

        if (call_geqrf(params) != 0)
            goto error;

        work_count = (fortran_int) *(ftyp*) params->WORK;

    }

    params->LWORK = fortran_int_max(fortran_int_max(1, n), work_count);

    work_size = (size_t) params->LWORK * sizeof(ftyp);
    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2)
        goto error;

    work = mem_buff2;

    params->WORK = (ftyp*)work;

    return 1;
 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));

    return 0;
}


static inline int
init_geqrf(GEQRF_PARAMS_t<fortran_doublecomplex> *params,
                   fortran_int m,
                   fortran_int n)
{
using ftyp = fortran_doublecomplex;
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *tau, *work;
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_min_m_n = min_m_n;
    size_t safe_m = m;
    size_t safe_n = n;

    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t tau_size = safe_min_m_n * sizeof(ftyp);

    fortran_int work_count;
    size_t work_size;
    fortran_int lda = fortran_int_max(1, m);

    mem_buff = (npy_uint8 *)malloc(a_size + tau_size);

    if (!mem_buff)
        goto error;

    a = mem_buff;
    tau = a + a_size;
    memset(tau, 0, tau_size);


    params->M = m;
    params->N = n;
    params->A = (ftyp*)a;
    params->TAU = (ftyp*)tau;
    params->LDA = lda;

    {
        /* compute optimal work size */

        ftyp work_size_query;

        params->WORK = &work_size_query;
        params->LWORK = -1;

        if (call_geqrf(params) != 0)
            goto error;

        work_count = (fortran_int) ((ftyp*)params->WORK)->r;

    }

    params->LWORK = fortran_int_max(fortran_int_max(1, n),
                                    work_count);

    work_size = (size_t) params->LWORK * sizeof(ftyp);

    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2)
        goto error;

    work = mem_buff2;

    params->WORK = (ftyp*)work;

    return 1;
 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));

    return 0;
}


template<typename ftyp>
static inline void
release_geqrf(GEQRF_PARAMS_t<ftyp>* params)
{
    /* A and WORK contain allocated blocks */
    free(params->A);
    free(params->WORK);
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static void
qr_r_raw(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;

    GEQRF_PARAMS_t<ftyp> params;
    int error_occurred = get_fp_invalid_and_clear();
    fortran_int n, m;

    INIT_OUTER_LOOP_2

    m = (fortran_int)dimensions[0];
    n = (fortran_int)dimensions[1];

    if (init_geqrf(&params, m, n)) {
        LINEARIZE_DATA_t a_in, tau_out;

        init_linearize_data(&a_in, n, m, steps[1], steps[0]);
        init_linearize_data(&tau_out, 1, fortran_int_min(m, n), 1, steps[2]);

        BEGIN_OUTER_LOOP_2
            int not_ok;
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            not_ok = call_geqrf(&params);
            if (!not_ok) {
                delinearize_matrix((typ*)args[0], (typ*)params.A, &a_in);
                delinearize_matrix((typ*)args[1], (typ*)params.TAU, &tau_out);
            } else {
                error_occurred = 1;
                nan_matrix((typ*)args[1], &tau_out);
            }
        END_OUTER_LOOP

        release_geqrf(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}


/* -------------------------------------------------------------------------- */
                 /* qr common code (modes - reduced and complete) */ 

template<typename typ>
struct GQR_PARAMS_t
{
    fortran_int M;
    fortran_int MC;
    fortran_int MN;
    void* A;
    typ *Q;
    fortran_int LDA;
    typ* TAU;
    typ *WORK;
    fortran_int LWORK;
} ;

static inline fortran_int
call_gqr(GQR_PARAMS_t<double> *params)
{
    fortran_int rv;
    LAPACK(dorgqr)(&params->M, &params->MC, &params->MN,
                          params->Q, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}
static inline fortran_int
call_gqr(GQR_PARAMS_t<f2c_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zungqr)(&params->M, &params->MC, &params->MN,
                          params->Q, &params->LDA,
                          params->TAU,
                          params->WORK, &params->LWORK,
                          &rv);
    return rv;
}

static inline int
init_gqr_common(GQR_PARAMS_t<fortran_doublereal> *params,
                          fortran_int m,
                          fortran_int n,
                          fortran_int mc)
{
using ftyp = fortran_doublereal;
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *q, *tau, *work;
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_mc = mc;
    size_t safe_min_m_n = min_m_n;
    size_t safe_m = m;
    size_t safe_n = n;
    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t q_size = safe_m * safe_mc * sizeof(ftyp);
    size_t tau_size = safe_min_m_n * sizeof(ftyp);

    fortran_int work_count;
    size_t work_size;
    fortran_int lda = fortran_int_max(1, m);

    mem_buff = (npy_uint8 *)malloc(q_size + tau_size + a_size);

    if (!mem_buff)
        goto error;

    q = mem_buff;
    tau = q + q_size;
    a = tau + tau_size;


    params->M = m;
    params->MC = mc;
    params->MN = min_m_n;
    params->A = a;
    params->Q = (ftyp*)q;
    params->TAU = (ftyp*)tau;
    params->LDA = lda;

    {
        /* compute optimal work size */
        ftyp work_size_query;

        params->WORK = &work_size_query;
        params->LWORK = -1;

        if (call_gqr(params) != 0)
            goto error;

        work_count = (fortran_int) *(ftyp*) params->WORK;

    }

    params->LWORK = fortran_int_max(fortran_int_max(1, n), work_count);

    work_size = (size_t) params->LWORK * sizeof(ftyp);

    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2)
        goto error;

    work = mem_buff2;

    params->WORK = (ftyp*)work;

    return 1;
 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));

    return 0;
}


static inline int
init_gqr_common(GQR_PARAMS_t<fortran_doublecomplex> *params,
                          fortran_int m,
                          fortran_int n,
                          fortran_int mc)
{
using ftyp=fortran_doublecomplex;
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *q, *tau, *work;
    fortran_int min_m_n = fortran_int_min(m, n);
    size_t safe_mc = mc;
    size_t safe_min_m_n = min_m_n;
    size_t safe_m = m;
    size_t safe_n = n;

    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t q_size = safe_m * safe_mc * sizeof(ftyp);
    size_t tau_size = safe_min_m_n * sizeof(ftyp);

    fortran_int work_count;
    size_t work_size;
    fortran_int lda = fortran_int_max(1, m);

    mem_buff = (npy_uint8 *)malloc(q_size + tau_size + a_size);

    if (!mem_buff)
        goto error;

    q = mem_buff;
    tau = q + q_size;
    a = tau + tau_size;


    params->M = m;
    params->MC = mc;
    params->MN = min_m_n;
    params->A = a;
    params->Q = (ftyp*)q;
    params->TAU = (ftyp*)tau;
    params->LDA = lda;

    {
        /* compute optimal work size */
        ftyp work_size_query;

        params->WORK = &work_size_query;
        params->LWORK = -1;

        if (call_gqr(params) != 0)
            goto error;

        work_count = (fortran_int) ((ftyp*)params->WORK)->r;

    }

    params->LWORK = fortran_int_max(fortran_int_max(1, n),
                                    work_count);

    work_size = (size_t) params->LWORK * sizeof(ftyp);

    mem_buff2 = (npy_uint8 *)malloc(work_size);
    if (!mem_buff2)
        goto error;

    work = mem_buff2;

    params->WORK = (ftyp*)work;
    params->LWORK = work_count;

    return 1;
 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));

    return 0;
}

/* -------------------------------------------------------------------------- */
                 /* qr (modes - reduced) */


template<typename typ>
static inline void
dump_gqr_params(const char *name,
                GQR_PARAMS_t<typ> *params)
{
    TRACE_TXT("\n%s:\n"\

              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n",

              name,

              "Q", params->Q,
              "TAU", params->TAU,
              "WORK", params->WORK,

              "M", (int)params->M,
              "MC", (int)params->MC,
              "MN", (int)params->MN,
              "LDA", (int)params->LDA,
              "LWORK", (int)params->LWORK);
}

template<typename ftyp>
static inline int
init_gqr(GQR_PARAMS_t<ftyp> *params,
                   fortran_int m,
                   fortran_int n)
{
    return init_gqr_common(
        params, m, n, 
        fortran_int_min(m, n));
}


template<typename typ>
static inline void
release_gqr(GQR_PARAMS_t<typ>* params)
{
    /* A and WORK contain allocated blocks */
    free(params->Q);
    free(params->WORK);
    memset(params, 0, sizeof(*params));
}

template<typename typ>
static void
qr_reduced(char **args, npy_intp const *dimensions, npy_intp const *steps,
                  void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;
    GQR_PARAMS_t<ftyp> params;
    int error_occurred = get_fp_invalid_and_clear();
    fortran_int n, m;

    INIT_OUTER_LOOP_3

    m = (fortran_int)dimensions[0];
    n = (fortran_int)dimensions[1];

    if (init_gqr(&params, m, n)) {
        LINEARIZE_DATA_t a_in, tau_in, q_out;

        init_linearize_data(&a_in, n, m, steps[1], steps[0]);
        init_linearize_data(&tau_in, 1, fortran_int_min(m, n), 1, steps[2]);
        init_linearize_data(&q_out, fortran_int_min(m, n), m, steps[4], steps[3]);

        BEGIN_OUTER_LOOP_3
            int not_ok;
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.Q, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.TAU, (typ*)args[1], &tau_in);
            not_ok = call_gqr(&params);
            if (!not_ok) {
                delinearize_matrix((typ*)args[2], (typ*)params.Q, &q_out);
            } else {
                error_occurred = 1;
                nan_matrix((typ*)args[2], &q_out);
            }
        END_OUTER_LOOP

        release_gqr(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}

/* -------------------------------------------------------------------------- */
                 /* qr (modes - complete) */

template<typename ftyp>
static inline int
init_gqr_complete(GQR_PARAMS_t<ftyp> *params,
                            fortran_int m,
                            fortran_int n)
{
    return init_gqr_common(params, m, n, m);
}


template<typename typ>
static void
qr_complete(char **args, npy_intp const *dimensions, npy_intp const *steps,
                  void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;
    GQR_PARAMS_t<ftyp> params;
    int error_occurred = get_fp_invalid_and_clear();
    fortran_int n, m;

    INIT_OUTER_LOOP_3

    m = (fortran_int)dimensions[0];
    n = (fortran_int)dimensions[1];


    if (init_gqr_complete(&params, m, n)) {
        LINEARIZE_DATA_t a_in, tau_in, q_out;

        init_linearize_data(&a_in, n, m, steps[1], steps[0]);
        init_linearize_data(&tau_in, 1, fortran_int_min(m, n), 1, steps[2]);
        init_linearize_data(&q_out, m, m, steps[4], steps[3]);

        BEGIN_OUTER_LOOP_3
            int not_ok;
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.Q, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.TAU, (typ*)args[1], &tau_in);
            not_ok = call_gqr(&params);
            if (!not_ok) {
                delinearize_matrix((typ*)args[2], (typ*)params.Q, &q_out);
            } else {
                error_occurred = 1;
                nan_matrix((typ*)args[2], &q_out);
            }
        END_OUTER_LOOP

        release_gqr(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}

/* -------------------------------------------------------------------------- */
                 /* least squares */

template<typename typ>
struct GELSD_PARAMS_t
{
    fortran_int M;
    fortran_int N;
    fortran_int NRHS;
    typ *A;
    fortran_int LDA;
    typ *B;
    fortran_int LDB;
    basetype_t<typ> *S;
    basetype_t<typ> *RCOND;
    fortran_int RANK;
    typ *WORK;
    fortran_int LWORK;
    basetype_t<typ> *RWORK;
    fortran_int *IWORK;
};

template<typename typ>
static inline void
dump_gelsd_params(const char *name,
                  GELSD_PARAMS_t<typ> *params)
{
    TRACE_TXT("\n%s:\n"\

              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\
              "%14s: %18p\n"\

              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\
              "%14s: %18d\n"\

              "%14s: %18p\n",

              name,

              "A", params->A,
              "B", params->B,
              "S", params->S,
              "WORK", params->WORK,
              "RWORK", params->RWORK,
              "IWORK", params->IWORK,

              "M", (int)params->M,
              "N", (int)params->N,
              "NRHS", (int)params->NRHS,
              "LDA", (int)params->LDA,
              "LDB", (int)params->LDB,
              "LWORK", (int)params->LWORK,
              "RANK", (int)params->RANK,

              "RCOND", params->RCOND);
}

static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_real> *params)
{
    fortran_int rv;
    LAPACK(sgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->IWORK,
                          &rv);
    return rv;
}


static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_doublereal> *params)
{
    fortran_int rv;
    LAPACK(dgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->IWORK,
                          &rv);
    return rv;
}


template<typename ftyp>
static inline int
init_gelsd(GELSD_PARAMS_t<ftyp> *params,
                   fortran_int m,
                   fortran_int n,
                   fortran_int nrhs,
scalar_trait)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *s, *work, *iwork;
    fortran_int min_m_n = fortran_int_min(m, n);
    fortran_int max_m_n = fortran_int_max(m, n);
    size_t safe_min_m_n = min_m_n;
    size_t safe_max_m_n = max_m_n;
    size_t safe_m = m;
    size_t safe_n = n;
    size_t safe_nrhs = nrhs;

    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t b_size = safe_max_m_n * safe_nrhs * sizeof(ftyp);
    size_t s_size = safe_min_m_n * sizeof(ftyp);

    fortran_int work_count;
    size_t work_size;
    size_t iwork_size;
    fortran_int lda = fortran_int_max(1, m);
    fortran_int ldb = fortran_int_max(1, fortran_int_max(m,n));

    size_t msize = a_size + b_size + s_size;
    mem_buff = (npy_uint8 *)malloc(msize != 0 ? msize : 1);

    if (!mem_buff) {
        goto no_memory;
    }
    a = mem_buff;
    b = a + a_size;
    s = b + b_size;

    params->M = m;
    params->N = n;
    params->NRHS = nrhs;
    params->A = (ftyp*)a;
    params->B = (ftyp*)b;
    params->S = (ftyp*)s;
    params->LDA = lda;
    params->LDB = ldb;

    {
        /* compute optimal work size */
        ftyp work_size_query;
        fortran_int iwork_size_query;

        params->WORK = &work_size_query;
        params->IWORK = &iwork_size_query;
        params->RWORK = NULL;
        params->LWORK = -1;

        if (call_gelsd(params) != 0) {
            goto error;
        }
        work_count = (fortran_int)work_size_query;

        work_size  = (size_t) work_size_query * sizeof(ftyp);
        iwork_size = (size_t)iwork_size_query * sizeof(fortran_int);
    }

    mem_buff2 = (npy_uint8 *)malloc(work_size + iwork_size);
    if (!mem_buff2) {
        goto no_memory;
    }
    work = mem_buff2;
    iwork = work + work_size;

    params->WORK = (ftyp*)work;
    params->RWORK = NULL;
    params->IWORK = (fortran_int*)iwork;
    params->LWORK = work_count;

    return 1;

 no_memory:
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API;
    PyErr_NoMemory();
    NPY_DISABLE_C_API;

 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));
    return 0;
}

static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_complex> *params)
{
    fortran_int rv;
    LAPACK(cgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->RWORK, (fortran_int*)params->IWORK,
                          &rv);
    return rv;
}

static inline fortran_int
call_gelsd(GELSD_PARAMS_t<fortran_doublecomplex> *params)
{
    fortran_int rv;
    LAPACK(zgelsd)(&params->M, &params->N, &params->NRHS,
                          params->A, &params->LDA,
                          params->B, &params->LDB,
                          params->S,
                          params->RCOND, &params->RANK,
                          params->WORK, &params->LWORK,
                          params->RWORK, (fortran_int*)params->IWORK,
                          &rv);
    return rv;
}


template<typename ftyp>
static inline int
init_gelsd(GELSD_PARAMS_t<ftyp> *params,
                   fortran_int m,
                   fortran_int n,
                   fortran_int nrhs,
complex_trait)
{
using frealtyp = basetype_t<ftyp>;
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *s, *work, *iwork, *rwork;
    fortran_int min_m_n = fortran_int_min(m, n);
    fortran_int max_m_n = fortran_int_max(m, n);
    size_t safe_min_m_n = min_m_n;
    size_t safe_max_m_n = max_m_n;
    size_t safe_m = m;
    size_t safe_n = n;
    size_t safe_nrhs = nrhs;

    size_t a_size = safe_m * safe_n * sizeof(ftyp);
    size_t b_size = safe_max_m_n * safe_nrhs * sizeof(ftyp);
    size_t s_size = safe_min_m_n * sizeof(frealtyp);

    fortran_int work_count;
    size_t work_size, rwork_size, iwork_size;
    fortran_int lda = fortran_int_max(1, m);
    fortran_int ldb = fortran_int_max(1, fortran_int_max(m,n));

    size_t msize = a_size + b_size + s_size;
    mem_buff = (npy_uint8 *)malloc(msize != 0 ? msize : 1);

    if (!mem_buff) {
        goto no_memory;
    }

    a = mem_buff;
    b = a + a_size;
    s = b + b_size;

    params->M = m;
    params->N = n;
    params->NRHS = nrhs;
    params->A = (ftyp*)a;
    params->B = (ftyp*)b;
    params->S = (frealtyp*)s;
    params->LDA = lda;
    params->LDB = ldb;

    {
        /* compute optimal work size */
        ftyp work_size_query;
        frealtyp rwork_size_query;
        fortran_int iwork_size_query;

        params->WORK = &work_size_query;
        params->IWORK = &iwork_size_query;
        params->RWORK = &rwork_size_query;
        params->LWORK = -1;

        if (call_gelsd(params) != 0) {
            goto error;
        }

        work_count = (fortran_int)work_size_query.r;

        work_size  = (size_t )work_size_query.r * sizeof(ftyp);
        rwork_size = (size_t)rwork_size_query * sizeof(frealtyp);
        iwork_size = (size_t)iwork_size_query * sizeof(fortran_int);
    }

    mem_buff2 = (npy_uint8 *)malloc(work_size + rwork_size + iwork_size);
    if (!mem_buff2) {
        goto no_memory;
    }

    work = mem_buff2;
    rwork = work + work_size;
    iwork = rwork + rwork_size;

    params->WORK = (ftyp*)work;
    params->RWORK = (frealtyp*)rwork;
    params->IWORK = (fortran_int*)iwork;
    params->LWORK = work_count;

    return 1;

 no_memory:
    NPY_ALLOW_C_API_DEF
    NPY_ALLOW_C_API;
    PyErr_NoMemory();
    NPY_DISABLE_C_API;

 error:
    TRACE_TXT("%s failed init\n", __FUNCTION__);
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));

    return 0;
}

template<typename ftyp>
static inline void
release_gelsd(GELSD_PARAMS_t<ftyp>* params)
{
    /* A and WORK contain allocated blocks */
    free(params->A);
    free(params->WORK);
    memset(params, 0, sizeof(*params));
}

/** Compute the squared l2 norm of a contiguous vector */
template<typename typ>
static basetype_t<typ>
abs2(typ *p, npy_intp n, scalar_trait) {
    npy_intp i;
    basetype_t<typ> res = 0;
    for (i = 0; i < n; i++) {
        typ el = p[i];
        res += el*el;
    }
    return res;
}
template<typename typ>
static basetype_t<typ>
abs2(typ *p, npy_intp n, complex_trait) {
    npy_intp i;
    basetype_t<typ> res = 0;
    for (i = 0; i < n; i++) {
        typ el = p[i];
        res += el.real*el.real + el.imag*el.imag;
    }
    return res;
}


template<typename typ>
static void
lstsq(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func))
{
using ftyp = fortran_type_t<typ>;
using basetyp = basetype_t<typ>;
    GELSD_PARAMS_t<ftyp> params;
    int error_occurred = get_fp_invalid_and_clear();
    fortran_int n, m, nrhs;
    fortran_int excess;

    INIT_OUTER_LOOP_7

    m = (fortran_int)dimensions[0];
    n = (fortran_int)dimensions[1];
    nrhs = (fortran_int)dimensions[2];
    excess = m - n;

    if (init_gelsd(&params, m, n, nrhs, dispatch_scalar<ftyp>{})) {
        LINEARIZE_DATA_t a_in, b_in, x_out, s_out, r_out;

        init_linearize_data(&a_in, n, m, steps[1], steps[0]);
        init_linearize_data_ex(&b_in, nrhs, m, steps[3], steps[2], fortran_int_max(n, m));
        init_linearize_data_ex(&x_out, nrhs, n, steps[5], steps[4], fortran_int_max(n, m));
        init_linearize_data(&r_out, 1, nrhs, 1, steps[6]);
        init_linearize_data(&s_out, 1, fortran_int_min(n, m), 1, steps[7]);

        BEGIN_OUTER_LOOP_7
            int not_ok;
            linearize_matrix((typ*)params.A, (typ*)args[0], &a_in);
            linearize_matrix((typ*)params.B, (typ*)args[1], &b_in);
            params.RCOND = (basetyp*)args[2];
            not_ok = call_gelsd(&params);
            if (!not_ok) {
                delinearize_matrix((typ*)args[3], (typ*)params.B, &x_out);
                *(npy_int*) args[5] = params.RANK;
                delinearize_matrix((basetyp*)args[6], (basetyp*)params.S, &s_out);

                /* Note that linalg.lstsq discards this when excess == 0 */
                if (excess >= 0 && params.RANK == n) {
                    /* Compute the residuals as the square sum of each column */
                    int i;
                    char *resid = args[4];
                    ftyp *components = (ftyp *)params.B + n;
                    for (i = 0; i < nrhs; i++) {
                        ftyp *vector = components + i*m;
                        /* Numpy and fortran floating types are the same size,
                         * so this cast is safe */
                        basetyp abs = abs2((typ *)vector, excess,
dispatch_scalar<typ>{});
                        memcpy(
                            resid + i*r_out.column_strides,
                            &abs, sizeof(abs));
                    }
                }
                else {
                    /* Note that this is always discarded by linalg.lstsq */
                    nan_matrix((basetyp*)args[4], &r_out);
                }
            } else {
                error_occurred = 1;
                nan_matrix((typ*)args[3], &x_out);
                nan_matrix((basetyp*)args[4], &r_out);
                *(npy_int*) args[5] = -1;
                nan_matrix((basetyp*)args[6], &s_out);
            }
        END_OUTER_LOOP

        release_gelsd(&params);
    }

    set_fp_invalid_or_clear(error_occurred);
}

#pragma GCC diagnostic pop

/* -------------------------------------------------------------------------- */
              /* gufunc registration  */

static void *array_of_nulls[] = {
    (void *)NULL,
    (void *)NULL,
    (void *)NULL,
    (void *)NULL,

    (void *)NULL,
    (void *)NULL,
    (void *)NULL,
    (void *)NULL,

    (void *)NULL,
    (void *)NULL,
    (void *)NULL,
    (void *)NULL,

    (void *)NULL,
    (void *)NULL,
    (void *)NULL,
    (void *)NULL
};

#define FUNC_ARRAY_NAME(NAME) NAME ## _funcs

#define GUFUNC_FUNC_ARRAY_REAL(NAME)                    \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        FLOAT_ ## NAME,                                 \
        DOUBLE_ ## NAME                                 \
    }

#define GUFUNC_FUNC_ARRAY_REAL_COMPLEX(NAME)            \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        FLOAT_ ## NAME,                                 \
        DOUBLE_ ## NAME,                                \
        CFLOAT_ ## NAME,                                \
        CDOUBLE_ ## NAME                                \
    }
#define GUFUNC_FUNC_ARRAY_REAL_COMPLEX_(NAME)            \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<npy_float, npy_float>,                                 \
        NAME<npy_double, npy_double>,                                \
        NAME<npy_cfloat, npy_float>,                                \
        NAME<npy_cdouble, npy_double>                                \
    }
#define GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(NAME)            \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<npy_float>,                                 \
        NAME<npy_double>,                                \
        NAME<npy_cfloat>,                                \
        NAME<npy_cdouble>                                \
    }

/* There are problems with eig in complex single precision.
 * That kernel is disabled
 */
#define GUFUNC_FUNC_ARRAY_EIG(NAME)                     \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<fortran_complex,fortran_real>,                                 \
        NAME<fortran_doublecomplex,fortran_doublereal>,                                \
        NAME<fortran_doublecomplex,fortran_doublecomplex>                                \
    }

/* The single precision functions are not used at all,
 * due to input data being promoted to double precision
 * in Python, so they are not implemented here.
 */
#define GUFUNC_FUNC_ARRAY_QR(NAME)                      \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        DOUBLE_ ## NAME,                                \
        CDOUBLE_ ## NAME                                \
    }
#define GUFUNC_FUNC_ARRAY_QR__(NAME)                      \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        NAME<npy_double>,                                \
        NAME<npy_cdouble>                                \
    }


GUFUNC_FUNC_ARRAY_REAL_COMPLEX_(slogdet);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX_(det);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eighlo);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eighup);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eigvalshlo);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(eigvalshup);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(solve);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(solve1);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(inv);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(cholesky_lo);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(svd_N);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(svd_S);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(svd_A);
GUFUNC_FUNC_ARRAY_QR__(qr_r_raw);
GUFUNC_FUNC_ARRAY_QR__(qr_reduced);
GUFUNC_FUNC_ARRAY_QR__(qr_complete);
GUFUNC_FUNC_ARRAY_REAL_COMPLEX__(lstsq);
GUFUNC_FUNC_ARRAY_EIG(eig);
GUFUNC_FUNC_ARRAY_EIG(eigvals);

static char equal_2_types[] = {
    NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_CFLOAT,
    NPY_CDOUBLE, NPY_CDOUBLE
};

static char equal_3_types[] = {
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_CFLOAT, NPY_CFLOAT,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE
};

/* second result is logdet, that will always be a REAL */
static char slogdet_types[] = {
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_CFLOAT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_DOUBLE
};

static char eigh_types[] = {
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT, NPY_CFLOAT,
    NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE
};

static char eighvals_types[] = {
    NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_DOUBLE
};

static char eig_types[] = {
    NPY_FLOAT, NPY_CFLOAT, NPY_CFLOAT,
    NPY_DOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE
};

static char eigvals_types[] = {
    NPY_FLOAT, NPY_CFLOAT,
    NPY_DOUBLE, NPY_CDOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE
};

static char svd_1_1_types[] = {
    NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_DOUBLE
};

static char svd_1_3_types[] = {
    NPY_FLOAT,   NPY_FLOAT,   NPY_FLOAT,  NPY_FLOAT,
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE, NPY_DOUBLE,
    NPY_CFLOAT,  NPY_CFLOAT,  NPY_FLOAT,  NPY_CFLOAT,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE
};

/* A, tau */
static char qr_r_raw_types[] = {
    NPY_DOUBLE,  NPY_DOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE,
};

/* A, tau, q */
static char qr_reduced_types[] = {
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
};

/* A, tau, q */
static char qr_complete_types[] = {
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_CDOUBLE,
};

/*  A,           b,           rcond,      x,           resid,      rank,    s,        */
static char lstsq_types[] = {
    NPY_FLOAT,   NPY_FLOAT,   NPY_FLOAT,  NPY_FLOAT,   NPY_FLOAT,  NPY_INT, NPY_FLOAT,
    NPY_DOUBLE,  NPY_DOUBLE,  NPY_DOUBLE, NPY_DOUBLE,  NPY_DOUBLE, NPY_INT, NPY_DOUBLE,
    NPY_CFLOAT,  NPY_CFLOAT,  NPY_FLOAT,  NPY_CFLOAT,  NPY_FLOAT,  NPY_INT, NPY_FLOAT,
    NPY_CDOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_CDOUBLE, NPY_DOUBLE, NPY_INT, NPY_DOUBLE,
};

typedef struct gufunc_descriptor_struct {
    const char *name;
    const char *signature;
    const char *doc;
    int ntypes;
    int nin;
    int nout;
    PyUFuncGenericFunction *funcs;
    char *types;
} GUFUNC_DESCRIPTOR_t;

GUFUNC_DESCRIPTOR_t gufunc_descriptors [] = {
    {
        "slogdet",
        "(m,m)->(),()",
        "slogdet on the last two dimensions and broadcast on the rest. \n"\
        "Results in two arrays, one with sign and the other with log of the"\
        " determinants. \n"\
        "    \"(m,m)->(),()\" \n",
        4, 1, 2,
        FUNC_ARRAY_NAME(slogdet),
        slogdet_types
    },
    {
        "det",
        "(m,m)->()",
        "det of the last two dimensions and broadcast on the rest. \n"\
        "    \"(m,m)->()\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(det),
        equal_2_types
    },
    {
        "eigh_lo",
        "(m,m)->(m),(m,m)",
        "eigh on the last two dimension and broadcast to the rest, using"\
        " lower triangle \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        "eigenvectors. \n"\
        "    \"(m,m)->(m),(m,m)\" \n",
        4, 1, 2,
        FUNC_ARRAY_NAME(eighlo),
        eigh_types
    },
    {
        "eigh_up",
        "(m,m)->(m),(m,m)",
        "eigh on the last two dimension and broadcast to the rest, using"\
        " upper triangle. \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        " eigenvectors. \n"\
        "    \"(m,m)->(m),(m,m)\" \n",
        4, 1, 2,
        FUNC_ARRAY_NAME(eighup),
        eigh_types
    },
    {
        "eigvalsh_lo",
        "(m,m)->(m)",
        "eigh on the last two dimension and broadcast to the rest, using"\
        " lower triangle. \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        "eigenvectors. \n"\
        "    \"(m,m)->(m)\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(eigvalshlo),
        eighvals_types
    },
    {
        "eigvalsh_up",
        "(m,m)->(m)",
        "eigvalsh on the last two dimension and broadcast to the rest,"\
        " using upper triangle. \n"\
        "Results in a vector of eigenvalues and a matrix with the"\
        "eigenvectors.\n"\
        "    \"(m,m)->(m)\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(eigvalshup),
        eighvals_types
    },
    {
        "solve",
        "(m,m),(m,n)->(m,n)",
        "solve the system a x = b, on the last two dimensions, broadcast"\
        " to the rest. \n"\
        "Results in a matrices with the solutions. \n"\
        "    \"(m,m),(m,n)->(m,n)\" \n",
        4, 2, 1,
        FUNC_ARRAY_NAME(solve),
        equal_3_types
    },
    {
        "solve1",
        "(m,m),(m)->(m)",
        "solve the system a x = b, for b being a vector, broadcast in"\
        " the outer dimensions. \n"\
        "Results in vectors with the solutions. \n"\
        "    \"(m,m),(m)->(m)\" \n",
        4, 2, 1,
        FUNC_ARRAY_NAME(solve1),
        equal_3_types
    },
    {
        "inv",
        "(m, m)->(m, m)",
        "compute the inverse of the last two dimensions and broadcast"\
        " to the rest. \n"\
        "Results in the inverse matrices. \n"\
        "    \"(m,m)->(m,m)\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(inv),
        equal_2_types
    },
    {
        "cholesky_lo",
        "(m,m)->(m,m)",
        "cholesky decomposition of hermitian positive-definite matrices. \n"\
        "Broadcast to all outer dimensions. \n"\
        "    \"(m,m)->(m,m)\" \n",
        4, 1, 1,
        FUNC_ARRAY_NAME(cholesky_lo),
        equal_2_types
    },
    {
        "svd_m",
        "(m,n)->(m)",
        "svd when n>=m. ",
        4, 1, 1,
        FUNC_ARRAY_NAME(svd_N),
        svd_1_1_types
    },
    {
        "svd_n",
        "(m,n)->(n)",
        "svd when n<=m",
        4, 1, 1,
        FUNC_ARRAY_NAME(svd_N),
        svd_1_1_types
    },
    {
        "svd_m_s",
        "(m,n)->(m,m),(m),(m,n)",
        "svd when m<=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_S),
        svd_1_3_types
    },
    {
        "svd_n_s",
        "(m,n)->(m,n),(n),(n,n)",
        "svd when m>=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_S),
        svd_1_3_types
    },
    {
        "svd_m_f",
        "(m,n)->(m,m),(m),(n,n)",
        "svd when m<=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_A),
        svd_1_3_types
    },
    {
        "svd_n_f",
        "(m,n)->(m,m),(n),(n,n)",
        "svd when m>=n",
        4, 1, 3,
        FUNC_ARRAY_NAME(svd_A),
        svd_1_3_types
    },
    {
        "eig",
        "(m,m)->(m),(m,m)",
        "eig on the last two dimension and broadcast to the rest. \n"\
        "Results in a vector with the  eigenvalues and a matrix with the"\
        " eigenvectors. \n"\
        "    \"(m,m)->(m),(m,m)\" \n",
        3, 1, 2,
        FUNC_ARRAY_NAME(eig),
        eig_types
    },
    {
        "eigvals",
        "(m,m)->(m)",
        "eigvals on the last two dimension and broadcast to the rest. \n"\
        "Results in a vector of eigenvalues. \n",
        3, 1, 1,
        FUNC_ARRAY_NAME(eigvals),
        eigvals_types
    },
    {
        "qr_r_raw_m",
        "(m,n)->(m)",
        "Compute TAU vector for the last two dimensions \n"\
        "and broadcast to the rest. For m <= n. \n",
        2, 1, 1,
        FUNC_ARRAY_NAME(qr_r_raw),
        qr_r_raw_types
    },
    {
        "qr_r_raw_n",
        "(m,n)->(n)",
        "Compute TAU vector for the last two dimensions \n"\
        "and broadcast to the rest. For m > n. \n",
        2, 1, 1,
        FUNC_ARRAY_NAME(qr_r_raw),
        qr_r_raw_types
    },
    {
        "qr_reduced",
        "(m,n),(k)->(m,k)",
        "Compute Q matrix for the last two dimensions \n"\
        "and broadcast to the rest. \n",
        2, 2, 1,
        FUNC_ARRAY_NAME(qr_reduced),
        qr_reduced_types
    },
    {
        "qr_complete",
        "(m,n),(n)->(m,m)",
        "Compute Q matrix for the last two dimensions \n"\
        "and broadcast to the rest. For m > n. \n",
        2, 2, 1,
        FUNC_ARRAY_NAME(qr_complete),
        qr_complete_types
    },
    {
        "lstsq_m",
        "(m,n),(m,nrhs),()->(n,nrhs),(nrhs),(),(m)",
        "least squares on the last two dimensions and broadcast to the rest. \n"\
        "For m <= n. \n",
        4, 3, 4,
        FUNC_ARRAY_NAME(lstsq),
        lstsq_types
    },
    {
        "lstsq_n",
        "(m,n),(m,nrhs),()->(n,nrhs),(nrhs),(),(n)",
        "least squares on the last two dimensions and broadcast to the rest. \n"\
        "For m >= n, meaning that residuals are produced. \n",
        4, 3, 4,
        FUNC_ARRAY_NAME(lstsq),
        lstsq_types
    }
};

static int
addUfuncs(PyObject *dictionary) {
    PyObject *f;
    int i;
    const int gufunc_count = sizeof(gufunc_descriptors)/
        sizeof(gufunc_descriptors[0]);
    for (i = 0; i < gufunc_count; i++) {
        GUFUNC_DESCRIPTOR_t* d = &gufunc_descriptors[i];
        f = PyUFunc_FromFuncAndDataAndSignature(d->funcs,
                                                array_of_nulls,
                                                d->types,
                                                d->ntypes,
                                                d->nin,
                                                d->nout,
                                                PyUFunc_None,
                                                d->name,
                                                d->doc,
                                                0,
                                                d->signature);
        if (f == NULL) {
            return -1;
        }
#if 0
        dump_ufunc_object((PyUFuncObject*) f);
#endif
        int ret = PyDict_SetItemString(dictionary, d->name, f);
        Py_DECREF(f);
        if (ret < 0) {
            return -1;
        }
    }
    return 0;
}



/* -------------------------------------------------------------------------- */
                  /* Module initialization stuff  */

static PyMethodDef UMath_LinAlgMethods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        UMATH_LINALG_MODULE_NAME,
        NULL,
        -1,
        UMath_LinAlgMethods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit__umath_linalg(void)
{
    PyObject *m;
    PyObject *d;
    PyObject *version;

    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    import_array();
    import_ufunc();

    d = PyModule_GetDict(m);
    if (d == NULL) {
        return NULL;
    }

    version = PyUnicode_FromString(umath_linalg_version_string);
    if (version == NULL) {
        return NULL;
    }
    int ret = PyDict_SetItemString(d, "__version__", version);
    Py_DECREF(version);
    if (ret < 0) {
        return NULL;
    }

    /* Load the ufunc operators into the module's namespace */
    if (addUfuncs(d) < 0) {
        return NULL;
    }

#ifdef HAVE_BLAS_ILP64
    PyDict_SetItemString(d, "_ilp64", Py_True);
#else
    PyDict_SetItemString(d, "_ilp64", Py_False);
#endif

    return m;
}
