/*This module contributed by Doug Heisterkamp
Modified by Jim Hugunin
More modifications by Jeff Whitaker
*/

#include "Python.h"
#include "numpy/noprefix.h"

#ifdef NO_APPEND_FORTRAN
# define FNAME(x) x
#else
# define FNAME(x) x##_
#endif

typedef struct { float r, i; } f2c_complex;
typedef struct { double r, i; } f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

extern int FNAME(dgeev)(char *jobvl, char *jobvr, int *n,
                         double a[], int *lda, double wr[], double wi[],
                         double vl[], int *ldvl, double vr[], int *ldvr,
                         double work[], int lwork[], int *info);
extern int FNAME(zgeev)(char *jobvl, char *jobvr, int *n,
                         f2c_doublecomplex a[], int *lda,
                         f2c_doublecomplex w[],
                         f2c_doublecomplex vl[], int *ldvl,
                         f2c_doublecomplex vr[], int *ldvr,
                         f2c_doublecomplex work[], int *lwork,
                         double rwork[], int *info);

extern int FNAME(dsyevd)(char *jobz, char *uplo, int *n,
                          double a[], int *lda, double w[], double work[],
                          int *lwork, int iwork[], int *liwork, int *info);
extern int FNAME(zheevd)(char *jobz, char *uplo, int *n,
                          f2c_doublecomplex a[], int *lda,
                          double w[], f2c_doublecomplex work[],
                          int *lwork, double rwork[], int *lrwork, int iwork[],
                          int *liwork, int *info);

extern int FNAME(dgelsd)(int *m, int *n, int *nrhs,
                          double a[], int *lda, double b[], int *ldb,
                          double s[], double *rcond, int *rank,
                          double work[], int *lwork, int iwork[], int *info);
extern int FNAME(zgelsd)(int *m, int *n, int *nrhs,
                          f2c_doublecomplex a[], int *lda,
                          f2c_doublecomplex b[], int *ldb,
                          double s[], double *rcond, int *rank,
                          f2c_doublecomplex work[], int *lwork,
                          double rwork[], int iwork[], int *info);

extern int FNAME(dgesv)(int *n, int *nrhs,
                         double a[], int *lda, int ipiv[],
                         double b[], int *ldb, int *info);
extern int FNAME(zgesv)(int *n, int *nrhs,
                         f2c_doublecomplex a[], int *lda, int ipiv[],
                         f2c_doublecomplex b[], int *ldb, int *info);

extern int FNAME(dgetrf)(int *m, int *n,
                          double a[], int *lda, int ipiv[], int *info);
extern int FNAME(zgetrf)(int *m, int *n,
                          f2c_doublecomplex a[], int *lda, int ipiv[],
                          int *info);

extern int FNAME(dpotrf)(char *uplo, int *n, double a[], int *lda, int *info);
extern int FNAME(zpotrf)(char *uplo, int *n,
                          f2c_doublecomplex a[], int *lda, int *info);

extern int FNAME(dgesdd)(char *jobz, int *m, int *n,
                          double a[], int *lda, double s[], double u[],
                          int *ldu, double vt[], int *ldvt, double work[],
                          int *lwork, int iwork[], int *info);
extern int FNAME(zgesdd)(char *jobz, int *m, int *n,
                          f2c_doublecomplex a[], int *lda,
                          double s[], f2c_doublecomplex u[], int *ldu,
                          f2c_doublecomplex vt[], int *ldvt,
                          f2c_doublecomplex work[], int *lwork,
                          double rwork[], int iwork[], int *info);

extern int FNAME(dgeqrf)(int *m, int *n, double a[], int *lda,
                          double tau[], double work[],
                          int *lwork, int *info);

extern int FNAME(zgeqrf)(int *m, int *n, f2c_doublecomplex a[], int *lda,
                          f2c_doublecomplex tau[], f2c_doublecomplex work[],
                          int *lwork, int *info);

extern int FNAME(dorgqr)(int *m, int *n, int *k, double a[], int *lda,
                          double tau[], double work[],
                          int *lwork, int *info);

extern int FNAME(zungqr)(int *m, int *n, int *k, f2c_doublecomplex a[],
                          int *lda, f2c_doublecomplex tau[],
                          f2c_doublecomplex work[], int *lwork, int *info);

static PyObject *LapackError;

#define TRY(E) if (!(E)) return NULL

static int
check_object(PyObject *ob, int t, char *obname,
                        char *tname, char *funname)
{
    if (!PyArray_Check(ob)) {
        PyErr_Format(LapackError,
                     "Expected an array for parameter %s in lapack_lite.%s",
                     obname, funname);
        return 0;
    } else if (!(((PyArrayObject *)ob)->flags & CONTIGUOUS)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not contiguous in lapack_lite.%s",
                     obname, funname);
        return 0;
    } else if (!(((PyArrayObject *)ob)->descr->type_num == t)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not of type %s in lapack_lite.%s",
                     obname, tname, funname);
        return 0;
    } else {
        return 1;
    }
}

#define CHDATA(p) ((char *) (((PyArrayObject *)p)->data))
#define SHDATA(p) ((short int *) (((PyArrayObject *)p)->data))
#define DDATA(p) ((double *) (((PyArrayObject *)p)->data))
#define FDATA(p) ((float *) (((PyArrayObject *)p)->data))
#define CDATA(p) ((f2c_complex *) (((PyArrayObject *)p)->data))
#define ZDATA(p) ((f2c_doublecomplex *) (((PyArrayObject *)p)->data))
#define IDATA(p) ((int *) (((PyArrayObject *)p)->data))

static PyObject *
lapack_lite_dgeev(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    char jobvl;
    char jobvr;
    int n;
    PyObject *a;
    int lda;
    PyObject *wr;
    PyObject *wi;
    PyObject *vl;
    int ldvl;
    PyObject *vr;
    int ldvr;
    PyObject *work;
    int lwork;
    int info;
    TRY(PyArg_ParseTuple(args,"cciOiOOOiOiOii",
                         &jobvl,&jobvr,&n,&a,&lda,&wr,&wi,&vl,&ldvl,
                         &vr,&ldvr,&work,&lwork,&info));

    TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dgeev"));
    TRY(check_object(wr,PyArray_DOUBLE,"wr","PyArray_DOUBLE","dgeev"));
    TRY(check_object(wi,PyArray_DOUBLE,"wi","PyArray_DOUBLE","dgeev"));
    TRY(check_object(vl,PyArray_DOUBLE,"vl","PyArray_DOUBLE","dgeev"));
    TRY(check_object(vr,PyArray_DOUBLE,"vr","PyArray_DOUBLE","dgeev"));
    TRY(check_object(work,PyArray_DOUBLE,"work","PyArray_DOUBLE","dgeev"));

    lapack_lite_status__ = \
            FNAME(dgeev)(&jobvl,&jobvr,&n,DDATA(a),&lda,DDATA(wr),DDATA(wi),
                         DDATA(vl),&ldvl,DDATA(vr),&ldvr,DDATA(work),&lwork,
                         &info);

    return Py_BuildValue("{s:i,s:c,s:c,s:i,s:i,s:i,s:i,s:i,s:i}","dgeev_",
                         lapack_lite_status__,"jobvl",jobvl,"jobvr",jobvr,
                         "n",n,"lda",lda,"ldvl",ldvl,"ldvr",ldvr,
                         "lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_dsyevd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    /*  Arguments */
    /*  ========= */

    char jobz;
    /*  JOBZ    (input) CHARACTER*1 */
    /*          = 'N':  Compute eigenvalues only; */
    /*          = 'V':  Compute eigenvalues and eigenvectors. */

    char uplo;
    /*  UPLO    (input) CHARACTER*1 */
    /*          = 'U':  Upper triangle of A is stored; */
    /*          = 'L':  Lower triangle of A is stored. */

    int n;
    /*  N       (input) INTEGER */
    /*          The order of the matrix A.  N >= 0. */

    PyObject *a;
    /*  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N) */
    /*          On entry, the symmetric matrix A.  If UPLO = 'U', the */
    /*          leading N-by-N upper triangular part of A contains the */
    /*          upper triangular part of the matrix A.  If UPLO = 'L', */
    /*          the leading N-by-N lower triangular part of A contains */
    /*          the lower triangular part of the matrix A. */
    /*          On exit, if JOBZ = 'V', then if INFO = 0, A contains the */
    /*          orthonormal eigenvectors of the matrix A. */
    /*          If JOBZ = 'N', then on exit the lower triangle (if UPLO='L') */
    /*          or the upper triangle (if UPLO='U') of A, including the */
    /*          diagonal, is destroyed. */

    int lda;
    /*  LDA     (input) INTEGER */
    /*          The leading dimension of the array A.  LDA >= max(1,N). */

    PyObject *w;
    /*  W       (output) DOUBLE PRECISION array, dimension (N) */
    /*          If INFO = 0, the eigenvalues in ascending order. */

    PyObject *work;
    /*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK) */
    /*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

    int lwork;
    /*  LWORK   (input) INTEGER */
    /*          The length of the array WORK.  LWORK >= max(1,3*N-1). */
    /*          For optimal efficiency, LWORK >= (NB+2)*N, */
    /*          where NB is the blocksize for DSYTRD returned by ILAENV. */

    PyObject *iwork;
    int liwork;

    int info;
    /*  INFO    (output) INTEGER */
    /*          = 0:  successful exit */
    /*          < 0:  if INFO = -i, the i-th argument had an illegal value */
    /*          > 0:  if INFO = i, the algorithm failed to converge; i */
    /*                off-diagonal elements of an intermediate tridiagonal */
    /*                form did not converge to zero. */

    int  lapack_lite_status__;

    TRY(PyArg_ParseTuple(args,"cciOiOOiOii",
                         &jobz,&uplo,&n,&a,&lda,&w,&work,&lwork,
                         &iwork,&liwork,&info));

    TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dsyevd"));
    TRY(check_object(w,PyArray_DOUBLE,"w","PyArray_DOUBLE","dsyevd"));
    TRY(check_object(work,PyArray_DOUBLE,"work","PyArray_DOUBLE","dsyevd"));
    TRY(check_object(iwork,PyArray_INT,"iwork","PyArray_INT","dsyevd"));

    lapack_lite_status__ = \
            FNAME(dsyevd)(&jobz,&uplo,&n,DDATA(a),&lda,DDATA(w),DDATA(work),
                          &lwork,IDATA(iwork),&liwork,&info);

    return Py_BuildValue("{s:i,s:c,s:c,s:i,s:i,s:i,s:i,s:i}","dsyevd_",
                         lapack_lite_status__,"jobz",jobz,"uplo",uplo,
                         "n",n,"lda",lda,"lwork",lwork,"liwork",liwork,"info",info);
}

static PyObject *
lapack_lite_zheevd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    /*  Arguments */
    /*  ========= */

    char jobz;
    /*  JOBZ    (input) CHARACTER*1 */
    /*          = 'N':  Compute eigenvalues only; */
    /*          = 'V':  Compute eigenvalues and eigenvectors. */

    char uplo;
    /*  UPLO    (input) CHARACTER*1 */
    /*          = 'U':  Upper triangle of A is stored; */
    /*          = 'L':  Lower triangle of A is stored. */

    int n;
    /*  N       (input) INTEGER */
    /*          The order of the matrix A.  N >= 0. */

    PyObject *a;
    /*  A       (input/output) COMPLEX*16 array, dimension (LDA, N) */
    /*          On entry, the Hermitian matrix A.  If UPLO = 'U', the */
    /*          leading N-by-N upper triangular part of A contains the */
    /*          upper triangular part of the matrix A.  If UPLO = 'L', */
    /*          the leading N-by-N lower triangular part of A contains */
    /*          the lower triangular part of the matrix A. */
    /*          On exit, if JOBZ = 'V', then if INFO = 0, A contains the */
    /*          orthonormal eigenvectors of the matrix A. */
    /*          If JOBZ = 'N', then on exit the lower triangle (if UPLO='L') */
    /*          or the upper triangle (if UPLO='U') of A, including the */
    /*          diagonal, is destroyed. */

    int lda;
    /*  LDA     (input) INTEGER */
    /*          The leading dimension of the array A.  LDA >= max(1,N). */

    PyObject *w;
    /*  W       (output) DOUBLE PRECISION array, dimension (N) */
    /*          If INFO = 0, the eigenvalues in ascending order. */

    PyObject *work;
    /*  WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK) */
    /*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

    int lwork;
    /*  LWORK   (input) INTEGER */
    /*          The length of the array WORK.  LWORK >= max(1,3*N-1). */
    /*          For optimal efficiency, LWORK >= (NB+2)*N, */
    /*          where NB is the blocksize for DSYTRD returned by ILAENV. */

    PyObject *rwork;
    /*  RWORK   (workspace) DOUBLE PRECISION array, dimension (max(1, 3*N-2)) */
    int lrwork;

    PyObject *iwork;
    int liwork;

    int info;
    /*  INFO    (output) INTEGER */
    /*          = 0:  successful exit */
    /*          < 0:  if INFO = -i, the i-th argument had an illegal value */
    /*          > 0:  if INFO = i, the algorithm failed to converge; i */
    /*                off-diagonal elements of an intermediate tridiagonal */
    /*                form did not converge to zero. */

    int  lapack_lite_status__;

    TRY(PyArg_ParseTuple(args,"cciOiOOiOiOii",
                         &jobz,&uplo,&n,&a,&lda,&w,&work,&lwork,&rwork,
                         &lrwork,&iwork,&liwork,&info));

    TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zheevd"));
    TRY(check_object(w,PyArray_DOUBLE,"w","PyArray_DOUBLE","zheevd"));
    TRY(check_object(work,PyArray_CDOUBLE,"work","PyArray_CDOUBLE","zheevd"));
    TRY(check_object(w,PyArray_DOUBLE,"rwork","PyArray_DOUBLE","zheevd"));
    TRY(check_object(iwork,PyArray_INT,"iwork","PyArray_INT","zheevd"));

    lapack_lite_status__ = \
    FNAME(zheevd)(&jobz,&uplo,&n,ZDATA(a),&lda,DDATA(w),ZDATA(work),
                  &lwork,DDATA(rwork),&lrwork,IDATA(iwork),&liwork,&info);

    return Py_BuildValue("{s:i,s:c,s:c,s:i,s:i,s:i,s:i,s:i,s:i}","zheevd_",
                         lapack_lite_status__,"jobz",jobz,"uplo",uplo,"n",n,
                         "lda",lda,"lwork",lwork,"lrwork",lrwork,
                         "liwork",liwork,"info",info);
}

static PyObject *
lapack_lite_dgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int m;
    int n;
    int nrhs;
    PyObject *a;
    int lda;
    PyObject *b;
    int ldb;
    PyObject *s;
    double rcond;
    int rank;
    PyObject *work;
    PyObject *iwork;
    int lwork;
    int info;
    TRY(PyArg_ParseTuple(args,"iiiOiOiOdiOiOi",
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&iwork,&info));

    TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dgelsd"));
    TRY(check_object(b,PyArray_DOUBLE,"b","PyArray_DOUBLE","dgelsd"));
    TRY(check_object(s,PyArray_DOUBLE,"s","PyArray_DOUBLE","dgelsd"));
    TRY(check_object(work,PyArray_DOUBLE,"work","PyArray_DOUBLE","dgelsd"));
    TRY(check_object(iwork,PyArray_INT,"iwork","PyArray_INT","dgelsd"));

    lapack_lite_status__ = \
            FNAME(dgelsd)(&m,&n,&nrhs,DDATA(a),&lda,DDATA(b),&ldb,
                          DDATA(s),&rcond,&rank,DDATA(work),&lwork,
                          IDATA(iwork),&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:d,s:i,s:i,s:i}","dgelsd_",
                         lapack_lite_status__,"m",m,"n",n,"nrhs",nrhs,
                         "lda",lda,"ldb",ldb,"rcond",rcond,"rank",rank,
                         "lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_dgesv(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int n;
    int nrhs;
    PyObject *a;
    int lda;
    PyObject *ipiv;
    PyObject *b;
    int ldb;
    int info;
    TRY(PyArg_ParseTuple(args,"iiOiOOii",&n,&nrhs,&a,&lda,&ipiv,&b,&ldb,&info));

    TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dgesv"));
    TRY(check_object(ipiv,PyArray_INT,"ipiv","PyArray_INT","dgesv"));
    TRY(check_object(b,PyArray_DOUBLE,"b","PyArray_DOUBLE","dgesv"));

    lapack_lite_status__ = \
    FNAME(dgesv)(&n,&nrhs,DDATA(a),&lda,IDATA(ipiv),DDATA(b),&ldb,&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i}","dgesv_",
                         lapack_lite_status__,"n",n,"nrhs",nrhs,"lda",lda,
                         "ldb",ldb,"info",info);
}

static PyObject *
lapack_lite_dgesdd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    char jobz;
    int m;
    int n;
    PyObject *a;
    int lda;
    PyObject *s;
    PyObject *u;
    int ldu;
    PyObject *vt;
    int ldvt;
    PyObject *work;
    int lwork;
    PyObject *iwork;
    int info;
    TRY(PyArg_ParseTuple(args,"ciiOiOOiOiOiOi",
                         &jobz,&m,&n,&a,&lda,&s,&u,&ldu,&vt,&ldvt,
                         &work,&lwork,&iwork,&info));

    TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dgesdd"));
    TRY(check_object(s,PyArray_DOUBLE,"s","PyArray_DOUBLE","dgesdd"));
    TRY(check_object(u,PyArray_DOUBLE,"u","PyArray_DOUBLE","dgesdd"));
    TRY(check_object(vt,PyArray_DOUBLE,"vt","PyArray_DOUBLE","dgesdd"));
    TRY(check_object(work,PyArray_DOUBLE,"work","PyArray_DOUBLE","dgesdd"));
    TRY(check_object(iwork,PyArray_INT,"iwork","PyArray_INT","dgesdd"));

    lapack_lite_status__ = \
            FNAME(dgesdd)(&jobz,&m,&n,DDATA(a),&lda,DDATA(s),DDATA(u),&ldu,
                          DDATA(vt),&ldvt,DDATA(work),&lwork,IDATA(iwork),
                          &info);

    if (info == 0 && lwork == -1) {
            /* We need to check the result because
               sometimes the "optimal" value is actually
               too small.
               Change it to the maximum of the minimum and the optimal.
            */
            long work0 = (long) *DDATA(work);
            int mn = MIN(m,n);
            int mx = MAX(m,n);

            switch(jobz){
            case 'N':
                    work0 = MAX(work0,3*mn + MAX(mx,6*mn)+500);
                    break;
            case 'O':
                    work0 = MAX(work0,3*mn*mn +                 \
                                MAX(mx,5*mn*mn+4*mn+500));
                    break;
            case 'S':
            case 'A':
                    work0 = MAX(work0,3*mn*mn +                 \
                                MAX(mx,4*mn*(mn+1))+500);
                    break;
            }
            *DDATA(work) = (double) work0;
    }
    return Py_BuildValue("{s:i,s:c,s:i,s:i,s:i,s:i,s:i,s:i,s:i}","dgesdd_",
                         lapack_lite_status__,"jobz",jobz,"m",m,"n",n,
                         "lda",lda,"ldu",ldu,"ldvt",ldvt,"lwork",lwork,
                         "info",info);
}

static PyObject *
lapack_lite_dgetrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int m;
    int n;
    PyObject *a;
    int lda;
    PyObject *ipiv;
    int info;
    TRY(PyArg_ParseTuple(args,"iiOiOi",&m,&n,&a,&lda,&ipiv,&info));

    TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dgetrf"));
    TRY(check_object(ipiv,PyArray_INT,"ipiv","PyArray_INT","dgetrf"));

    lapack_lite_status__ = \
            FNAME(dgetrf)(&m,&n,DDATA(a),&lda,IDATA(ipiv),&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i}","dgetrf_",lapack_lite_status__,
                         "m",m,"n",n,"lda",lda,"info",info);
}

static PyObject *
lapack_lite_dpotrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int n;
    PyObject *a;
    int lda;
    char uplo;
    int info;

    TRY(PyArg_ParseTuple(args,"ciOii",&uplo,&n,&a,&lda,&info));
    TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dpotrf"));

    lapack_lite_status__ = \
            FNAME(dpotrf)(&uplo,&n,DDATA(a),&lda,&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i}","dpotrf_",lapack_lite_status__,
                         "n",n,"lda",lda,"info",info);
}

static PyObject *
lapack_lite_dgeqrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int  lapack_lite_status__;
        int m, n, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiOiOOii",&m,&n,&a,&lda,&tau,&work,&lwork,&info));

        /* check objects and convert to right storage order */
        TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dgeqrf"));
        TRY(check_object(tau,PyArray_DOUBLE,"tau","PyArray_DOUBLE","dgeqrf"));
        TRY(check_object(work,PyArray_DOUBLE,"work","PyArray_DOUBLE","dgeqrf"));

        lapack_lite_status__ = \
                FNAME(dgeqrf)(&m, &n, DDATA(a), &lda, DDATA(tau),
                              DDATA(work), &lwork, &info);

        return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i}","dgeqrf_",
                             lapack_lite_status__,"m",m,"n",n,"lda",lda,
                             "lwork",lwork,"info",info);
}


static PyObject *
lapack_lite_dorgqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int  lapack_lite_status__;
        int m, n, k, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiiOiOOii",  &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
        TRY(check_object(a,PyArray_DOUBLE,"a","PyArray_DOUBLE","dorgqr"));
        TRY(check_object(tau,PyArray_DOUBLE,"tau","PyArray_DOUBLE","dorgqr"));
        TRY(check_object(work,PyArray_DOUBLE,"work","PyArray_DOUBLE","dorgqr"));
        lapack_lite_status__ = \
        FNAME(dorgqr)(&m, &n, &k, DDATA(a), &lda, DDATA(tau), DDATA(work), &lwork, &info);

        return Py_BuildValue("{s:i,s:i}","dorgqr_",lapack_lite_status__,
                             "info",info);
}


static PyObject *
lapack_lite_zgeev(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    char jobvl;
    char jobvr;
    int n;
    PyObject *a;
    int lda;
    PyObject *w;
    PyObject *vl;
    int ldvl;
    PyObject *vr;
    int ldvr;
    PyObject *work;
    int lwork;
    PyObject *rwork;
    int info;
    TRY(PyArg_ParseTuple(args,"cciOiOOiOiOiOi",
                         &jobvl,&jobvr,&n,&a,&lda,&w,&vl,&ldvl,
                         &vr,&ldvr,&work,&lwork,&rwork,&info));

    TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zgeev"));
    TRY(check_object(w,PyArray_CDOUBLE,"w","PyArray_CDOUBLE","zgeev"));
    TRY(check_object(vl,PyArray_CDOUBLE,"vl","PyArray_CDOUBLE","zgeev"));
    TRY(check_object(vr,PyArray_CDOUBLE,"vr","PyArray_CDOUBLE","zgeev"));
    TRY(check_object(work,PyArray_CDOUBLE,"work","PyArray_CDOUBLE","zgeev"));
    TRY(check_object(rwork,PyArray_DOUBLE,"rwork","PyArray_DOUBLE","zgeev"));

    lapack_lite_status__ = \
            FNAME(zgeev)(&jobvl,&jobvr,&n,ZDATA(a),&lda,ZDATA(w),ZDATA(vl),
                         &ldvl,ZDATA(vr),&ldvr,ZDATA(work),&lwork,
                         DDATA(rwork),&info);

    return Py_BuildValue("{s:i,s:c,s:c,s:i,s:i,s:i,s:i,s:i,s:i}","zgeev_",
                         lapack_lite_status__,"jobvl",jobvl,"jobvr",jobvr,
                         "n",n,"lda",lda,"ldvl",ldvl,"ldvr",ldvr,
                         "lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_zgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int m;
    int n;
    int nrhs;
    PyObject *a;
    int lda;
    PyObject *b;
    int ldb;
    PyObject *s;
    double rcond;
    int rank;
    PyObject *work;
    int lwork;
    PyObject *rwork;
    PyObject *iwork;
    int info;
    TRY(PyArg_ParseTuple(args,"iiiOiOiOdiOiOOi",
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&rwork,&iwork,&info));

    TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zgelsd"));
    TRY(check_object(b,PyArray_CDOUBLE,"b","PyArray_CDOUBLE","zgelsd"));
    TRY(check_object(s,PyArray_DOUBLE,"s","PyArray_DOUBLE","zgelsd"));
    TRY(check_object(work,PyArray_CDOUBLE,"work","PyArray_CDOUBLE","zgelsd"));
    TRY(check_object(rwork,PyArray_DOUBLE,"rwork","PyArray_DOUBLE","zgelsd"));
    TRY(check_object(iwork,PyArray_INT,"iwork","PyArray_INT","zgelsd"));

    lapack_lite_status__ = \
    FNAME(zgelsd)(&m,&n,&nrhs,ZDATA(a),&lda,ZDATA(b),&ldb,DDATA(s),&rcond,
                  &rank,ZDATA(work),&lwork,DDATA(rwork),IDATA(iwork),&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i}","zgelsd_",
                         lapack_lite_status__,"m",m,"n",n,"nrhs",nrhs,"lda",lda,
                         "ldb",ldb,"rank",rank,"lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_zgesv(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int n;
    int nrhs;
    PyObject *a;
    int lda;
    PyObject *ipiv;
    PyObject *b;
    int ldb;
    int info;
    TRY(PyArg_ParseTuple(args,"iiOiOOii",&n,&nrhs,&a,&lda,&ipiv,&b,&ldb,&info));

    TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zgesv"));
    TRY(check_object(ipiv,PyArray_INT,"ipiv","PyArray_INT","zgesv"));
    TRY(check_object(b,PyArray_CDOUBLE,"b","PyArray_CDOUBLE","zgesv"));

    lapack_lite_status__ = \
    FNAME(zgesv)(&n,&nrhs,ZDATA(a),&lda,IDATA(ipiv),ZDATA(b),&ldb,&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i}","zgesv_",
                         lapack_lite_status__,"n",n,"nrhs",nrhs,"lda",lda,
                         "ldb",ldb,"info",info);
}

static PyObject *
lapack_lite_zgesdd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    char jobz;
    int m;
    int n;
    PyObject *a;
    int lda;
    PyObject *s;
    PyObject *u;
    int ldu;
    PyObject *vt;
    int ldvt;
    PyObject *work;
    int lwork;
    PyObject *rwork;
    PyObject *iwork;
    int info;
    TRY(PyArg_ParseTuple(args,"ciiOiOOiOiOiOOi",
                         &jobz,&m,&n,&a,&lda,&s,&u,&ldu,
                         &vt,&ldvt,&work,&lwork,&rwork,&iwork,&info));

    TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zgesdd"));
    TRY(check_object(s,PyArray_DOUBLE,"s","PyArray_DOUBLE","zgesdd"));
    TRY(check_object(u,PyArray_CDOUBLE,"u","PyArray_CDOUBLE","zgesdd"));
    TRY(check_object(vt,PyArray_CDOUBLE,"vt","PyArray_CDOUBLE","zgesdd"));
    TRY(check_object(work,PyArray_CDOUBLE,"work","PyArray_CDOUBLE","zgesdd"));
    TRY(check_object(rwork,PyArray_DOUBLE,"rwork","PyArray_DOUBLE","zgesdd"));
    TRY(check_object(iwork,PyArray_INT,"iwork","PyArray_INT","zgesdd"));

    lapack_lite_status__ = \
    FNAME(zgesdd)(&jobz,&m,&n,ZDATA(a),&lda,DDATA(s),ZDATA(u),&ldu,
                  ZDATA(vt),&ldvt,ZDATA(work),&lwork,DDATA(rwork),
                  IDATA(iwork),&info);

    return Py_BuildValue("{s:i,s:c,s:i,s:i,s:i,s:i,s:i,s:i,s:i}","zgesdd_",
                         lapack_lite_status__,"jobz",jobz,"m",m,"n",n,
                         "lda",lda,"ldu",ldu,"ldvt",ldvt,"lwork",lwork,
                         "info",info);
}

static PyObject *
lapack_lite_zgetrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int m;
    int n;
    PyObject *a;
    int lda;
    PyObject *ipiv;
    int info;
    TRY(PyArg_ParseTuple(args,"iiOiOi",&m,&n,&a,&lda,&ipiv,&info));

    TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zgetrf"));
    TRY(check_object(ipiv,PyArray_INT,"ipiv","PyArray_INT","zgetrf"));

    lapack_lite_status__ = \
    FNAME(zgetrf)(&m,&n,ZDATA(a),&lda,IDATA(ipiv),&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i}","zgetrf_",
                         lapack_lite_status__,"m",m,"n",n,"lda",lda,"info",info);
}

static PyObject *
lapack_lite_zpotrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int  lapack_lite_status__;
    int n;
    PyObject *a;
    int lda;
    char uplo;
    int info;

    TRY(PyArg_ParseTuple(args,"ciOii",&uplo,&n,&a,&lda,&info));
    TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zpotrf"));
    lapack_lite_status__ = \
    FNAME(zpotrf)(&uplo,&n,ZDATA(a),&lda,&info);

    return Py_BuildValue("{s:i,s:i,s:i,s:i}","zpotrf_",
                         lapack_lite_status__,"n",n,"lda",lda,"info",info);
}

static PyObject *
lapack_lite_zgeqrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int  lapack_lite_status__;
        int m, n, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiOiOOii",&m,&n,&a,&lda,&tau,&work,&lwork,&info));

/* check objects and convert to right storage order */
        TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zgeqrf"));
        TRY(check_object(tau,PyArray_CDOUBLE,"tau","PyArray_CDOUBLE","zgeqrf"));
        TRY(check_object(work,PyArray_CDOUBLE,"work","PyArray_CDOUBLE","zgeqrf"));

        lapack_lite_status__ = \
        FNAME(zgeqrf)(&m, &n, ZDATA(a), &lda, ZDATA(tau), ZDATA(work), &lwork, &info);

        return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i}","zgeqrf_",lapack_lite_status__,"m",m,"n",n,"lda",lda,"lwork",lwork,"info",info);
}


static PyObject *
lapack_lite_zungqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int  lapack_lite_status__;
        int m, n, k, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiiOiOOii",  &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
        TRY(check_object(a,PyArray_CDOUBLE,"a","PyArray_CDOUBLE","zungqr"));
        TRY(check_object(tau,PyArray_CDOUBLE,"tau","PyArray_CDOUBLE","zungqr"));
        TRY(check_object(work,PyArray_CDOUBLE,"work","PyArray_CDOUBLE","zungqr"));


        lapack_lite_status__ = \
        FNAME(zungqr)(&m, &n, &k, ZDATA(a), &lda, ZDATA(tau), ZDATA(work),
                      &lwork, &info);

        return Py_BuildValue("{s:i,s:i}","zungqr_",lapack_lite_status__,
                             "info",info);
}



#define STR(x) #x
#define lameth(name) {STR(name), lapack_lite_##name, METH_VARARGS, NULL}
static struct PyMethodDef lapack_lite_module_methods[] = {
    lameth(zheevd),
    lameth(dsyevd),
    lameth(dgeev),
    lameth(dgelsd),
    lameth(dgesv),
    lameth(dgesdd),
    lameth(dgetrf),
    lameth(dpotrf),
    lameth(dgeqrf),
    lameth(dorgqr),
    lameth(zgeev),
    lameth(zgelsd),
    lameth(zgesv),
    lameth(zgesdd),
    lameth(zgetrf),
    lameth(zpotrf),
    lameth(zgeqrf),
    lameth(zungqr),
    { NULL,NULL,0, NULL}
};

static char lapack_lite_module_documentation[] = "";


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "lapack_lite",
        NULL,
        -1,
        lapack_lite_module_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

/* Initialization function for the module */
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyObject *PyInit_lapack_lite(void)
#else
#define RETVAL
PyMODINIT_FUNC
initlapack_lite(void)
#endif
{
    PyObject *m,*d;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("lapack_lite", lapack_lite_module_methods,
		       lapack_lite_module_documentation,
		       (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL) {
        return RETVAL;
    }
    import_array();
    d = PyModule_GetDict(m);
    LapackError = PyErr_NewException("lapack_lite.LapackError", NULL, NULL);
    PyDict_SetItemString(d, "LapackError", LapackError);

    return RETVAL;
}
