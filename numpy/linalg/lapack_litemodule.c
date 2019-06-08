/*This module contributed by Doug Heisterkamp
Modified by Jim Hugunin
More modifications by Jeff Whitaker
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"

#ifdef NO_APPEND_FORTRAN
# define FNAME(x) x
#else
# define FNAME(x) x##_
#endif

typedef struct { float r, i; } f2c_complex;
typedef struct { double r, i; } f2c_doublecomplex;
/* typedef long int (*L_fp)(); */

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

extern int FNAME(xerbla)(char *srname, int *info);

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
    }
    else if (!PyArray_IS_C_CONTIGUOUS((PyArrayObject *)ob)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not contiguous in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    else if (!(PyArray_TYPE((PyArrayObject *)ob) == t)) {
        PyErr_Format(LapackError,
                     "Parameter %s is not of type %s in lapack_lite.%s",
                     obname, tname, funname);
        return 0;
    }
    else if (PyArray_ISBYTESWAPPED((PyArrayObject *)ob)) {
        PyErr_Format(LapackError,
                     "Parameter %s has non-native byte order in lapack_lite.%s",
                     obname, funname);
        return 0;
    }
    else {
        return 1;
    }
}

#define CHDATA(p) ((char *) PyArray_DATA((PyArrayObject *)p))
#define SHDATA(p) ((short int *) PyArray_DATA((PyArrayObject *)p))
#define DDATA(p) ((double *) PyArray_DATA((PyArrayObject *)p))
#define FDATA(p) ((float *) PyArray_DATA((PyArrayObject *)p))
#define CDATA(p) ((f2c_complex *) PyArray_DATA((PyArrayObject *)p))
#define ZDATA(p) ((f2c_doublecomplex *) PyArray_DATA((PyArrayObject *)p))
#define IDATA(p) ((int *) PyArray_DATA((PyArrayObject *)p))

static PyObject *
lapack_lite_dgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int lapack_lite_status;
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
    TRY(PyArg_ParseTuple(args,"iiiOiOiOdiOiOi:dgelsd",
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&iwork,&info));

    TRY(check_object(a,NPY_DOUBLE,"a","NPY_DOUBLE","dgelsd"));
    TRY(check_object(b,NPY_DOUBLE,"b","NPY_DOUBLE","dgelsd"));
    TRY(check_object(s,NPY_DOUBLE,"s","NPY_DOUBLE","dgelsd"));
    TRY(check_object(work,NPY_DOUBLE,"work","NPY_DOUBLE","dgelsd"));
    TRY(check_object(iwork,NPY_INT,"iwork","NPY_INT","dgelsd"));

    lapack_lite_status =
            FNAME(dgelsd)(&m,&n,&nrhs,DDATA(a),&lda,DDATA(b),&ldb,
                          DDATA(s),&rcond,&rank,DDATA(work),&lwork,
                          IDATA(iwork),&info);
    if (PyErr_Occurred()) {
        return NULL;
    }

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:d,s:i,s:i,s:i}","dgelsd_",
                         lapack_lite_status,"m",m,"n",n,"nrhs",nrhs,
                         "lda",lda,"ldb",ldb,"rcond",rcond,"rank",rank,
                         "lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_dgeqrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int lapack_lite_status;
        int m, n, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiOiOOii:dgeqrf",&m,&n,&a,&lda,&tau,&work,&lwork,&info));

        /* check objects and convert to right storage order */
        TRY(check_object(a,NPY_DOUBLE,"a","NPY_DOUBLE","dgeqrf"));
        TRY(check_object(tau,NPY_DOUBLE,"tau","NPY_DOUBLE","dgeqrf"));
        TRY(check_object(work,NPY_DOUBLE,"work","NPY_DOUBLE","dgeqrf"));

        lapack_lite_status =
                FNAME(dgeqrf)(&m, &n, DDATA(a), &lda, DDATA(tau),
                              DDATA(work), &lwork, &info);
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i}","dgeqrf_",
                             lapack_lite_status,"m",m,"n",n,"lda",lda,
                             "lwork",lwork,"info",info);
}


static PyObject *
lapack_lite_dorgqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int lapack_lite_status;
        int m, n, k, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiiOiOOii:dorgqr",  &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
        TRY(check_object(a,NPY_DOUBLE,"a","NPY_DOUBLE","dorgqr"));
        TRY(check_object(tau,NPY_DOUBLE,"tau","NPY_DOUBLE","dorgqr"));
        TRY(check_object(work,NPY_DOUBLE,"work","NPY_DOUBLE","dorgqr"));
        lapack_lite_status =
            FNAME(dorgqr)(&m, &n, &k, DDATA(a), &lda, DDATA(tau), DDATA(work),
                          &lwork, &info);
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue("{s:i,s:i}","dorgqr_",lapack_lite_status,
                             "info",info);
}


static PyObject *
lapack_lite_zgelsd(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int lapack_lite_status;
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
    TRY(PyArg_ParseTuple(args,"iiiOiOiOdiOiOOi:zgelsd",
                         &m,&n,&nrhs,&a,&lda,&b,&ldb,&s,&rcond,
                         &rank,&work,&lwork,&rwork,&iwork,&info));

    TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zgelsd"));
    TRY(check_object(b,NPY_CDOUBLE,"b","NPY_CDOUBLE","zgelsd"));
    TRY(check_object(s,NPY_DOUBLE,"s","NPY_DOUBLE","zgelsd"));
    TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zgelsd"));
    TRY(check_object(rwork,NPY_DOUBLE,"rwork","NPY_DOUBLE","zgelsd"));
    TRY(check_object(iwork,NPY_INT,"iwork","NPY_INT","zgelsd"));

    lapack_lite_status =
        FNAME(zgelsd)(&m,&n,&nrhs,ZDATA(a),&lda,ZDATA(b),&ldb,DDATA(s),&rcond,
                      &rank,ZDATA(work),&lwork,DDATA(rwork),IDATA(iwork),&info);
    if (PyErr_Occurred()) {
        return NULL;
    }

    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i}","zgelsd_",
                         lapack_lite_status,"m",m,"n",n,"nrhs",nrhs,"lda",lda,
                         "ldb",ldb,"rank",rank,"lwork",lwork,"info",info);
}

static PyObject *
lapack_lite_zgeqrf(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int lapack_lite_status;
        int m, n, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiOiOOii:zgeqrf",&m,&n,&a,&lda,&tau,&work,&lwork,&info));

/* check objects and convert to right storage order */
        TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zgeqrf"));
        TRY(check_object(tau,NPY_CDOUBLE,"tau","NPY_CDOUBLE","zgeqrf"));
        TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zgeqrf"));

        lapack_lite_status =
            FNAME(zgeqrf)(&m, &n, ZDATA(a), &lda, ZDATA(tau), ZDATA(work),
                          &lwork, &info);
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i}","zgeqrf_",lapack_lite_status,"m",m,"n",n,"lda",lda,"lwork",lwork,"info",info);
}


static PyObject *
lapack_lite_zungqr(PyObject *NPY_UNUSED(self), PyObject *args)
{
        int lapack_lite_status;
        int m, n, k, lwork;
        PyObject *a, *tau, *work;
        int lda;
        int info;

        TRY(PyArg_ParseTuple(args,"iiiOiOOii:zungqr",  &m, &n, &k, &a, &lda, &tau, &work, &lwork, &info));
        TRY(check_object(a,NPY_CDOUBLE,"a","NPY_CDOUBLE","zungqr"));
        TRY(check_object(tau,NPY_CDOUBLE,"tau","NPY_CDOUBLE","zungqr"));
        TRY(check_object(work,NPY_CDOUBLE,"work","NPY_CDOUBLE","zungqr"));


        lapack_lite_status =
            FNAME(zungqr)(&m, &n, &k, ZDATA(a), &lda, ZDATA(tau), ZDATA(work),
                          &lwork, &info);
        if (PyErr_Occurred()) {
            return NULL;
        }

        return Py_BuildValue("{s:i,s:i}","zungqr_",lapack_lite_status,
                             "info",info);
}


static PyObject *
lapack_lite_xerbla(PyObject *NPY_UNUSED(self), PyObject *args)
{
    int info = -1;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;
    FNAME(xerbla)("test", &info);
    NPY_END_THREADS;

    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_RETURN_NONE;
}


#define STR(x) #x
#define lameth(name) {STR(name), lapack_lite_##name, METH_VARARGS, NULL}
static struct PyMethodDef lapack_lite_module_methods[] = {
    lameth(dgelsd),
    lameth(dgeqrf),
    lameth(dorgqr),
    lameth(zgelsd),
    lameth(zgeqrf),
    lameth(zungqr),
    lameth(xerbla),
    { NULL,NULL,0, NULL}
};


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
#define RETVAL(x) x
PyMODINIT_FUNC PyInit_lapack_lite(void)
#else
#define RETVAL(x)
PyMODINIT_FUNC
initlapack_lite(void)
#endif
{
    PyObject *m,*d;
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule4("lapack_lite", lapack_lite_module_methods,
                       "", (PyObject*)NULL,PYTHON_API_VERSION);
#endif
    if (m == NULL) {
        return RETVAL(NULL);
    }
    import_array();
    d = PyModule_GetDict(m);
    LapackError = PyErr_NewException("lapack_lite.LapackError", NULL, NULL);
    PyDict_SetItemString(d, "LapackError", LapackError);

    return RETVAL(m);
}
