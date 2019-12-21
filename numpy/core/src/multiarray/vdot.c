#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#include <Python.h>
#include "common.h"
#include "vdot.h"
#include "npy_cblas.h"


/*
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
CFLOAT_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
            char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
#if defined(HAVE_CBLAS)
    CBLAS_INT is1b = blas_stride(is1, sizeof(npy_cfloat));
    CBLAS_INT is2b = blas_stride(is2, sizeof(npy_cfloat));

    if (is1b && is2b) {
        double sum[2] = {0., 0.};  /* double for stability */

        while (n > 0) {
            CBLAS_INT chunk = n < NPY_CBLAS_CHUNK ? n : NPY_CBLAS_CHUNK;
            float tmp[2];

            CBLAS_FUNC(cblas_cdotc_sub)((CBLAS_INT)n, ip1, is1b, ip2, is2b, tmp);
            sum[0] += (double)tmp[0];
            sum[1] += (double)tmp[1];
            /* use char strides here */
            ip1 += chunk * is1;
            ip2 += chunk * is2;
            n -= chunk;
        }
        ((float *)op)[0] = (float)sum[0];
        ((float *)op)[1] = (float)sum[1];
    }
    else
#endif
    {
        float sumr = (float)0.0;
        float sumi = (float)0.0;
        npy_intp i;

        for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
            const float ip1r = ((float *)ip1)[0];
            const float ip1i = ((float *)ip1)[1];
            const float ip2r = ((float *)ip2)[0];
            const float ip2i = ((float *)ip2)[1];

            sumr += ip1r * ip2r + ip1i * ip2i;
            sumi += ip1r * ip2i - ip1i * ip2r;
        }
        ((float *)op)[0] = sumr;
        ((float *)op)[1] = sumi;
    }
}


/*
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
CDOUBLE_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
             char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
#if defined(HAVE_CBLAS)
    CBLAS_INT is1b = blas_stride(is1, sizeof(npy_cdouble));
    CBLAS_INT is2b = blas_stride(is2, sizeof(npy_cdouble));

    if (is1b && is2b) {
        double sum[2] = {0., 0.};  /* double for stability */

        while (n > 0) {
            CBLAS_INT chunk = n < NPY_CBLAS_CHUNK ? n : NPY_CBLAS_CHUNK;
            double tmp[2];

            CBLAS_FUNC(cblas_zdotc_sub)((CBLAS_INT)n, ip1, is1b, ip2, is2b, tmp);
            sum[0] += (double)tmp[0];
            sum[1] += (double)tmp[1];
            /* use char strides here */
            ip1 += chunk * is1;
            ip2 += chunk * is2;
            n -= chunk;
        }
        ((double *)op)[0] = (double)sum[0];
        ((double *)op)[1] = (double)sum[1];
    }
    else
#endif
    {
        double sumr = (double)0.0;
        double sumi = (double)0.0;
        npy_intp i;

        for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
            const double ip1r = ((double *)ip1)[0];
            const double ip1i = ((double *)ip1)[1];
            const double ip2r = ((double *)ip2)[0];
            const double ip2i = ((double *)ip2)[1];

            sumr += ip1r * ip2r + ip1i * ip2i;
            sumi += ip1r * ip2i - ip1i * ip2r;
        }
        ((double *)op)[0] = sumr;
        ((double *)op)[1] = sumi;
    }
}


/*
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
CLONGDOUBLE_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2,
                 char *op, npy_intp n, void *NPY_UNUSED(ignore))
{
    npy_longdouble tmpr = 0.0L;
    npy_longdouble tmpi = 0.0L;
    npy_intp i;

    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        const npy_longdouble ip1r = ((npy_longdouble *)ip1)[0];
        const npy_longdouble ip1i = ((npy_longdouble *)ip1)[1];
        const npy_longdouble ip2r = ((npy_longdouble *)ip2)[0];
        const npy_longdouble ip2i = ((npy_longdouble *)ip2)[1];

        tmpr += ip1r * ip2r + ip1i * ip2i;
        tmpi += ip1r * ip2i - ip1i * ip2r;
    }
    ((npy_longdouble *)op)[0] = tmpr;
    ((npy_longdouble *)op)[1] = tmpi;
}

/*
 * All data is assumed aligned.
 */
NPY_NO_EXPORT void
OBJECT_vdot(char *ip1, npy_intp is1, char *ip2, npy_intp is2, char *op, npy_intp n,
            void *NPY_UNUSED(ignore))
{
    npy_intp i;
    PyObject *tmp0, *tmp1, *tmp2, *tmp = NULL;
    PyObject **tmp3;
    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        if ((*((PyObject **)ip1) == NULL) || (*((PyObject **)ip2) == NULL)) {
            tmp1 = Py_False;
            Py_INCREF(Py_False);
        }
        else {
            tmp0 = PyObject_CallMethod(*((PyObject **)ip1), "conjugate", NULL);
            if (tmp0 == NULL) {
                Py_XDECREF(tmp);
                return;
            }
            tmp1 = PyNumber_Multiply(tmp0, *((PyObject **)ip2));
            Py_DECREF(tmp0);
            if (tmp1 == NULL) {
                Py_XDECREF(tmp);
                return;
            }
        }
        if (i == 0) {
            tmp = tmp1;
        }
        else {
            tmp2 = PyNumber_Add(tmp, tmp1);
            Py_XDECREF(tmp);
            Py_XDECREF(tmp1);
            if (tmp2 == NULL) {
                return;
            }
            tmp = tmp2;
        }
    }
    tmp3 = (PyObject**) op;
    tmp2 = *tmp3;
    *((PyObject **)op) = tmp;
    Py_XDECREF(tmp2);
}
