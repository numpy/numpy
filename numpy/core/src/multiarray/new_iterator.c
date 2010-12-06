#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "structmember.h"

#define _MULTIARRAYMODULE
#include <numpy/ndarrayobject.h>

#include "new_iterator.h"


#define NPY_ITER_FLAGS_HASPERM  0x001
#define NPY_ITER_FLAGS_HASINDEX 0x002
#define NPY_ITER_FLAGS_HASCOORD 0x004

/* Size of all the data before the AXISDATA starts */
#define NIT_SIZEOF_BASEDATA(itflags, ndim, niter) ( \
        /* uint32 itflags AND uint16 ndim AND uint16 niter */ \
        8 + \
        /* PyArray_Descr* dtypes[niter] */ \
        /* npy_intp itemsizes[niter] */ \
        2*(NPY_SIZEOF_INTP)*(niter) \
        )

/* Size of one AXISDATA struct within the iterator */
#define NIT_SIZEOF_AXISDATA(itflags, ndim, niter) (( \
        /* intp shape */ \
        1 + \
        /* intp coord */ \
        1 + \
        /* intp stride[niter] AND char* ptr[niter] */ \
        2*(niter))*NPY_SIZEOF_INTP + \
        /* intp indexstride AND intp index (when index is provided) */ \
        ((itflags&NPY_ITER_FLAGS_HASINDEX) ? 2 : 0)) \

/* Size of the whole iterator */
#define NIT_SIZEOF_ITERATOR(itflags, ndim, niter) ( \
        NIT_SIZEOF_BASEDATA(itflags, ndim, niter) + \
        NIT_SIZEOF_AXISDATA(itflags, ndim, niter)*(ndim) + \
        NPY_SIZEOF_INTP*(ndim) + \
        NPY_SIZEOF_INTP*(niter))

/* Internal-only ITERATOR DATA MEMBER ACCESS */
#define NIT_FLAGS(iter) \
        (*((npy_uint32*)(iter)))
#define NIT_NDIM(iter) \
        (*((npy_uint16*)(iter) + 2))
#define NIT_NITER(iter) \
        (*((npy_uint16*)(iter) + 3))
#define NIT_DTYPES(iter, itflags, ndim, niter) \
        ((PyArray_Descr**)((char*)(iter) + 8))
#define NIT_ITEMSIZES(iter, itflags, ndim, niter) \
        ((npy_intp*)((char*)(iter) + 8 + NPY_SIZEOF_INTP*(niter)))
#define NIT_AXISDATA(iter, itflags, ndim, niter) \
        ((char*)(iter) + NIT_SIZEOF_BASEDATA(itflags, ndim, niter))
#define NIT_PERM(iter, itflags, ndim, niter)  ((npy_intp*)( \
        (char*)(iter) + NIT_SIZEOF_BASEDATA(itflags, ndim, niter) + \
        NIT_SIZEOF_AXISDATA(itflags, ndim, niter) *(ndim))
#define NIT_OBJECTS(iter, itflags, ndim, niter) ((PyObject**)( \
        (char*)(iter) + NIT_SIZEOF_BASEDATA(itflags, ndim, niter) + \
        (NIT_SIZEOF_AXISDATA(itflags, ndim, niter) + \
         (((itflags)&NPY_ITER_FLAGS_HASPERM) ? NPY_SIZEOF_INTP : 0) \
        )*(ndim)))

/* Internal-only AXISDATA MEMBER ACCESS. */
#define NAD_SHAPE(axisdata) (*((npy_intp*)(axisdata)))
#define NAD_COORD(axisdata) (*((npy_intp*)(axisdata) + 1))
#define NAD_STRIDES(axisdata) ((npy_intp*)(axisdata) + 2)
#define NAD_PTRS(axisdata, nstrides) ((char**)(axisdata) + 2 + (nstrides))
#define NAD_NSTRIDES(itflags, ndim, niter) \
        ((niter) + ((itflags&NPY_ITER_FLAGS_HASPERM) ? 1 : 0))

void* NpyIter_New(PyObject* op, npy_uint32 flags, PyArray_Descr* dtype,
                  int min_depth, int max_depth)
{
    npy_uint32 itflags = 0;
    npy_intp idim, ndim = 1, niter = 1,
            nstrides, sizeof_axisdata;
    PyArray_Descr* opdtype;
    void* iter = 0;
    char* axisdata = 0;

    /* Currently only work with arrays */
    if (!PyArray_Check(op)) {
        PyErr_SetString(PyExc_ValueError,
                "Can only create an iterator for an array");
        return NULL;
    }

    /* Currently only do 1 dimension */
    if (PyArray_NDIM(op) != 1) {
        PyErr_SetString(PyExc_ValueError,
                "Can only create a one-dimensional iterator");
        return NULL;
    }

    /* Get the data type of the array */
    opdtype = PyArray_DESCR(op);
    if (opdtype == NULL) {
        PyErr_SetString(PyExc_ValueError,
                "Input object has no dtype descr");
        return NULL;
    }
    if (dtype != NULL && !PyArray_EquivTypes(opdtype, dtype)) {
        PyErr_SetString(PyExc_ValueError,
                "Don't support automatic dtype conversions yet");
        return NULL;
    }
    /* Take a reference to the dtype for the iterator */
    Py_INCREF(opdtype);

    /* Allocate memory for the iterator */
    iter = malloc(NIT_SIZEOF_ITERATOR(itflags, ndim, niter));

    /* Fill in the base data */
    NIT_FLAGS(iter) = itflags;
    NIT_NDIM(iter) = ndim;
    NIT_NITER(iter) = niter;
    NIT_DTYPES(iter, itflags, ndim, niter)[0] = opdtype;
    NIT_ITEMSIZES(iter, itflags, ndim, niter)[0] = PyArray_ITEMSIZE(op);

    /* Fill in the axis data */
    sizeof_axisdata = NIT_SIZEOF_AXISDATA(itflags, ndim, niter);
    axisdata = NIT_AXISDATA(iter, itflags, ndim, niter);
    nstrides = NAD_NSTRIDES(itflags, ndim, niter);
    for(idim = 0; idim < ndim; ++idim, axisdata += sizeof_axisdata) {
        NAD_SHAPE(axisdata) = PyArray_DIM(op, idim);
        NAD_COORD(axisdata) = 0;
        NAD_STRIDES(axisdata)[0] = PyArray_STRIDE(op, idim);
        NAD_PTRS(axisdata, nstrides)[0] = PyArray_DATA(op);
    }

    /* Fill in the destruction data */
    NIT_OBJECTS(iter, itflags, ndim, niter)[0] = op;

    return iter;
}

int NpyIter_Deallocate(void* iter)
{
    npy_uint32 itflags = NIT_FLAGS(iter);
    npy_intp ndim = NIT_NDIM(iter), niter = NIT_NITER(iter);
    npy_intp i;
    PyArray_Descr **dtypes = NIT_DTYPES(iter, itflags, ndim, niter);
    PyObject **arrays = NIT_OBJECTS(iter, itflags, ndim, niter);

    /* Deallocate all the dtypes and objects that were iterated */
    for(i = 0; i < niter; ++i) {
        Py_XDECREF(dtypes[i]);
        Py_XDECREF(arrays[i]);
    }

    /* Deallocate the iterator memory */
    free(iter);

    return NPY_SUCCEED;
}

/* Specialized iternext */
int npyiter_iternext_1dim_1iter_noitflags(void *iter)
{
    char* axisdata = NIT_AXISDATA(iter, 0, 1, 1);
    /* Increment the 1 pointer */
    NAD_PTRS(axisdata, 1)[0] += NAD_STRIDES(axisdata)[0];
    /* Increment the 1 coordinate */
    NAD_COORD(axisdata)++;
    /* Finished when the coordinate equals the shape */
    return NAD_COORD(axisdata) < NAD_SHAPE(axisdata);
}

NpyIter_IterNext_Fn NpyIter_GetIterNext(void *iter)
{
    npy_uint32 itflags = NIT_FLAGS(iter);
    npy_intp ndim = NIT_NDIM(iter), niter = NIT_NITER(iter);

    /* Switch statements let the compiler optimize this most effectively */
    switch (itflags) {
        case 0:
            switch (ndim) {
                case 1:
                    switch (niter) {
                        case 1:
                            return &npyiter_iternext_1dim_1iter_noitflags;
                        default:
                            return NULL;
                    }
                default:
                    return NULL;
            }
        default:
            return NULL;
    }
}

char **NpyIter_GetDataPtrArray(void *iter)
{
    npy_uint32 itflags = NIT_FLAGS(iter);
    npy_intp ndim = NIT_NDIM(iter), niter = NIT_NITER(iter);
    npy_intp nstrides = NAD_NSTRIDES(itflags, ndim, niter);
    char* axisdata = NIT_AXISDATA(iter, itflags, ndim, niter);

    return NAD_PTRS(axisdata, nstrides);
}

npy_intp *NpyIter_GetItemSizeArray(void *iter)
{
    npy_uint32 itflags = NIT_FLAGS(iter);
    npy_intp ndim = NIT_NDIM(iter), niter = NIT_NITER(iter);
    
    return NIT_ITEMSIZES(iter, itflags, ndim, niter);
}

