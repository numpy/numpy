#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"
#include "common/npy_argparse.h"

#include "common/dlpack/dlpack.h"
#include "common/npy_dlpack.h"

static void
array_dlpack_deleter(DLManagedTensor *self)
{
    PyArrayObject *array = (PyArrayObject *)self->manager_ctx;
    // This will also free the strides as it's one allocation.
    PyMem_Free(self->dl_tensor.shape);
    PyMem_Free(self);
    Py_XDECREF(array);
}

/* This is exactly as mandated by dlpack */
static void dlpack_capsule_deleter(PyObject *self) {
    if (PyCapsule_IsValid(self, "used_dltensor")) {
        return;
    }

    /* an exception may be in-flight, we must save it in case we create another one */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    DLManagedTensor *managed = (DLManagedTensor *)PyCapsule_GetPointer(self, "dltensor");
    if (managed == NULL) {
        PyErr_WriteUnraisable(self);
        goto done;
    }
    /* the spec says the deleter can be NULL if there is no way for the caller to provide a reasonable destructor. */
    if (managed->deleter) {
        managed->deleter(managed);
        /* TODO: is the deleter allowed to set a python exception? */
        assert(!PyErr_Occurred());
    }

done:
    PyErr_Restore(type, value, traceback);
}

// This function cannot return NULL, but it can fail,
// So call PyErr_Occurred to check if it failed after
// calling it.
static DLDevice
array_get_dl_device(PyArrayObject *self) {
    DLDevice ret;
    ret.device_type = kDLCPU;
    ret.device_id = 0;
    PyObject *base = PyArray_BASE(self);
    // The outer if is due to the fact that NumPy arrays are on the CPU
    // by default (if not created from DLPack).
    if (PyCapsule_IsValid(base, NPY_DLPACK_INTERNAL_CAPSULE_NAME)) {
        DLManagedTensor *managed = PyCapsule_GetPointer(
                base, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
        if (managed == NULL) {
            return ret;
        }
        return managed->dl_tensor.device;
    }
    return ret;
}

static char *
array_get_dl_data(PyArrayObject *self) {
    PyObject *base = PyArray_BASE(self);
    if (PyCapsule_IsValid(base, NPY_DLPACK_INTERNAL_CAPSULE_NAME)) {
        DLManagedTensor *managed = PyCapsule_GetPointer(
                base, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
        if (managed == NULL) {
            return NULL;
        }
        return managed->dl_tensor.data;
    }
    return PyArray_DATA(self);
}

/* used internally */
static void array_dlpack_internal_capsule_deleter(PyObject *self)
{
    DLManagedTensor *managed = 
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
    if (managed == NULL) {
        return;
    }
    managed->deleter(managed);
}

PyObject *
array_dlpack(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *stream = Py_None;
    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("__dlpack__", args, len_args, kwnames,
            "$stream", NULL, &stream, NULL, NULL, NULL)) {
        return NULL;
    }

    if (stream != Py_None) {
        PyErr_SetString(PyExc_RuntimeError, "NumPy only supports "
                "stream=None.");
        return NULL;
    }

    npy_intp itemsize = PyArray_ITEMSIZE(self);
    int ndim = PyArray_NDIM(self);
    npy_intp *strides = PyArray_STRIDES(self);
    npy_intp *shape = PyArray_SHAPE(self);

    if (!PyArray_IS_C_CONTIGUOUS(self) && PyArray_SIZE(self) != 1) {
        for (int i = 0; i < ndim; ++i) {
            if (strides[i] % itemsize != 0) {
                PyErr_SetString(PyExc_RuntimeError,
                        "DLPack only supports strides which are a multiple of "
                        "itemsize.");
                return NULL;
            }
        }
    }

    DLDataType managed_dtype;
    PyArray_Descr *dtype = PyArray_DESCR(self);

    if (PyDataType_ISBYTESWAPPED(dtype)) {
        PyErr_SetString(PyExc_TypeError, "DLPack only supports native "
                    "byte swapping.");
            return NULL;
    }

    managed_dtype.bits = 8 * itemsize;
    managed_dtype.lanes = 1;

    if (PyDataType_ISSIGNED(dtype)) {
        managed_dtype.code = kDLInt;
    } else if (PyDataType_ISUNSIGNED(dtype)) {
        managed_dtype.code = kDLUInt;
    } else if (PyDataType_ISFLOAT(dtype)) {
        // We can't be sure that the dtype is
        // IEEE or padded.
        if (itemsize > 8) {
            PyErr_SetString(PyExc_TypeError, "DLPack only supports IEEE "
                    "floating point types without padding.");
            return NULL;
        }
        managed_dtype.code = kDLFloat;
    } else if (PyDataType_ISCOMPLEX(dtype)) {
        // We can't be sure that the dtype is
        // IEEE or padded.
        if (itemsize > 16) {
            PyErr_SetString(PyExc_TypeError, "DLPack only supports IEEE "
                    "complex point types without padding.");
            return NULL;
        }
        managed_dtype.code = kDLComplex;
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "DLPack only supports signed/unsigned integers, float "
                        "and complex dtypes.");
        return NULL;
    }

    DLDevice device = array_get_dl_device(self);
    if (PyErr_Occurred()) {
        return NULL;
    }
    char *data = array_get_dl_data(self);
    if (data == NULL) {
        return NULL;
    }

    DLManagedTensor *managed = PyMem_Malloc(sizeof(DLManagedTensor));
    if (managed == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    managed->dl_tensor.data = data;
    managed->dl_tensor.device = device;
    managed->dl_tensor.dtype = managed_dtype;


    int64_t *managed_shape_strides = PyMem_Malloc(sizeof(int64_t) * ndim * 2);
    if (managed_shape_strides == NULL) {
        PyErr_NoMemory();
        PyMem_Free(managed);
        return NULL;
    }

    int64_t *managed_shape = managed_shape_strides;
    int64_t *managed_strides = managed_shape_strides + ndim;
    for (int i = 0; i < ndim; ++i) {
        managed_shape[i] = shape[i];
        // Strides in DLPack are items; in NumPy are bytes.
        managed_strides[i] = strides[i] / itemsize;
    }

    managed->dl_tensor.ndim = ndim;
    managed->dl_tensor.shape = managed_shape;
    managed->dl_tensor.strides = NULL;
    if (PyArray_SIZE(self) != 1 && !PyArray_IS_C_CONTIGUOUS(self)) {
        managed->dl_tensor.strides = managed_strides;
    }
    managed->dl_tensor.byte_offset = (char *)PyArray_DATA(self) - data;
    managed->manager_ctx = self;
    managed->deleter = array_dlpack_deleter;

    PyObject *capsule = PyCapsule_New(managed, NPY_DLPACK_CAPSULE_NAME,
            dlpack_capsule_deleter);
    if (capsule == NULL) {
        PyMem_Free(managed);
        PyMem_Free(managed_shape_strides);
        return NULL;
    }

    // the capsule holds a reference
    Py_INCREF(self);
    return capsule;
}

PyObject *
array_dlpack_device(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    DLDevice device = array_get_dl_device(self);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("ii", device.device_type, device.device_id);
}

NPY_NO_EXPORT PyObject *
from_dlpack(PyObject *NPY_UNUSED(self), PyObject *obj) {
    PyObject *capsule = PyObject_CallMethod((PyObject *)obj->ob_type,
            "__dlpack__", "O", obj);
    if (capsule == NULL) {
        return NULL;
    }

    DLManagedTensor *managed = 
        (DLManagedTensor *)PyCapsule_GetPointer(capsule,
        NPY_DLPACK_CAPSULE_NAME);

    if (managed == NULL) {
        Py_XDECREF(capsule);
        return NULL;
    }

    const int ndim = managed->dl_tensor.ndim;
    if (ndim > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_RuntimeError,
                "maxdims of DLPack tensor is higher than the supported "
                "maxdims.");
        Py_XDECREF(capsule);
        return NULL;
    }

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    if (device_type != kDLCPU &&
            device_type != kDLCUDAHost &&
            device_type != kDLROCMHost &&
            device_type != kDLCUDAManaged) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported device in DLTensor.");
        Py_XDECREF(capsule);
        return NULL;
    }

    if (managed->dl_tensor.dtype.lanes != 1) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported lanes in DLTensor dtype.");
        Py_XDECREF(capsule);
        return NULL;
    }

    int typenum = -1;
    const uint8_t bits = managed->dl_tensor.dtype.bits;
    const npy_intp itemsize = bits / 8;
    switch(managed->dl_tensor.dtype.code) {
    case kDLInt:
        switch (bits)
        {
            case 8: typenum = NPY_INT8; break;
            case 16: typenum = NPY_INT16; break;
            case 32: typenum = NPY_INT32; break;
            case 64: typenum = NPY_INT64; break;
        }
        break;
    case kDLUInt:
        switch (bits)
        {
            case 8: typenum = NPY_UINT8; break;
            case 16: typenum = NPY_UINT16; break;
            case 32: typenum = NPY_UINT32; break;
            case 64: typenum = NPY_UINT64; break;
        }
        break;
    case kDLFloat:
        switch (bits)
        {
            case 16: typenum = NPY_FLOAT16; break;
            case 32: typenum = NPY_FLOAT32; break;
            case 64: typenum = NPY_FLOAT64; break;
        }
        break;
    case kDLComplex:
        switch (bits)
        {
            case 64: typenum = NPY_COMPLEX64; break;
            case 128: typenum = NPY_COMPLEX128; break;
        }
        break;
    }

    if (typenum == -1) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported dtype in DLTensor.");
        Py_XDECREF(capsule);
        return NULL;
    }

    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr == NULL) {
        Py_XDECREF(capsule);
        return NULL;
    }

    npy_intp shape[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];

    for (int i = 0; i < ndim; ++i) {
        shape[i] = managed->dl_tensor.shape[i];
        // DLPack has elements as stride units, NumPy has bytes.
        if (managed->dl_tensor.strides != NULL)
        {
            strides[i] = managed->dl_tensor.strides[i] * itemsize;
        }
    }

    char *data = (char *)managed->dl_tensor.data +
            managed->dl_tensor.byte_offset;

    PyObject *ret = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, shape,
            managed->dl_tensor.strides != NULL ? strides : NULL, data, 0, NULL);
    if (ret == NULL) {
        Py_XDECREF(capsule);
        Py_XDECREF(descr);
        return NULL;
    }

    PyObject *new_capsule = PyCapsule_New(managed,
            NPY_DLPACK_INTERNAL_CAPSULE_NAME,
            array_dlpack_internal_capsule_deleter);
    if (new_capsule == NULL) {
        Py_XDECREF(capsule);
        Py_XDECREF(ret);
        return NULL;
    }

    if (PyArray_SetBaseObject((PyArrayObject *)ret, new_capsule) < 0) {
        Py_XDECREF(capsule);
        Py_XDECREF(ret);
        return NULL;
    }

    if (PyCapsule_SetName(capsule, NPY_DLPACK_USED_CAPSULE_NAME) < 0) {
        Py_XDECREF(capsule);
        Py_XDECREF(ret);
        return NULL;
    }
    
    Py_XDECREF(capsule);
    return ret;
}


