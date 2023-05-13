#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "dlpack/dlpack.h"
#include "numpy/arrayobject.h"
#include "npy_argparse.h"
#include "npy_dlpack.h"

static void
array_dlpack_deleter(DLManagedTensor *self)
{
    /*
     * Leak the pyobj if not initialized.  This can happen if we are running
     * exit handlers that are destructing c++ objects with residual (owned)
     * PyObjects stored in them after the Python runtime has already been
     * terminated.
     */
    if (!Py_IsInitialized()) {
        return;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    PyArrayObject *array = (PyArrayObject *)self->manager_ctx;
    // This will also free the shape and strides as it's one allocation.
    PyMem_Free(self);
    Py_XDECREF(array);

    PyGILState_Release(state);
}

/* This is exactly as mandated by dlpack */
static void dlpack_capsule_deleter(PyObject *self) {
    if (PyCapsule_IsValid(self, NPY_DLPACK_USED_CAPSULE_NAME)) {
        return;
    }

    /* an exception may be in-flight, we must save it in case we create another one */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    DLManagedTensor *managed =
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_CAPSULE_NAME);
    if (managed == NULL) {
        PyErr_WriteUnraisable(self);
        goto done;
    }
    /*
     *  the spec says the deleter can be NULL if there is no way for the caller
     * to provide a reasonable destructor.
     */
    if (managed->deleter) {
        managed->deleter(managed);
        /* TODO: is the deleter allowed to set a python exception? */
        assert(!PyErr_Occurred());
    }

done:
    PyErr_Restore(type, value, traceback);
}

/* used internally, almost identical to dlpack_capsule_deleter() */
static void array_dlpack_internal_capsule_deleter(PyObject *self)
{
    /* an exception may be in-flight, we must save it in case we create another one */
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);

    DLManagedTensor *managed =
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
    if (managed == NULL) {
        PyErr_WriteUnraisable(self);
        goto done;
    }
    /*
     *  the spec says the deleter can be NULL if there is no way for the caller
     * to provide a reasonable destructor.
     */
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

    // walk the bases (see gh-20340)
    while (base != NULL && PyArray_Check(base)) {
        base = PyArray_BASE((PyArrayObject *)base);
    }

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
        PyErr_SetString(PyExc_RuntimeError,
                "NumPy only supports stream=None.");
        return NULL;
    }

    if ( !(PyArray_FLAGS(self) & NPY_ARRAY_WRITEABLE)) {
        PyErr_SetString(PyExc_BufferError,
            "Cannot export readonly array since signalling readonly "
            "is unsupported by DLPack.");
        return NULL;
    }

    npy_intp itemsize = PyArray_ITEMSIZE(self);
    int ndim = PyArray_NDIM(self);
    npy_intp *strides = PyArray_STRIDES(self);
    npy_intp *shape = PyArray_SHAPE(self);

    if (!PyArray_IS_C_CONTIGUOUS(self) && PyArray_SIZE(self) != 1) {
        for (int i = 0; i < ndim; ++i) {
            if (shape[i] != 1 && strides[i] % itemsize != 0) {
                PyErr_SetString(PyExc_BufferError,
                        "DLPack only supports strides which are a multiple of "
                        "itemsize.");
                return NULL;
            }
        }
    }

    DLDataType managed_dtype;
    PyArray_Descr *dtype = PyArray_DESCR(self);

    if (PyDataType_ISBYTESWAPPED(dtype)) {
        PyErr_SetString(PyExc_BufferError,
                "DLPack only supports native byte order.");
            return NULL;
    }

    managed_dtype.bits = 8 * itemsize;
    managed_dtype.lanes = 1;

    if (PyDataType_ISBOOL(dtype)) {
        managed_dtype.code = kDLBool;
    }
    else if (PyDataType_ISSIGNED(dtype)) {
        managed_dtype.code = kDLInt;
    }
    else if (PyDataType_ISUNSIGNED(dtype)) {
        managed_dtype.code = kDLUInt;
    }
    else if (PyDataType_ISFLOAT(dtype)) {
        // We can't be sure that the dtype is
        // IEEE or padded.
        if (itemsize > 8) {
            PyErr_SetString(PyExc_BufferError,
                    "DLPack only supports IEEE floating point types "
                    "without padding (longdouble typically is not IEEE).");
            return NULL;
        }
        managed_dtype.code = kDLFloat;
    }
    else if (PyDataType_ISCOMPLEX(dtype)) {
        // We can't be sure that the dtype is
        // IEEE or padded.
        if (itemsize > 16) {
            PyErr_SetString(PyExc_BufferError,
                    "DLPack only supports IEEE floating point types "
                    "without padding (longdouble typically is not IEEE).");
            return NULL;
        }
        managed_dtype.code = kDLComplex;
    }
    else {
        PyErr_SetString(PyExc_BufferError,
                "DLPack only supports signed/unsigned integers, float "
                "and complex dtypes.");
        return NULL;
    }

    DLDevice device = array_get_dl_device(self);
    if (PyErr_Occurred()) {
        return NULL;
    }

    // ensure alignment
    int offset = sizeof(DLManagedTensor) % sizeof(void *);
    void *ptr = PyMem_Malloc(sizeof(DLManagedTensor) + offset +
        (sizeof(int64_t) * ndim * 2));
    if (ptr == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    DLManagedTensor *managed = ptr;

    /*
     * Note: the `dlpack.h` header suggests/standardizes that `data` must be
     * 256-byte aligned.  We ignore this intentionally, because `__dlpack__`
     * standardizes that `byte_offset` must be 0 (for now) to not break pytorch:
     * https://github.com/data-apis/array-api/issues/293#issuecomment-964111413
     *
     * We further assume that exporting fully unaligned data is OK even without
     * `byte_offset` since the standard does not reject it.
     * Presumably, pytorch will support importing `byte_offset != 0` and NumPy
     * can choose to use it starting about 2023.  At that point, it may be
     * that NumPy MUST use `byte_offset` to adhere to the standard (as
     * specified in the header)!
     */
    managed->dl_tensor.data = PyArray_DATA(self);
    managed->dl_tensor.byte_offset = 0;
    managed->dl_tensor.device = device;
    managed->dl_tensor.dtype = managed_dtype;

    int64_t *managed_shape_strides = (int64_t *)((char *)ptr +
        sizeof(DLManagedTensor) + offset);

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
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = self;
    managed->deleter = array_dlpack_deleter;

    PyObject *capsule = PyCapsule_New(managed, NPY_DLPACK_CAPSULE_NAME,
            dlpack_capsule_deleter);
    if (capsule == NULL) {
        PyMem_Free(ptr);
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
        Py_DECREF(capsule);
        return NULL;
    }

    const int ndim = managed->dl_tensor.ndim;
    if (ndim > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_RuntimeError,
                "maxdims of DLPack tensor is higher than the supported "
                "maxdims.");
        Py_DECREF(capsule);
        return NULL;
    }

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    if (device_type != kDLCPU &&
            device_type != kDLCUDAHost &&
            device_type != kDLROCMHost &&
            device_type != kDLCUDAManaged) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported device in DLTensor.");
        Py_DECREF(capsule);
        return NULL;
    }

    if (managed->dl_tensor.dtype.lanes != 1) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported lanes in DLTensor dtype.");
        Py_DECREF(capsule);
        return NULL;
    }

    int typenum = -1;
    const uint8_t bits = managed->dl_tensor.dtype.bits;
    const npy_intp itemsize = bits / 8;
    switch (managed->dl_tensor.dtype.code) {
    case kDLBool:
        if (bits == 8) {
            typenum = NPY_BOOL;
        }
        break;
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
        Py_DECREF(capsule);
        return NULL;
    }

    npy_intp shape[NPY_MAXDIMS];
    npy_intp strides[NPY_MAXDIMS];

    for (int i = 0; i < ndim; ++i) {
        shape[i] = managed->dl_tensor.shape[i];
        // DLPack has elements as stride units, NumPy has bytes.
        if (managed->dl_tensor.strides != NULL) {
            strides[i] = managed->dl_tensor.strides[i] * itemsize;
        }
    }

    char *data = (char *)managed->dl_tensor.data +
            managed->dl_tensor.byte_offset;

    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr == NULL) {
        Py_DECREF(capsule);
        return NULL;
    }

    PyObject *ret = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, shape,
            managed->dl_tensor.strides != NULL ? strides : NULL, data, 0, NULL);
    if (ret == NULL) {
        Py_DECREF(capsule);
        return NULL;
    }

    PyObject *new_capsule = PyCapsule_New(managed,
            NPY_DLPACK_INTERNAL_CAPSULE_NAME,
            array_dlpack_internal_capsule_deleter);
    if (new_capsule == NULL) {
        Py_DECREF(capsule);
        Py_DECREF(ret);
        return NULL;
    }

    if (PyArray_SetBaseObject((PyArrayObject *)ret, new_capsule) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(ret);
        return NULL;
    }

    if (PyCapsule_SetName(capsule, NPY_DLPACK_USED_CAPSULE_NAME) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(ret);
        return NULL;
    }

    Py_DECREF(capsule);
    return ret;
}


