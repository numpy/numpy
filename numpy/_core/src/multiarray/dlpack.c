#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "dlpack/dlpack.h"
#include "numpy/arrayobject.h"
#include "npy_argparse.h"
#include "npy_dlpack.h"
#include "npy_static_data.h"
#include "conversion_utils.h"


/*
 * Deleter for a NumPy exported dlpack DLManagedTensor(Versioned).
 */
static void
array_dlpack_deleter(DLManagedTensorVersioned *self)
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

/* TODO: Basically same as above until dlpack v0 is removed: */
static void
array_dlpack_deleter_unversioned(DLManagedTensor *self)
{
    if (!Py_IsInitialized()) {
        return;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    PyArrayObject *array = (PyArrayObject *)self->manager_ctx;
    PyMem_Free(self);
    Py_XDECREF(array);

    PyGILState_Release(state);
}


/*
 * Deleter for a DLPack capsule wrapping a DLManagedTensor(Versioned).
 *
 * This is exactly as mandated by dlpack
 */
static void
dlpack_capsule_deleter(PyObject *self) {
    if (PyCapsule_IsValid(self, NPY_DLPACK_VERSIONED_USED_CAPSULE_NAME)) {
        return;
    }

    DLManagedTensorVersioned *managed =
        (DLManagedTensorVersioned *)PyCapsule_GetPointer(
            self, NPY_DLPACK_VERSIONED_CAPSULE_NAME);
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
    }
    /*
     * The spec says the deleter can be NULL if there is no way for the caller
     * to provide a reasonable destructor.
     */
    if (managed->deleter) {
        managed->deleter(managed);
    }
}

/* TODO: Basically same as above until dlpack v0 is removed: */
static void
dlpack_capsule_deleter_unversioned(PyObject *self) {
    if (PyCapsule_IsValid(self, NPY_DLPACK_USED_CAPSULE_NAME)) {
        return;
    }

    DLManagedTensor *managed =
        (DLManagedTensor *)PyCapsule_GetPointer(self, NPY_DLPACK_CAPSULE_NAME);
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
    }

    if (managed->deleter) {
        managed->deleter(managed);
    }
}


/*
 * Deleter for the capsule used as a `base` in `from_dlpack`.
 *
 * This is almost identical to the above used internally as the base for our array
 * so that we can consume (rename) the original capsule.
 */
static void
array_dlpack_internal_capsule_deleter(PyObject *self)
{
    DLManagedTensorVersioned *managed =
        (DLManagedTensorVersioned *)PyCapsule_GetPointer(
            self, NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME);
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
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
}

/* TODO: Basically same as above until dlpack v0 is removed: */
static void
array_dlpack_internal_capsule_deleter_unversioned(PyObject *self)
{
    DLManagedTensor *managed =
        (DLManagedTensor *)PyCapsule_GetPointer(
            self, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
    if (managed == NULL) {
        PyErr_WriteUnraisable(NULL);
        return;
    }

    if (managed->deleter) {
        managed->deleter(managed);
        assert(!PyErr_Occurred());
    }
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
        DLManagedTensor *managed = (DLManagedTensor *)PyCapsule_GetPointer(
                base, NPY_DLPACK_INTERNAL_CAPSULE_NAME);
        if (managed == NULL) {
            return ret;
        }
        return managed->dl_tensor.device;
    }
    else if (PyCapsule_IsValid(base, NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME)) {
        DLManagedTensorVersioned *managed = (DLManagedTensorVersioned *)PyCapsule_GetPointer(
                base, NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME);
        if (managed == NULL) {
            return ret;
        }
        return managed->dl_tensor.device;
    }
    return ret;
}


/*
 * Fill the dl_tensor struct from the `self` array.
 * This struct could be versioned, but as of now is not.
 */
static int
fill_dl_tensor_information(
    DLTensor *dl_tensor, PyArrayObject *self, DLDevice *result_device)
{
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
                return -1;
            }
        }
    }

    DLDataType managed_dtype;
    PyArray_Descr *dtype = PyArray_DESCR(self);

    if (PyDataType_ISBYTESWAPPED(dtype)) {
        PyErr_SetString(PyExc_BufferError,
                "DLPack only supports native byte order.");
            return -1;
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
            return -1;
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
            return -1;
        }
        managed_dtype.code = kDLComplex;
    }
    else {
        PyErr_SetString(PyExc_BufferError,
                "DLPack only supports signed/unsigned integers, float "
                "and complex dtypes.");
        return -1;
    }

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
    dl_tensor->data = PyArray_DATA(self);
    dl_tensor->byte_offset = 0;
    dl_tensor->device = *result_device;
    dl_tensor->dtype = managed_dtype;

    for (int i = 0; i < ndim; ++i) {
        dl_tensor->shape[i] = shape[i];
        // Strides in DLPack are items; in NumPy are bytes.
        dl_tensor->strides[i] = strides[i] / itemsize;
    }

    dl_tensor->ndim = ndim;
    if (PyArray_IS_C_CONTIGUOUS(self)) {
        /* No need to pass strides, so just NULL it again */
        dl_tensor->strides = NULL;
    }
    dl_tensor->byte_offset = 0;

    return 0;
}


static PyObject *
create_dlpack_capsule(
        PyArrayObject *self, int versioned, DLDevice *result_device, int copied)
{
    int ndim = PyArray_NDIM(self);

    /*
     * We align shape and strides at the end but need to align them, offset
     * gives the offset of the shape (and strides) including the struct size.
     */
    size_t align = sizeof(int64_t);
    size_t struct_size = (
        versioned ? sizeof(DLManagedTensorVersioned) : sizeof(DLManagedTensor));

    size_t offset = (struct_size + align - 1) / align * align;
    void *ptr = PyMem_Malloc(offset + (sizeof(int64_t) * ndim * 2));
    if (ptr == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    DLTensor *dl_tensor;
    PyCapsule_Destructor capsule_deleter;
    const char *capsule_name;

    if (versioned) {
        DLManagedTensorVersioned *managed = (DLManagedTensorVersioned *)ptr;
        capsule_name = NPY_DLPACK_VERSIONED_CAPSULE_NAME;
        capsule_deleter = (PyCapsule_Destructor)dlpack_capsule_deleter;
        managed->deleter = array_dlpack_deleter;
        managed->manager_ctx = self;

        dl_tensor = &managed->dl_tensor;

        /* The versioned tensor has additional fields that we need to set */
        managed->version.major = 1;
        managed->version.minor = 0;

        managed->flags = 0;
        if (!PyArray_CHKFLAGS(self, NPY_ARRAY_WRITEABLE)) {
            managed->flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
        }
        if (copied) {
            managed->flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
        }
    }
    else {
        DLManagedTensor *managed = (DLManagedTensor *)ptr;
        capsule_name = NPY_DLPACK_CAPSULE_NAME;
        capsule_deleter = (PyCapsule_Destructor)dlpack_capsule_deleter_unversioned;
        managed->deleter = array_dlpack_deleter_unversioned;
        managed->manager_ctx = self;

        dl_tensor = &managed->dl_tensor;
    }

    dl_tensor->shape = (int64_t *)((char *)ptr + offset);
    /* Note that strides may be set to NULL later if C-contiguous */
    dl_tensor->strides = dl_tensor->shape + ndim;

    if (fill_dl_tensor_information(dl_tensor, self, result_device) < 0) {
        PyMem_Free(ptr);
        return NULL;
    }

    PyObject *capsule = PyCapsule_New(ptr, capsule_name, capsule_deleter);
    if (capsule == NULL) {
        PyMem_Free(ptr);
        return NULL;
    }

    // the capsule holds a reference
    Py_INCREF(self);

    return capsule;
}


static int
device_converter(PyObject *obj, DLDevice *result_device)
{
    int type, id;
    if (obj == Py_None) {
        return NPY_SUCCEED;
    }
    if (!PyTuple_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "dl_device must be a tuple");
        return NPY_FAIL;
    }
    if (!PyArg_ParseTuple(obj, "ii", &type, &id)) {
        return NPY_FAIL;
    }
    /* We can honor the request if matches the existing one or is CPU */
    if (type == result_device->device_type && id == result_device->device_id) {
        return NPY_SUCCEED;
    }
    if (type == kDLCPU && id == 0) {
        result_device->device_type = type;
        result_device->device_id = id;
        return NPY_SUCCEED;
    }

    PyErr_SetString(PyExc_ValueError, "unsupported device requested");
    return NPY_FAIL;
}


NPY_NO_EXPORT PyObject *
array_dlpack(PyArrayObject *self,
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *stream = Py_None;
    PyObject *max_version = Py_None;
    NPY_COPYMODE copy_mode = NPY_COPY_IF_NEEDED;
    long major_version = 0;
    /* We allow the user to request a result device in principle. */
    DLDevice result_device = array_get_dl_device(self);
    if (PyErr_Occurred()) {
        return NULL;
    }

    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("__dlpack__", args, len_args, kwnames,
            "$stream", NULL, &stream,
            "$max_version", NULL, &max_version,
            "$dl_device", &device_converter, &result_device,
            "$copy", &PyArray_CopyConverter, &copy_mode,
            NULL, NULL, NULL)) {
        return NULL;
    }

    if (max_version != Py_None) {
        if (!PyTuple_Check(max_version) || PyTuple_GET_SIZE(max_version) != 2) {
            PyErr_SetString(PyExc_TypeError,
                    "max_version must be None or a tuple with two elements.");
            return NULL;
        }
        major_version = PyLong_AsLong(PyTuple_GET_ITEM(max_version, 0));
        if (major_version == -1 && PyErr_Occurred()) {
            return NULL;
        }
    }

    if (stream != Py_None) {
        PyErr_SetString(PyExc_RuntimeError,
                "NumPy only supports stream=None.");
        return NULL;
    }

    /* If the user requested a copy be made, honor that here already */
    if (copy_mode == NPY_COPY_ALWAYS) {
        /* TODO: It may be good to check ability to export dtype first. */
        self = (PyArrayObject *)PyArray_NewCopy(self, NPY_KEEPORDER);
        if (self == NULL) {
            return NULL;
        }
    }
    else {
        Py_INCREF(self);
    }

    if (major_version < 1 && !(PyArray_FLAGS(self) & NPY_ARRAY_WRITEABLE)) {
        PyErr_SetString(PyExc_BufferError,
            "Cannot export readonly array since signalling readonly "
            "is unsupported by DLPack (supported by newer DLPack version).");
        Py_DECREF(self);
        return NULL;
    }

    /*
     * TODO: The versioned and non-versioned structs of DLPack are very
     * similar but not ABI compatible so that the function called here requires
     * branching (templating didn't seem worthwhile).
     *
     * Version 0 support should be deprecated in NumPy 2.1 and the branches
     * can then be removed again.
     */
    PyObject *res = create_dlpack_capsule(
            self, major_version >= 1, &result_device,
            copy_mode == NPY_COPY_ALWAYS);
    Py_DECREF(self);

    return res;
}

NPY_NO_EXPORT PyObject *
array_dlpack_device(PyArrayObject *self, PyObject *NPY_UNUSED(args))
{
    DLDevice device = array_get_dl_device(self);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return Py_BuildValue("ii", device.device_type, device.device_id);
}

NPY_NO_EXPORT PyObject *
from_dlpack(PyObject *NPY_UNUSED(self),
        PyObject *const *args, Py_ssize_t len_args, PyObject *kwnames)
{
    PyObject *obj, *copy = Py_None, *device = Py_None;
    NPY_PREPARE_ARGPARSER;
    if (npy_parse_arguments("from_dlpack", args, len_args, kwnames,
            "obj", NULL, &obj,
            "$copy", NULL, &copy,
            "$device", NULL, &device,
            NULL, NULL, NULL) < 0) {
        return NULL;
    }

    /* 
     * Prepare arguments for the full call. We always forward copy and pass
     * our max_version. `device` is always passed as `None`, but if the user
     * provided a device, we will replace it with the "cpu": (1, 0).
     */
    PyObject *call_args[] = {obj, Py_None, copy, npy_static_pydata.dl_max_version};
    Py_ssize_t nargsf = 1 | PY_VECTORCALL_ARGUMENTS_OFFSET;

    /* If device is passed it must be "cpu" and replace it with (1, 0) */
    if (device != Py_None) {
        /* test that device is actually CPU */
        NPY_DEVICE device_request = NPY_DEVICE_CPU;
        if (!PyArray_DeviceConverterOptional(device, &device_request)) {
            return NULL;
        }
        assert(device_request == NPY_DEVICE_CPU);
        call_args[1] = npy_static_pydata.dl_cpu_device_tuple;
    }


    PyObject *capsule = PyObject_VectorcallMethod(
            npy_interned_str.__dlpack__, call_args, nargsf,
            npy_static_pydata.dl_call_kwnames);
    if (capsule == NULL) {
        /*
         * TODO: This path should be deprecated in NumPy 2.1.  Once deprecated
         * the below code can be simplified w.r.t. to versioned/unversioned.
         *
         * We try without any arguments if both device and copy are None,
         * since the exporter may not support older versions of the protocol.
         */
        if (PyErr_ExceptionMatches(PyExc_TypeError)
                && device == Py_None && copy == Py_None) {
            /* max_version may be unsupported, try without kwargs */
            PyErr_Clear();
            capsule = PyObject_VectorcallMethod(
                npy_interned_str.__dlpack__, call_args, nargsf, NULL);
        }
        if (capsule == NULL) {
            return NULL;
        }
    }

    void *managed_ptr;
    DLTensor dl_tensor;
    int readonly;
    int versioned = PyCapsule_IsValid(capsule, NPY_DLPACK_VERSIONED_CAPSULE_NAME);
    if (versioned) {
        managed_ptr = PyCapsule_GetPointer(capsule, NPY_DLPACK_VERSIONED_CAPSULE_NAME);
        DLManagedTensorVersioned *managed = (DLManagedTensorVersioned *)managed_ptr;
        if (managed == NULL) {
            Py_DECREF(capsule);
            return NULL;
        }

        if (managed->version.major > 1) {
            PyErr_SetString(PyExc_BufferError,
                "from_dlpack(): the exported DLPack major version is too "
                "high to be imported by this version of NumPy.");
            Py_DECREF(capsule);
            return NULL;
        }

        dl_tensor = managed->dl_tensor;
        readonly = (managed->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
    }
    else {
        managed_ptr = PyCapsule_GetPointer(capsule, NPY_DLPACK_CAPSULE_NAME);
        DLManagedTensor *managed = (DLManagedTensor *)managed_ptr;
        if (managed == NULL) {
            Py_DECREF(capsule);
            return NULL;
        }
        dl_tensor = managed->dl_tensor;
        readonly = 1;
    }

    const int ndim = dl_tensor.ndim;
    if (ndim > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_RuntimeError,
                "maxdims of DLPack tensor is higher than the supported "
                "maxdims.");
        Py_DECREF(capsule);
        return NULL;
    }

    DLDeviceType device_type = dl_tensor.device.device_type;
    if (device_type != kDLCPU &&
            device_type != kDLCUDAHost &&
            device_type != kDLROCMHost &&
            device_type != kDLCUDAManaged) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported device in DLTensor.");
        Py_DECREF(capsule);
        return NULL;
    }

    if (dl_tensor.dtype.lanes != 1) {
        PyErr_SetString(PyExc_RuntimeError,
                "Unsupported lanes in DLTensor dtype.");
        Py_DECREF(capsule);
        return NULL;
    }

    int typenum = -1;
    const uint8_t bits = dl_tensor.dtype.bits;
    const npy_intp itemsize = bits / 8;
    switch (dl_tensor.dtype.code) {
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
        shape[i] = dl_tensor.shape[i];
        // DLPack has elements as stride units, NumPy has bytes.
        if (dl_tensor.strides != NULL) {
            strides[i] = dl_tensor.strides[i] * itemsize;
        }
    }

    char *data = (char *)dl_tensor.data + dl_tensor.byte_offset;

    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr == NULL) {
        Py_DECREF(capsule);
        return NULL;
    }

    PyObject *ret = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, shape,
            dl_tensor.strides != NULL ? strides : NULL, data, readonly ? 0 :
            NPY_ARRAY_WRITEABLE, NULL);

    if (ret == NULL) {
        Py_DECREF(capsule);
        return NULL;
    }

    PyObject *new_capsule;
    if (versioned) {
        new_capsule = PyCapsule_New(managed_ptr,
            NPY_DLPACK_VERSIONED_INTERNAL_CAPSULE_NAME,
            (PyCapsule_Destructor)array_dlpack_internal_capsule_deleter);
    }
    else {
        new_capsule = PyCapsule_New(managed_ptr,
            NPY_DLPACK_INTERNAL_CAPSULE_NAME,
            (PyCapsule_Destructor)array_dlpack_internal_capsule_deleter_unversioned);
    }

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

    const char *new_name = (
        versioned ? NPY_DLPACK_VERSIONED_USED_CAPSULE_NAME
                  : NPY_DLPACK_USED_CAPSULE_NAME);
    if (PyCapsule_SetName(capsule, new_name) < 0) {
        Py_DECREF(capsule);
        Py_DECREF(ret);
        return NULL;
    }

    Py_DECREF(capsule);
    return ret;
}


