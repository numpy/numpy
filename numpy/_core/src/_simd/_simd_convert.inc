/**
 * This file is included by `_simd.dispatch.c.src`. Its contents are affected by the simd configuration, and
 * therefore must be built multiple times. Making it a standalone `.c` file with `NPY_VISIBILITY_HIDDEN`
 * symbols would require judicious use of `NPY_CPU_DISPATCH_DECLARE` and `NPY_CPU_DISPATCH_CURFX`, which was
 * deemed too harmful to readability.
 */
/************************************
 ** Protected Definitions
 ************************************/
static simd_data
simd_scalar_from_number(PyObject *obj, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(info->is_scalar && info->lane_size > 0);
    simd_data data;
    if (info->is_float) {
        data.f64 = PyFloat_AsDouble(obj);
        if (dtype == simd_data_f32){
            data.f32 = (float)data.f64;
        }
    } else {
        data.u64 = PyLong_AsUnsignedLongLongMask(obj);
    #if NPY_SIMD_BIGENDIAN
        int leftb = (sizeof(npyv_lanetype_u64) - info->lane_size) * 8;
        data.u64 <<= leftb;
    #endif
    }
    return data;
}

static PyObject *
simd_scalar_to_number(simd_data data, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(info->is_scalar && info->lane_size > 0);
    if (info->is_float) {
        if (dtype == simd_data_f32) {
            return PyFloat_FromDouble(data.f32);
        }
        return PyFloat_FromDouble(data.f64);
    }
    int leftb = (sizeof(npyv_lanetype_u64) - info->lane_size) * 8;
#if !NPY_SIMD_BIGENDIAN
    data.u64 <<= leftb;
#endif
    if (info->is_signed) {
        return PyLong_FromLongLong(data.s64 >> leftb);
    }
    return PyLong_FromUnsignedLongLong(data.u64 >> leftb);
}

typedef struct {
    Py_ssize_t len;
    void *ptr;
} simd__alloc_data;

static void *
simd_sequence_new(Py_ssize_t len, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(len > 0 && info->is_sequence && info->lane_size > 0);
    size_t size = sizeof(simd__alloc_data) + len * info->lane_size + NPY_SIMD_WIDTH;
    void *ptr = malloc(size);
    if (ptr == NULL) {
        return PyErr_NoMemory();
    }
    // align the pointer
    simd__alloc_data *a_ptr = (simd__alloc_data *)(
        ((uintptr_t)ptr + sizeof(simd__alloc_data) + NPY_SIMD_WIDTH) & ~(uintptr_t)(NPY_SIMD_WIDTH-1)
    );
    a_ptr[-1].len = len;
    a_ptr[-1].ptr = ptr;
    return a_ptr;
}

static Py_ssize_t
simd_sequence_len(void const *ptr)
{
    return ((simd__alloc_data const*)ptr)[-1].len;
}

static void
simd_sequence_free(void *ptr)
{
    free(((simd__alloc_data *)ptr)[-1].ptr);
}

static void *
simd_sequence_from_iterable(PyObject *obj, simd_data_type dtype, Py_ssize_t min_size)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    assert(info->is_sequence && info->lane_size > 0);
    PyObject *seq_obj = PySequence_Fast(obj, "expected a sequence");
    if (seq_obj == NULL) {
        return NULL;
    }
    Py_ssize_t seq_size = PySequence_Fast_GET_SIZE(seq_obj);
    if (seq_size < min_size) {
        PyErr_Format(PyExc_ValueError,
            "minimum acceptable size of the required sequence is %d, given(%d)",
            min_size, seq_size
        );
        Py_DECREF(seq_obj);
        return NULL;
    }
    npyv_lanetype_u8 *dst = simd_sequence_new(seq_size, dtype);
    if (dst == NULL) {
        return NULL;
    }
    PyObject **seq_items = PySequence_Fast_ITEMS(seq_obj);
    for (Py_ssize_t i = 0; i < seq_size; ++i) {
        simd_data data = simd_scalar_from_number(seq_items[i], info->to_scalar);
        npyv_lanetype_u8 *sdst = dst + i * info->lane_size;
        memcpy(sdst, &data.u64, info->lane_size);
    }
    Py_DECREF(seq_obj);

    if (PyErr_Occurred()) {
        simd_sequence_free(dst);
        return NULL;
    }
    return dst;
}

static int
simd_sequence_fill_iterable(PyObject *obj, const void *ptr, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    if (!PySequence_Check(obj)) {
        PyErr_Format(PyExc_TypeError,
            "a sequence object is required to fill %s", info->pyname
        );
        return -1;
    }
    const npyv_lanetype_u8 *src = ptr;
    Py_ssize_t seq_len = simd_sequence_len(ptr);
    for (Py_ssize_t i = 0; i < seq_len; ++i) {
        const npyv_lanetype_u8 *ssrc = src + i * info->lane_size;
        simd_data data;
        memcpy(&data.u64, ssrc, info->lane_size);
        PyObject *item = simd_scalar_to_number(data, info->to_scalar);
        if (item == NULL) {
            return -1;
        }
        int res = PySequence_SetItem(obj, i, item);
        Py_DECREF(item);
        if (res < 0) {
            return -1;
        }
    }
    return 0;
}

static PyObject *
simd_sequence_to_list(const void *ptr, simd_data_type dtype)
{
    PyObject *list = PyList_New(simd_sequence_len(ptr));
    if (list == NULL) {
        return NULL;
    }
    if (simd_sequence_fill_iterable(list, ptr, dtype) < 0) {
        Py_DECREF(list);
        return NULL;
    }
    return list;
}

static simd_data
simd_vectorx_from_tuple(PyObject *obj, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    // NPYV currently only supports x2 and x3
    assert(info->is_vectorx > 1 && info->is_vectorx < 4);

    simd_data data = {.u64 = 0};
    if (!PyTuple_Check(obj) || PyTuple_GET_SIZE(obj) != info->is_vectorx) {
        PyErr_Format(PyExc_TypeError,
            "a tuple of %d vector type %s is required",
            info->is_vectorx, simd_data_getinfo(info->to_vector)->pyname
        );
        return data;
    }
    for (int i = 0; i < info->is_vectorx; ++i) {
        PyObject *item = PyTuple_GET_ITEM(obj, i);
        // get the max multi-vec and let the compiler do the rest
        data.vu64x3.val[i] = PySIMDVector_AsData((PySIMDVectorObject*)item, info->to_vector).vu64;
        if (PyErr_Occurred()) {
            return data;
        }
    }
    return data;
}

static PyObject *
simd_vectorx_to_tuple(simd_data data, simd_data_type dtype)
{
    const simd_data_info *info = simd_data_getinfo(dtype);
    // NPYV currently only supports x2 and x3
    assert(info->is_vectorx > 1 && info->is_vectorx < 4);

    PyObject *tuple = PyTuple_New(info->is_vectorx);
    if (tuple == NULL) {
        return NULL;
    }
    for (int i = 0; i < info->is_vectorx; ++i) {
        // get the max multi-vector and let the compiler handle the rest
        simd_data vdata = {.vu64 = data.vu64x3.val[i]};
        PyObject *item = (PyObject*)PySIMDVector_FromData(vdata, info->to_vector);
        if (item == NULL) {
            // TODO: improve log add item number
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, i, item);
    }
    return tuple;
}
