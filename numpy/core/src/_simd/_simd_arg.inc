/**
 * This file is included by `_simd.dispatch.c.src`. Its contents are affected by the simd configuration, and
 * therefore must be built multiple times. Making it a standalone `.c` file with `NPY_VISIBILITY_HIDDEN`
 * symbols would require judicious use of `NPY_CPU_DISPATCH_DECLARE` and `NPY_CPU_DISPATCH_CURFX`, which was
 * deemed too harmful to readability.
 */
/************************************
 ** Protected Definitions
 ************************************/
static int
simd_arg_from_obj(PyObject *obj, simd_arg *arg)
{
    assert(arg->dtype != 0);
    const simd_data_info *info = simd_data_getinfo(arg->dtype);
    if (info->is_scalar) {
        arg->data = simd_scalar_from_number(obj, arg->dtype);
    }
    else if (info->is_sequence) {
        unsigned min_seq_size = simd_data_getinfo(info->to_vector)->nlanes;
        arg->data.qu8 = simd_sequence_from_iterable(obj, arg->dtype, min_seq_size);
    }
    else if (info->is_vectorx) {
        arg->data = simd_vectorx_from_tuple(obj, arg->dtype);
    }
    else if (info->is_vector) {
        arg->data = PySIMDVector_AsData((PySIMDVectorObject*)obj, arg->dtype);
    } else {
        arg->data.u64 = 0;
        PyErr_Format(PyExc_RuntimeError,
            "unhandled arg from obj type id:%d, name:%s", arg->dtype, info->pyname
        );
        return -1;
    }
    if (PyErr_Occurred()) {
        return -1;
    }
    return 0;
}

static PyObject *
simd_arg_to_obj(const simd_arg *arg)
{
    assert(arg->dtype != 0);
    const simd_data_info *info = simd_data_getinfo(arg->dtype);
    if (info->is_scalar) {
        return simd_scalar_to_number(arg->data, arg->dtype);
    }
    if (info->is_sequence) {
        return simd_sequence_to_list(arg->data.qu8, arg->dtype);
    }
    if (info->is_vectorx) {
        return simd_vectorx_to_tuple(arg->data, arg->dtype);
    }
    if (info->is_vector) {
        return (PyObject*)PySIMDVector_FromData(arg->data, arg->dtype);
    }
    PyErr_Format(PyExc_RuntimeError,
        "unhandled arg to object type id:%d, name:%s", arg->dtype, info->pyname
    );
    return NULL;
}

static void
simd_arg_free(simd_arg *arg)
{
    const simd_data_info *info = simd_data_getinfo(arg->dtype);
    if (info->is_sequence) {
        simd_sequence_free(arg->data.qu8);
    }
}

static int
simd_arg_converter(PyObject *obj, simd_arg *arg)
{
    if (obj != NULL) {
        if (simd_arg_from_obj(obj, arg) < 0) {
            return 0;
        }
        arg->obj = obj;
        return Py_CLEANUP_SUPPORTED;
    } else {
        simd_arg_free(arg);
    }
    return 1;
}
